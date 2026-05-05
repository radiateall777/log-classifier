import argparse
import json
import math
import os
import shutil
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from log_classifier.teacher.data import ClassificationDataset, read_dataset
from log_classifier.teacher.metrics import compute_classification_metrics
from log_classifier.teacher.model import CodeBERTClassifier
from log_classifier.teacher.utils import ensure_dir, get_device, load_yaml, save_json, set_seed

CONFUSION_NAMES = ["动态规划", "排序算法", "搜索算法"]
CONFUSION_PAIR_NAMES = [
    ("动态规划", "搜索算法"),
    ("动态规划", "排序算法"),
    ("搜索算法", "排序算法"),
]


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum(dim=1).mean()


def collate_fn(batch, tokenizer, max_length):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    ids = [item.get("id") for item in batch]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    result = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }
    if any(x is not None for x in ids):
        result["ids"] = ids
    return result


def normalize_id2label(id2label: dict[Any, Any]) -> dict[int, str]:
    return {int(k): str(v) for k, v in id2label.items()}


def normalize_label2id(label2id: dict[Any, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in label2id.items()}


def load_label_mapping(path: str) -> tuple[dict[str, int], dict[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "label2id" in data and "id2label" in data:
        label2id = normalize_label2id(data["label2id"])
        id2label = normalize_id2label(data["id2label"])
    elif isinstance(data, dict):
        # Fallback for a raw {"0": "label"} / {0: "label"} mapping.
        try:
            id2label = normalize_id2label(data)
            label2id = {label: idx for idx, label in id2label.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unsupported label mapping format in {path}. Expected "
                "{'label2id': ..., 'id2label': ...} or {id: label}."
            ) from exc
    else:
        raise ValueError(f"Unsupported label mapping format in {path}: {type(data)}")

    if set(label2id.values()) != set(id2label.keys()):
        raise ValueError(
            f"label2id/id2label are inconsistent in {path}: "
            f"label ids={sorted(label2id.values())}, id2label keys={sorted(id2label.keys())}"
        )
    return label2id, id2label


def save_label_mapping_snapshot(label2id: dict[str, int], id2label: dict[int, str], path: str) -> None:
    save_json(
        {
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
        },
        path,
    )


def validate_confusion_labels(label2id: dict[str, int]) -> None:
    missing = [name for name in CONFUSION_NAMES if name not in label2id]
    if missing:
        raise ValueError(
            "Cannot compute confusion pair margin loss because label_mapping is missing "
            f"required labels: {missing}. Available labels: {sorted(label2id.keys())}"
        )


def confusion_pair_margin_loss(logits, labels, label2id, margin=0.5):
    validate_confusion_labels(label2id)
    confusion_ids = torch.tensor([label2id[name] for name in CONFUSION_NAMES], device=logits.device)

    mask = (labels.unsqueeze(1) == confusion_ids.unsqueeze(0)).any(dim=1)
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    logits_sub = logits[mask]
    labels_sub = labels[mask]

    true_logits = logits_sub.gather(1, labels_sub.unsqueeze(1)).squeeze(1)

    confusion_logits = logits_sub[:, confusion_ids].clone()
    for i, label in enumerate(labels_sub):
        local_idx = (confusion_ids == label).nonzero(as_tuple=True)[0]
        confusion_logits[i, local_idx] = -1e4

    max_confuser_logits = confusion_logits.max(dim=1).values
    return F.relu(margin - (true_logits - max_confuser_logits)).mean()


def get_transformer_layers(model: CodeBERTClassifier):
    candidate_paths = [
        ("model.encoder.encoder.layer", lambda m: m.encoder.encoder.layer),
        ("model.encoder.roberta.encoder.layer", lambda m: m.encoder.roberta.encoder.layer),
        ("model.encoder.base_model.encoder.layer", lambda m: m.encoder.base_model.encoder.layer),
    ]
    for path, getter in candidate_paths:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                print(f"Found transformer layers at {path}, num_layers={len(layers)}")
                return layers
        except AttributeError:
            continue
    raise RuntimeError(
        "Cannot locate transformer layers for partial encoder fine-tuning. Tried: "
        + ", ".join(path for path, _ in candidate_paths)
    )


def freeze_encoder(model: CodeBERTClassifier) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_last_n_encoder_layers(model: CodeBERTClassifier, n_layers: int) -> None:
    freeze_encoder(model)
    layers = get_transformer_layers(model)
    if n_layers <= 0:
        print("unfreeze_last_n_layers <= 0, encoder remains frozen.")
        return
    n_layers = min(int(n_layers), len(layers))
    for layer in layers[-n_layers:]:
        for p in layer.parameters():
            p.requires_grad = True
    print(f"Unfroze last {n_layers}/{len(layers)} transformer layers.")


def set_head_trainable(model: CodeBERTClassifier) -> None:
    for module in [model.classifier, model.metric_projector, model.metric_head]:
        for p in module.parameters():
            p.requires_grad = True


def trainable_params(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad]


def build_optimizer_and_scheduler(model, center_loss_fn, train_loader, config, epochs):
    encoder_trainable_params = trainable_params(model.encoder)
    param_groups = []

    if encoder_trainable_params:
        param_groups.append({"params": encoder_trainable_params, "lr": float(config.get("learning_rate_encoder", 1e-6))})

    classifier_params = trainable_params(model.classifier)
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": float(config.get("learning_rate_classifier", 5e-6))})

    metric_projector_params = trainable_params(model.metric_projector)
    if metric_projector_params:
        param_groups.append({"params": metric_projector_params, "lr": float(config.get("learning_rate_metric_head", 1e-4))})

    metric_head_params = trainable_params(model.metric_head)
    if metric_head_params:
        param_groups.append({"params": metric_head_params, "lr": float(config.get("learning_rate_metric_head", 1e-4))})

    center_params = trainable_params(center_loss_fn)
    if center_params:
        param_groups.append({"params": center_params, "lr": float(config.get("learning_rate_center", 1e-3))})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")

    optimizer = AdamW(param_groups, weight_decay=float(config.get("weight_decay", 0.01)))
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
    total_steps = math.ceil(len(train_loader) / grad_accum_steps) * int(epochs)
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.0)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def load_stage1_weights(model: CodeBERTClassifier, checkpoint_dir: str, device: torch.device) -> None:
    checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    state_obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_obj, dict) and "model_state_dict" in state_obj:
        state_dict = state_obj["model_state_dict"]
    else:
        state_dict = state_obj

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    print(f"Loaded Stage 1 checkpoint from {checkpoint_path} with strict=False")
    print(f"missing_keys: {missing_keys}")
    print(f"unexpected_keys: {unexpected_keys}")

    allowed_missing_prefixes = ("metric_projector.", "metric_head.")
    suspicious_missing = [k for k in missing_keys if not k.startswith(allowed_missing_prefixes)]
    if suspicious_missing:
        raise RuntimeError(
            "Stage 1 checkpoint did not load cleanly into encoder/classifier. "
            f"Unexpected missing keys outside new metric modules: {suspicious_missing}"
        )


def save_best_checkpoint(model, center_loss_fn, tokenizer, label2id, id2label, config, eval_results, output_dir):
    best_dir = os.path.join(output_dir, "best")
    ensure_dir(best_dir)
    tokenizer.save_pretrained(best_dir)
    save_label_mapping_snapshot(label2id, id2label, os.path.join(best_dir, "label_mapping.json"))
    torch.save(model.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "center_loss_state_dict": center_loss_fn.state_dict(),
            "config": config,
        },
        os.path.join(best_dir, "checkpoint.pt"),
    )
    save_json(eval_results, os.path.join(best_dir, "eval_results.json"))
    if "embedding_diagnostics" in eval_results:
        save_json(eval_results["embedding_diagnostics"], os.path.join(best_dir, "embedding_diagnostics.json"))


def compute_embedding_diagnostics(features: torch.Tensor, labels: torch.Tensor, id2label: dict[int, str], label2id: dict[str, int]):
    features = F.normalize(features.float(), p=2, dim=-1)
    labels = labels.long()

    center_by_id: dict[int, torch.Tensor] = {}
    valid_label_ids = []
    valid_label_names = []
    intra_distances = []

    for class_id in sorted(id2label.keys()):
        mask = labels == class_id
        if not mask.any():
            continue
        class_features = features[mask]
        center = F.normalize(class_features.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
        center_by_id[class_id] = center
        valid_label_ids.append(class_id)
        valid_label_names.append(id2label[class_id])

        cosine_to_center = (class_features * center.unsqueeze(0)).sum(dim=1)
        intra_distances.append(1.0 - cosine_to_center)

    if not center_by_id:
        return {
            "mean_intra_cos_dist": None,
            "mean_inter_cos_dist": None,
            "inter_intra_ratio": None,
            "center_labels": [],
            "center_cosine_matrix": [],
            "center_cosine_matrix_with_labels": {},
            "confusion_pair_center_cosine": {},
        }

    intra_all = torch.cat(intra_distances) if intra_distances else torch.empty(0)
    mean_intra = float(intra_all.mean().item()) if intra_all.numel() > 0 else None

    centers = torch.stack([center_by_id[class_id] for class_id in valid_label_ids], dim=0)
    center_cos = centers @ centers.t()

    if len(valid_label_ids) > 1:
        upper_mask = torch.triu(torch.ones_like(center_cos, dtype=torch.bool), diagonal=1)
        inter_dist = 1.0 - center_cos[upper_mask]
        mean_inter = float(inter_dist.mean().item())
    else:
        mean_inter = None

    ratio = None
    if mean_intra is not None and mean_inter is not None and mean_intra > 1e-12:
        ratio = mean_inter / mean_intra

    matrix = center_cos.detach().cpu().tolist()
    matrix_with_labels = {
        row_label: {col_label: float(matrix[i][j]) for j, col_label in enumerate(valid_label_names)}
        for i, row_label in enumerate(valid_label_names)
    }

    pair_cosine = {}
    for a, b in CONFUSION_PAIR_NAMES:
        if a not in label2id or b not in label2id:
            pair_cosine[f"{a} vs {b}"] = None
            continue
        a_id = label2id[a]
        b_id = label2id[b]
        if a_id not in center_by_id or b_id not in center_by_id:
            pair_cosine[f"{a} vs {b}"] = None
            continue
        pair_cosine[f"{a} vs {b}"] = float((center_by_id[a_id] * center_by_id[b_id]).sum().item())

    return {
        "mean_intra_cos_dist": mean_intra,
        "mean_inter_cos_dist": mean_inter,
        "inter_intra_ratio": ratio,
        "center_labels": valid_label_names,
        "center_cosine_matrix": matrix,
        "center_cosine_matrix_with_labels": matrix_with_labels,
        "confusion_pair_center_cosine": pair_cosine,
    }


def prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def evaluate_metric(model, dataloader, device, labels_list, id2label, label2id, alpha=0.5):
    model.eval()
    all_labels = []
    all_ce_preds = []
    all_metric_preds = []
    all_fusion_preds = []
    all_features = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids,
                attention_mask,
                labels=None,
                return_features=True,
                return_metric=True,
                metric_labels=None,  # no CosFace margin during evaluation
            )
            ce_logits = outputs["logits"]
            metric_logits = outputs["metric_logits"]
            metric_features = outputs["metric_features"]
            final_logits = ce_logits + float(alpha) * metric_logits

            loss = F.cross_entropy(final_logits, labels)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_ce_preds.extend(torch.argmax(ce_logits, dim=-1).cpu().numpy().tolist())
            all_metric_preds.extend(torch.argmax(metric_logits, dim=-1).cpu().numpy().tolist())
            all_fusion_preds.extend(torch.argmax(final_logits, dim=-1).cpu().numpy().tolist())
            all_features.append(metric_features.detach().cpu())

    ce_metrics = compute_classification_metrics(all_labels, all_ce_preds, labels_list, id2label)
    metric_metrics = compute_classification_metrics(all_labels, all_metric_preds, labels_list, id2label)
    fusion_metrics = compute_classification_metrics(all_labels, all_fusion_preds, labels_list, id2label)

    features = torch.cat(all_features, dim=0) if all_features else torch.empty(0)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)
    diagnostics = compute_embedding_diagnostics(features, label_tensor, id2label, label2id)

    results = dict(fusion_metrics)
    results["eval_loss"] = total_loss / max(len(dataloader), 1)
    results["ce_only"] = ce_metrics
    results["metric_only"] = metric_metrics
    results["fusion"] = fusion_metrics
    results["embedding_diagnostics"] = diagnostics

    # Convenience flat keys for dashboards/logs.
    results["ce_accuracy"] = ce_metrics.get("accuracy")
    results["ce_macro_f1"] = ce_metrics.get("macro_f1")
    results["metric_accuracy"] = metric_metrics.get("accuracy")
    results["metric_macro_f1"] = metric_metrics.get("macro_f1")
    results["fusion_accuracy"] = fusion_metrics.get("accuracy")
    results["fusion_macro_f1"] = fusion_metrics.get("macro_f1")
    results["fusion_weighted_f1"] = fusion_metrics.get("weighted_f1")
    results["metric_logit_alpha"] = float(alpha)
    return results


def print_eval_summary(eval_results, title: str):
    diagnostics = eval_results.get("embedding_diagnostics", {})
    pair_cosine = diagnostics.get("confusion_pair_center_cosine", {})
    print(
        f"{title} | fusion macro_f1={eval_results.get('fusion_macro_f1', 0.0):.4f} | "
        f"ce macro_f1={eval_results.get('ce_macro_f1', 0.0):.4f} | "
        f"metric macro_f1={eval_results.get('metric_macro_f1', 0.0):.4f} | "
        f"inter_intra_ratio={diagnostics.get('inter_intra_ratio')}"
    )
    print(
        "Embedding diagnostics | "
        f"mean_intra_cos_dist={diagnostics.get('mean_intra_cos_dist')} | "
        f"mean_inter_cos_dist={diagnostics.get('mean_inter_cos_dist')} | "
        f"inter_intra_ratio={diagnostics.get('inter_intra_ratio')}"
    )
    for pair_name in ["动态规划 vs 搜索算法", "动态规划 vs 排序算法", "搜索算法 vs 排序算法"]:
        print(f"Center cosine {pair_name}: {pair_cosine.get(pair_name)}")


def train_one_phase(
    phase_name,
    model,
    center_loss_fn,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    config,
    labels_list,
    id2label,
    label2id,
    tokenizer,
    output_dir,
    best_score,
):
    epochs = int(config[phase_name])
    if epochs <= 0:
        return best_score

    phase_display_name = "head warmup" if phase_name == "epochs_head_warmup" else "partial encoder finetune"
    print(f"Starting Stage 2 {phase_display_name} for {epochs} epoch(s)...")

    grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
    use_fp16 = bool(config.get("fp16", False) and device.type == "cuda")
    lambda_metric_ce = float(config.get("lambda_metric_ce", 0.5))
    lambda_center = float(config.get("lambda_center", 0.01))
    lambda_pair_margin = float(config.get("lambda_pair_margin", 0.1))
    pair_margin = float(config.get("pair_margin", 0.5))
    alpha = float(config.get("metric_logit_alpha", 0.5))

    for epoch in range(1, epochs + 1):
        model.train()
        center_loss_fn.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss_meter = 0.0

        progress_bar = tqdm(train_loader, desc=f"{phase_display_name} epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=use_fp16):
                outputs = model(
                    input_ids,
                    attention_mask,
                    labels=None,
                    return_features=True,
                    return_metric=True,
                    metric_labels=labels,
                )

                ce_logits = outputs["logits"]
                metric_logits = outputs["metric_logits"]
                metric_features = outputs["metric_features"]

                ce_loss = F.cross_entropy(ce_logits, labels)
                metric_ce_loss = F.cross_entropy(metric_logits, labels)
                center_loss_value = center_loss_fn(metric_features, labels)
                pair_loss = confusion_pair_margin_loss(metric_logits, labels, label2id, margin=pair_margin)

                loss = (
                    ce_loss
                    + lambda_metric_ce * metric_ce_loss
                    + lambda_center * center_loss_value
                    + lambda_pair_margin * pair_loss
                )
                loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()
            total_loss_meter += loss.item()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix(
                {
                    "total_loss": f"{loss.item():.4f}",
                    "ce_loss": f"{ce_loss.item():.4f}",
                    "metric_ce": f"{metric_ce_loss.item():.4f}",
                    "center": f"{center_loss_value.item():.4f}",
                    "pair": f"{pair_loss.item():.4f}",
                }
            )

        avg_loss = total_loss_meter / max(len(train_loader), 1)
        eval_results = evaluate_metric(model, valid_loader, device, labels_list, id2label, label2id, alpha=alpha)
        eval_results["phase"] = phase_display_name
        eval_results["train_loss"] = avg_loss
        print_eval_summary(eval_results, title=f"Epoch done: {phase_display_name} {epoch}/{epochs}")

        epoch_results_path = os.path.join(
            output_dir,
            f"eval_{phase_display_name.replace(' ', '_')}_epoch_{epoch}.json",
        )
        save_json(eval_results, epoch_results_path)

        current_score = eval_results.get("fusion_macro_f1", 0.0) or 0.0
        if current_score > best_score:
            best_score = current_score
            save_best_checkpoint(
                model,
                center_loss_fn,
                tokenizer,
                label2id,
                id2label,
                config,
                eval_results,
                output_dir,
            )
            print(f"-> Saved new best Stage 2 model with fusion Macro F1: {best_score:.4f}")

    return best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))
    device = get_device()

    output_dir = config["output_dir"]
    ensure_dir(output_dir)
    save_json(config, os.path.join(output_dir, "stage2_config_snapshot.json"))

    label2id, id2label = load_label_mapping(config["label_mapping_path"])
    validate_confusion_labels(label2id)
    num_labels = len(label2id)
    labels_list = sorted(id2label.keys())

    print("Loading clean original data for Metric Space Refinement...")
    train_data = read_dataset(config.get("data_path", "data/random_samples_splits.json"), split="train")
    valid_data = read_dataset(config.get("data_path", "data/random_samples_splits.json"), split="dev")
    train_dataset = ClassificationDataset(train_data, label2id)
    valid_dataset = ClassificationDataset(valid_data, label2id)

    tokenizer = AutoTokenizer.from_pretrained(config["stage1_checkpoint_dir"])
    tokenizer.save_pretrained(output_dir)
    save_label_mapping_snapshot(label2id, id2label, os.path.join(output_dir, "label_mapping.json"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, int(config.get("max_length", 512))),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, int(config.get("max_length", 512))),
    )

    print("Initializing CodeBERTClassifier with metric branch...")
    model = CodeBERTClassifier(
        model_name=config.get("model_name", "microsoft/codebert-base"),
        num_labels=num_labels,
        dropout_prob=float(config.get("dropout_prob", 0.1)),
        metric_dim=int(config.get("metric_dim", 256)),
        cosface_scale=float(config.get("cosface_scale", 16.0)),
        cosface_margin=float(config.get("cosface_margin", 0.15)),
    ).to(device)
    load_stage1_weights(model, config["stage1_checkpoint_dir"], device)

    center_loss_fn = CenterLoss(num_classes=num_labels, feat_dim=int(config.get("metric_dim", 256))).to(device)
    use_fp16 = bool(config.get("fp16", False) and device.type == "cuda")
    scaler = GradScaler(enabled=use_fp16)

    alpha = float(config.get("metric_logit_alpha", 0.5))
    print("Stage 1 baseline before Stage 2...")
    baseline_results = evaluate_metric(model, valid_loader, device, labels_list, id2label, label2id, alpha=0.0)
    baseline_results["note"] = "Stage 1 CE-only baseline before metric refinement; fusion uses alpha=0.0 because metric head is newly initialized."
    save_json(baseline_results, os.path.join(output_dir, "baseline_eval_results.json"))
    print_eval_summary(baseline_results, title="Stage 1 baseline before Stage 2")

    # Initial best is the Stage 1 CE baseline, so best/ is never empty even if
    # fusion after Stage 2 does not beat the original classifier.
    best_score = baseline_results.get("ce_macro_f1", 0.0) or 0.0
    save_best_checkpoint(
        model,
        center_loss_fn,
        tokenizer,
        label2id,
        id2label,
        config,
        baseline_results,
        output_dir,
    )

    # Stage 2.1: head warmup. Encoder frozen; classifier, metric projector/head,
    # and centers are trainable.
    freeze_encoder(model)
    set_head_trainable(model)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        center_loss_fn,
        train_loader,
        config,
        epochs=int(config.get("epochs_head_warmup", 1)),
    )
    best_score = train_one_phase(
        "epochs_head_warmup",
        model,
        center_loss_fn,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        scaler,
        device,
        config,
        labels_list,
        id2label,
        label2id,
        tokenizer,
        output_dir,
        best_score,
    )

    # Stage 2.2: partial encoder fine-tuning. Rebuild optimizer/scheduler because
    # the trainable parameter set changes.
    unfreeze_last_n_encoder_layers(model, int(config.get("unfreeze_last_n_layers", 4)))
    set_head_trainable(model)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        center_loss_fn,
        train_loader,
        config,
        epochs=int(config.get("epochs_finetune", 3)),
    )
    best_score = train_one_phase(
        "epochs_finetune",
        model,
        center_loss_fn,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        scaler,
        device,
        config,
        labels_list,
        id2label,
        label2id,
        tokenizer,
        output_dir,
        best_score,
    )

    print(f"Stage 2 Metric Space Refinement complete. Best score threshold used: {best_score:.4f}")
    print(f"Best artifacts saved under: {os.path.join(output_dir, 'best')}")


if __name__ == "__main__":
    main()
