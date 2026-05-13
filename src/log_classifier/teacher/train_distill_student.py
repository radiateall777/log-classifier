import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from log_classifier.teacher.data import ClassificationDataset, read_dataset
from log_classifier.teacher.model import CodeBERTClassifier
from log_classifier.teacher.train_stage1_ce import (
    evaluate,
    forward_ce_only,
    optimizer_step_with_amp,
    save_best_checkpoint,
)
from log_classifier.teacher.utils import (
    ensure_dir,
    get_device,
    load_label_mapping,
    load_yaml,
    save_json,
    save_label_mapping,
    set_seed,
)


def collate_fn_distill(
    batch,
    teacher_tokenizer,
    student_tokenizer,
    teacher_max_length,
    student_max_length,
):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    teacher_inputs = teacher_tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=teacher_max_length,
        return_tensors="pt",
    )
    student_inputs = student_tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=student_max_length,
        return_tensors="pt",
    )

    collated = {
        "teacher_input_ids": teacher_inputs["input_ids"],
        "teacher_attention_mask": teacher_inputs["attention_mask"],
        "student_input_ids": student_inputs["input_ids"],
        "student_attention_mask": student_inputs["attention_mask"],
        "labels": labels,
    }

    return collated


def tokenize_texts(tokenizer, texts, max_length):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


def collate_fn_student_eval(batch, student_tokenizer, student_max_length):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    encodings = student_tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=student_max_length,
        return_tensors="pt",
    )

    return {
        "ids": [item["id"] for item in batch],
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }


def load_classifier_from_checkpoint(model_name, checkpoint_dir, num_labels, device, dropout_prob=0.1):
    model = CodeBERTClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout_prob=dropout_prob,
    )
    state_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    return model.to(device)


def forward_with_features(model, input_ids, attention_mask, labels):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_features=True,
    )


def forward_with_hidden_states(model, input_ids, attention_mask, labels):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_features=True,
        return_hidden_states=True,
    )


def kd_kl_loss(student_logits, teacher_logits, temperature):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)


def symmetric_consistency_loss(clean_logits, noisy_logits, temperature):
    clean_probs = F.softmax(clean_logits.detach() / temperature, dim=-1)
    noisy_probs = F.softmax(noisy_logits / temperature, dim=-1)
    clean_to_noisy = F.kl_div(
        F.log_softmax(noisy_logits / temperature, dim=-1),
        clean_probs,
        reduction="batchmean",
    )
    noisy_to_clean = F.kl_div(
        F.log_softmax(clean_logits / temperature, dim=-1),
        noisy_probs.detach(),
        reduction="batchmean",
    )
    return 0.5 * (clean_to_noisy + noisy_to_clean) * (temperature**2)


def feature_alignment_loss(student_features, teacher_features):
    student_features = F.normalize(student_features, dim=-1)
    teacher_features = F.normalize(teacher_features, dim=-1)
    return F.mse_loss(student_features, teacher_features)


def hidden_state_alignment_loss(student_hidden_states, teacher_hidden_states):
    if not student_hidden_states or not teacher_hidden_states:
        raise ValueError("hidden states are required for hidden-state distillation")

    student_layers = list(student_hidden_states[1:])
    teacher_layers = list(teacher_hidden_states[1:])

    num_student_layers = len(student_layers)
    num_teacher_layers = len(teacher_layers)
    if num_student_layers == 0 or num_teacher_layers == 0:
        return teacher_layers[-1].new_tensor(0.0)

    mapped_teacher_indices = torch.linspace(
        0,
        num_teacher_layers - 1,
        steps=num_student_layers,
    ).round().long().tolist()

    losses = []
    for student_idx, teacher_idx in enumerate(mapped_teacher_indices):
        student_state = F.normalize(student_layers[student_idx], dim=-1)
        teacher_state = F.normalize(teacher_layers[teacher_idx].detach(), dim=-1)
        losses.append(F.mse_loss(student_state, teacher_state))

    return torch.stack(losses).mean()


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature,
    ce_weight,
    kd_weight,
    label_smoothing=0.0,
    student_features=None,
    teacher_features=None,
    projected_student_features=None,
    feature_weight=0.0,
    student_hidden_states=None,
    teacher_hidden_states=None,
    hidden_state_weight=0.0,
):
    ce_loss = F.cross_entropy(student_logits, labels, label_smoothing=float(label_smoothing))
    kd_loss = kd_kl_loss(student_logits, teacher_logits, temperature)
    if feature_weight > 0 and teacher_features is not None:
        features_for_loss = projected_student_features
        if features_for_loss is None:
            features_for_loss = student_features
        feature_loss = feature_alignment_loss(features_for_loss, teacher_features)
    else:
        feature_loss = student_logits.new_tensor(0.0)

    if hidden_state_weight > 0 and student_hidden_states is not None and teacher_hidden_states is not None:
        hidden_state_loss = hidden_state_alignment_loss(student_hidden_states, teacher_hidden_states)
    else:
        hidden_state_loss = student_logits.new_tensor(0.0)

    loss = (
        ce_weight * ce_loss
        + kd_weight * kd_loss
        + feature_weight * feature_loss
        + hidden_state_weight * hidden_state_loss
    )
    return loss, {
        "ce_loss": ce_loss.detach(),
        "kd_loss": kd_loss.detach(),
        "feature_loss": feature_loss.detach(),
        "hidden_state_loss": hidden_state_loss.detach(),
    }


def rdrop_loss(student_logits_a, student_logits_b, temperature):
    return symmetric_consistency_loss(
        clean_logits=student_logits_a,
        noisy_logits=student_logits_b,
        temperature=temperature,
    )


def infer_hidden_size(model):
    return int(model.encoder.config.hidden_size)


def infer_feature_size(model):
    return int(getattr(model, "feature_dim", infer_hidden_size(model)))


def build_feature_projector(student, teacher, device):
    student_feature_size = infer_feature_size(student)
    teacher_feature_size = infer_feature_size(teacher)
    if student_feature_size == teacher_feature_size:
        return nn.Identity().to(device)
    return nn.Linear(student_feature_size, teacher_feature_size).to(device)


def find_transformer_layers(encoder):
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return encoder.encoder.layer
    if hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layer"):
        return encoder.transformer.layer
    if hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layers"):
        return encoder.transformer.layers
    return None


def truncate_student_layers(student, keep_layers):
    if keep_layers is None:
        return

    layers = find_transformer_layers(student.encoder)
    if layers is None:
        raise ValueError("Could not find transformer layers to truncate for this student model.")

    original_layers = len(layers)
    keep_layers = int(keep_layers)
    if keep_layers <= 0 or keep_layers > original_layers:
        raise ValueError(f"student_keep_layers must be in [1, {original_layers}], got {keep_layers}.")

    if keep_layers == original_layers:
        print(f"Student keeps all {original_layers} transformer layers.")
        return

    selected_indices = torch.linspace(0, original_layers - 1, steps=keep_layers).round().long().tolist()
    selected_layers = [layers[index] for index in selected_indices]
    truncated_layers = nn.ModuleList(selected_layers)

    if hasattr(student.encoder, "encoder") and hasattr(student.encoder.encoder, "layer"):
        student.encoder.encoder.layer = truncated_layers
    elif hasattr(student.encoder, "transformer") and hasattr(student.encoder.transformer, "layer"):
        student.encoder.transformer.layer = truncated_layers
    else:
        student.encoder.transformer.layers = truncated_layers

    if hasattr(student.encoder.config, "num_hidden_layers"):
        student.encoder.config.num_hidden_layers = keep_layers
    if hasattr(student.encoder.config, "n_layers"):
        student.encoder.config.n_layers = keep_layers

    print(
        "Truncated student encoder layers: "
        f"{original_layers} -> {keep_layers} using indices {selected_indices}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml(args.config)

    set_seed(int(config.get("seed", 42)))
    device = get_device()

    output_dir = config["output_dir"]
    ensure_dir(output_dir)
    save_json(config, os.path.join(output_dir, "config_snapshot.json"))

    teacher_checkpoint_dir = config["teacher_checkpoint_dir"]
    teacher_model_name = config.get("teacher_model_name", "microsoft/unixcoder-base")
    student_model_name = config.get("student_model_name", "distilbert-base-uncased")

    print("Loading teacher label mapping...")
    id2label, label2id = load_label_mapping(os.path.join(teacher_checkpoint_dir, "label_mapping.json"))
    num_labels = len(label2id)
    labels_list = sorted(id2label.keys())

    save_label_mapping(
        id2label,
        label2id,
        os.path.join(output_dir, "label_mapping.json"),
    )

    print("Loading data...")
    train_data = read_dataset(
        config.get("data_path", "data/random_samples_splits.json"),
        split="train",
    )
    valid_data = read_dataset(
        config.get("data_path", "data/random_samples_splits.json"),
        split="dev",
    )

    train_dataset = ClassificationDataset(train_data, label2id)
    valid_dataset = ClassificationDataset(valid_data, label2id)

    print("Loading tokenizers...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_checkpoint_dir)
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_tokenizer.save_pretrained(output_dir)

    teacher_max_length = int(config.get("teacher_max_length", config.get("max_length", 512)))
    student_max_length = int(config.get("student_max_length", config.get("max_length", 384)))
    use_rdrop = bool(config.get("use_rdrop", True))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=lambda b: collate_fn_distill(
            b,
            teacher_tokenizer,
            student_tokenizer,
            teacher_max_length,
            student_max_length,
        ),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config.get("eval_batch_size", config["batch_size"])),
        shuffle=False,
        collate_fn=lambda b: collate_fn_student_eval(
            b,
            student_tokenizer,
            student_max_length,
        ),
    )

    print(f"Loading teacher: {teacher_model_name}")
    teacher = load_classifier_from_checkpoint(
        model_name=teacher_model_name,
        checkpoint_dir=teacher_checkpoint_dir,
        num_labels=num_labels,
        device=device,
        dropout_prob=float(config.get("teacher_dropout_prob", 0.1)),
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"Initializing student: {student_model_name}")
    student = CodeBERTClassifier(
        model_name=student_model_name,
        num_labels=num_labels,
        dropout_prob=float(config.get("dropout_prob", 0.1)),
        pooling_mode=config.get("pooling_mode", "cls"),
        classifier_hidden_dim=int(config.get("classifier_hidden_dim", 0)),
        multi_sample_dropout_num=int(config.get("multi_sample_dropout_num", 1)),
    ).to(device)
    truncate_student_layers(student, config.get("student_keep_layers"))
    feature_projector = build_feature_projector(student, teacher, device)

    trainable_parameters = list(student.parameters())
    trainable_parameters.extend(
        param for param in feature_projector.parameters() if param.requires_grad
    )
    optimizer = AdamW(
        trainable_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )

    epochs = int(config["epochs"])
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
    total_steps = math.ceil(len(train_loader) / grad_accum_steps) * epochs
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.1)))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_fp16 = bool(config.get("fp16", False) and device.type == "cuda")
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler("cuda", enabled=use_fp16)

    temperature = float(config.get("temperature", 2.0))
    ce_weight = float(config.get("ce_weight", 0.45))
    kd_weight = float(config.get("kd_weight", 0.45))
    feature_weight = float(config.get("feature_weight", 0.1))
    hidden_state_weight = float(config.get("hidden_state_weight", 0.15))
    label_smoothing = float(config.get("label_smoothing", 0.05))
    rdrop_weight = float(config.get("rdrop_weight", 0.2))

    best_macro_f1 = -1.0

    print("Starting teacher-student distillation...")
    print(
        f"epochs={epochs}, total_steps={total_steps}, warmup_steps={warmup_steps}, "
        f"temperature={temperature}, ce_weight={ce_weight}, kd_weight={kd_weight}, "
        f"feature_weight={feature_weight}, use_rdrop={use_rdrop}, "
        f"hidden_state_weight={hidden_state_weight}, "
        f"label_smoothing={label_smoothing}, rdrop_weight={rdrop_weight}"
    )

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        total_feature_loss = 0.0
        total_hidden_state_loss = 0.0
        total_rdrop_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f"Distill Epoch {epoch}/{epochs}")

        for step, batch in enumerate(progress_bar):
            teacher_input_ids = batch["teacher_input_ids"].to(device)
            teacher_attention_mask = batch["teacher_attention_mask"].to(device)
            student_input_ids = batch["student_input_ids"].to(device)
            student_attention_mask = batch["student_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.inference_mode():
                teacher_outputs = forward_with_hidden_states(
                    teacher,
                    teacher_input_ids,
                    teacher_attention_mask,
                    labels,
                )
                teacher_logits = teacher_outputs["logits"]
                teacher_features = teacher_outputs["features"]
                teacher_hidden_states = teacher_outputs["hidden_states"]

            with autocast(device_type=amp_device_type, enabled=use_fp16):
                student_outputs = forward_with_hidden_states(
                    student,
                    student_input_ids,
                    student_attention_mask,
                    labels,
                )
                student_logits = student_outputs["logits"]
                student_features = student_outputs["features"]
                student_hidden_states = student_outputs["hidden_states"]
                projected_student_features = feature_projector(student_features)

                if use_rdrop:
                    student_outputs_b = forward_ce_only(
                        student,
                        student_input_ids,
                        student_attention_mask,
                        labels,
                    )
                    student_logits_b = student_outputs_b["logits"]
                    rdrop_term = rdrop_loss(
                        student_logits_a=student_logits,
                        student_logits_b=student_logits_b,
                        temperature=temperature,
                    )
                else:
                    rdrop_term = student_logits.new_tensor(0.0)

                clean_loss, clean_parts = distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    temperature=temperature,
                    ce_weight=ce_weight,
                    kd_weight=kd_weight,
                    label_smoothing=label_smoothing,
                    student_features=student_features,
                    teacher_features=teacher_features,
                    projected_student_features=projected_student_features,
                    feature_weight=feature_weight,
                    student_hidden_states=student_hidden_states,
                    teacher_hidden_states=teacher_hidden_states,
                    hidden_state_weight=hidden_state_weight,
                )
                loss = clean_loss + rdrop_weight * rdrop_term
                loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()

            total_loss += loss.item()
            total_ce_loss += clean_parts["ce_loss"].item()
            total_kd_loss += clean_parts["kd_loss"].item()
            total_feature_loss += clean_parts["feature_loss"].item()
            total_hidden_state_loss += clean_parts["hidden_state_loss"].item()
            total_rdrop_loss += rdrop_term.item()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer_step_with_amp(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    use_fp16=use_fp16,
                )

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{clean_parts['ce_loss'].item():.4f}",
                    "kd": f"{clean_parts['kd_loss'].item():.4f}",
                    "feat": f"{clean_parts['feature_loss'].item():.4f}",
                    "hid": f"{clean_parts['hidden_state_loss'].item():.4f}",
                    "rdrop": f"{rdrop_term.item():.4f}",
                }
            )

        train_metrics = {
            "train_loss": total_loss / max(len(train_loader), 1),
            "train_ce_loss": total_ce_loss / max(len(train_loader), 1),
            "train_kd_loss": total_kd_loss / max(len(train_loader), 1),
            "train_feature_loss": total_feature_loss / max(len(train_loader), 1),
            "train_hidden_state_loss": total_hidden_state_loss / max(len(train_loader), 1),
            "train_rdrop_loss": total_rdrop_loss / max(len(train_loader), 1),
            "epoch": epoch,
        }

        val_metrics, dev_logits, dev_labels = evaluate(
            student,
            valid_loader,
            device,
            labels_list,
            id2label,
        )
        val_metrics.update(train_metrics)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {val_metrics['train_loss']:.4f} | "
            f"CE: {val_metrics['train_ce_loss']:.4f} | "
            f"KD: {val_metrics['train_kd_loss']:.4f} | "
            f"Feat: {val_metrics['train_feature_loss']:.4f} | "
            f"Hidden: {val_metrics['train_hidden_state_loss']:.4f} | "
            f"RDrop: {val_metrics['train_rdrop_loss']:.4f} | "
            f"Eval Loss: {val_metrics['eval_loss']:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f}"
        )

        save_json(val_metrics, os.path.join(output_dir, f"eval_epoch_{epoch}.json"))

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_best_checkpoint(
                model=student,
                tokenizer=student_tokenizer,
                output_dir=output_dir,
                id2label=id2label,
                label2id=label2id,
                config=config,
                eval_metrics=val_metrics,
                dev_logits=dev_logits,
                dev_labels=dev_labels,
            )
            print(f"-> Saved new best student with Macro F1: {best_macro_f1:.4f}")

    print("Distillation complete.")
    print(f"Best student Macro F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
