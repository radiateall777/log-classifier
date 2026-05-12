import argparse
import math
import os
import random
import re

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


class RobustTextAugmenter:
    """Small label-preserving perturbations for robust distillation."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.role_pattern = re.compile(
            r"\b(user|assistant|Objective|Task|Details|Example)\s*:\s*",
            re.IGNORECASE,
        )
        self.md_pattern = re.compile(r"```[a-zA-Z]*\n|```", re.IGNORECASE)
        self.line_comment_pattern = re.compile(r"//.*$", re.MULTILINE)
        self.block_comment_pattern = re.compile(r"/\*.*?\*/", re.DOTALL)
        self.python_comment_pattern = re.compile(r"#.*$", re.MULTILINE)
        self.noises = [
            "Example usage:",
            "The following is a simple implementation.",
            "Here is the code:",
            "Note:",
            "Solution:",
        ]
        self.augmentations = [
            self.remove_role_markers,
            self.remove_markdown_noise,
            self.remove_code_comments,
            self.normalize_whitespace,
            self.add_harmless_noise,
            self.truncate_assistant_explanation,
        ]

    def augment(self, text):
        aug_func = self.rng.choice(self.augmentations)
        try:
            augmented = aug_func(text)
            return augmented if augmented.strip() else text
        except Exception:
            return text

    def remove_role_markers(self, text):
        return self.role_pattern.sub(" ", text).strip()

    def remove_markdown_noise(self, text):
        return self.md_pattern.sub("", text).strip()

    def remove_code_comments(self, text):
        text = self.block_comment_pattern.sub("", text)
        text = self.line_comment_pattern.sub("", text)
        text = self.python_comment_pattern.sub("", text)
        return text.strip()

    def normalize_whitespace(self, text):
        return re.sub(r"[ \t]{2,}", " ", text)

    def add_harmless_noise(self, text):
        noise = self.rng.choice(self.noises)
        return f"{noise}\n{text}" if self.rng.random() > 0.5 else f"{text}\n{noise}"

    def truncate_assistant_explanation(self, text):
        parts = text.split("assistant:", 1)
        if len(parts) != 2:
            return text
        code_blocks = re.split(r"(```.*?```)", parts[1], flags=re.DOTALL)
        if len(code_blocks) < 3:
            return text
        return parts[0] + "assistant:" + "".join(code_blocks[:-1])


def collate_fn_distill(
    batch,
    teacher_tokenizer,
    student_tokenizer,
    teacher_max_length,
    student_max_length,
    augmenter=None,
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

    if augmenter is not None:
        noisy_texts = [augmenter.augment(text) for text in texts]
        teacher_noisy_inputs = tokenize_texts(teacher_tokenizer, noisy_texts, teacher_max_length)
        student_noisy_inputs = tokenize_texts(student_tokenizer, noisy_texts, student_max_length)
        collated.update(
            {
                "teacher_noisy_input_ids": teacher_noisy_inputs["input_ids"],
                "teacher_noisy_attention_mask": teacher_noisy_inputs["attention_mask"],
                "student_noisy_input_ids": student_noisy_inputs["input_ids"],
                "student_noisy_attention_mask": student_noisy_inputs["attention_mask"],
            }
        )

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


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature,
    ce_weight,
    kd_weight,
    student_features=None,
    teacher_features=None,
    projected_student_features=None,
    feature_weight=0.0,
):
    ce_loss = F.cross_entropy(student_logits, labels)
    kd_loss = kd_kl_loss(student_logits, teacher_logits, temperature)
    if feature_weight > 0 and teacher_features is not None:
        features_for_loss = projected_student_features
        if features_for_loss is None:
            features_for_loss = student_features
        feature_loss = feature_alignment_loss(features_for_loss, teacher_features)
    else:
        feature_loss = student_logits.new_tensor(0.0)

    loss = ce_weight * ce_loss + kd_weight * kd_loss + feature_weight * feature_loss
    return loss, {
        "ce_loss": ce_loss.detach(),
        "kd_loss": kd_loss.detach(),
        "feature_loss": feature_loss.detach(),
    }


def robust_distillation_loss(
    student_clean_logits,
    student_noisy_logits,
    teacher_noisy_logits,
    labels,
    temperature,
    robust_ce_weight,
    robust_kd_weight,
    consistency_weight,
):
    robust_ce_loss = F.cross_entropy(student_noisy_logits, labels)
    robust_kd_loss = kd_kl_loss(student_noisy_logits, teacher_noisy_logits, temperature)
    consistency_loss = symmetric_consistency_loss(
        clean_logits=student_clean_logits,
        noisy_logits=student_noisy_logits,
        temperature=temperature,
    )
    loss = (
        robust_ce_weight * robust_ce_loss
        + robust_kd_weight * robust_kd_loss
        + consistency_weight * consistency_loss
    )
    return loss, {
        "robust_ce_loss": robust_ce_loss.detach(),
        "robust_kd_loss": robust_kd_loss.detach(),
        "consistency_loss": consistency_loss.detach(),
    }


def infer_hidden_size(model):
    return int(model.encoder.config.hidden_size)


def build_feature_projector(student, teacher, device):
    student_hidden_size = infer_hidden_size(student)
    teacher_hidden_size = infer_hidden_size(teacher)
    if student_hidden_size == teacher_hidden_size:
        return nn.Identity().to(device)
    return nn.Linear(student_hidden_size, teacher_hidden_size).to(device)


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


def move_encoding_to_device(encodings, device):
    return {key: value.to(device) for key, value in encodings.items()}


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
    use_robust_distill = bool(config.get("robust_distill", True))
    augmenter = RobustTextAugmenter(seed=int(config.get("seed", 42))) if use_robust_distill else None

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
            augmenter,
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
    robust_ce_weight = float(config.get("robust_ce_weight", 0.15))
    robust_kd_weight = float(config.get("robust_kd_weight", 0.35))
    consistency_weight = float(config.get("consistency_weight", 0.15))

    best_macro_f1 = -1.0

    print("Starting teacher-student distillation...")
    print(
        f"epochs={epochs}, total_steps={total_steps}, warmup_steps={warmup_steps}, "
        f"temperature={temperature}, ce_weight={ce_weight}, kd_weight={kd_weight}, "
        f"feature_weight={feature_weight}, robust_distill={use_robust_distill}"
    )

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        total_feature_loss = 0.0
        total_robust_ce_loss = 0.0
        total_robust_kd_loss = 0.0
        total_consistency_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f"Distill Epoch {epoch}/{epochs}")

        for step, batch in enumerate(progress_bar):
            teacher_input_ids = batch["teacher_input_ids"].to(device)
            teacher_attention_mask = batch["teacher_attention_mask"].to(device)
            student_input_ids = batch["student_input_ids"].to(device)
            student_attention_mask = batch["student_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.inference_mode():
                teacher_outputs = forward_with_features(
                    teacher,
                    teacher_input_ids,
                    teacher_attention_mask,
                    labels,
                )
                teacher_logits = teacher_outputs["logits"]
                teacher_features = teacher_outputs["features"]

            with autocast(device_type=amp_device_type, enabled=use_fp16):
                student_outputs = forward_with_features(
                    student,
                    student_input_ids,
                    student_attention_mask,
                    labels,
                )
                student_logits = student_outputs["logits"]
                student_features = student_outputs["features"]
                projected_student_features = feature_projector(student_features)
                clean_loss, clean_parts = distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    temperature=temperature,
                    ce_weight=ce_weight,
                    kd_weight=kd_weight,
                    student_features=student_features,
                    teacher_features=teacher_features,
                    projected_student_features=projected_student_features,
                    feature_weight=feature_weight,
                )

                if use_robust_distill:
                    teacher_noisy_inputs = {
                        "input_ids": batch["teacher_noisy_input_ids"].to(device),
                        "attention_mask": batch["teacher_noisy_attention_mask"].to(device),
                    }
                    student_noisy_inputs = {
                        "input_ids": batch["student_noisy_input_ids"].to(device),
                        "attention_mask": batch["student_noisy_attention_mask"].to(device),
                    }
                    with torch.inference_mode():
                        teacher_noisy_outputs = forward_ce_only(
                            teacher,
                            teacher_noisy_inputs["input_ids"],
                            teacher_noisy_inputs["attention_mask"],
                            labels,
                        )
                        teacher_noisy_logits = teacher_noisy_outputs["logits"]

                    student_noisy_outputs = forward_ce_only(
                        student,
                        student_noisy_inputs["input_ids"],
                        student_noisy_inputs["attention_mask"],
                        labels,
                    )
                    robust_loss, robust_parts = robust_distillation_loss(
                        student_clean_logits=student_logits,
                        student_noisy_logits=student_noisy_outputs["logits"],
                        teacher_noisy_logits=teacher_noisy_logits,
                        labels=labels,
                        temperature=temperature,
                        robust_ce_weight=robust_ce_weight,
                        robust_kd_weight=robust_kd_weight,
                        consistency_weight=consistency_weight,
                    )
                else:
                    robust_loss = student_logits.new_tensor(0.0)
                    robust_parts = {
                        "robust_ce_loss": student_logits.new_tensor(0.0),
                        "robust_kd_loss": student_logits.new_tensor(0.0),
                        "consistency_loss": student_logits.new_tensor(0.0),
                    }

                loss = clean_loss + robust_loss
                loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()

            total_loss += loss.item()
            total_ce_loss += clean_parts["ce_loss"].item()
            total_kd_loss += clean_parts["kd_loss"].item()
            total_feature_loss += clean_parts["feature_loss"].item()
            total_robust_ce_loss += robust_parts["robust_ce_loss"].item()
            total_robust_kd_loss += robust_parts["robust_kd_loss"].item()
            total_consistency_loss += robust_parts["consistency_loss"].item()

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
                    "rob": f"{robust_loss.item():.4f}",
                }
            )

        train_metrics = {
            "train_loss": total_loss / max(len(train_loader), 1),
            "train_ce_loss": total_ce_loss / max(len(train_loader), 1),
            "train_kd_loss": total_kd_loss / max(len(train_loader), 1),
            "train_feature_loss": total_feature_loss / max(len(train_loader), 1),
            "train_robust_ce_loss": total_robust_ce_loss / max(len(train_loader), 1),
            "train_robust_kd_loss": total_robust_kd_loss / max(len(train_loader), 1),
            "train_consistency_loss": total_consistency_loss / max(len(train_loader), 1),
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
            f"RobustKD: {val_metrics['train_robust_kd_loss']:.4f} | "
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
