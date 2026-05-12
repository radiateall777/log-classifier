import argparse
import math
import os

import torch
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


def collate_fn_distill(batch, teacher_tokenizer, student_tokenizer, teacher_max_length, student_max_length):
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

    return {
        "teacher_input_ids": teacher_inputs["input_ids"],
        "teacher_attention_mask": teacher_inputs["attention_mask"],
        "student_input_ids": student_inputs["input_ids"],
        "student_attention_mask": student_inputs["attention_mask"],
        "labels": labels,
    }


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


def distillation_loss(student_logits, teacher_logits, labels, temperature, ce_weight, kd_weight):
    ce_loss = F.cross_entropy(student_logits, labels)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    loss = ce_weight * ce_loss + kd_weight * kd_loss
    return loss, ce_loss.detach(), kd_loss.detach()


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
    ).to(device)

    optimizer = AdamW(
        student.parameters(),
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
    ce_weight = float(config.get("ce_weight", 0.3))
    kd_weight = float(config.get("kd_weight", 0.7))

    best_macro_f1 = -1.0

    print("Starting teacher-student distillation...")
    print(
        f"epochs={epochs}, total_steps={total_steps}, warmup_steps={warmup_steps}, "
        f"temperature={temperature}, ce_weight={ce_weight}, kd_weight={kd_weight}"
    )

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f"Distill Epoch {epoch}/{epochs}")

        for step, batch in enumerate(progress_bar):
            teacher_input_ids = batch["teacher_input_ids"].to(device)
            teacher_attention_mask = batch["teacher_attention_mask"].to(device)
            student_input_ids = batch["student_input_ids"].to(device)
            student_attention_mask = batch["student_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.inference_mode():
                teacher_outputs = forward_ce_only(
                    teacher,
                    teacher_input_ids,
                    teacher_attention_mask,
                    labels,
                )
                teacher_logits = teacher_outputs["logits"]

            with autocast(device_type=amp_device_type, enabled=use_fp16):
                student_outputs = forward_ce_only(
                    student,
                    student_input_ids,
                    student_attention_mask,
                    labels,
                )
                student_logits = student_outputs["logits"]
                loss, ce_loss, kd_loss = distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    temperature=temperature,
                    ce_weight=ce_weight,
                    kd_weight=kd_weight,
                )
                loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_kd_loss += kd_loss.item()

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
                    "ce": f"{ce_loss.item():.4f}",
                    "kd": f"{kd_loss.item():.4f}",
                }
            )

        train_metrics = {
            "train_loss": total_loss / max(len(train_loader), 1),
            "train_ce_loss": total_ce_loss / max(len(train_loader), 1),
            "train_kd_loss": total_kd_loss / max(len(train_loader), 1),
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
