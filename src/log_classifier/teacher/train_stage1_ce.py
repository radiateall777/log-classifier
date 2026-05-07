import argparse
import math
import os

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from log_classifier.teacher.data import ClassificationDataset, build_label_mapping, read_dataset
from log_classifier.teacher.metrics import compute_classification_metrics
from log_classifier.teacher.model import CodeBERTClassifier
from log_classifier.teacher.utils import (
    ensure_dir,
    get_device,
    load_yaml,
    save_json,
    save_label_mapping,
    set_seed,
)


def collate_fn(batch, tokenizer, max_length):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }


def forward_ce_only(model, input_ids, attention_mask, labels):
    """
    Compatible with both old and extended CodeBERTClassifier.
    """
    try:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_features=False,
            return_metric=False,
            return_confusion=False,
            return_dp_search=False,
        )
    except TypeError:
        try:
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_features=False,
                return_metric=False,
                return_confusion=False,
            )
        except TypeError:
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_features=False,
                return_metric=False,
            )


def evaluate(model, dataloader, device, labels_list, id2label):
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = forward_ce_only(model, input_ids, attention_mask, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_logits.append(logits.detach().cpu())

    metrics = compute_classification_metrics(
        all_labels,
        all_preds,
        labels_list,
        id2label,
    )
    metrics["eval_loss"] = total_loss / max(len(dataloader), 1)

    logits_all = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0)
    labels_all = torch.tensor(all_labels, dtype=torch.long)

    return metrics, logits_all, labels_all


def optimizer_step_with_amp(optimizer, scheduler, scaler, use_fp16):
    if use_fp16:
        old_scale = scaler.get_scale()

        scaler.step(optimizer)
        scaler.update()

        new_scale = scaler.get_scale()

        if new_scale >= old_scale:
            scheduler.step()
    else:
        optimizer.step()
        scheduler.step()

    optimizer.zero_grad(set_to_none=True)


def save_best_checkpoint(
    model,
    tokenizer,
    output_dir,
    id2label,
    label2id,
    config,
    eval_metrics,
    dev_logits,
    dev_labels,
):
    best_dir = os.path.join(output_dir, "best")
    ensure_dir(best_dir)

    torch.save(model.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(best_dir)

    save_label_mapping(
        id2label,
        label2id,
        os.path.join(best_dir, "label_mapping.json"),
    )

    save_json(eval_metrics, os.path.join(best_dir, "eval_results.json"))
    save_json(config, os.path.join(best_dir, "config_snapshot.json"))

    torch.save(
        {
            "labels": dev_labels.cpu(),
            "logits": dev_logits.cpu(),
        },
        os.path.join(best_dir, "dev_logits.pt"),
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

    print("Loading data...")
    train_data = read_dataset(
        config.get("data_path", "data/random_samples_splits.json"),
        split="train",
    )
    valid_data = read_dataset(
        config.get("data_path", "data/random_samples_splits.json"),
        split="dev",
    )

    label2id, id2label = build_label_mapping(train_data)
    num_labels = len(label2id)
    labels_list = sorted(id2label.keys())

    save_label_mapping(
        id2label,
        label2id,
        os.path.join(output_dir, "label_mapping.json"),
    )

    train_dataset = ClassificationDataset(train_data, label2id)
    valid_dataset = ClassificationDataset(valid_data, label2id)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.save_pretrained(output_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b,
            tokenizer,
            int(config["max_length"]),
        ),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b,
            tokenizer,
            int(config["max_length"]),
        ),
    )

    print(f"Initializing CE-only classifier: {config['model_name']}")
    model = CodeBERTClassifier(
        model_name=config["model_name"],
        num_labels=num_labels,
        dropout_prob=float(config.get("dropout_prob", 0.1)),
        metric_dim=int(config.get("metric_dim", 256)),
        cosface_scale=float(config.get("cosface_scale", 16.0)),
        cosface_margin=float(config.get("cosface_margin", 0.15)),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
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

    best_macro_f1 = -1.0

    print("Starting CE-only training...")
    print(f"epochs={epochs}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=amp_device_type, enabled=use_fp16):
                outputs = forward_ce_only(model, input_ids, attention_mask, labels)
                loss = outputs["loss"]
                loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()
            total_train_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer_step_with_amp(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    use_fp16=use_fp16,
                )

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        val_metrics, dev_logits, dev_labels = evaluate(
            model,
            valid_loader,
            device,
            labels_list,
            id2label,
        )

        val_metrics["train_loss"] = avg_train_loss
        val_metrics["epoch"] = epoch

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Eval Loss: {val_metrics['eval_loss']:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f}"
        )

        save_json(val_metrics, os.path.join(output_dir, f"eval_epoch_{epoch}.json"))

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]

            save_best_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                id2label=id2label,
                label2id=label2id,
                config=config,
                eval_metrics=val_metrics,
                dev_logits=dev_logits,
                dev_labels=dev_labels,
            )

            print(f"-> Saved new best model with Macro F1: {best_macro_f1:.4f}")

    print("CE-only training complete.")
    print(f"Best Macro F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()