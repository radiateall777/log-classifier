import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from log_classifier.teacher.augment import TextCodeAugmenter
from log_classifier.teacher.data import ClassificationDataset, read_dataset
from log_classifier.teacher.hard_negative import compute_hard_scores, load_hard_weights, save_hard_weights, scores_to_weights
from log_classifier.teacher.losses import supervised_contrastive_loss, symmetric_kl_loss, weighted_cross_entropy
from log_classifier.teacher.model import CodeBERTClassifier
from log_classifier.teacher.train_stage1 import evaluate
from log_classifier.teacher.utils import ensure_dir, get_device, load_label_mapping, load_yaml, save_json, save_label_mapping, set_seed


def collate_fn_stage2(batch, tokenizer, max_length, augmenter):
    texts_orig = []
    texts_aug = []
    labels = []
    ids = []
    
    for item in batch:
        texts_orig.append(item["text"])
        # Generate 1 augmented view
        texts_aug.append(augmenter.augment(item["text"], n=1)[0])
        labels.append(item["label"])
        ids.append(item["id"])
        
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    enc_orig = tokenizer(texts_orig, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    enc_aug = tokenizer(texts_aug, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    
    return {
        "ids": ids,
        "input_ids": enc_orig["input_ids"],
        "attention_mask": enc_orig["attention_mask"],
        "input_ids_aug": enc_aug["input_ids"],
        "attention_mask_aug": enc_aug["attention_mask"],
        "labels": labels_tensor
    }


def generate_hard_weights(model, dataloader, device, output_path, gamma, max_weight):
    model.eval()
    all_features = []
    all_labels = []
    all_probs = []
    all_preds = []
    all_ids = []

    print("Generating hard negative weights from Stage 1 model...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, return_features=True)
            logits = outputs["logits"]
            features = outputs["features"]
            
            probs = F.softmax(logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(confidences.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_ids.extend(batch["ids"])

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    scores = compute_hard_scores(all_features, all_labels, all_probs, all_preds)
    weights = scores_to_weights(scores, gamma=gamma, max_weight=max_weight)
    
    save_hard_weights(output_path, all_ids, weights)
    print(f"Hard negative weights saved to {output_path}")


def collate_fn_eval(batch, tokenizer, max_length):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return {
        "ids": [item["id"] for item in batch],
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml(args.config)
    
    set_seed(config.get("seed", 42))
    device = get_device()
    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    stage1_checkpoint_dir = config["stage1_checkpoint_dir"]
    id2label, label2id = load_label_mapping(config["label_mapping_path"])
    num_labels = len(label2id)
    labels_list = list(id2label.keys())

    print("Loading data...")
    train_data = read_dataset(config.get("data_path", "data/random_samples_splits.json"), split="train")
    valid_data = read_dataset(config.get("data_path", "data/random_samples_splits.json"), split="dev")

    train_dataset = ClassificationDataset(train_data, label2id)
    valid_dataset = ClassificationDataset(valid_data, label2id)

    tokenizer = AutoTokenizer.from_pretrained(stage1_checkpoint_dir)
    tokenizer.save_pretrained(output_dir)
    save_label_mapping(id2label, label2id, os.path.join(output_dir, "label_mapping.json"))

    print("Initializing Stage 1 model...")
    model = CodeBERTClassifier(
        model_name=config["model_name"],
        num_labels=num_labels
    )
    model.load_state_dict(torch.load(os.path.join(stage1_checkpoint_dir, "pytorch_model.bin"), map_location="cpu"))
    model.to(device)

    hard_weights_path = config.get("hard_weights_path", os.path.join(output_dir, "hard_weights.json"))
    if not os.path.exists(hard_weights_path):
        train_loader_hn = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=lambda b: collate_fn_eval(b, tokenizer, config["max_length"])
        )
        generate_hard_weights(
            model, train_loader_hn, device, hard_weights_path,
            gamma=config.get("hard_gamma", 1.0),
            max_weight=config.get("hard_max_weight", 3.0)
        )

    hard_weights_dict = load_hard_weights(hard_weights_path)

    augmenter = TextCodeAugmenter(seed=config.get("seed", 42))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn_stage2(b, tokenizer, config["max_length"], augmenter)
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn_eval(b, tokenizer, config["max_length"])
    )

    optimizer = AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=config.get("weight_decay", 0.01))
    
    epochs = config["epochs"]
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    use_fp16 = config.get("fp16", False)
    scaler = GradScaler(enabled=use_fp16)

    lambda_scl = config.get("lambda_scl", 0.2)
    lambda_cons = config.get("lambda_consistency", 0.1)
    scl_temp = config.get("scl_temperature", 0.07)
    cons_temp = config.get("consistency_temperature", 1.0)

    baseline_metrics = evaluate(model, valid_loader, device, labels_list, id2label)
    best_macro_f1 = baseline_metrics["macro_f1"]

    print(
        f"Stage 1 baseline before Stage 2 | "
        f"Eval Loss: {baseline_metrics['eval_loss']:.4f} | "
        f"Macro F1: {baseline_metrics['macro_f1']:.4f} | "
        f"Acc: {baseline_metrics['accuracy']:.4f}"
    )

    print("Starting Stage 2 Robust Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_ids_aug = batch["input_ids_aug"].to(device)
            attention_mask_aug = batch["attention_mask_aug"].to(device)
            labels = batch["labels"].to(device)
            
            # Fetch sample weights
            batch_weights = torch.tensor(
                [hard_weights_dict.get(str(i), 1.0) for i in batch["ids"]],
                device=device
            )

            # Normalize weights to avoid amplifying the overall CE loss scale
            batch_weights = batch_weights / batch_weights.mean().clamp_min(1e-6)

            with autocast(enabled=use_fp16):
                outputs_orig = model(input_ids, attention_mask, return_features=True)
                outputs_aug = model(input_ids_aug, attention_mask_aug, return_features=True)
                
                ce_loss_orig = weighted_cross_entropy(outputs_orig["logits"], labels, batch_weights)
                ce_loss_aug = weighted_cross_entropy(outputs_aug["logits"], labels, batch_weights)
                
                features_all = torch.cat([outputs_orig["features"], outputs_aug["features"]], dim=0)
                labels_all = torch.cat([labels, labels], dim=0)
                scl_loss = supervised_contrastive_loss(features_all, labels_all, temperature=scl_temp)
                
                kl_loss = symmetric_kl_loss(outputs_orig["logits"], outputs_aug["logits"], temperature=cons_temp)
                
                loss = 0.5 * (ce_loss_orig + ce_loss_aug) \
                        + lambda_scl * scl_loss \
                        + lambda_cons * kl_loss

                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            total_train_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        # Validation
        val_metrics = evaluate(model, valid_loader, device, labels_list, id2label)
        print(f"Epoch {epoch} Eval Loss: {val_metrics['eval_loss']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_dir = os.path.join(output_dir, "best")
            ensure_dir(best_dir)
            torch.save(model.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(best_dir)
            save_json(val_metrics, os.path.join(best_dir, "eval_results.json"))
            print(f"-> Saved new best robust model with Macro F1: {best_macro_f1:.4f}")

    print("Stage 2 Training Complete.")

if __name__ == "__main__":
    main()
