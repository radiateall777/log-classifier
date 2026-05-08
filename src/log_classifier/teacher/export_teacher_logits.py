import argparse
import json
import os
from typing import Any

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from log_classifier.teacher.data import ClassificationDataset, read_dataset
from log_classifier.teacher.model import CodeBERTClassifier
from log_classifier.teacher.utils import ensure_dir, get_device, load_yaml, set_seed


def load_label_mapping(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "label2id" in data and "id2label" in data:
        label2id = {str(k): int(v) for k, v in data["label2id"].items()}
        id2label = {int(k): str(v) for k, v in data["id2label"].items()}
    elif isinstance(data, dict):
        # fallback: {"0": "Java Spring相关", ...}
        id2label = {int(k): str(v) for k, v in data.items()}
        label2id = {v: k for k, v in id2label.items()}
    else:
        raise ValueError(f"Unsupported label mapping format: {path}")

    return label2id, id2label


def collate_fn(batch, tokenizer, max_length: int):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)

    ids = []
    for i, item in enumerate(batch):
        ids.append(item.get("id", str(i)))

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
        "ids": ids,
        "texts": texts,
    }


def load_teacher_model(config: dict[str, Any], num_labels: int, device: torch.device):
    model = CodeBERTClassifier(
        model_name=config["model_name"],
        num_labels=num_labels,
        dropout_prob=float(config.get("dropout_prob", 0.1)),
    )

    checkpoint_path = os.path.join(
        config["teacher_checkpoint_dir"],
        "pytorch_model.bin",
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    incompatible = model.load_state_dict(state_dict, strict=False)

    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    print(f"Loaded teacher checkpoint from: {checkpoint_path}")
    print(f"missing_keys: {missing_keys}")
    print(f"unexpected_keys: {unexpected_keys}")

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def export_split_logits(
    model,
    tokenizer,
    dataset,
    split_name: str,
    output_dir: str,
    config: dict[str, Any],
    device: torch.device,
):
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 16)),
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b,
            tokenizer,
            int(config.get("max_length", 512)),
        ),
    )

    use_fp16 = bool(config.get("fp16", False) and device.type == "cuda")
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"

    all_ids = []
    all_texts = []
    all_labels = []
    all_logits = []

    model.eval()

    for batch in tqdm(dataloader, desc=f"Exporting {split_name} logits"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        with autocast(device_type=amp_device_type, enabled=use_fp16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                return_features=False,
            )

            logits = outputs["logits"]

        all_ids.extend(batch["ids"])
        all_texts.extend(batch["texts"])
        all_labels.append(labels.cpu())
        all_logits.append(logits.detach().cpu().float())

    labels_tensor = torch.cat(all_labels, dim=0)
    logits_tensor = torch.cat(all_logits, dim=0)
    probs_tensor = torch.softmax(logits_tensor, dim=-1)

    payload = {
        "split": split_name,
        "ids": all_ids,
        "texts": all_texts,
        "labels": labels_tensor,
        "logits": logits_tensor,
        "probs": probs_tensor,
    }

    output_path = os.path.join(
        output_dir,
        f"{split_name}_teacher_logits.pt",
    )

    torch.save(payload, output_path)

    print(f"Saved {split_name} teacher logits to: {output_path}")
    print(f"logits shape: {tuple(logits_tensor.shape)}")
    print(f"labels shape: {tuple(labels_tensor.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))

    device = get_device()

    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    label2id, id2label = load_label_mapping(config["label_mapping_path"])
    num_labels = len(label2id)

    print(f"Loaded label mapping from: {config['label_mapping_path']}")
    print(f"num_labels={num_labels}")
    print(f"id2label={id2label}")

    tokenizer = AutoTokenizer.from_pretrained(config["teacher_checkpoint_dir"])

    model = load_teacher_model(
        config=config,
        num_labels=num_labels,
        device=device,
    )

    export_splits = config.get("export_splits", ["train", "dev"])

    for split_name in export_splits:
        print(f"Loading split: {split_name}")

        raw_data = read_dataset(
            config["data_path"],
            split=split_name,
        )

        dataset = ClassificationDataset(raw_data, label2id)

        export_split_logits(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            split_name=split_name,
            output_dir=output_dir,
            config=config,
            device=device,
        )

    print("Teacher logits export complete.")


if __name__ == "__main__":
    main()