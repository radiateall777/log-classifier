import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from log_classifier.teacher.data import ClassificationDataset, read_dataset
from log_classifier.teacher.metrics import compute_classification_metrics
from log_classifier.teacher.model import CodeBERTClassifier
from log_classifier.teacher.train_distill_student import RobustTextAugmenter, truncate_student_layers
from log_classifier.teacher.utils import (
    get_device,
    load_label_mapping,
    load_yaml,
    save_json,
    set_seed,
)


def collate_fn_eval(batch, tokenizer, max_length):
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


def build_loader(samples, label2id, tokenizer, max_length, batch_size):
    dataset = ClassificationDataset(samples, label2id)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=lambda b: collate_fn_eval(b, tokenizer, int(max_length)),
    )


@torch.inference_mode()
def evaluate_loader(model, dataloader, device, labels_list, id2label, desc):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_features=False,
        )

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        total_loss += outputs["loss"].item()
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_classification_metrics(
        all_labels,
        all_preds,
        labels_list,
        id2label,
    )
    metrics["eval_loss"] = total_loss / max(len(dataloader), 1)
    return metrics


@torch.inference_mode()
def measure_throughput(model, dataloader, device, rounds=5, warmup_rounds=1):
    model.eval()
    total_samples = len(dataloader.dataset)

    for _ in range(int(warmup_rounds)):
        for batch in dataloader:
            model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                return_features=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed_values = []
    for round_idx in range(int(rounds)):
        start = time.perf_counter()
        for batch in dataloader:
            model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                return_features=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        elapsed_values.append(elapsed)
        print(f"Throughput round {round_idx + 1}: {total_samples / elapsed:.2f} samples/sec")

    avg_elapsed = sum(elapsed_values) / max(len(elapsed_values), 1)
    return total_samples / avg_elapsed


def build_robust_samples(samples, seed):
    augmenter = RobustTextAugmenter(seed=int(seed))
    robust_types = [
        ("remove_role_markers", augmenter.remove_role_markers),
        ("remove_markdown_noise", augmenter.remove_markdown_noise),
        ("remove_code_comments", augmenter.remove_code_comments),
        ("normalize_whitespace", augmenter.normalize_whitespace),
        ("truncate_assistant_explanation", augmenter.truncate_assistant_explanation),
        ("add_harmless_noise", augmenter.add_harmless_noise),
    ]

    robust_samples_by_type = {}
    for name, aug_func in robust_types:
        perturbed = []
        for item in samples:
            new_item = item.copy()
            try:
                augmented = aug_func(item["text"])
                new_item["text"] = augmented if augmented.strip() else item["text"]
            except Exception:
                new_item["text"] = item["text"]
            perturbed.append(new_item)
        robust_samples_by_type[name] = perturbed
    return robust_samples_by_type


def average_metrics(metrics_list):
    return {
        "accuracy": sum(m["accuracy"] for m in metrics_list) / len(metrics_list),
        "macro_f1": sum(m["macro_f1"] for m in metrics_list) / len(metrics_list),
        "weighted_f1": sum(m["weighted_f1"] for m in metrics_list) / len(metrics_list),
        "eval_loss": sum(m["eval_loss"] for m in metrics_list) / len(metrics_list),
    }


def resolve_target(value, default):
    return float(default if value is None else value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))

    device = get_device()
    checkpoint_dir = config["checkpoint_dir"]
    model_name = config.get("model_name", "microsoft/unixcoder-base")
    max_length = int(config.get("max_length", 512))
    batch_size = int(config.get("batch_size", 64))
    throughput_batch_size = int(config.get("throughput_batch_size", batch_size))

    id2label, label2id = load_label_mapping(os.path.join(checkpoint_dir, "label_mapping.json"))
    labels_list = sorted(id2label.keys())

    print("Loading data...")
    samples = read_dataset(
        config.get("data_path", "data/random_samples_splits.json"),
        split=config.get("split", "test"),
    )

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = CodeBERTClassifier(model_name=model_name, num_labels=len(label2id))
    truncate_student_layers(model, config.get("student_keep_layers"))

    state_dict = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"), map_location="cpu")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"missing_keys: {list(incompatible.missing_keys)}")
        print(f"unexpected_keys: {list(incompatible.unexpected_keys)}")

    model.to(device)
    if bool(config.get("fp16_eval", False)) and device.type == "cuda":
        model.half()

    clean_loader = build_loader(samples, label2id, tokenizer, max_length, batch_size)
    throughput_loader = build_loader(
        samples,
        label2id,
        tokenizer,
        max_length,
        throughput_batch_size,
    )

    print("Evaluating clean accuracy...")
    clean_metrics = evaluate_loader(
        model,
        clean_loader,
        device,
        labels_list,
        id2label,
        desc="Clean eval",
    )

    robust_by_type = {}
    robust_metrics = []
    if bool(config.get("robust_eval", True)):
        print("Evaluating robustness...")
        for name, robust_samples in build_robust_samples(samples, config.get("seed", 42)).items():
            loader = build_loader(robust_samples, label2id, tokenizer, max_length, batch_size)
            metrics = evaluate_loader(
                model,
                loader,
                device,
                labels_list,
                id2label,
                desc=f"Robust eval: {name}",
            )
            robust_by_type[name] = metrics
            robust_metrics.append(metrics)

    robust_average = average_metrics(robust_metrics) if robust_metrics else None

    print("Measuring throughput...")
    throughput = measure_throughput(
        model,
        throughput_loader,
        device,
        rounds=int(config.get("throughput_rounds", 5)),
        warmup_rounds=int(config.get("throughput_warmup_rounds", 1)),
    )

    baseline_throughput = config.get("baseline_throughput_samples_per_sec")
    throughput_target_ratio = resolve_target(config.get("throughput_target_ratio"), 1.2)
    clean_accuracy_target = resolve_target(config.get("clean_accuracy_target"), 0.9)
    robust_accuracy_target = config.get("robust_accuracy_target")

    throughput_pass = None
    throughput_lift = None
    if baseline_throughput is not None:
        baseline_throughput = float(baseline_throughput)
        throughput_lift = throughput / baseline_throughput
        throughput_pass = throughput_lift >= throughput_target_ratio

    robust_pass = None
    if robust_accuracy_target is not None and robust_average is not None:
        robust_pass = robust_average["accuracy"] >= float(robust_accuracy_target)

    summary = {
        "clean_accuracy": clean_metrics["accuracy"],
        "clean_accuracy_target": clean_accuracy_target,
        "clean_accuracy_pass": clean_metrics["accuracy"] >= clean_accuracy_target,
        "robust_average_accuracy": None if robust_average is None else robust_average["accuracy"],
        "robust_accuracy_target": robust_accuracy_target,
        "robust_accuracy_pass": robust_pass,
        "throughput_samples_per_sec": float(throughput),
        "baseline_throughput_samples_per_sec": baseline_throughput,
        "throughput_lift": throughput_lift,
        "throughput_target_ratio": throughput_target_ratio,
        "throughput_pass": throughput_pass,
    }

    report = {
        "checkpoint_dir": checkpoint_dir,
        "model_name": model_name,
        "student_keep_layers": config.get("student_keep_layers"),
        "max_length": max_length,
        "batch_size": batch_size,
        "throughput_batch_size": throughput_batch_size,
        "clean": clean_metrics,
        "robust_by_type": robust_by_type,
        "robust_average": robust_average,
        "summary": summary,
    }

    output_report = config["output_report"]
    save_json(report, output_report)

    print("\nAcceptance summary")
    print(f"Clean accuracy: {summary['clean_accuracy']:.4f} target={clean_accuracy_target:.4f} pass={summary['clean_accuracy_pass']}")
    if robust_average is not None:
        print(f"Robust avg accuracy: {robust_average['accuracy']:.4f} target={robust_accuracy_target} pass={robust_pass}")
    if throughput_lift is None:
        print(f"Throughput: {throughput:.2f} samples/sec baseline=<not set> pass=<not checked>")
    else:
        print(
            f"Throughput: {throughput:.2f} samples/sec baseline={baseline_throughput:.2f} "
            f"lift={throughput_lift:.3f} target={throughput_target_ratio:.3f} pass={throughput_pass}"
        )
    print(f"Report saved to {output_report}")


if __name__ == "__main__":
    main()
