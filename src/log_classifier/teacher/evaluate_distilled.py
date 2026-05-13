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
from log_classifier.teacher.token_noise import apply_unk_token_noise
from log_classifier.teacher.train_distill_student import truncate_student_layers
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
def evaluate_loader(
    model,
    dataloader,
    device,
    labels_list,
    id2label,
    desc,
    tokenizer=None,
    noise_prob=None,
    min_keep_tokens=1,
    seed=42,
):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    generator = None

    if noise_prob is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when noise_prob is set")
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    for batch in tqdm(dataloader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if noise_prob is not None:
            input_ids, _ = apply_unk_token_noise(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                noise_prob=float(noise_prob),
                min_keep_tokens=int(min_keep_tokens),
                generator=generator,
            )

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
def measure_throughput(
    model,
    dataloader,
    device,
    rounds=5,
    warmup_rounds=1,
    tokenizer=None,
    noise_prob=None,
    min_keep_tokens=1,
    seed=42,
):
    model.eval()
    total_samples = len(dataloader.dataset)
    generator = None

    if noise_prob is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when noise_prob is set")
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    for _ in range(int(warmup_rounds)):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if noise_prob is not None:
                input_ids, _ = apply_unk_token_noise(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=tokenizer,
                    noise_prob=float(noise_prob),
                    min_keep_tokens=int(min_keep_tokens),
                    generator=generator,
                )
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_features=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed_values = []
    for round_idx in range(int(rounds)):
        start = time.perf_counter()
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if noise_prob is not None:
                input_ids, _ = apply_unk_token_noise(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=tokenizer,
                    noise_prob=float(noise_prob),
                    min_keep_tokens=int(min_keep_tokens),
                    generator=generator,
                )
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_features=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        elapsed_values.append(elapsed)
        print(f"Throughput round {round_idx + 1}: {total_samples / elapsed:.2f} samples/sec")

    avg_elapsed = sum(elapsed_values) / max(len(elapsed_values), 1)
    return total_samples / avg_elapsed


def average_metrics(metrics_list):
    return {
        "accuracy": sum(m["accuracy"] for m in metrics_list) / len(metrics_list),
        "macro_f1": sum(m["macro_f1"] for m in metrics_list) / len(metrics_list),
        "weighted_f1": sum(m["weighted_f1"] for m in metrics_list) / len(metrics_list),
        "eval_loss": sum(m["eval_loss"] for m in metrics_list) / len(metrics_list),
    }


def resolve_target(value, default):
    return float(default if value is None else value)


def infer_keep_layers_from_state_dict(state_dict):
    layer_prefix = "encoder.encoder.layer."
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith(layer_prefix):
            remainder = key[len(layer_prefix):]
            index_str = remainder.split(".", 1)[0]
            if index_str.isdigit():
                layer_indices.add(int(index_str))
    if not layer_indices:
        return None
    return max(layer_indices) + 1


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
    min_keep_tokens = int(config.get("min_keep_tokens", 1))
    noise_probs = config.get("noise_probs", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    id2label, label2id = load_label_mapping(os.path.join(checkpoint_dir, "label_mapping.json"))
    labels_list = sorted(id2label.keys())

    print("Loading data...")
    samples = read_dataset(
        config.get("data_path", "data/random_samples_splits.json"),
        split=config.get("split", "test"),
    )

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = CodeBERTClassifier(
        model_name=model_name,
        num_labels=len(label2id),
        dropout_prob=float(config.get("dropout_prob", 0.1)),
        pooling_mode=config.get("pooling_mode", "cls"),
        classifier_hidden_dim=int(config.get("classifier_hidden_dim", 0)),
        multi_sample_dropout_num=int(config.get("multi_sample_dropout_num", 1)),
    )
    state_dict = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"), map_location="cpu")
    inferred_keep_layers = config.get("student_keep_layers")
    if inferred_keep_layers is None:
        inferred_keep_layers = infer_keep_layers_from_state_dict(state_dict)
        if inferred_keep_layers is not None:
            print(f"Inferred student_keep_layers={inferred_keep_layers} from checkpoint weights")
    truncate_student_layers(model, inferred_keep_layers)
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

    robust_by_prob = {}
    robust_metrics = []
    if bool(config.get("robust_eval", True)):
        print("Evaluating UNK-token robustness...")
        for noise_prob in noise_probs:
            noise_prob = float(noise_prob)
            metrics = evaluate_loader(
                model,
                clean_loader,
                device,
                labels_list,
                id2label,
                desc=f"Robust eval UNK p={noise_prob}",
                tokenizer=tokenizer,
                noise_prob=noise_prob,
                min_keep_tokens=min_keep_tokens,
                seed=int(config.get("seed", 42)) + int(noise_prob * 1000),
            )
            metrics["noise_prob"] = noise_prob
            robust_by_prob[str(noise_prob)] = metrics
            robust_metrics.append(metrics)

    robust_average = average_metrics(robust_metrics) if robust_metrics else None

    print("Measuring throughput...")
    throughput = measure_throughput(
        model,
        throughput_loader,
        device,
        rounds=int(config.get("throughput_rounds", 5)),
        warmup_rounds=int(config.get("throughput_warmup_rounds", 1)),
        tokenizer=tokenizer,
        noise_prob=float(config["throughput_noise_prob"]) if config.get("throughput_noise_prob") is not None else None,
        min_keep_tokens=min_keep_tokens,
        seed=int(config.get("seed", 42)),
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
        "noise_type": "input_id_level_unk_replacement_fixed_ratio",
        "noise_probs": [float(p) for p in noise_probs],
        "min_keep_tokens": min_keep_tokens,
        "clean": clean_metrics,
        "unk_noise_by_prob": robust_by_prob,
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
