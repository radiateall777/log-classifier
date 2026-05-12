import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from log_classifier.teacher.data import ClassificationDataset, read_dataset
from log_classifier.teacher.metrics import compute_classification_metrics
from log_classifier.teacher.model import CodeBERTClassifier
from log_classifier.teacher.token_noise import apply_unk_token_noise
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
        "ids": ids,
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }


@torch.no_grad()
def evaluate_with_optional_unk_noise(
    model,
    dataloader,
    device,
    labels_list,
    id2label,
    tokenizer,
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
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    desc = "Evaluating clean" if noise_prob is None else f"Evaluating UNK p={noise_prob}"

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

        loss = outputs["loss"]
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_classification_metrics(
        all_labels,
        all_preds,
        labels_list,
        id2label,
    )
    metrics["eval_loss"] = total_loss / max(len(dataloader), 1)

    return metrics, all_preds, all_labels


def compute_prediction_consistency(clean_preds, noisy_preds):
    if len(clean_preds) != len(noisy_preds):
        raise ValueError(
            f"clean_preds and noisy_preds length mismatch: "
            f"{len(clean_preds)} vs {len(noisy_preds)}"
        )

    if not clean_preds:
        return None

    same = sum(int(a == b) for a, b in zip(clean_preds, noisy_preds))
    return same / len(clean_preds)


def mean_or_none(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize_trial_metrics(trial_metrics, trial_consistency, noise_prob, num_trials):
    acc_values = [m["accuracy"] for m in trial_metrics]
    macro_f1_values = [m["macro_f1"] for m in trial_metrics]
    weighted_f1_values = [m["weighted_f1"] for m in trial_metrics]
    eval_loss_values = [m["eval_loss"] for m in trial_metrics]
    consistency_values = [c for c in trial_consistency if c is not None]

    return {
        "noise_prob": float(noise_prob),
        "num_trials": int(num_trials),
        "accuracy": mean_or_none(acc_values),
        "macro_f1": mean_or_none(macro_f1_values),
        "weighted_f1": mean_or_none(weighted_f1_values),
        "eval_loss": mean_or_none(eval_loss_values),
        "consistency_with_clean_pred": mean_or_none(consistency_values),
        "trials": trial_metrics,
    }


def summarize_prob_summaries(prob_summaries):
    if not prob_summaries:
        return {
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "eval_loss": None,
            "consistency_with_clean_pred": None,
        }

    return {
        "accuracy": mean_or_none([m.get("accuracy") for m in prob_summaries]),
        "macro_f1": mean_or_none([m.get("macro_f1") for m in prob_summaries]),
        "weighted_f1": mean_or_none([m.get("weighted_f1") for m in prob_summaries]),
        "eval_loss": mean_or_none([m.get("eval_loss") for m in prob_summaries]),
        "consistency_with_clean_pred": mean_or_none(
            [m.get("consistency_with_clean_pred") for m in prob_summaries]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml(args.config)

    set_seed(config.get("seed", 42))
    device = get_device()

    checkpoint_dir = config["checkpoint_dir"]
    id2label, label2id = load_label_mapping(
        os.path.join(checkpoint_dir, "label_mapping.json")
    )

    num_labels = len(label2id)
    labels_list = list(id2label.keys())

    print("Loading data...")
    split = config.get("split", "test")
    test_data = read_dataset(
        config.get("data_path", "data/random_samples_splits.json"),
        split=split,
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    if tokenizer.unk_token_id is None:
        raise ValueError(
            f"Tokenizer loaded from {checkpoint_dir} has no unk_token_id. "
            "UNK noise evaluation cannot run."
        )

    print(f"tokenizer.unk_token: {tokenizer.unk_token}")
    print(f"tokenizer.unk_token_id: {tokenizer.unk_token_id}")

    print("Initializing model...")
    model = CodeBERTClassifier(
        model_name=config.get("model_name", "microsoft/unixcoder-base"),
        num_labels=num_labels,
    )

    state_dict = torch.load(
        os.path.join(checkpoint_dir, "pytorch_model.bin"),
        map_location="cpu",
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {list(incompatible.missing_keys)}")
    print(f"unexpected_keys: {list(incompatible.unexpected_keys)}")

    model.to(device)
    model.eval()

    dataset = ClassificationDataset(test_data, label2id)

    dataloader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=lambda b: collate_fn_eval(
            b,
            tokenizer,
            int(config["max_length"]),
        ),
    )

    print("Evaluating on clean set...")
    clean_metrics, clean_preds, clean_labels = evaluate_with_optional_unk_noise(
        model=model,
        dataloader=dataloader,
        device=device,
        labels_list=labels_list,
        id2label=id2label,
        tokenizer=tokenizer,
        noise_prob=None,
        seed=int(config.get("seed", 42)),
    )

    report = {
        "split": split,
        "checkpoint_dir": checkpoint_dir,
        "model_name": config.get("model_name"),
        "noise_type": "input_id_level_unk_replacement_fixed_ratio",
        "min_keep_tokens": int(config.get("min_keep_tokens", 1)),
        "num_noise_trials": int(config.get("num_noise_trials", 5)),
        "clean": clean_metrics,
    }

    if config.get("unk_noise_eval", True):
        noise_probs = config.get(
            "noise_probs",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
        num_trials = int(config.get("num_noise_trials", 5))
        min_keep_tokens = int(config.get("min_keep_tokens", 1))

        robust_by_prob = {}

        summaries_01_05 = []
        summaries_01_09 = []

        print("Evaluating UNK-token noise robustness...")

        for p in noise_probs:
            p = float(p)

            trial_metrics = []
            trial_consistency = []

            for trial in range(num_trials):
                trial_seed = int(config.get("seed", 42)) + trial + int(p * 1000)

                metrics, noisy_preds, _ = evaluate_with_optional_unk_noise(
                    model=model,
                    dataloader=dataloader,
                    device=device,
                    labels_list=labels_list,
                    id2label=id2label,
                    tokenizer=tokenizer,
                    noise_prob=p,
                    min_keep_tokens=min_keep_tokens,
                    seed=trial_seed,
                )

                consistency = compute_prediction_consistency(
                    clean_preds,
                    noisy_preds,
                )

                metrics["consistency_with_clean_pred"] = consistency
                metrics["trial"] = trial
                metrics["seed"] = trial_seed

                trial_metrics.append(metrics)
                trial_consistency.append(consistency)

            p_summary = summarize_trial_metrics(
                trial_metrics=trial_metrics,
                trial_consistency=trial_consistency,
                noise_prob=p,
                num_trials=num_trials,
            )

            robust_by_prob[str(p)] = p_summary

            summaries_01_09.append(p_summary)

            if 0.1 <= p <= 0.5:
                summaries_01_05.append(p_summary)

            print(
                f"UNK p={p:.1f} | "
                f"Acc={p_summary['accuracy']:.4f} | "
                f"MacroF1={p_summary['macro_f1']:.4f} | "
                f"WeightedF1={p_summary['weighted_f1']:.4f} | "
                f"Consistency={p_summary['consistency_with_clean_pred']:.4f}"
            )

        report["unk_noise_by_prob"] = robust_by_prob

        report["unk_noise_average_0.1_0.5"] = summarize_prob_summaries(
            summaries_01_05
        )

        report["unk_noise_average_0.1_0.9"] = summarize_prob_summaries(
            summaries_01_09
        )

        # 与 baseline 字段名对齐：全部缺失率下的平均值。
        report["average_all_noise_ratios"] = report["unk_noise_average_0.1_0.9"]

    output_report = config["output_report"]
    save_json(report, output_report)

    print(f"Evaluation report saved to {output_report}")


if __name__ == "__main__":
    main()