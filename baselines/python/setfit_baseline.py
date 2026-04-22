"""SetFit baseline for small-dataset text classification.

SetFit (Sentence Transformer Fine-tuning) uses contrastive learning
with Sentence Transformers + a classification head. Excellent for
small datasets (100-5000 samples).

Tunstall et al., "Efficient Few-Shot Learning Without Prompts", 2022.

Usage::
    python3 baselines/python/setfit_baseline.py
    python3 baselines/python/setfit_baseline.py --model_name BAAI/bge-base-en-v1.5
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from log_classifier.data.preprocess import (
    assign_label_ids,
    build_label_maps,
    build_samples,
    filter_rare_classes,
    load_json_data,
    split_dataset,
)


def evaluate_predictions(y_true, y_pred, id2label):
    all_label_ids = sorted(id2label.keys())
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=all_label_ids, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0),
    }
    report = classification_report(
        y_true, y_pred,
        labels=all_label_ids,
        target_names=[id2label[i] for i in all_label_ids],
        digits=4, zero_division=0, output_dict=True,
    )
    return {"metrics": metrics, "classification_report": report}


def run_setfit(
    train_data, dev_data, test_data, label_list, label2id, id2label,
    model_name="sentence-transformers/all-mpnet-base-v2",
    num_epochs=1, batch_size=16, num_iterations=20, seed=42,
):
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    train_ds = Dataset.from_list([
        {"text": s["text"], "label": s["labels"]} for s in train_data
    ])
    dev_ds = Dataset.from_list([
        {"text": s["text"], "label": s["labels"]} for s in dev_data
    ])
    test_ds = Dataset.from_list([
        {"text": s["text"], "label": s["labels"]} for s in test_data
    ])

    model = SetFitModel.from_pretrained(
        model_name,
        labels=label_list,
    )

    training_args = TrainingArguments(
        output_dir=f"./baseline_results/setfit_{model_name.replace('/', '_')}",
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        seed=seed,
        evaluation_strategy="epoch",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
    )

    print(f"\n[SetFit] 模型: {model_name}")
    print(f"[SetFit] 训练集: {len(train_data)}, 验证集: {len(dev_data)}, 测试集: {len(test_data)}")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"[SetFit] 训练完成，耗时: {train_time:.2f}s")

    dev_preds = model.predict(dev_ds["text"])
    dev_labels = dev_ds["label"]
    dev_f1 = f1_score(dev_labels, dev_preds, average="macro", zero_division=0)
    print(f"[SetFit] Dev macro_f1: {dev_f1:.4f}")

    t0 = time.time()
    test_preds = model.predict(test_ds["text"])
    infer_time = time.time() - t0
    test_labels = test_ds["label"]
    throughput = len(test_data) / infer_time if infer_time > 0 else float("inf")

    if hasattr(test_preds, "tolist"):
        test_preds = test_preds.tolist()
    if hasattr(test_labels, "tolist"):
        test_labels = test_labels.tolist()

    results = evaluate_predictions(test_labels, test_preds, id2label)
    results["method"] = f"setfit_{model_name.replace('/', '_')}"
    results["method_name"] = f"SetFit ({model_name.split('/')[-1]})"
    results["model_name"] = model_name
    results["train_time_seconds"] = train_time
    results["infer_time_seconds"] = infer_time
    results["throughput_samples_per_second"] = throughput
    results["test_samples"] = len(test_data)
    results["num_labels"] = len(id2label)
    results["dev_macro_f1"] = dev_f1
    results["setfit_params"] = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "num_iterations": num_iterations,
    }

    print(f"\n{'=' * 60}")
    print(f"SetFit ({model_name})")
    print(f"{'=' * 60}")
    m = results["metrics"]
    print(f"Accuracy:        {m['accuracy']:.4f}")
    print(f"Macro F1:        {m['macro_f1']:.4f}")
    print(f"Weighted F1:     {m['weighted_f1']:.4f}")
    print(f"Macro Precision: {m['macro_precision']:.4f}")
    print(f"Macro Recall:    {m['macro_recall']:.4f}")

    print("\n" + classification_report(
        test_labels, test_preds,
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        digits=4, zero_division=0,
    ))

    return results


def main():
    parser = argparse.ArgumentParser(description="SetFit Baseline")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--data_path", type=str, default="./data/random_samples.jsonl")
    parser.add_argument("--output_dir", type=str, default="./baseline_results")
    parser.add_argument("--label_field", type=str, default="label3")
    parser.add_argument("--text_mode", type=str, default="user_assistant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_iterations", type=int, default=20)
    args = parser.parse_args()

    np.random.seed(args.seed)

    raw_data = load_json_data(args.data_path)
    samples = build_samples(raw_data, args.label_field, args.text_mode)
    samples = filter_rare_classes(samples, min_count=2)
    label_list, label2id, id2label = build_label_maps(samples)
    assign_label_ids(samples, label2id)

    train_data, dev_data, test_data = split_dataset(
        samples, seed=args.seed, test_size=0.1, dev_size=0.1,
    )

    results = run_setfit(
        train_data, dev_data, test_data,
        label_list, label2id, id2label,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{results['method']}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
