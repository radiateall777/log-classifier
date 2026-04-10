"""Baseline 模型训练脚本。

用法::
    python3 baselines/run_baseline_train.py --model bert-base-uncased
    python3 baselines/run_baseline_train.py --model microsoft/deberta-v3-base
"""

import os
import sys

# 设置 HuggingFace 镜像（国内加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
)

# =============================================================================
# 数据处理（与 src/log_classifier/data/preprocess.py 保持一致）
# =============================================================================

def load_json_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    return data


def flatten_messages(
    messages: List[Dict[str, str]],
    text_mode: str,
    item: Dict[str, Any],
) -> str:
    user_texts, assistant_texts = [], []
    for msg in messages:
        role = msg.get("role", "").strip()
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            user_texts.append(content)
        elif role == "assistant":
            assistant_texts.append(content)

    user_text = " ".join(user_texts).strip()
    assistant_text = " ".join(assistant_texts).strip()

    if text_mode == "user_only":
        return f"user: {user_text}"
    if text_mode == "assistant_only":
        return f"assistant: {assistant_text}"
    if text_mode == "user_assistant":
        return f"user: {user_text} assistant: {assistant_text}"
    if text_mode == "with_meta":
        language = str(item.get("language", "")).strip()
        dataset_name = str(item.get("dataset", "")).strip()
        return (
            f"language: {language} "
            f"dataset: {dataset_name} "
            f"user: {user_text} "
            f"assistant: {assistant_text}"
        )
    raise ValueError(f"不支持的 text_mode: {text_mode}")


def build_samples(
    raw_data: List[Dict[str, Any]],
    label_field: str,
    text_mode: str,
) -> List[Dict[str, Any]]:
    samples = []
    for item in raw_data:
        messages = item.get("messages", [])
        if not isinstance(messages, list):
            continue
        label = item.get(label_field, None)
        if label is None:
            continue
        text = flatten_messages(messages, text_mode, item)
        if not text:
            continue
        samples.append({
            "id": item.get("id", None),
            "text": text,
            "label_text": str(label).strip(),
        })
    return samples


def filter_rare_classes(
    samples: List[Dict[str, Any]],
    min_count: int = 2,
) -> List[Dict[str, Any]]:
    from collections import Counter
    counter = Counter(x["label_text"] for x in samples)
    kept = [x for x in samples if counter[x["label_text"]] >= min_count]
    removed = len(samples) - len(kept)
    if removed > 0:
        print(f"[Info] 移除了 {removed} 条极少数类样本（每类少于 {min_count} 条）。")
    return kept


def build_label_maps(samples: List[Dict[str, Any]]):
    label_list = sorted({x["label_text"] for x in samples})
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_list, label2id, id2label


def assign_label_ids(samples: List[Dict[str, Any]], label2id: Dict[str, int]) -> None:
    for x in samples:
        x["labels"] = label2id[x["label_text"]]


def split_dataset(
    samples: List[Dict[str, Any]],
    seed: int,
    test_size: float,
    dev_size: float,
):
    labels = [x["labels"] for x in samples]
    indices = list(range(len(samples)))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels,
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    relative_dev_size = dev_size / (1.0 - test_size)
    train_idx, dev_idx = train_test_split(
        train_val_idx, test_size=relative_dev_size, random_state=seed,
        stratify=train_val_labels,
    )
    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in dev_idx],
        [samples[i] for i in test_idx],
    )


# =============================================================================
# HuggingFace Dataset 适配
# =============================================================================

from datasets import Dataset, DatasetDict


def build_hf_dataset_dict(
    train_data: List[Dict[str, Any]],
    dev_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
) -> DatasetDict:
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(dev_data),
        "test": Dataset.from_list(test_data),
    })


def tokenize_datasets(dataset_dict: DatasetDict, tokenizer, max_length: int):
    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset_dict.map(_tokenize, batched=True)
    keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized["train"].column_names:
        keep.append("token_type_ids")
    tokenized.set_format(type="torch", columns=keep)
    return tokenized


# =============================================================================
# 评估指标
# =============================================================================

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


# =============================================================================
# 训练
# =============================================================================

def _compute_class_weights(
    train_data: List[Dict[str, Any]],
    enabled: bool,
) -> Optional[torch.Tensor]:
    if not enabled:
        return None
    train_labels = np.array([x["labels"] for x in train_data])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    return torch.tensor(weights, dtype=torch.float)


class WeightedTrainer:
    """简化版 WeightedTrainer，对齐 src/log_classifier/training/weighted_trainer.py"""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        # 直接用 transformers.Trainer，避免重复造轮子
        from transformers import Trainer
        self._trainer = Trainer(**kwargs)
        self.class_weights = class_weights

    def __getattr__(self, name):
        return getattr(self._trainer, name)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def train(self):
        return self._trainer.train()

    def evaluate(self, *args, **kwargs):
        return self._trainer.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self._trainer.predict(*args, **kwargs)

    def save_model(self, *args, **kwargs):
        self._trainer.save_model(*args, **kwargs)


def run_baseline_training(
    model_name: str,
    data_path: str,
    output_dir: str,
    text_mode: str = "user_assistant",
    label_field: str = "label3",
    max_length: int = 256,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    seed: int = 42,
    test_size: float = 0.1,
    dev_size: float = 0.1,
    use_class_weights: bool = True,
    logging_steps: int = 50,
    save_total_limit: int = 2,
    early_stopping_patience: int = 2,
    fp16: bool = False,
) -> Dict[str, Any]:
    """运行单个 baseline 模型训练"""

    # ---- 随机种子 ----
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- 数据 ----
    print(f"\n{'='*60}")
    print(f"训练 Baseline: {model_name}")
    print(f"{'='*60}")
    print(f"加载数据: {data_path}")

    raw_data = load_json_data(data_path)
    samples = build_samples(raw_data, label_field, text_mode)
    samples = filter_rare_classes(samples, min_count=2)
    print(f"总样本数: {len(samples)}")

    label_list, label2id, id2label = build_label_maps(samples)
    assign_label_ids(samples, label2id)
    print(f"标签数: {len(label_list)}")
    print(f"标签列表: {label_list}")

    train_data, dev_data, test_data = split_dataset(
        samples, seed=seed, test_size=test_size, dev_size=dev_size,
    )
    print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

    # ---- HF Dataset ----
    dataset_dict = build_hf_dataset_dict(train_data, dev_data, test_data)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenized_datasets = tokenize_datasets(dataset_dict, tokenizer, max_length)

    # ---- 模型 ----
    print(f"加载模型: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # ---- 训练参数 ----
    steps_per_epoch = max(1, len(tokenized_datasets["train"]) // train_batch_size)
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    model_output_dir = os.path.join(output_dir, model_name.replace("/", "_"))

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=save_total_limit,
        fp16=fp16,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
    )

    # ---- Trainer ----
    class_weights = _compute_class_weights(train_data, use_class_weights)
    if class_weights is not None:
        print(f"[Info] 使用 class weights: {class_weights.tolist()}")

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # ---- 训练 ----
    print("\n========== 开始训练 ==========")
    t_start = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - t_start
    print(f"训练完成! 耗时: {train_elapsed:.2f}s")

    # ---- 验证集评估 ----
    print("\n========== 验证集结果 ==========")
    val_metrics = trainer.evaluate(tokenized_datasets["validation"])
    print(val_metrics)

    # ---- 测试集评估 ----
    print("\n========== 测试集结果 ==========")
    num_test = len(tokenized_datasets["test"])
    t_test_start = time.perf_counter()
    test_output = trainer.predict(tokenized_datasets["test"])
    test_elapsed = time.perf_counter() - t_test_start

    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_output.label_ids
    throughput = num_test / test_elapsed if test_elapsed > 0 else float("inf")

    test_acc = accuracy_score(test_labels, test_preds)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)

    print({
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_samples": num_test,
        "test_elapsed_sec": round(test_elapsed, 3),
        "test_throughput_samples_per_sec": round(throughput, 2),
    })

    print("\n========== 分类报告 ==========")
    print(classification_report(
        test_labels, test_preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4, zero_division=0,
    ))

    # ---- 保存模型 ----
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    label_map_path = os.path.join(model_output_dir, "label_mappings.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
        }, f, ensure_ascii=False, indent=2)

    # 保存 splits
    splits_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_splits.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump({
            "train": train_data,
            "dev": dev_data,
            "test": test_data,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存到: {model_output_dir}")

    return {
        "model_name": model_name,
        "val_metrics": val_metrics,
        "test_metrics": {
            "accuracy": test_acc,
            "macro_f1": test_macro_f1,
            "weighted_f1": test_weighted_f1,
            "precision": precision_score(test_labels, test_preds, average="macro", zero_division=0),
            "recall": recall_score(test_labels, test_preds, average="macro", zero_division=0),
        },
        "elapsed_seconds": round(train_elapsed + test_elapsed, 3),
        "train_elapsed_seconds": round(train_elapsed, 3),
        "test_elapsed_seconds": round(test_elapsed, 3),
        "test_throughput_samples_per_sec": round(throughput, 2),
        "test_samples": num_test,
        "num_labels": len(label_list),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline 模型训练")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="模型名称 (HuggingFace ID)")
    parser.add_argument("--data_path", type=str,
                        default="./data/random_samples.jsonl",
                        help="数据路径")
    parser.add_argument("--output_dir", type=str,
                        default="./baseline_results",
                        help="输出目录")
    parser.add_argument("--text_mode", type=str, default="user_assistant",
                        help="文本模式")
    parser.add_argument("--label_field", type=str, default="label3",
                        help="标签字段")
    parser.add_argument("--max_length", type=int, default=256,
                        help="最大序列长度")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="训练批大小")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="评估批大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="测试集比例")
    parser.add_argument("--dev_size", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--use_class_weights", type=lambda x: x.lower() == "true",
                        default=True,
                        help="是否使用类别权重")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="日志记录步数")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="最多保存几个 checkpoint")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                        help="早停耐心值")
    parser.add_argument("--fp16", type=lambda x: x.lower() == "true",
                        default=str(torch.cuda.is_available()).lower(),
                        help="是否使用 FP16")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = run_baseline_training(
        model_name=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        text_mode=args.text_mode,
        label_field=args.label_field,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        test_size=args.test_size,
        dev_size=args.dev_size,
        use_class_weights=args.use_class_weights,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        early_stopping_patience=args.early_stopping_patience,
        fp16=args.fp16,
    )

    # 保存结果
    result_path = os.path.join(
        args.output_dir,
        f"{args.model.replace('/', '_')}_train_results.json",
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n训练结果已保存: {result_path}")
    print(f"Macro F1: {results['test_metrics']['macro_f1']:.4f}")
    return results


if __name__ == "__main__":
    main()
