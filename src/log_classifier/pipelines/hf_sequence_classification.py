"""HuggingFace Trainer 序列分类训练流水线。

职责仅为「编排」：按顺序调用 data / models / training 子模块，
本身不包含数据处理或模型构建细节。

所有配置通过 DataConfig / ModelConfig / TrainConfig 传入，
训练结果以结构化 dict 形式返回，供调用方持久化。
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
)

from log_classifier.config import DataConfig, ModelConfig, TrainConfig
from log_classifier.data.hf_dataset import build_hf_dataset_dict, tokenize_datasets
from log_classifier.data.preprocess import (
    assign_label_ids, build_label_maps, build_samples,
    filter_rare_classes, load_json_data, split_dataset,
)
from log_classifier.models.hf_classifier import build_model, build_tokenizer
from log_classifier.training.metrics import compute_metrics
from log_classifier.training.weighted_trainer import WeightedTrainer


# ------------------------------------------------------------------
# 内部辅助
# ------------------------------------------------------------------

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
    tensor = torch.tensor(weights, dtype=torch.float)
    print(f"[Info] 使用 class weights: {tensor.tolist()}")
    return tensor


def _make_training_args(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    total_training_steps: int,
) -> TrainingArguments:
    warmup_steps = int(total_training_steps * train_cfg.warmup_ratio)
    return TrainingArguments(
        output_dir=train_cfg.output_dir,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=train_cfg.logging_steps,

        per_device_train_batch_size=train_cfg.train_batch_size,
        per_device_eval_batch_size=train_cfg.eval_batch_size,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        num_train_epochs=train_cfg.num_train_epochs,
        warmup_steps=warmup_steps,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=train_cfg.save_total_limit,

        fp16=train_cfg.fp16,
        report_to="none",
        seed=train_cfg.seed,
    )


def _evaluate_and_report(
    trainer: WeightedTrainer,
    tokenized_datasets,
    id2label: Dict[int, str],
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, np.ndarray, float]:
    """评估并返回结构化结果。"""
    print("\n========== 验证集结果 ==========")
    val_metrics = trainer.evaluate(tokenized_datasets["validation"])
    print(val_metrics)

    print("\n========== 测试集结果 ==========")
    num_test_samples = len(tokenized_datasets["test"])
    t_start = time.perf_counter()
    test_output = trainer.predict(tokenized_datasets["test"])
    t_elapsed = time.perf_counter() - t_start

    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_output.label_ids
    throughput = num_test_samples / t_elapsed if t_elapsed > 0 else float("inf")

    test_metrics = {
        "accuracy": accuracy_score(test_labels, test_preds),
        "macro_f1": f1_score(test_labels, test_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(test_labels, test_preds, average="weighted", zero_division=0),
        "precision": precision_score(test_labels, test_preds, average="macro", zero_division=0),
        "recall": recall_score(test_labels, test_preds, average="macro", zero_division=0),
    }

    print({
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_samples": num_test_samples,
        "test_elapsed_sec": round(t_elapsed, 3),
        "test_throughput_samples_per_sec": round(throughput, 2),
    })

    print("\n========== 分类报告 ==========")
    print(classification_report(
        test_labels, test_preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4, zero_division=0,
    ))

    return val_metrics, test_metrics, test_preds, test_labels, throughput


def _load_fixed_splits(
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
) -> Optional[Dict[str, Any]]:
    """尝试从预先生成的固定划分文件加载数据。

    若 `{data_path}_splits.json` 存在且 seed/test_size/dev_size 与配置一致，
    直接返回划分好的数据；否则返回 None，由调用方执行 split_dataset。
    """
    splits_path = data_cfg.data_path.replace(".jsonl", "_splits.json").replace(".json", "_splits.json")
    if not os.path.exists(splits_path):
        return None

    try:
        with open(splits_path, "r", encoding="utf-8") as f:
            saved = json.load(f)

        # 校验划分参数一致性（seed 及比例写在 splits 文件元数据中）
        meta = saved.get("_meta", {})
        if (meta.get("seed") != train_cfg.seed
                or abs(meta.get("test_size", 0) - data_cfg.test_size) > 1e-6
                or abs(meta.get("dev_size", 0) - data_cfg.dev_size) > 1e-6):
            print(f"[Warning] 固定划分参数与当前配置不一致，重新划分。")
            return None

        print(f"[Info] 使用固定数据划分: {splits_path}")
        return {
            "train": saved["train"],
            "dev": saved["dev"],
            "test": saved["test"],
            "label_list": saved.get("label_list"),
            "label2id": saved.get("label2id"),
            "id2label": saved.get("id2label"),
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Warning] 固定划分文件读取失败 ({e})，重新划分。")
        return None


def _save_artifacts(
    trainer: WeightedTrainer,
    tokenizer,
    output_dir: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "label_mappings.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
            f, ensure_ascii=False, indent=2,
        )

    # 导出最优权重文件，方便推理时直接 load_state_dict
    best_weights_path = os.path.join(output_dir, "best_model.pt")
    torch.save(trainer.model.state_dict(), best_weights_path)
    print(f"最优权重已保存: {best_weights_path}")

    print(f"\n模型和标签映射已保存到: {output_dir}")


# ------------------------------------------------------------------
# 公开接口
# ------------------------------------------------------------------

def run_hf_sequence_classification(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> Dict[str, Any]:
    """执行完整训练流程，返回结构化结果字典。

    调用方负责持久化结果（save_model / save_pretrained 已在内部完成）。
    """
    # ---- 数据（优先使用固定划分文件） ----
    fixed = _load_fixed_splits(data_cfg, train_cfg)

    if fixed is not None:
        train_data = fixed["train"]
        dev_data = fixed["dev"]
        test_data = fixed["test"]
        label_list = fixed.get("label_list")
        label2id = fixed.get("label2id")
        id2label = fixed.get("id2label")

        # 恢复 labels 字段（splits 中只有 label_text）
        if "labels" not in train_data[0]:
            for s in train_data:
                s["labels"] = label2id[s["label_text"]]
            for s in dev_data:
                s["labels"] = label2id[s["label_text"]]
            for s in test_data:
                s["labels"] = label2id[s["label_text"]]

        print(f"总样本数（固定划分）: {len(train_data) + len(dev_data) + len(test_data)}")
        print(f"标签数: {len(label_list)}")
        print(f"标签列表: {label_list}")
        print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")
    else:
        raw_data = load_json_data(data_cfg.data_path)
        samples = build_samples(raw_data, data_cfg.label_field, data_cfg.text_mode)
        samples = filter_rare_classes(samples, min_count=data_cfg.min_class_count)
        print(f"总样本数: {len(samples)}")

        label_list, label2id, id2label = build_label_maps(samples)
        assign_label_ids(samples, label2id)
        print(f"标签数: {len(label_list)}")
        print(f"标签列表: {label_list}")

        train_data, dev_data, test_data = split_dataset(
            samples, seed=train_cfg.seed,
            test_size=data_cfg.test_size, dev_size=data_cfg.dev_size,
        )
        print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

    # ---- HF 适配 ----
    dataset_dict = build_hf_dataset_dict(train_data, dev_data, test_data)
    tokenizer = build_tokenizer(model_cfg.model_name)
    tokenized_datasets = tokenize_datasets(dataset_dict, tokenizer, model_cfg.max_length)

    # ---- 模型 ----
    model = build_model(model_cfg.model_name, len(label_list), id2label, label2id)

    # ---- Trainer ----
    class_weights = _compute_class_weights(train_data, train_cfg.use_class_weights)
    steps_per_epoch = math.ceil(len(tokenized_datasets["train"]) / train_cfg.train_batch_size)
    total_training_steps = steps_per_epoch * train_cfg.num_train_epochs
    training_args = _make_training_args(train_cfg, model_cfg, total_training_steps)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=train_cfg.early_stopping_patience),
        ],
    )

    # ---- 训练 & 评估 & 保存 ----
    print("\n========== 开始训练 ==========")
    t0 = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - t0
    print(f"训练完成，耗时: {train_elapsed:.2f}s")

    val_metrics, test_metrics, test_preds, test_labels, throughput \
        = _evaluate_and_report(trainer, tokenized_datasets, id2label)

    _save_artifacts(
        trainer, tokenizer, train_cfg.output_dir,
        label2id, id2label,
    )

    return {
        "model_name": model_cfg.model_name,
        "label_list": label_list,
        "label2id": label2id,
        "id2label": id2label,
        "train_samples": len(train_data),
        "dev_samples": len(dev_data),
        "test_samples": len(test_data),
        "num_labels": len(label_list),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_elapsed_seconds": round(train_elapsed, 3),
        "test_throughput_samples_per_sec": round(throughput, 2),
        "best_weights_path": os.path.join(train_cfg.output_dir, "best_model.pt"),
    }
