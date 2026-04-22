"""K-fold × 多 seed Transformer 训练包装器。

对同一套训练配置：
1. 先划分出固定 test 集（seed=42），剩余部分做 K 折交叉验证。
2. 对每一折，再跑 N 个 seed，共 K × N 个模型。
3. 每个模型在本折 train 上训练、本折 val 上早停，最终保存：
   - 对「本折 val」样本（= train 集子集）的概率 → 聚合为训练集 OOF 矩阵
   - 对固定 test 集的概率 → K × N 份取平均，作为集成预测

输出（默认到 <output_dir>）:
    oof_probs.npy        [N_trainval, num_classes]   无泄漏 OOF
    oof_labels.npy       [N_trainval]
    oof_index.npy        [N_trainval]  每行对应 samples_trainval 的原始 index
    test_probs.npy       [N_test, num_classes]       K×N 模型平均
    test_labels.npy      [N_test]
    kfold_summary.json   每折/每 seed 的指标 + 最终指标

用法::
    python3 baselines/run_kfold_train.py \\
        --model_name microsoft/deberta-v3-base \\
        --output_dir ./baseline_results/deberta_v3_base_kfold_fgm \\
        --k_folds 5 --seeds 42 123 2024 \\
        --use_adversarial --use_rdrop --label_smoothing 0.1 \\
        --use_layerwise_lr_decay
"""

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from log_classifier.config import DataConfig, ModelConfig, TrainConfig
from log_classifier.data.preprocess import (
    assign_label_ids, build_label_maps, build_samples,
    filter_rare_classes, load_json_data,
)
from log_classifier.pipelines import run_hf_sequence_classification
from log_classifier.utils import seed_everything


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="K-fold × 多 seed Transformer 训练")

    # 数据
    p.add_argument("--data_path", default="./data/random_samples.jsonl")
    p.add_argument("--text_mode", default="user_assistant")
    p.add_argument("--label_field", default="label3")
    p.add_argument("--test_size", type=float, default=0.1,
                   help="固定 test 集比例（从 seed=42 划分）")
    p.add_argument("--min_class_count", type=int, default=2)

    # K-fold
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--test_split_seed", type=int, default=42,
                   help="固定 test 集的划分 seed")

    # 模型
    p.add_argument("--model_name", required=True)
    p.add_argument("--max_length", type=int, default=512)

    # 输出
    p.add_argument("--output_dir", required=True,
                   help="主输出目录。中间 per-fold 模型会落到子目录后清理。")
    p.add_argument("--keep_fold_models", action="store_true", default=False,
                   help="保留每折的 model.safetensors（默认清理节省磁盘）")

    # 训练超参
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--bf16", action="store_true", default=False,
                   help="使用 bfloat16（Ampere+ 推荐，DeBERTa 必须）；与 fp16 互斥")
    p.add_argument("--no_fp16", dest="fp16_default", action="store_false", default=True,
                   help="禁用 fp16")
    p.add_argument("--use_class_weights", action="store_true", default=True)
    p.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")

    # Phase 3 & Phase A 增强
    p.add_argument("--use_focal_loss", action="store_true", default=False)
    p.add_argument("--focal_loss_gamma", type=float, default=2.0)
    p.add_argument("--use_adversarial", action="store_true", default=False)
    p.add_argument("--adversarial_method", type=str, default="fgm", choices=["fgm", "pgd"])
    p.add_argument("--adversarial_epsilon", type=float, default=1.0)
    p.add_argument("--use_layerwise_lr_decay", action="store_true", default=False)
    p.add_argument("--layerwise_lr_decay_rate", type=float, default=0.95)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--use_rdrop", action="store_true", default=False)
    p.add_argument("--rdrop_alpha", type=float, default=1.0)
    p.add_argument("--use_eda", action="store_true", default=False)
    p.add_argument("--augment_target_classes", type=str, nargs="*", default=None)
    p.add_argument("--num_aug_per_sample", type=int, default=2)

    return p


# ------------------------------------------------------------------
# 数据准备
# ------------------------------------------------------------------

def _prepare_samples(args) -> Tuple[
    List[Dict[str, Any]],      # all samples（含 labels）
    List[str],                  # label_list
    Dict[str, int],             # label2id
    Dict[int, str],             # id2label
]:
    raw_data = load_json_data(args.data_path)
    samples = build_samples(raw_data, args.label_field, args.text_mode)
    samples = filter_rare_classes(samples, min_count=args.min_class_count)

    label_list, label2id, id2label = build_label_maps(samples)
    assign_label_ids(samples, label2id)
    print(f"总样本数: {len(samples)} | 标签数: {len(label_list)}")
    print(f"标签列表: {label_list}")
    return samples, label_list, label2id, id2label


def _split_fixed_test(
    samples: List[Dict[str, Any]],
    test_size: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """返回 (train_val_indices, test_indices) —— 索引均指向原 samples 列表。"""
    labels = [x["labels"] for x in samples]
    indices = list(range(len(samples)))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels,
    )
    return train_val_idx, test_idx


# ------------------------------------------------------------------
# 训练单次 fold × seed
# ------------------------------------------------------------------

def _build_train_cfg(args, seed: int, fold_output_dir: str) -> TrainConfig:
    return TrainConfig(
        output_dir=fold_output_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        seed=seed,
        use_class_weights=args.use_class_weights,
        fp16=torch.cuda.is_available() and args.fp16_default and not args.bf16,
        bf16=args.bf16 and torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=1,
        early_stopping_patience=args.early_stopping_patience,
        use_focal_loss=args.use_focal_loss,
        focal_loss_gamma=args.focal_loss_gamma,
        use_adversarial=args.use_adversarial,
        adversarial_method=args.adversarial_method,
        adversarial_epsilon=args.adversarial_epsilon,
        use_layerwise_lr_decay=args.use_layerwise_lr_decay,
        layerwise_lr_decay_rate=args.layerwise_lr_decay_rate,
        label_smoothing=args.label_smoothing,
        use_rdrop=args.use_rdrop,
        rdrop_alpha=args.rdrop_alpha,
        use_eda=args.use_eda,
        augment_target_classes=args.augment_target_classes,
        num_aug_per_sample=args.num_aug_per_sample,
    )


def _run_single(
    args,
    seed: int,
    fold_idx: int,
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    label_list: List[str],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    fold_output_dir: str,
) -> Dict[str, Any]:
    seed_everything(seed)

    data_cfg = DataConfig(
        data_path=args.data_path,
        text_mode=args.text_mode,
        label_field=args.label_field,
        test_size=args.test_size,
        dev_size=0.0,      # 不使用，外部已指定 dev
        min_class_count=args.min_class_count,
    )
    model_cfg = ModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
    )
    train_cfg = _build_train_cfg(args, seed, fold_output_dir)

    preloaded = {
        "train": train_samples,
        "dev": val_samples,      # 这里 dev = 本折 val（用于早停）
        "test": test_samples,
        "label_list": label_list,
        "label2id": label2id,
        "id2label": id2label,
    }

    result = run_hf_sequence_classification(
        data_cfg, model_cfg, train_cfg,
        preloaded_splits=preloaded,
        save_model=False,
    )

    return result


# ------------------------------------------------------------------
# 指标聚合
# ------------------------------------------------------------------

def _evaluate_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    preds = np.argmax(probs, axis=-1)
    num_classes = probs.shape[1]
    all_label_ids = list(range(num_classes))
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=all_label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, labels=all_label_ids, average="weighted", zero_division=0)),
        "precision": float(precision_score(labels, preds, labels=all_label_ids, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, labels=all_label_ids, average="macro", zero_division=0)),
    }
    report = classification_report(
        labels, preds,
        labels=all_label_ids,
        target_names=[id2label[i] for i in all_label_ids],
        digits=4, zero_division=0, output_dict=True,
    )
    return {"metrics": metrics, "classification_report": report}


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def main():
    args = _build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 载入全部样本并划分固定 test 集
    samples, label_list, label2id, id2label = _prepare_samples(args)
    trainval_idx, test_idx = _split_fixed_test(
        samples, args.test_size, seed=args.test_split_seed,
    )
    print(f"\n固定 test 集: {len(test_idx)} 条 | K-fold 池: {len(trainval_idx)} 条 | K={args.k_folds} | seeds={args.seeds}")

    trainval_samples = [samples[i] for i in trainval_idx]
    test_samples = [samples[i] for i in test_idx]
    trainval_labels = np.array([s["labels"] for s in trainval_samples])

    n_trainval = len(trainval_samples)
    n_test = len(test_samples)
    num_classes = len(label_list)

    # 2. 初始化 OOF 累加缓冲
    oof_probs = np.zeros((n_trainval, num_classes), dtype=np.float32)
    oof_seed_count = np.zeros(n_trainval, dtype=np.int32)   # 每个样本被多少 seed 累加过
    test_probs_accum = np.zeros((n_test, num_classes), dtype=np.float32)
    test_model_count = 0
    oof_labels = trainval_labels.copy()

    # 3. 遍历每个 seed（用于生成不同的 K 折划分 + 独立训练种子）
    per_run_summaries: List[Dict[str, Any]] = []
    global_t0 = time.time()

    for seed_i, seed in enumerate(args.seeds):
        print("\n" + "=" * 70)
        print(f"[Seed {seed_i+1}/{len(args.seeds)}] seed={seed}")
        print("=" * 70)

        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=seed)
        fold_iter = skf.split(np.zeros(n_trainval), trainval_labels)

        for fold_i, (tr_idx, val_idx) in enumerate(fold_iter):
            tag = f"seed{seed}_fold{fold_i+1}"
            fold_output_dir = os.path.join(args.output_dir, "_scratch", tag)

            train_samples = [trainval_samples[i] for i in tr_idx]
            val_samples = [trainval_samples[i] for i in val_idx]

            print(f"\n--- {tag}: train={len(train_samples)} val={len(val_samples)} test={n_test} ---")
            t0 = time.time()
            result = _run_single(
                args, seed, fold_i, train_samples, val_samples, test_samples,
                label_list, label2id, id2label, fold_output_dir,
            )
            elapsed = time.time() - t0

            dev_probs = result["_dev_probs"]       # 本折 val 概率（∈ trainval）
            t_probs = result["_test_probs"]        # 本折 test 概率（固定 test）

            # 累加 OOF（注意：val_idx 是相对 trainval 的索引）
            oof_probs[val_idx] += dev_probs
            oof_seed_count[val_idx] += 1

            # 累加 test probs（K×S 全部都评估同一个 test）
            test_probs_accum += t_probs
            test_model_count += 1

            fold_metrics = result["test_metrics"]
            val_metrics = result["val_metrics"]
            per_run_summaries.append({
                "seed": seed, "fold": fold_i + 1,
                "val_macro_f1": float(val_metrics.get("eval_macro_f1", 0.0)),
                "test_macro_f1": float(fold_metrics["macro_f1"]),
                "test_accuracy": float(fold_metrics["accuracy"]),
                "train_elapsed_seconds": round(elapsed, 2),
            })
            print(f"[{tag}] val_macro_f1={val_metrics.get('eval_macro_f1', 0.0):.4f}  "
                  f"test_macro_f1={fold_metrics['macro_f1']:.4f}  耗时 {elapsed:.1f}s")

            # 清理中间 checkpoint
            if not args.keep_fold_models:
                shutil.rmtree(fold_output_dir, ignore_errors=True)

    # 4. 归一化 OOF / test 概率
    if oof_seed_count.min() == 0:
        raise RuntimeError("存在样本没有被任何 fold 评估到，检查 K-fold 划分。")
    oof_probs /= oof_seed_count[:, None]
    test_probs = test_probs_accum / test_model_count

    # 5. 评估 OOF 与 test 集成效果
    print("\n" + "=" * 70)
    print("K-fold 集成总结")
    print("=" * 70)

    test_labels_arr = np.array([s["labels"] for s in test_samples])
    oof_eval = _evaluate_from_probs(oof_probs, oof_labels, id2label)
    test_eval = _evaluate_from_probs(test_probs, test_labels_arr, id2label)

    print(f"\n[OOF]  acc={oof_eval['metrics']['accuracy']:.4f}  macro_f1={oof_eval['metrics']['macro_f1']:.4f}")
    print(f"[Test ensemble] acc={test_eval['metrics']['accuracy']:.4f}  macro_f1={test_eval['metrics']['macro_f1']:.4f}")

    # 每 run macro_f1 的均值与方差
    per_test_f1 = [r["test_macro_f1"] for r in per_run_summaries]
    print(f"\n单模型 test_macro_f1 统计：mean={np.mean(per_test_f1):.4f} "
          f"std={np.std(per_test_f1):.4f}  min={np.min(per_test_f1):.4f}  max={np.max(per_test_f1):.4f}")

    # 6. 落盘
    np.save(os.path.join(args.output_dir, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(args.output_dir, "oof_labels.npy"), oof_labels)
    np.save(os.path.join(args.output_dir, "oof_index.npy"), np.array(trainval_idx))
    np.save(os.path.join(args.output_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(args.output_dir, "test_labels.npy"), test_labels_arr)

    summary = {
        "model_name": args.model_name,
        "k_folds": args.k_folds,
        "seeds": args.seeds,
        "num_models": test_model_count,
        "label_list": label_list,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "config": {
            "max_length": args.max_length,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "use_focal_loss": args.use_focal_loss,
            "use_adversarial": args.use_adversarial,
            "adversarial_method": args.adversarial_method,
            "use_layerwise_lr_decay": args.use_layerwise_lr_decay,
            "label_smoothing": args.label_smoothing,
            "use_rdrop": args.use_rdrop,
            "rdrop_alpha": args.rdrop_alpha,
            "use_eda": args.use_eda,
            "augment_target_classes": args.augment_target_classes,
        },
        "oof_metrics": oof_eval["metrics"],
        "oof_classification_report": oof_eval["classification_report"],
        "test_ensemble_metrics": test_eval["metrics"],
        "test_ensemble_classification_report": test_eval["classification_report"],
        "per_run_summaries": per_run_summaries,
        "total_elapsed_seconds": round(time.time() - global_t0, 2),
    }

    summary_path = os.path.join(args.output_dir, "kfold_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 清理 scratch
    shutil.rmtree(os.path.join(args.output_dir, "_scratch"), ignore_errors=True)

    print(f"\n全部完成，输出已写入: {args.output_dir}")
    print(f"- kfold_summary.json")
    print(f"- oof_probs.npy / oof_labels.npy / oof_index.npy")
    print(f"- test_probs.npy / test_labels.npy")
    print(f"\n集成 test macro_f1: {test_eval['metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
