"""单折 worker：只跑 K-fold 中指定 (seed, fold_idx) 的一个 fold。

被 `run_kfold_parallel.py` 调度器在独立 subprocess 里调用，通过 `CUDA_VISIBLE_DEVICES`
指定 GPU。worker 把本折的 dev 概率、test 概率与指标写到 `--output_path`（npz），
主进程读取后聚合为 OOF + K×S 平均 test 概率。

用法示例：
    CUDA_VISIBLE_DEVICES=3 python baselines/run_fold_worker.py \\
        --model_name roberta-large \\
        --seed 42 --fold_idx 0 --kfold_k 5 \\
        --test_split_seed 42 --test_size 0.1 \\
        --output_path ./baseline_results/roberta_large_sota/_scratch/seed42_fold0/result.npz \\
        ... (其余训练参数与 run_kfold_train.py 相同)
"""
import argparse
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from log_classifier.config import DataConfig, ModelConfig, TrainConfig
from log_classifier.data.preprocess import (
    assign_label_ids, build_label_maps, build_samples,
    filter_rare_classes, load_json_data,
)
from log_classifier.pipelines import run_hf_sequence_classification
from log_classifier.utils import seed_everything


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="K-fold single fold worker")

    # 数据
    p.add_argument("--data_path", default="./data/random_samples.jsonl")
    p.add_argument("--text_mode", default="user_assistant")
    p.add_argument("--label_field", default="label3")
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--test_split_seed", type=int, default=42)
    p.add_argument("--min_class_count", type=int, default=2)

    # 本折定位
    p.add_argument("--seed", type=int, required=True, help="本折的训练 seed（也用于 K-fold 划分）")
    p.add_argument("--fold_idx", type=int, required=True, help="0-indexed fold 编号")
    p.add_argument("--kfold_k", type=int, default=5)

    # 模型
    p.add_argument("--model_name", required=True)
    p.add_argument("--max_length", type=int, default=512)

    # 输出
    p.add_argument("--output_path", required=True, help="结果 npz 保存路径")
    p.add_argument("--scratch_dir", required=True, help="本折训练中间文件的落地目录")

    # 训练超参
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=20)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--early_stopping_patience", type=int, default=4)
    p.add_argument("--use_class_weights", action="store_true", default=True)
    p.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--no_fp16", dest="fp16_default", action="store_false", default=True)

    # Phase A 技巧
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


def main():
    args = _build_parser().parse_args()
    seed_everything(args.seed)

    # 1. 读全量样本、做固定 test 划分（必须与 parallel 调度器一致）
    raw_data = load_json_data(args.data_path)
    samples = build_samples(raw_data, args.label_field, args.text_mode)
    samples = filter_rare_classes(samples, min_count=args.min_class_count)
    label_list, label2id, id2label = build_label_maps(samples)
    assign_label_ids(samples, label2id)

    labels_all = [x["labels"] for x in samples]
    indices = list(range(len(samples)))
    trainval_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.test_split_seed,
        stratify=labels_all,
    )
    trainval_samples = [samples[i] for i in trainval_idx]
    test_samples = [samples[i] for i in test_idx]
    trainval_labels = np.array([s["labels"] for s in trainval_samples])

    # 2. 按本 seed 的 K-fold 划分，找本折的 train / val 索引
    skf = StratifiedKFold(n_splits=args.kfold_k, shuffle=True, random_state=args.seed)
    splits = list(skf.split(np.zeros(len(trainval_samples)), trainval_labels))
    if not (0 <= args.fold_idx < args.kfold_k):
        raise ValueError(f"fold_idx {args.fold_idx} 超出 [0, {args.kfold_k})")
    tr_idx, val_idx = splits[args.fold_idx]

    train_data = [trainval_samples[i] for i in tr_idx]
    val_data = [trainval_samples[i] for i in val_idx]

    print(f"[Worker seed={args.seed} fold={args.fold_idx}] "
          f"train={len(train_data)} val={len(val_data)} test={len(test_samples)}")

    # 3. 调用 pipeline
    data_cfg = DataConfig(
        data_path=args.data_path,
        text_mode=args.text_mode,
        label_field=args.label_field,
        test_size=args.test_size,
        dev_size=0.0,
        min_class_count=args.min_class_count,
    )
    model_cfg = ModelConfig(model_name=args.model_name, max_length=args.max_length)
    train_cfg = TrainConfig(
        output_dir=args.scratch_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
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

    preloaded = {
        "train": train_data,
        "dev": val_data,
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

    # 4. 持久化
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez_compressed(
        args.output_path,
        val_indices=val_idx.astype(np.int64),         # 相对 trainval 的索引
        val_probs=result["_dev_probs"].astype(np.float32),
        test_probs=result["_test_probs"].astype(np.float32),
        val_macro_f1=np.float32(result["val_metrics"].get("eval_macro_f1", 0.0)),
        test_macro_f1=np.float32(result["test_metrics"]["macro_f1"]),
        test_accuracy=np.float32(result["test_metrics"]["accuracy"]),
        train_elapsed_seconds=np.float32(result.get("train_elapsed_seconds", 0.0)),
        seed=np.int64(args.seed),
        fold_idx=np.int64(args.fold_idx),
    )
    print(f"[Worker seed={args.seed} fold={args.fold_idx}] "
          f"val_f1={result['val_metrics'].get('eval_macro_f1', 0):.4f} "
          f"test_f1={result['test_metrics']['macro_f1']:.4f} "
          f"已写入 {args.output_path}")


if __name__ == "__main__":
    main()
