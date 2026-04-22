"""Baseline 模型统一训练脚本。

完全复用 src/log_classifier 训练框架，通过原生 config 对象
（DataConfig / ModelConfig / TrainConfig）调用训练流水线。

用法::
    # 单模型
    python3 baselines/python/train_single.py --model_name bert-base-uncased

    # 全套 Phase A 增强（R-Drop + Label Smoothing + Layerwise LR + EDA 搜索算法 + FGM）
    python3 baselines/python/train_single.py \\
        --model_name microsoft/deberta-v3-base \\
        --use_adversarial --adversarial_method fgm \\
        --use_rdrop --rdrop_alpha 1.0 \\
        --label_smoothing 0.1 \\
        --use_layerwise_lr_decay --layerwise_lr_decay_rate 0.9 \\
        --use_eda --augment_target_classes 搜索算法 --num_aug_per_sample 3
"""

import os
import sys
import json
import argparse

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch
from log_classifier.config import DataConfig, ModelConfig, TrainConfig
from log_classifier.pipelines import run_hf_sequence_classification
from log_classifier.utils import seed_everything


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline 模型统一训练")

    # 数据
    p.add_argument("--data_path", default="./data/random_samples.jsonl")
    p.add_argument("--text_mode", default="user_assistant")
    p.add_argument("--label_field", default="label3")
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--dev_size", type=float, default=0.1)
    p.add_argument("--min_class_count", type=int, default=2)

    # 模型
    p.add_argument("--model_name", required=True)
    p.add_argument("--max_length", type=int, default=256)

    # 训练输出
    p.add_argument("--output_dir", default=None,
                   help="默认为 ./baseline_results/<model_name>")

    # 训练超参（一一对应 TrainConfig）
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="梯度累积步数：等效 batch = batch × grad_accum，用于大模型")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_class_weights", dest="use_class_weights",
                   action="store_true", default=True)
    p.add_argument("--no_class_weights", dest="use_class_weights",
                   action="store_false")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--bf16", action="store_true", default=False,
                   help="使用 bfloat16（Ampere+ 推荐，DeBERTa 必须）；与 fp16 互斥")
    p.add_argument("--no_fp16", dest="fp16_default", action="store_false", default=True,
                   help="禁用 fp16（例如 DeBERTa 应禁 fp16 并配合 --bf16）")

    # Phase 3 enhancements
    p.add_argument("--use_focal_loss", action="store_true", default=False)
    p.add_argument("--focal_loss_gamma", type=float, default=2.0)
    p.add_argument("--use_adversarial", action="store_true", default=False)
    p.add_argument("--adversarial_method", type=str, default="fgm", choices=["fgm", "pgd"])
    p.add_argument("--adversarial_epsilon", type=float, default=1.0)
    p.add_argument("--use_layerwise_lr_decay", action="store_true", default=False)
    p.add_argument("--layerwise_lr_decay_rate", type=float, default=0.95)

    # Phase A（0.95+）增强
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="标签平滑 ε；推荐 0.1。与 focal 可叠加。")
    p.add_argument("--use_rdrop", action="store_true", default=False,
                   help="启用 R-Drop 正则化（同 batch 两次前向 + KL）")
    p.add_argument("--rdrop_alpha", type=float, default=1.0)
    p.add_argument("--use_eda", action="store_true", default=False,
                   help="启用 EDA 数据增强（仅训练集）")
    p.add_argument("--augment_target_classes", type=str, nargs="*", default=None,
                   help="只增强指定类别（用于弱势类定向增强）；不指定则全类增强")
    p.add_argument("--num_aug_per_sample", type=int, default=2)
    p.add_argument("--eda_alpha_ri", type=float, default=0.1)
    p.add_argument("--eda_alpha_rs", type=float, default=0.1)
    p.add_argument("--eda_p_rd", type=float, default=0.1)

    # Phase B 协作：是否额外保存 dev/test 概率到 npz（供 Stacking 用）
    p.add_argument("--save_probs", action="store_true", default=False,
                   help="保存 dev/test softmax 概率到 output_dir/probs.npz")

    return p


def main():
    args = _build_parser().parse_args()

    # 每个模型输出到独立目录
    if args.output_dir is None:
        args.output_dir = f"./baseline_results/{args.model_name.replace('/', '_')}"

    # 用原生 dataclass 配置，不做任何转换
    data_cfg = DataConfig(
        data_path=args.data_path,
        text_mode=args.text_mode,
        label_field=args.label_field,
        test_size=args.test_size,
        dev_size=args.dev_size,
        min_class_count=args.min_class_count,
    )
    model_cfg = ModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
    )
    train_cfg = TrainConfig(
        output_dir=args.output_dir,
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
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
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
        eda_alpha_ri=args.eda_alpha_ri,
        eda_alpha_rs=args.eda_alpha_rs,
        eda_p_rd=args.eda_p_rd,
    )

    seed_everything(train_cfg.seed)

    # 调用原生 pipeline，训练、评估、保存全在里面
    result = run_hf_sequence_classification(data_cfg, model_cfg, train_cfg)

    # 可选：保存概率矩阵（供 Stacking 复用）
    if args.save_probs:
        probs_path = os.path.join(args.output_dir, "probs.npz")
        np.savez_compressed(
            probs_path,
            dev_probs=result["_dev_probs"],
            test_probs=result["_test_probs"],
            test_preds=result["_test_preds"],
            test_labels=result["_test_labels"],
        )
        print(f"概率已保存: {probs_path}")

    # 持久化结构化结果（去掉内部 numpy 字段）
    serializable = {k: v for k, v in result.items() if not k.startswith("_")}
    result_path = os.path.join(
        args.output_dir,
        f"{args.model_name.replace('/', '_')}_train_results.json",
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {result_path}")
    print(f"Macro F1: {result['test_metrics']['macro_f1']:.4f}")
    return result


if __name__ == "__main__":
    main()
