"""Baseline 模型统一训练脚本。

完全复用 src/log_classifier 训练框架，通过原生 config 对象
（DataConfig / ModelConfig / TrainConfig）调用训练流水线。

用法::
    # 单模型
    python3 baselines/run_baseline_train.py --model_name bert-base-uncased

    # 所有 baseline
    bash baselines/run_all_baselines_train.sh
"""

import os
import sys
import json
import argparse

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
    p.add_argument("--early_stopping_patience", type=int, default=2)

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
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        use_class_weights=args.use_class_weights,
        fp16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        early_stopping_patience=args.early_stopping_patience,
    )

    seed_everything(train_cfg.seed)

    # 调用原生 pipeline，训练、评估、保存全在里面
    result = run_hf_sequence_classification(data_cfg, model_cfg, train_cfg)

    # 持久化结构化结果（pipeline 内部已保存模型/tokenizer/splits）
    result_path = os.path.join(
        args.output_dir,
        f"{args.model_name.replace('/', '_')}_train_results.json",
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {result_path}")
    print(f"Macro F1: {result['test_metrics']['macro_f1']:.4f}")
    return result


if __name__ == "__main__":
    main()
