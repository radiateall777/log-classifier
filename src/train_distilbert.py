"""DistilBERT 序列分类训练入口。

DistilBERT 相较 BERT 层数减半（6 层 vs 12 层），参数量约为 66M（BERT-base 110M），
推理速度提升约 60%，适合作为轻量级 baseline 对比。

用法::
    python3 src/train_distilbert.py
    python3 src/train_distilbert.py --cpu          # 强制 CPU
    python3 src/train_distilbert.py --batch_size 8  # 自定义 batch size
"""

import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from log_classifier.config import DataConfig, ModelConfig, TrainConfig
from log_classifier.pipelines import run_hf_sequence_classification
from log_classifier.utils import seed_everything


def _gpu_has_enough_memory(min_free_gb: float = 2.0) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / (1024 ** 3)
        print(f"[Info] GPU 0: {free_gb:.1f} GiB 空闲 / {total / (1024**3):.1f} GiB 总计")
        return free_gb >= min_free_gb
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="DistilBERT 序列分类训练")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 训练")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    use_gpu = (not args.cpu) and _gpu_has_enough_memory(min_free_gb=2.0)
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[Info] 使用 CPU 训练")
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        print("[Info] 使用 GPU 训练")

    data_cfg = DataConfig()
    model_cfg = ModelConfig(
        model_name="distilbert-base-uncased",
        max_length=256,
    )
    train_cfg = TrainConfig(
        output_dir="./output/distilbert_base_label3",
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        early_stopping_patience=3,
        fp16=use_gpu,
    )

    seed_everything(train_cfg.seed)
    result = run_hf_sequence_classification(data_cfg, model_cfg, train_cfg)
    print(f"\n训练完成，Macro F1: {result['test_metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
