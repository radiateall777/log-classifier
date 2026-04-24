from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class DataConfig:
    """数据读取、样本构造、划分相关配置。"""
    data_path: str = "./data/random_samples.jsonl"
    text_mode: str = "user_assistant"      # user_only / assistant_only / user_assistant / with_meta
    label_field: str = "label3"
    test_size: float = 0.1
    dev_size: float = 0.1
    min_class_count: int = 2


@dataclass
class ModelConfig:
    """模型 & tokenizer 相关配置。"""
    model_name: str = "bert-base-uncased"
    max_length: int = 512


@dataclass
class TrainConfig:
    """训练超参、运行时、保存策略等配置。"""
    output_dir: str = "./output/bert_base_label3"
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 5
    warmup_ratio: float = 0.1
    seed: int = 42

    use_class_weights: bool = True
    fp16: bool = field(default_factory=lambda: torch.cuda.is_available())
    # 使用 bfloat16 替代 fp16；bf16 有更大指数范围、不需要 GradScaler，
    # DeBERTa-v3 等模型在 fp16 下会触发 "Attempting to unscale FP16 gradients" 错误，必须用 bf16。
    # Ampere+ (RTX 3090/A100 等) 硬件支持 bf16。
    bf16: bool = False
    logging_steps: int = 50
    save_total_limit: int = 2
    early_stopping_patience: int = 5

    # Baseline 不需要其它花哨的配置
