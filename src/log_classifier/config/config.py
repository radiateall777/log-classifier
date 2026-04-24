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

    # Phase 3：Focal Loss
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0

    # Phase 3：对抗训练
    use_adversarial: bool = False
    adversarial_method: str = "fgm"
    adversarial_epsilon: float = 1.0

    # Phase 3：分层学习率衰减
    use_layerwise_lr_decay: bool = False
    layerwise_lr_decay_rate: float = 0.95

    # Phase A（0.95+ 升级）：Label Smoothing
    label_smoothing: float = 0.0

    # Phase A：R-Drop 正则化
    use_rdrop: bool = False
    rdrop_alpha: float = 1.0

    # Phase A：EDA 数据增强（训练集）
    use_eda: bool = False
    augment_target_classes: Optional[List[str]] = None   # None 表示所有类都增强
    num_aug_per_sample: int = 2
    eda_alpha_ri: float = 0.1
    eda_alpha_rs: float = 0.1
    eda_p_rd: float = 0.1

    # Phase A：梯度累积（支持大模型小 batch 等效大 batch）
    gradient_accumulation_steps: int = 1
