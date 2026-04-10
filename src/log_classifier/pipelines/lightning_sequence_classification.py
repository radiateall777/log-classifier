"""PyTorch Lightning 序列分类训练流水线。

职责与 hf_sequence_classification.py 对称：
只做编排，不包含数据处理或模型构建细节。
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from log_classifier.config import DataConfig, ModelConfig, TrainConfig
from log_classifier.data.lightning_datamodule import LogClassifierDataModule
from log_classifier.models.hf_classifier import build_model
from log_classifier.models.lightning_classifier import LitSequenceClassifier


# ------------------------------------------------------------------
# 内部辅助
# ------------------------------------------------------------------

def _compute_class_weights(
    train_labels: List[int],
    enabled: bool,
) -> Optional[torch.Tensor]:
    if not enabled:
        return None
    labels_arr = np.array(train_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_arr),
        y=labels_arr,
    )
    tensor = torch.tensor(weights, dtype=torch.float)
    print(f"[Info] 使用 class weights: {tensor.tolist()}")
    return tensor  # type: ignore[return-value]


def _save_artifacts(
    output_dir: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    label_map_path = os.path.join(output_dir, "label_mappings.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            f, ensure_ascii=False, indent=2,
        )


# ------------------------------------------------------------------
# 公开接口
# ------------------------------------------------------------------

def run_lightning_sequence_classification(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> None:
    # ---- 数据 ----
    dm = LogClassifierDataModule(data_cfg, model_cfg, train_cfg)
    dm.setup()

    # ---- 模型 ----
    backbone = build_model(
        model_cfg.model_name,
        dm.num_labels,
        dm.id2label,
        dm.label2id,
    )

    class_weights = _compute_class_weights(dm.train_labels, train_cfg.use_class_weights)

    steps_per_epoch = len(dm.train_dataloader())
    num_training_steps = steps_per_epoch * train_cfg.num_train_epochs

    lit_model = LitSequenceClassifier(
        backbone=backbone,
        num_labels=dm.num_labels,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        num_training_steps=num_training_steps,
        class_weights=class_weights,
    )

    # ---- Callbacks ----
    checkpoint_cb = ModelCheckpoint(
        dirpath=train_cfg.output_dir,
        filename="best-{epoch}-{val_macro_f1:.4f}",
        monitor="val_macro_f1",
        mode="max",
        save_top_k=train_cfg.save_total_limit,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=train_cfg.early_stopping_patience,
    )

    # ---- Trainer ----
    precision = "16-mixed" if train_cfg.fp16 else "32-true"
    trainer = Trainer(
        default_root_dir=train_cfg.output_dir,
        max_epochs=train_cfg.num_train_epochs,
        precision=precision,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=train_cfg.logging_steps,
        deterministic=True,
        enable_progress_bar=True,
    )

    # ---- 训练 ----
    print("\n========== 开始训练 (Lightning) ==========")
    trainer.fit(lit_model, datamodule=dm)

    # ---- 测试 ----
    print("\n========== 测试集结果 ==========")
    num_test_samples = len(dm.test_dataloader().dataset)
    t_start = time.perf_counter()
    trainer.test(lit_model, datamodule=dm, ckpt_path="best")
    t_elapsed = time.perf_counter() - t_start

    throughput = num_test_samples / t_elapsed if t_elapsed > 0 else float("inf")
    print({
        "test_samples": num_test_samples,
        "test_elapsed_sec": round(t_elapsed, 3),
        "test_throughput_samples_per_sec": round(throughput, 2),
    })

    # ---- 分类报告 ----
    test_preds = lit_model.last_test_preds
    test_labels = lit_model.last_test_labels

    if test_preds is not None and len(test_preds) > 0:
        print("\n========== 分类报告 ==========")
        print(classification_report(
            test_labels, test_preds,
            target_names=[dm.id2label[i] for i in range(dm.num_labels)],
            digits=4, zero_division=0,
        ))

    # ---- 保存 HF 模型 + tokenizer（与 HF pipeline 格式一致） ----
    best_model_path = checkpoint_cb.best_model_path
    if best_model_path:
        best_lit = LitSequenceClassifier.load_from_checkpoint(
            best_model_path, backbone=backbone,
        )
        best_lit.backbone.save_pretrained(train_cfg.output_dir)
    else:
        lit_model.backbone.save_pretrained(train_cfg.output_dir)

    dm.tokenizer.save_pretrained(train_cfg.output_dir)
    _save_artifacts(train_cfg.output_dir, dm.label2id, dm.id2label)

    print(f"\n模型和标签映射已保存到: {train_cfg.output_dir}")
