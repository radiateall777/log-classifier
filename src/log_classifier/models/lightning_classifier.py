"""LightningModule 封装：backbone 无关的序列分类器。"""

from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup


class LitSequenceClassifier(pl.LightningModule):
    """通用序列分类 LightningModule。

    Parameters
    ----------
    backbone : nn.Module
        任意 backbone，forward 需返回含 ``logits`` 键的对象或 dict。
    num_labels : int
        分类标签数。
    learning_rate, weight_decay, warmup_ratio : float
        优化器 / scheduler 超参。
    num_training_steps : int
        总训练步数，用于 linear warmup scheduler。
    class_weights : torch.Tensor | None
        类别权重，传入则使用带权 CrossEntropyLoss。
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_labels: int,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        num_training_steps: int = 1000,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "class_weights"])

        self.backbone = backbone
        self.num_labels = num_labels

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.class_weights = None
            self.loss_fn = nn.CrossEntropyLoss()

        self._val_preds: List[np.ndarray] = []
        self._val_labels: List[np.ndarray] = []
        self._test_preds: List[np.ndarray] = []
        self._test_labels: List[np.ndarray] = []

        self.last_test_preds: Optional[np.ndarray] = None
        self.last_test_labels: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs: Any) -> Any:
        return self.backbone(**kwargs)

    def _shared_step(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = batch.pop("labels")
        outputs = self.backbone(**batch)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits, labels

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, logits, labels = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, logits, labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        preds = torch.argmax(logits, dim=-1)
        self._val_preds.append(preds.cpu().numpy())
        self._val_labels.append(labels.cpu().numpy())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = np.concatenate(self._val_preds)
        labels = np.concatenate(self._val_labels)
        self._val_preds.clear()
        self._val_labels.clear()

        self.log("val_accuracy", accuracy_score(labels, preds), prog_bar=True, sync_dist=True)
        self.log("val_macro_f1", f1_score(labels, preds, average="macro", zero_division=0), prog_bar=True, sync_dist=True)
        self.log("val_weighted_f1", f1_score(labels, preds, average="weighted", zero_division=0), sync_dist=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, logits, labels = self._shared_step(batch)
        self.log("test_loss", loss, sync_dist=True)

        preds = torch.argmax(logits, dim=-1)
        self._test_preds.append(preds.cpu().numpy())
        self._test_labels.append(labels.cpu().numpy())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        preds = np.concatenate(self._test_preds)
        labels = np.concatenate(self._test_labels)
        self._test_preds.clear()
        self._test_labels.clear()

        self.last_test_preds = preds
        self.last_test_labels = labels

        self.log("test_accuracy", accuracy_score(labels, preds), sync_dist=True)
        self.log("test_macro_f1", f1_score(labels, preds, average="macro", zero_division=0), sync_dist=True)
        self.log("test_weighted_f1", f1_score(labels, preds, average="weighted", zero_division=0), sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        no_decay = {"bias", "LayerNorm.weight"}
        params = [
            {
                "params": [p for n, p in self.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.backbone.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)

        warmup_steps = int(self.hparams.num_training_steps * self.hparams.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
