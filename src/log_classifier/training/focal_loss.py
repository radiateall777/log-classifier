"""Focal Loss for handling class imbalance and hard examples.

Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
Adapted for multi-class text classification.

支持可选的 label smoothing（Szegedy et al., 2016），把硬标签
one-hot 软化成 (1-ε) · onehot + ε/K，缓解过拟合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class weighting and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            smooth_targets = torch.full_like(log_probs, eps / num_classes)
            smooth_targets.scatter_(
                dim=-1,
                index=targets.unsqueeze(-1),
                value=1.0 - eps + eps / num_classes,
            )
            pt = (probs * smooth_targets).sum(dim=-1)
            ce_loss = -(log_probs * smooth_targets).sum(dim=-1)
        else:
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            pt = (probs * targets_one_hot).sum(dim=-1)
            ce_loss = -(log_probs * targets_one_hot).sum(dim=-1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.weight is not None:
            w = self.weight.to(logits.device)
            class_weights = w[targets]
            loss = loss * class_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
