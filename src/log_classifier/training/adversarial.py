"""Adversarial training methods (FGM / PGD) for robustness.

Miyato et al., "Adversarial Training Methods for Semi-Supervised Text Classification", ICLR 2017.
"""

import torch


class FGM:
    """Fast Gradient Method for adversarial perturbation on embeddings."""

    def __init__(self, model: torch.nn.Module, epsilon: float = 1.0, emb_name: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self._backup: dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self._backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    perturbation = self.epsilon * param.grad / norm
                    param.data.add_(perturbation)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self._backup:
                    param.data = self._backup[name]
        self._backup = {}


class PGD:
    """Projected Gradient Descent for adversarial perturbation."""

    def __init__(
        self,
        model: torch.nn.Module,
        epsilon: float = 1.0,
        alpha: float = 0.3,
        emb_name: str = "word_embeddings",
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_name = emb_name
        self._backup: dict[str, torch.Tensor] = {}
        self._grad_backup: dict[str, torch.Tensor | None] = {}

    def attack(self, is_first: bool = False) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first:
                    self._backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    perturbation = self.alpha * param.grad / norm
                    param.data.add_(perturbation)
                    param.data = self._project(name, param.data)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self._backup:
                    param.data = self._backup[name]
        self._backup = {}

    def _project(self, param_name: str, param_data: torch.Tensor) -> torch.Tensor:
        if param_name in self._backup:
            delta = param_data - self._backup[param_name]
            norm = torch.norm(delta)
            if norm > self.epsilon:
                delta = self.epsilon * delta / norm
            return self._backup[param_name] + delta
        return param_data

    def backup_grad(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self._grad_backup[name] = param.grad.clone()

    def restore_grad(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self._grad_backup:
                    param.grad = self._grad_backup[name]
        self._grad_backup = {}
