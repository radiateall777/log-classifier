"""扩展版 HuggingFace Trainer，支持：

- 类别权重 + Focal Loss + Label Smoothing
- FGM / PGD 对抗训练
- R-Drop 正则化（Wu et al. 2021）
- 分层学习率衰减（ULMFiT-style）

所有增强通过构造参数开关，默认行为与原生 Trainer 一致。
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from log_classifier.training.focal_loss import FocalLoss
from log_classifier.training.adversarial import FGM, PGD


class WeightedTrainer(Trainer):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_loss_gamma: float = 2.0,
        use_adversarial: bool = False,
        adversarial_method: str = "fgm",
        adversarial_epsilon: float = 1.0,
        label_smoothing: float = 0.0,
        use_rdrop: bool = False,
        rdrop_alpha: float = 1.0,
        use_layerwise_lr_decay: bool = False,
        layerwise_lr_decay_rate: float = 0.95,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_loss_gamma = focal_loss_gamma
        self.use_adversarial = use_adversarial
        self.adversarial_method = adversarial_method
        self.adversarial_epsilon = adversarial_epsilon
        self.label_smoothing = label_smoothing
        self.use_rdrop = use_rdrop
        self.rdrop_alpha = rdrop_alpha
        self.use_layerwise_lr_decay = use_layerwise_lr_decay
        self.layerwise_lr_decay_rate = layerwise_lr_decay_rate
        self._adv_module = None

    # ------------------------------------------------------------------
    # 损失函数
    # ------------------------------------------------------------------

    def _get_loss_fn(self, device: torch.device) -> nn.Module:
        weight = self.class_weights.to(device) if self.class_weights is not None else None
        if self.use_focal_loss:
            return FocalLoss(
                gamma=self.focal_loss_gamma,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.label_smoothing,
        )

    def _plain_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor, num_labels: int) -> torch.Tensor:
        loss_fct = self._get_loss_fn(logits.device)
        return loss_fct(logits.view(-1, num_labels), labels.view(-1))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        num_labels = model.config.num_labels

        outputs = model(**inputs)
        logits = outputs.get("logits")
        ce_loss = self._plain_ce_loss(logits, labels, num_labels)

        if not self.use_rdrop:
            loss = ce_loss
        else:
            outputs2 = model(**inputs)
            logits2 = outputs2.get("logits")
            ce_loss2 = self._plain_ce_loss(logits2, labels, num_labels)

            log_p1 = F.log_softmax(logits, dim=-1)
            log_p2 = F.log_softmax(logits2, dim=-1)
            p1 = log_p1.exp()
            p2 = log_p2.exp()
            kl_12 = F.kl_div(log_p1, p2, reduction="batchmean")
            kl_21 = F.kl_div(log_p2, p1, reduction="batchmean")
            kl = 0.5 * (kl_12 + kl_21)

            loss = 0.5 * (ce_loss + ce_loss2) + self.rdrop_alpha * kl

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # 训练 step：对抗训练（与 R-Drop 兼容，compute_loss 自动生效）
    # ------------------------------------------------------------------

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self.use_adversarial:
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        if self._adv_module is None:
            if self.adversarial_method == "fgm":
                self._adv_module = FGM(model, epsilon=self.adversarial_epsilon)
            else:
                self._adv_module = PGD(model, epsilon=self.adversarial_epsilon)

        if isinstance(self._adv_module, FGM):
            self._adv_module.attack()
            with self.compute_loss_context_manager():
                adv_loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                adv_loss = adv_loss.mean()
            self.accelerator.backward(adv_loss)
            self._adv_module.restore()
        else:
            self._adv_module.backup_grad()
            for step in range(3):
                self._adv_module.attack(is_first=(step == 0))
                if step != 2:
                    model.zero_grad()
                with self.compute_loss_context_manager():
                    adv_loss = self.compute_loss(model, inputs)
                if self.args.n_gpu > 1:
                    adv_loss = adv_loss.mean()
                self.accelerator.backward(adv_loss)
            self._adv_module.restore()
            self._adv_module.restore_grad()

        return loss.detach() / self.args.gradient_accumulation_steps

    # ------------------------------------------------------------------
    # 分层学习率衰减
    # ------------------------------------------------------------------

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if not self.use_layerwise_lr_decay:
            return super().create_optimizer()

        base_lr = self.args.learning_rate
        weight_decay = self.args.weight_decay
        decay = self.layerwise_lr_decay_rate
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")

        model = self.model
        try:
            encoder_layers = model.base_model.encoder.layer
        except AttributeError:
            encoder_layers = None

        if encoder_layers is None:
            # 非标准结构（如 DistilBERT 用 transformer.layer，或纯 encoder）
            # 退化为原生优化器
            print("[Warning] 未找到 encoder.layer，分层 LR 退化为单 LR。")
            return super().create_optimizer()

        num_layers = len(encoder_layers)
        param_groups: list[dict] = []
        used_param_ids: set[int] = set()

        def _add_group(params: list[tuple[str, torch.nn.Parameter]], lr: float):
            decay_params = [p for n, p in params if not any(nd in n for nd in no_decay)]
            no_decay_params = [p for n, p in params if any(nd in n for nd in no_decay)]
            if decay_params:
                param_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
                used_param_ids.update(id(p) for p in decay_params)
            if no_decay_params:
                param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
                used_param_ids.update(id(p) for p in no_decay_params)

        named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        # 1. 分类头（最高 LR）
        head = [(n, p) for n, p in named_params if ("classifier" in n) or ("pooler" in n)]
        _add_group(head, lr=base_lr)

        # 2. 编码器各层：底层 LR 小
        for idx in range(num_layers):
            layer_lr = base_lr * (decay ** (num_layers - idx - 1))
            layer_name = f"encoder.layer.{idx}."
            layer_params = [(n, p) for n, p in named_params if layer_name in n]
            _add_group(layer_params, lr=layer_lr)

        # 3. Embeddings（最低 LR）
        embed_lr = base_lr * (decay ** num_layers)
        embed = [(n, p) for n, p in named_params if "embeddings" in n]
        _add_group(embed, lr=embed_lr)

        # 4. 兜底：其余未分组的参数（如 task-specific 投影）
        leftover = [(n, p) for n, p in named_params if id(p) not in used_param_ids]
        if leftover:
            _add_group(leftover, lr=base_lr)

        # 使用 Trainer 原生 optimizer 类（通常是 AdamW）
        optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        # learning_rate 已在 param_groups 里，移除默认值避免覆盖
        optim_kwargs.pop("lr", None)
        self.optimizer = optim_cls(param_groups, **optim_kwargs)

        head_lr = base_lr
        bottom_lr = base_lr * (decay ** num_layers)
        print(f"[Info] 分层 LR 衰减 rate={decay}, 共 {num_layers} 层，head_lr={head_lr:.2e} → embed_lr={bottom_lr:.2e}")
        return self.optimizer
