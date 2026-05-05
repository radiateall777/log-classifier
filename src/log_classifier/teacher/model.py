import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class CosFaceHead(nn.Module):
    """CosFace classification head over normalized metric features.

    During training, the cosine logit for the ground-truth class is reduced by
    ``margin`` before applying the global ``scale``. At inference time, pass
    ``labels=None`` to get scaled cosine logits without margin.
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 16.0, margin: float = 0.15):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=-1)
        weight = F.normalize(self.weight, p=2, dim=-1)
        logits = F.linear(features, weight)

        if labels is not None:
            if labels.dim() != 1:
                labels = labels.view(-1)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            logits = logits - one_hot * self.margin

        return logits * self.scale


class CodeBERTClassifier(nn.Module):
    """CodeBERT classifier with an optional metric-learning branch.

    The default path remains unchanged for Stage 1:
        CodeBERT encoder -> CLS pooled feature -> dropout -> linear classifier.

    The metric branch is activated only when ``return_metric=True`` and therefore
    remains backward-compatible with older Stage 1 training code and checkpoints.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout_prob: float = 0.1,
        metric_dim: int = 256,
        cosface_scale: float = 16.0,
        cosface_margin: float = 0.15,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.metric_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, metric_dim),
        )
        self.metric_head = CosFaceHead(
            in_features=metric_dim,
            num_classes=num_labels,
            scale=cosface_scale,
            margin=cosface_margin,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        return_features: bool = False,
        return_metric: bool = False,
        metric_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs,
        }
        # RoBERTa/CodeBERT does not need token_type_ids, but keeping this makes
        # the module tolerant to callers that pass them for BERT-like models.
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        encoder_outputs = self.encoder(**encoder_kwargs)
        pooled = encoder_outputs.last_hidden_state[:, 0]
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)

        outputs: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            outputs["loss"] = self.loss_fn(logits, labels)

        if return_features:
            outputs["features"] = pooled

        if return_metric:
            metric_features = F.normalize(self.metric_projector(pooled), p=2, dim=-1)
            head_labels = metric_labels if metric_labels is not None else labels
            metric_logits = self.metric_head(metric_features, labels=head_labels)
            outputs["metric_features"] = metric_features
            outputs["metric_logits"] = metric_logits

        return outputs
