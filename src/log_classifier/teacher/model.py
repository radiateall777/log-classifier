import torch
import torch.nn as nn
from transformers import AutoModel


class CodeBERTClassifier(nn.Module):
    """Clean encoder classifier for CE-only training.

    Default path:
        AutoModel encoder -> CLS pooled feature -> dropout -> linear classifier

    This class is intentionally minimal for fair backbone comparison:
        - microsoft/codebert-base
        - microsoft/graphcodebert-base
        - microsoft/unixcoder-base

    No metric branch, no confusion head, no DP/Search head.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout_prob: float = 0.1,
        pooling_mode: str = "cls",
        classifier_hidden_dim: int = 0,
        multi_sample_dropout_num: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.pooling_mode = pooling_mode
        self.multi_sample_dropout_num = max(1, int(multi_sample_dropout_num))

        if self.pooling_mode not in {"cls", "cls_mean"}:
            raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}")

        self.dropout = nn.Dropout(dropout_prob)
        head_input_dim = hidden_size if self.pooling_mode == "cls" else hidden_size * 2

        if int(classifier_hidden_dim) > 0:
            self.pre_classifier = nn.Sequential(
                nn.LayerNorm(head_input_dim),
                nn.Linear(head_input_dim, int(classifier_hidden_dim)),
                nn.GELU(),
            )
            classifier_input_dim = int(classifier_hidden_dim)
        else:
            self.pre_classifier = nn.Identity()
            classifier_input_dim = head_input_dim

        self.feature_dim = classifier_input_dim
        self.classifier = nn.Linear(classifier_input_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def pool_features(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None):
        cls_pooled = last_hidden_state[:, 0]

        if self.pooling_mode == "cls":
            return cls_pooled

        if attention_mask is None:
            mean_pooled = last_hidden_state.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            masked_hidden = last_hidden_state * mask
            token_count = mask.sum(dim=1).clamp_min(1.0)
            mean_pooled = masked_hidden.sum(dim=1) / token_count

        return torch.cat([cls_pooled, mean_pooled], dim=-1)

    def compute_logits(self, pooled: torch.Tensor):
        features = self.pre_classifier(pooled)
        if self.multi_sample_dropout_num == 1:
            return self.classifier(self.dropout(features))

        logits = 0.0
        for _ in range(self.multi_sample_dropout_num):
            logits = logits + self.classifier(self.dropout(features))
        return logits / float(self.multi_sample_dropout_num), features

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        return_features: bool = False,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Some BERT-like models accept token_type_ids. RoBERTa-like models do not need it.
        # Passing it only when provided keeps this compatible with multiple backbones.
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        if return_hidden_states:
            encoder_kwargs["output_hidden_states"] = True

        encoder_outputs = self.encoder(**encoder_kwargs)

        pooled = self.pool_features(encoder_outputs.last_hidden_state, attention_mask)
        logits_result = self.compute_logits(pooled)
        if isinstance(logits_result, tuple):
            logits, features = logits_result
        else:
            logits = logits_result
            features = self.pre_classifier(pooled)

        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
        }

        if labels is not None:
            outputs["loss"] = self.loss_fn(logits, labels)

        if return_features:
            outputs["features"] = features

        if return_hidden_states:
            outputs["hidden_states"] = encoder_outputs.hidden_states

        return outputs
