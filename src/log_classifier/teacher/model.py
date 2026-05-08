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
        **kwargs,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        return_features: bool = False,
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

        encoder_outputs = self.encoder(**encoder_kwargs)

        # Use CLS token representation.
        pooled = encoder_outputs.last_hidden_state[:, 0]

        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)

        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
        }

        if labels is not None:
            outputs["loss"] = self.loss_fn(logits, labels)

        if return_features:
            outputs["features"] = pooled

        return outputs