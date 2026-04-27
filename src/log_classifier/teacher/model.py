import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class CodeBERTClassifier(nn.Module):
    """
    CodeBERT-based text/code classifier with an additional projection head
    for Supervised Contrastive Learning (SCL).
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        projection_dim: int = 256,
    ):
        super().__init__()
        self.num_labels = num_labels
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        
        hidden_size = self.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Projection head for SCL: Linear -> ReLU -> Linear -> L2 Norm
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        return_features: bool = True
    ) -> dict:
        """
        Forward pass.
        Returns:
            dict containing:
                - logits: Classification logits
                - features: Normalized SCL projection features (if return_features=True)
                - loss: Cross entropy loss (if labels are provided)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use pooler_output if available, else use CLS token (index 0)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        result = {"logits": logits}
        
        if return_features:
            proj_features = self.projection(pooled_output)
            # Normalize to unit sphere for contrastive loss
            proj_features = F.normalize(proj_features, p=2, dim=1)
            result["features"] = proj_features
            
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            result["loss"] = loss
            
        return result
