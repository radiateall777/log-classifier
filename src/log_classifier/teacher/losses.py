import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Compute Supervised Contrastive Loss.
    Args:
        features: [batch_size, dim] tensor of normalized features.
        labels: [batch_size] tensor of labels.
        temperature: Temperature parameter.
    Returns:
        Scalar loss.
    """
    device = features.device
    batch_size = features.shape[0]

    # Handle edge case: very small batch or no valid labels
    if batch_size < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    labels = labels.contiguous().view(-1, 1)
    
    # Mask of positives: mask[i, j] = 1 if labels[i] == labels[j]
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Remove self-contrast: mask[i, i] = 0
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    
    # Compute similarity logits
    # similarity: [batch_size, batch_size]
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    
    # For numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    
    # Mask out self-contrast from denominator
    exp_logits = torch.exp(logits) * logits_mask
    
    # Log_prob
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
    
    # Compute mean of log-likelihood over positive
    # Ensure denominator is at least 1 to avoid division by zero (for anchors with no positives)
    mask_sum = mask.sum(1)
    valid_anchors = mask_sum > 0
    
    # If no anchors have positives in the batch, return 0 loss
    if not valid_anchors.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / mask_sum[valid_anchors]
    
    # Loss
    loss = -mean_log_prob_pos.mean()
    return loss


def symmetric_kl_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute symmetric KL divergence: KL(p_a || p_b) + KL(p_b || p_a)
    Args:
        logits_a: [batch_size, num_classes]
        logits_b: [batch_size, num_classes]
        temperature: Temperature for sharpening.
    """
    p_a = F.log_softmax(logits_a / temperature, dim=-1)
    p_b = F.log_softmax(logits_b / temperature, dim=-1)
    
    prob_a = F.softmax(logits_a / temperature, dim=-1)
    prob_b = F.softmax(logits_b / temperature, dim=-1)
    
    kl_a2b = F.kl_div(p_b, prob_a, reduction='batchmean')
    kl_b2a = F.kl_div(p_a, prob_b, reduction='batchmean')
    
    return kl_a2b + kl_b2a


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute weighted cross entropy loss.
    If sample_weights is None, falls back to normal CE.
    Args:
        logits: [batch_size, num_classes]
        labels: [batch_size]
        sample_weights: [batch_size]
    """
    if sample_weights is None:
        return F.cross_entropy(logits, labels)
        
    ce = F.cross_entropy(logits, labels, reduction='none')
    weighted_ce = ce * sample_weights
    return weighted_ce.mean()
