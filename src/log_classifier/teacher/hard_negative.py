from typing import Dict

import numpy as np


def compute_hard_scores(
    features: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    top_k: int = 10,
    a_error: float = 1.0,
    b_low_conf: float = 1.0,
    c_nearest_neg: float = 1.0,
) -> np.ndarray:
    """
    Compute hard negative scores for each sample based on:
    - classification error
    - low confidence
    - nearest negative cosine similarity

    Args:
        features: [N, D] array of normalized embeddings.
        labels: [N] array of true labels.
        probs: [N] array of prediction confidences (probability of predicted class).
        preds: [N] array of predicted labels.
        top_k: (Unused explicitly in matrix version) Number of negatives to consider.
        a_error: Weight for classification error.
        b_low_conf: Weight for low confidence.
        c_nearest_neg: Weight for nearest negative similarity.

    Returns:
        scores: [N] normalized array of hard scores in [0, 1].
    """
    N = len(labels)
    scores = np.zeros(N, dtype=np.float32)

    # Convert to batched/matrix ops if dataset isn't huge.
    # For very large datasets, this might need batching, but we'll use a chunked approach.
    chunk_size = 1000
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk_features = features[i:end]  # [C, D]
        chunk_labels = labels[i:end]      # [C]
        chunk_preds = preds[i:end]        # [C]
        chunk_probs = probs[i:end]        # [C]

        # 1. Error score: 1 if pred != label else 0
        error_score = (chunk_preds != chunk_labels).astype(np.float32)

        # 2. Low confidence score: 1 - confidence
        conf_score = 1.0 - chunk_probs

        # 3. Nearest negative similarity
        # similarity matrix between chunk and all features: [C, N]
        sim_matrix = np.dot(chunk_features, features.T)
        
        # Mask out positives (same label) by setting similarity to -inf
        mask = (chunk_labels[:, None] == labels[None, :])
        sim_matrix[mask] = -1e9
        
        # Find max similarity for negatives
        neg_sim = np.max(sim_matrix, axis=1)
        
        # Bound similarity to [-1, 1] for safety, but it's typically in [-1, 1] for L2 normed
        neg_sim = np.clip(neg_sim, -1.0, 1.0)
        
        chunk_score = (
            a_error * error_score +
            b_low_conf * conf_score +
            c_nearest_neg * neg_sim
        )
        scores[i:end] = chunk_score

    # Normalize scores to [0, 1]
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score > min_score:
        scores = (scores - min_score) / (max_score - min_score)
    else:
        scores = np.zeros_like(scores)

    return scores


def scores_to_weights(scores: np.ndarray, gamma: float = 1.0, max_weight: float = 3.0) -> np.ndarray:
    """
    Convert normalized hard scores to sample weights.
    w_i = min(1.0 + gamma * score_i, max_weight)
    """
    weights = 1.0 + gamma * scores
    return np.clip(weights, 1.0, max_weight)


def save_hard_weights(path: str, ids: list, weights: np.ndarray) -> None:
    """
    Save mapping of id -> weight.
    """
    import os
    import json
    
    if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    data = {str(k): float(v) for k, v in zip(ids, weights)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_hard_weights(path: str) -> Dict[str, float]:
    """
    Load mapping of id -> weight.
    """
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}
