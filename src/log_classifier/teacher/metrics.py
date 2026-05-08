from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: List[int],
    id2label: Dict[int, str]
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: List of ground truth label IDs.
        y_pred: List of predicted label IDs.
        labels: List of all possible label IDs.
        id2label: Mapping from label ID to label name.
        
    Returns:
        Dictionary containing accuracy, macro-F1, weighted-F1, per-class metrics,
        and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    per_class = {}
    for i, label_id in enumerate(labels):
        label_name = id2label[label_id]
        per_class[label_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]) if support is not None else 0
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }
