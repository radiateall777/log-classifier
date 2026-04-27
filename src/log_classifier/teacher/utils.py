import json
import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: str) -> None:
    """Save data to a JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    """Ensure that a directory exists."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_device() -> torch.device:
    """Get the available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_label_mapping(id2label: dict, label2id: dict, path: str) -> None:
    """Save label mapping to a JSON file."""
    data = {"id2label": id2label, "label2id": label2id}
    save_json(data, path)


def load_label_mapping(path: str) -> tuple[dict, dict]:
    """Load label mapping from a JSON file."""
    data = load_json(path)
    # Convert string keys back to int for id2label
    id2label = {int(k): v for k, v in data["id2label"].items()}
    label2id = data["label2id"]
    return id2label, label2id
