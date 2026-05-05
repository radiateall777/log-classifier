import csv
import json
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from log_classifier.data.preprocess import build_label_maps, load_json_data


def read_dataset(path: str, split: str = None) -> List[Dict[str, Any]]:
    """
    Read dataset from a JSONL, JSON, or CSV file.
    Expects each sample to have 'id', 'text', and 'label_text'.
    If 'label_text' is missing but 'label' is present, maps it to 'label_text'.
    """
    if path.endswith(".csv"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = {
                    "id": row.get("id"),
                    "text": row.get("text", ""),
                    "label_text": str(row.get("label_text", row.get("label", ""))).strip(),
                }
                data.append(sample)
        return data
    else:
        # Reuse existing robust JSON/JSONL loader
        raw_data = load_json_data(path)
        
        if isinstance(raw_data, dict):
            if split and split in raw_data:
                raw_data = raw_data[split]
            elif "train" in raw_data or "test" in raw_data:
                # If no split is specified but it looks like a split dict, default to train or return all?
                # Actually, if we have a split, we should definitely use it.
                pass
                
        data = []
        for item in raw_data:
            sample = {
                "id": item.get("id"),
                "text": item.get("text", ""),
                "label_text": str(item.get("label_text", item.get("label", ""))).strip(),
            }
            data.append(sample)
        return data


def build_label_mapping(samples: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build label mapping from samples.
    Returns:
        label2id: Dict mapping label string to integer ID.
        id2label: Dict mapping integer ID to label string.
    """
    _, label2id, id2label = build_label_maps(samples)
    return label2id, id2label


class ClassificationDataset(Dataset):
    """
    Dataset class that serves the samples and converts label strings to IDs.
    Does not modify the original text (preserving it for Stage 2 augmentation).
    """

    def __init__(self, samples: List[Dict[str, Any]], label2id: Dict[str, int]):
        self.samples = samples
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        label_text = item["label_text"]
        label_id = self.label2id[label_text]

        return {
            "id": item["id"],
            "text": item["text"],
            "label": label_id,
            "label_text": label_text,
        }
