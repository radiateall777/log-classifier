"""领域数据处理：与训练框架无关的纯数据逻辑。"""

import json
from collections import Counter
from typing import Any, Dict, List, Tuple

from sklearn.model_selection import train_test_split


def load_json_data(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        # 寻找第一个非空白字符
        while True:
            char = f.read(1)
            if not char or not char.isspace():
                break
        f.seek(0)

        if char in ("[", "{"):
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    if not isinstance(data, (list, dict)):
        raise ValueError("数据文件必须是 JSON list, JSON dict 或 JSONL 格式。")
    return data


def flatten_messages(
    messages: List[Dict[str, str]],
    text_mode: str,
    item: Dict[str, Any],
) -> str:
    user_texts: List[str] = []
    assistant_texts: List[str] = []

    for msg in messages:
        role = msg.get("role", "").strip()
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            user_texts.append(content)
        elif role == "assistant":
            assistant_texts.append(content)

    user_text = " ".join(user_texts).strip()
    assistant_text = " ".join(assistant_texts).strip()

    if text_mode == "user_only":
        return f"user: {user_text}".strip()

    if text_mode == "assistant_only":
        return f"assistant: {assistant_text}".strip()

    if text_mode == "user_assistant":
        return f"user: {user_text} assistant: {assistant_text}".strip()

    if text_mode == "with_meta":
        language = str(item.get("language", "")).strip()
        dataset_name = str(item.get("dataset", "")).strip()
        return (
            f"language: {language} "
            f"dataset: {dataset_name} "
            f"user: {user_text} "
            f"assistant: {assistant_text}"
        ).strip()

    raise ValueError(f"不支持的 text_mode: {text_mode}")


def build_samples(
    raw_data: List[Dict[str, Any]],
    label_field: str,
    text_mode: str,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for item in raw_data:
        messages = item.get("messages", [])
        if not isinstance(messages, list):
            continue

        label = item.get(label_field, None)
        if label is None:
            continue

        text = flatten_messages(messages, text_mode, item)
        if not text:
            continue

        samples.append({
            "id": item.get("id", None),
            "text": text,
            "label_text": str(label).strip(),
        })

    if not samples:
        raise ValueError("没有构造出有效样本，请检查数据格式。")
    return samples


def filter_rare_classes(
    samples: List[Dict[str, Any]],
    min_count: int = 2,
) -> List[Dict[str, Any]]:
    counter = Counter(x["label_text"] for x in samples)
    kept = [x for x in samples if counter[x["label_text"]] >= min_count]
    removed = len(samples) - len(kept)
    if removed > 0:
        print(f"[Info] 移除了 {removed} 条属于极少数类的样本（每类少于 {min_count} 条）。")
    return kept


def build_label_maps(
    samples: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    label_list = sorted({x["label_text"] for x in samples})
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_list, label2id, id2label


def assign_label_ids(
    samples: List[Dict[str, Any]],
    label2id: Dict[str, int],
) -> None:
    for x in samples:
        x["labels"] = label2id[x["label_text"]]


def split_dataset(
    samples: List[Dict[str, Any]],
    seed: int,
    test_size: float,
    dev_size: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    labels = [x["labels"] for x in samples]
    indices = list(range(len(samples)))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels,
    )

    train_val_labels = [labels[i] for i in train_val_idx]
    relative_dev_size = dev_size / (1.0 - test_size)

    train_idx, dev_idx = train_test_split(
        train_val_idx, test_size=relative_dev_size, random_state=seed,
        stratify=train_val_labels,
    )

    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in dev_idx],
        [samples[i] for i in test_idx],
    )
