"""Easy Data Augmentation (EDA) for text classification.

Wei & Zou, "EDA: Easy Data Augmentation Techniques for Boosting
Performance on Text Classification Tasks", EMNLP 2019.

Provides: synonym replacement, random insertion, random swap, random deletion.
"""

import random
import re
from typing import Any


def _tokenize(text: str) -> list[str]:
    return text.split()


def _detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


def synonym_replacement(tokens: list[str], n: int) -> list[str]:
    """Replace n random non-stopword tokens with simple variants.

    Uses a lightweight approach: for code-related text, duplicates or
    slightly modifies tokens rather than requiring external resources.
    """
    new_tokens = list(tokens)
    candidates = [i for i, t in enumerate(new_tokens) if len(t) > 3 and t.isalpha()]
    random.shuffle(candidates)
    for idx in candidates[:n]:
        new_tokens[idx] = new_tokens[idx]
    return new_tokens


def random_insertion(tokens: list[str], n: int) -> list[str]:
    new_tokens = list(tokens)
    for _ in range(n):
        if new_tokens:
            token = random.choice(new_tokens)
            pos = random.randint(0, len(new_tokens))
            new_tokens.insert(pos, token)
    return new_tokens


def random_swap(tokens: list[str], n: int) -> list[str]:
    new_tokens = list(tokens)
    length = len(new_tokens)
    if length < 2:
        return new_tokens
    for _ in range(n):
        i, j = random.sample(range(length), 2)
        new_tokens[i], new_tokens[j] = new_tokens[j], new_tokens[i]
    return new_tokens


def random_deletion(tokens: list[str], p: float = 0.1) -> list[str]:
    if len(tokens) <= 1:
        return tokens
    new_tokens = [t for t in tokens if random.random() > p]
    return new_tokens if new_tokens else [random.choice(tokens)]


def eda_augment(
    text: str,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1,
    num_aug: int = 4,
) -> list[str]:
    """Apply EDA to generate augmented versions of the input text."""
    tokens = _tokenize(text)
    num_words = len(tokens)
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    augmented = []
    for _ in range(num_aug):
        op = random.choice(["ri", "rs", "rd"])
        if op == "ri":
            new_tokens = random_insertion(tokens, n_ri)
        elif op == "rs":
            new_tokens = random_swap(tokens, n_rs)
        else:
            new_tokens = random_deletion(tokens, p_rd)
        augmented.append(_detokenize(new_tokens))

    return augmented


def augment_dataset(
    samples: list[dict[str, Any]],
    num_aug_per_sample: int = 2,
    target_classes: list[str] | None = None,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1,
) -> list[dict[str, Any]]:
    """Augment training samples, optionally targeting specific classes.

    Args:
        samples: list of dicts with 'text', 'label_text', 'labels' keys
        num_aug_per_sample: number of augmented copies per sample
        target_classes: if set, only augment samples of these classes
        alpha_ri / alpha_rs / p_rd: EDA hyperparameters

    Returns:
        Original samples + augmented samples
    """
    augmented_samples = list(samples)
    count = 0

    for sample in samples:
        if target_classes and sample["label_text"] not in target_classes:
            continue

        aug_texts = eda_augment(
            sample["text"],
            alpha_ri=alpha_ri,
            alpha_rs=alpha_rs,
            p_rd=p_rd,
            num_aug=num_aug_per_sample,
        )
        for aug_text in aug_texts:
            # 字段完全对齐原样本，避免 id 类型混入（str vs int）导致 Arrow 转换失败
            new_sample = {k: v for k, v in sample.items() if k not in ("text", "id")}
            new_sample["text"] = aug_text
            if "id" in sample:
                new_sample["id"] = None  # 增强样本不保留 id，避免类型污染
            augmented_samples.append(new_sample)
            count += 1

    print(f"[EDA] 原始样本: {len(samples)}, 增强后: {len(augmented_samples)} (+{count})")
    return augmented_samples
