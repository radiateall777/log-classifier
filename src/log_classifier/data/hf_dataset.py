"""HuggingFace Dataset 适配层：将通用样本转为 HF DatasetDict 并完成 tokenize。"""

from typing import Any, Dict, List

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase


def build_hf_dataset_dict(
    train_data: List[Dict[str, Any]],
    dev_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
) -> DatasetDict:
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(dev_data),
        "test": Dataset.from_list(test_data),
    })


def tokenize_datasets(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> DatasetDict:
    def _tokenize(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset_dict.map(_tokenize, batched=True)

    keep_columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized["train"].column_names:
        keep_columns.append("token_type_ids")

    tokenized.set_format(type="torch", columns=keep_columns)
    return tokenized
