"""PyTorch Lightning DataModule：复用 preprocess.py 产出标准 DataLoader。"""

from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from log_classifier.config import DataConfig, ModelConfig, TrainConfig
from log_classifier.data.preprocess import (
    assign_label_ids,
    build_label_maps,
    build_samples,
    filter_rare_classes,
    load_json_data,
    split_dataset,
)


class _TokenizedDataset(Dataset):
    """将 preprocess 产出的 sample list 做 tokenize 后包装为 PyTorch Dataset。"""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        texts = [s["text"] for s in samples]
        self.labels = [s["labels"] for s in samples]
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length, padding="max_length",
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class LogClassifierDataModule(pl.LightningDataModule):
    """封装数据加载全流程，对外暴露 label 映射供 pipeline / model 使用。"""

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.label_list: List[str] = []
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}

        self._train_ds: Optional[_TokenizedDataset] = None
        self._val_ds: Optional[_TokenizedDataset] = None
        self._test_ds: Optional[_TokenizedDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        raw_data = load_json_data(self.data_cfg.data_path)
        samples = build_samples(raw_data, self.data_cfg.label_field, self.data_cfg.text_mode)
        samples = filter_rare_classes(samples, min_count=self.data_cfg.min_class_count)

        self.label_list, self.label2id, self.id2label = build_label_maps(samples)
        assign_label_ids(samples, self.label2id)

        train_data, dev_data, test_data = split_dataset(
            samples,
            seed=self.train_cfg.seed,
            test_size=self.data_cfg.test_size,
            dev_size=self.data_cfg.dev_size,
        )

        print(f"总样本数: {len(samples)}")
        print(f"标签数: {len(self.label_list)}")
        print(f"标签列表: {self.label_list}")
        print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name, use_fast=True)

        self._train_ds = _TokenizedDataset(train_data, self.tokenizer, self.model_cfg.max_length)
        self._val_ds = _TokenizedDataset(dev_data, self.tokenizer, self.model_cfg.max_length)
        self._test_ds = _TokenizedDataset(test_data, self.tokenizer, self.model_cfg.max_length)

        self._train_labels = [s["labels"] for s in train_data]

    @property
    def num_labels(self) -> int:
        return len(self.label_list)

    @property
    def train_labels(self) -> List[int]:
        return self._train_labels

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.train_cfg.train_batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self.train_cfg.eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None
        return DataLoader(
            self._test_ds,
            batch_size=self.train_cfg.eval_batch_size,
            shuffle=False,
            num_workers=0,
        )
