"""BERT 序列分类训练入口（PyTorch Lightning）。

用法::
    uv run python src/train_bert_lightning.py
"""

from log_classifier.config import DataConfig, ModelConfig, TrainConfig
from log_classifier.pipelines import run_lightning_sequence_classification
from log_classifier.utils import seed_everything


def main() -> None:
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    seed_everything(train_cfg.seed)
    run_lightning_sequence_classification(data_cfg, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
