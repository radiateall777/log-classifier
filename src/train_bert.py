"""BERT 序列分类训练入口。

用法::
    python3 src/train_bert.py
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from log_classifier.config import DataConfig, ModelConfig, TrainConfig
from log_classifier.pipelines import run_hf_sequence_classification
from log_classifier.utils import seed_everything


def main() -> None:
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    seed_everything(train_cfg.seed)
    result = run_hf_sequence_classification(data_cfg, model_cfg, train_cfg)
    print(f"\n训练完成，Macro F1: {result['test_metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
