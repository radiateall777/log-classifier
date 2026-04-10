# log-classifier

日志分类多分类任务训练框架，支持 HuggingFace `Trainer` 和 PyTorch Lightning 两种训练方式。

## 📁 项目结构

```
log-classifier/
├── src/
│   ├── log_classifier/
│   │   ├── config/          # 数据、模型、训练配置 (DataConfig / ModelConfig / TrainConfig)
│   │   ├── data/            # 数据加载、预处理、Lightning DataModule
│   │   │   ├── preprocess.py        # 纯数据逻辑，与框架无关
│   │   │   ├── hf_dataset.py        # HuggingFace Dataset 适配层
│   │   │   └── lightning_datamodule.py
│   │   ├── models/          # 模型 & tokenizer 构建
│   │   ├── pipelines/        # 训练流水线 (HF Trainer / Lightning Trainer)
│   │   ├── training/         # WeightedTrainer / 评估指标
│   │   └── utils/            # 随机种子设置
│   ├── train_bert.py         # HuggingFace Trainer 训练入口
│   └── train_bert_lightning.py  # PyTorch Lightning 训练入口
├── baselines/                 # Baseline 对比实验
│   ├── run_baseline.py       # 单模型评估脚本
│   └── run_all_baselines.sh  # 一键运行所有 baseline
├── data/
│   └── random_samples.jsonl   # 5000 条样本，5分类 (label3)
└── baseline_results/          # baseline 运行结果
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -e .
```

或使用 pip 安装核心依赖：

```bash
pip install transformers>=5.5.0 torch>=2.11.0 pytorch-lightning>=2.0.0 \
    scikit-learn>=1.8.0 datasets>=4.8.4 accelerate>=1.13.0
```

### 运行训练

```bash
# HuggingFace Trainer 方式
python src/train_bert.py

# PyTorch Lightning 方式
python src/train_bert_lightning.py
```

### 运行 Baseline 对比

```bash
# 单模型
python baselines/run_baseline.py --model bert-base-uncased

# 所有 baseline
bash baselines/run_all_baselines.sh
```

## ⚙️ 配置说明

| 配置类 | 关键参数 | 默认值 |
|--------|---------|--------|
| `DataConfig` | `data_path` | `./data/random_samples.jsonl` |
| | `label_field` | `label3` |
| | `text_mode` | `user_assistant` |
| | `test_size` / `dev_size` | `0.1` / `0.1` |
| `ModelConfig` | `model_name` | `bert-base-uncased` |
| | `max_length` | `256` |
| `TrainConfig` | `train_batch_size` | `16` |
| | `learning_rate` | `2e-5` |
| | `num_train_epochs` | `5` |
| | `use_class_weights` | `True` |

### text_mode 选项

- `user_only` — 仅使用用户消息
- `assistant_only` — 仅使用助手回复
- `user_assistant` — 拼接 `user: ... assistant: ...`
- `with_meta` — 包含 language / dataset 元信息

## 📊 Baseline 模型

| 模型 | HuggingFace ID | 备注 |
|------|---------------|------|
| BERT | `bert-base-uncased` | |
| RoBERTa | `roberta-base` | |
| DeBERTa-v3 | `microsoft/deberta-v3-base` | |
| ERNIE-2.0 | `nghuyong/ernie-2.0-base-en` | |
| MacBERT | `hfl/chinese-macbert-base` | |
| XLNet | `xlnet-base-cased` | |
| ALBERT | `albert-base-v2` | |
| ELECTRA | `google/electra-base-discriminator` | |
| CodeBERT | `microsoft/codebert-base` | |
