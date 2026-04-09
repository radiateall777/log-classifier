# 日志分类多分类任务 Baseline 对比方法

本目录包含2022-2024年间主流的Transformer-based文本/代码分类模型作为Baseline对比。

## 📋 支持的Baseline模型

| 模型 | HuggingFace ID | 简介 | 发表时间 |
|------|----------------|------|----------|
| BERT | `bert-base-uncased` | Google经典预训练模型 | 2018 |
| RoBERTa | `roberta-base` | Facebook优化版BERT | 2019 |
| DeBERTa-v3 | `microsoft/deberta-v3-base` | Microsoft增强版Transformer | 2021 (广泛使用于2022-2024) |
| ERNIE-2.0 | `nghuyong/ernie-2.0-base-en` | 百度知识增强预训练模型 | 2022 |
| MacBERT | `hfl/chinese-macbert-base` | 哈工大中文BERT替代模型 | 2020 (广泛使用于2022-2024) |
| XLNet | `xlnet-base-cased` | Google置换自回归模型 | 2019 |
| ALBERT | `albert-base-v2` | Google轻量级BERT | 2019 |
| ELECTRA | `google/electra-base-discriminator` | Google高效预训练模型 | 2020 |
| CodeBERT | `microsoft/codebert-base` | 微软代码理解预训练模型 | 2020 |
| GraphCodeBERT | `microsoft/graphcodebert-base` | 微软代码结构理解模型 | 2021 |
| UnixCoder | `Microsoft/unixcoder-base` | 微软统一代码理解模型 | 2022 |

## 🚀 快速开始

### 运行单个Baseline

```bash
python baselines/run_baseline.py --model bert-base-uncased
```

### 运行所有Baseline (一键脚本)

```bash
bash baselines/run_all_baselines.sh
```

### 自定义参数

```bash
python baselines/run_baseline.py \
    --model microsoft/deberta-v3-base \
    --max_length 256 \
    --batch_size 16 \
    --data_path ./data/random_samples.jsonl \
    --output_dir ./baseline_results \
    --label_field label3 \
    --text_mode user_assistant \
    --seed 42
```

## 📊 数据说明

- **数据文件**: `data/random_samples.jsonl`
- **标签字段**: `label3` (5分类任务)
- **类别**: 搜索算法、代码补全、动态规划、排序算法、Java Spring相关
- **样本数**: 5000条 (每类1000条)

## 📁 输出文件

运行后会在 `baseline_results/` 目录下生成：

- `{model_name}_results.json` - 详细评估结果
- `{model_name}_splits.json` - 数据划分结果
- `summary.json` - 所有Baseline汇总对比

## 📈 评估指标

- Accuracy (准确率)
- Macro F1 (宏平均F1)
- Weighted F1 (加权F1)
- Macro Precision (宏平均精确率)
- Macro Recall (宏平均召回率)
- Throughput (吞吐量 samples/s)

## ⚙️ 模型选择说明

本实现遵循以下原则：

1. **方法逻辑不变**: 严格采用各Baseline原始论文的实现方法
2. **仅修改数据加载**: 根据本项目数据格式调整输入输出
3. **仅修改评测部分**: 使用统一的评估指标和流程
4. **CPU推理**: 无需GPU即可运行所有模型

## 🔧 依赖

```
transformers>=4.30
torch>=2.0
scikit-learn>=1.0
numpy
pandas
```

安装依赖:
```bash
pip install transformers torch scikit-learn numpy pandas
```

## 📝 添加新的Baseline

如需添加新的Baseline模型，只需在 `run_all_baselines.sh` 中的 `BASELINES` 数组添加新的模型：

```bash
declare -a BASELINES=(
    "BERT:bert-base-uncased"
    "你的模型:model-id"
)
```
