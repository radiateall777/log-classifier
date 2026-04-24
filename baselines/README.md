# 日志分类多分类任务 Baseline 对比方法

本目录包含2022-2024年间主流的Transformer-based文本/代码分类模型作为Baseline对比。

## 📋 支持的Baseline模型

| 模型 | HuggingFace ID | 简介 | 发表时间 |
|------|----------------|------|----------|
| BERT | `bert-base-uncased` | Google经典预训练模型 | 2018 |
| RoBERTa | `roberta-base` | Facebook优化版BERT | 2019 |
| DeBERTa-v3 | `microsoft/deberta-v3-base` | Microsoft增强版Transformer | 2021 |
| ERNIE-2.0 | `nghuyong/ernie-2.0-base-en` | 百度知识增强预训练模型 | 2019 |
| MacBERT | `hfl/chinese-macbert-base` | 哈工大中文BERT替代模型 | 2020 |
| XLNet | `xlnet-base-cased` | Google置换自回归模型 | 2019 |
| ALBERT | `albert-base-v2` | Google轻量级BERT | 2019 |
| ELECTRA | `google/electra-base-discriminator` | Google高效预训练模型 | 2020 |
| CodeBERT | `microsoft/codebert-base` | 微软代码理解预训练模型 | 2020 |

---

## 🚀 工作流：训练与评估

整个评测分为两个步骤：**第一步：微调训练** -> **第二步：鲁棒性与吞吐评估**。

### 第一步：模型训练 (微调)

你可以使用 `train.py` 单独微调一个模型：

```bash
python baselines/dl/train.py --model_name bert-base-uncased
```

或者使用一键脚本训练所有 Baseline 模型：

```bash
bash baselines/dl/run_all_train.sh
```

### 第二步：模型评估 (噪音鲁棒度与吞吐量)

使用 `eval.py` 对**已经完成微调**的模型单独进行噪音测试与吞吐量性能测试：

```bash
python baselines/dl/eval.py --model_dir ./outputs/baselines/dl/bert-base-uncased
```
或者一键评测 `outputs/baselines/dl` 目录下的所有微调模型，并汇总出评估报告：

```bash
bash baselines/dl/run_all_eval.sh
```

## 📊 数据说明

- **数据文件**: `data/random_samples.jsonl`
- **标签字段**: `label3` (5分类任务)
- **类别**: 搜索算法、代码补全、动态规划、排序算法、Java Spring相关
- **样本数**: 5000条 (每类1000条)

## 📁 输出文件目录结构

运行完整的两步命令后，`outputs/baselines/dl/` 目录下将生成如下内容：

```text
outputs/baselines/dl/
├── bert-base-uncased/
│   ├── best_model.pt                  # 抽取的轻量优选 checkpoint
│   ├── config.json / model.safetensors# HuggingFace 模型权重
│   ├── label_mappings.json            # 类别 ID 的映射关系
│   ├── noise_robustness_results.json  # 该模型不同噪音容忍下的测试结果
│   ...
├── summary_train.json                 # 训练阶段基于准确率的全局排序
└── summary_eval_robustness.json       # 评估阶段噪音与吞吐的全局汇总
```

## 📈 评估指标

- Accuracy (准确率)
- Macro F1 (宏平均F1)
- Weighted F1 (加权F1)
- Throughput (吞吐量 samples/s)

这几个指标均会在不同参数 `[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]` 的 UNK replacement 中统计以获取其降级曲线。

## 🔧 依赖

```text
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.0.0
numpy>=1.20.0
pandas>=1.3.0
datasets>=2.0.0
accelerate>=0.20.0
tiktoken>=0.4.0
```

安装依赖:
```bash
pip install -r baselines/requirements.txt
```
