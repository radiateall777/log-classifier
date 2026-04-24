# baselines/ — 实验运行脚本

按 **Phase** 组织的日志分类实验入口。所有结果输出到 `../baseline_results/`，
排行榜见 [`../baseline_results/README.md`](../baseline_results/README.md)。

## 📁 目录结构

| 模型 | HuggingFace ID | 简介 | 发表时间 |
|------|----------------|------|----------|
| BERT | `bert-base-uncased` | Google经典预训练模型 | 2018 |
| DistilBERT | `distilbert-base-uncased` | 经知识蒸馏的轻量BERT | 2019 |
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

## 📊 数据与依赖

- 数据：`../data/random_samples.jsonl`（5000 条，5 类各 1000）
- 标签字段：`label3`；固定 test 划分：`test_split_seed=42, test_size=0.1`
- 依赖：见项目根 [`pyproject.toml`](../pyproject.toml)，安装 `uv sync` 或 `pip install -e ..`

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

## ⚠️ 已知问题

- **DeBERTa-v3** 在本 pipeline 上不收敛（train loss 震荡 1.6，eval f1 稳定 0.067），已改用 `roberta-large` 作为强 backbone。
- 训练完成后 transformer fold 权重**不保留**（`_scratch` 目录会被清理），只保存 OOF/test 概率的 `.npy`；Stacking 依赖这些 `.npy` 即可复现。如需对新样本推理，请使用 Phase 3 的 `model.safetensors`（本地保留，未推送 git）。

---

_此 README 记录脚本的运行方式；实验结果排行榜由 `python/regen_summary.py` 自动生成到 `../baseline_results/README.md`。_
