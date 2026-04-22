# 日志分类器优化方案

本文档说明在原有 baseline 基础上所做的全部改进、新增方法，以及运行指南。

---

## 1 现状诊断

### 1.1 数据集概况

| 项目 | 值 |
|------|----|
| 数据文件 | `data/random_samples.jsonl` |
| 总样本 | 5 000（每类 1 000，均衡） |
| 划分 | Train 3 999 / Dev 501 / Test 500（分层抽样） |
| 标签字段 | `label3`（5 类） |
| 类别 | Java Spring 相关、代码补全、动态规划、排序算法、搜索算法 |
| 文本构造 | `user: … assistant: …` 拼接 |

### 1.2 原始 Baseline 结果（test macro F1）

**传统 ML 方法**

| 方法 | Macro F1 | Accuracy | 训练耗时 |
|------|----------|----------|----------|
| FastText | 0.8948 | 0.896 | 6.1s |
| TF-IDF + Linear SVM | 0.8907 | 0.892 | 0.7s |
| TF-IDF + Logistic Regression | 0.8887 | 0.890 | 37.7s |
| TF-IDF + Naive Bayes | 0.8743 | 0.876 | 0.02s |

**Transformer 微调**

| 模型 | Macro F1 | Accuracy | 训练耗时 |
|------|----------|----------|----------|
| bert-base-uncased | 0.8803 | 0.882 | 279s |
| distilbert-base-uncased | 0.8746 | 0.876 | 103s |
| nghuyong/ernie-2.0-base-en | 0.8628 | 0.864 | 330s |
| roberta-base | 0.8529 | 0.858 | 195s |
| hfl/chinese-macbert-base | 0.8399 | 0.842 | 154s |

> DeBERTa-v3、ALBERT、ELECTRA、CodeBERT 训练失败，无结果。

### 1.3 发现的问题

1. **Transformer 全面低于 ML 方法**：FastText (0.895) > BERT (0.880)，差距 1.5 个百分点
2. **"搜索算法"类是瓶颈**：所有方法中该类 recall 仅 0.72–0.78，F1 仅 0.78–0.80
3. **max_length=256 截断严重**：代码对话文本通常较长，截断丢失了关键信息
4. **early_stopping_patience=2 过于激进**：模型可能在还能提升时就被终止
5. **多个强模型未跑通**：DeBERTa-v3（base 级别最强分类模型）无结果
6. **无高级训练技术**：未使用数据增强、对抗训练、focal loss 等
7. **无集成方法**：未尝试模型融合

---

## 2 优化方案总览

改进分为四个阶段，按优先级从高到低排列：

```
Phase 1  修复 Transformer 训练配置     ── 预期 0.90–0.92
Phase 2  引入新方法                    ── 预期 0.92–0.94
Phase 3  高级训练技术                  ── 预期 0.93–0.95
Phase 4  模型集成                      ── 预期 0.95+
```

---

## 3 Phase 1：修复 Transformer 训练

### 3.1 max_length 256 → 512

代码对话文本较长，增大输入长度避免截断关键信息。

**修改文件**：
- `src/log_classifier/config/config.py` — `ModelConfig.max_length` 默认值 256 → 512
- `baselines/run_all_baselines_train.sh` — `MAX_LENGTH=512`

### 3.2 early_stopping_patience 2 → 5

原始设置过于激进，给模型更多收敛时间。

**修改文件**：
- `src/log_classifier/config/config.py` — `TrainConfig.early_stopping_patience` 默认值 2 → 5
- `baselines/run_all_baselines_train.sh`
- `baselines/run_baseline_train.py`

### 3.3 修复 DeBERTa-v3 等失败模型

DeBERTa-v3 在众多 NLU benchmark 上是 base 级别最强模型。脚本已保留该模型在训练列表中，增大 patience 后可望正常收敛。

### 3.4 text_mode=with_meta 对比实验

新增实验：在文本前添加 `language:` 和 `dataset:` 元信息，可能帮助区分语义相似的类别（如"搜索算法" vs "动态规划"）。

**运行脚本**：`baselines/run_enhanced_train.sh`（含 4 模型 × 4 策略 + with_meta 对比）

---

## 4 Phase 2：引入新方法

### 4.1 SetFit — 小数据集专用

SetFit 使用 Sentence Transformers 做对比学习 + 轻量分类头，在小数据集（100–5000 样本）上显著优于标准微调。

**新增文件**：`baselines/run_setfit_baseline.py`

```bash
python3 baselines/run_setfit_baseline.py \
    --model_name sentence-transformers/all-mpnet-base-v2
```

### 4.2 Sentence-Transformer 嵌入 + 传统分类器

用预训练 Sentence-BERT 生成稠密嵌入向量（768 维），再接传统分类器，结合了深度语义表示与传统方法的优势。

提供 4 种分类器组合：
- SBERT + Logistic Regression
- SBERT + SVM (RBF kernel)
- SBERT + XGBoost
- SBERT + LightGBM

**新增文件**：`baselines/run_embedding_baselines.py`

```bash
python3 baselines/run_embedding_baselines.py --method all \
    --encoder sentence-transformers/all-mpnet-base-v2
```

### 4.3 依赖

新增依赖：`sentence-transformers`、`setfit`、`xgboost`、`lightgbm`（已更新 `pyproject.toml` 和 `baselines/requirements.txt`）。

---

## 5 Phase 3：高级训练技术

### 5.1 Focal Loss

针对难分类样本（如"搜索算法"类）的损失函数，通过降低易分类样本的权重来聚焦难例。

**新增文件**：`src/log_classifier/training/focal_loss.py`

公式：`FL(pt) = -α(1-pt)^γ · log(pt)`，默认 `γ=2.0`。

```bash
python3 baselines/run_baseline_train.py \
    --model_name bert-base-uncased \
    --use_focal_loss --focal_loss_gamma 2.0
```

### 5.2 FGM / PGD 对抗训练

在 embedding 层注入对抗扰动，提升模型鲁棒性和泛化能力。训练时间增加约 30%，但不影响推理速度。

**新增文件**：`src/log_classifier/training/adversarial.py`

- **FGM**（Fast Gradient Method）：单步扰动，实现简单，开销小
- **PGD**（Projected Gradient Descent）：多步扰动，效果更强但更慢

```bash
python3 baselines/run_baseline_train.py \
    --model_name bert-base-uncased \
    --use_adversarial --adversarial_method fgm
```

### 5.3 EDA 数据增强

Easy Data Augmentation：通过随机插入、交换、删除操作生成增强样本，支持定向增强弱势类。

**新增文件**：`src/log_classifier/data/augmentation.py`

提供的 API：
```python
from log_classifier.data.augmentation import augment_dataset

augmented = augment_dataset(
    train_data,
    num_aug_per_sample=2,
    target_classes=["搜索算法"],  # 只增强弱势类
)
```

### 5.4 分层学习率衰减

TrainConfig 中预留了 `use_layerwise_lr_decay` 和 `layerwise_lr_decay_rate` 配置，Transformer 底层学习率小于顶层，防止预训练知识被破坏。

---

## 6 Phase 4：Stacking 集成

收集多个基模型的预测概率，训练元学习器进行二阶融合。

**新增文件**：`baselines/run_ensemble.py`

基模型层（Level-0）：
- TF-IDF + LR / SVM / NB
- FastText

元学习器层（Level-1）：
- Logistic Regression (C=1.0, C=10.0)
- XGBoost（若已安装）

```bash
python3 baselines/run_ensemble.py
```

---

## 7 文件变更一览

### 修改的文件

| 文件 | 变更内容 |
|------|---------|
| `src/log_classifier/config/config.py` | max_length 512，patience 5，新增 focal/adversarial/layerwise-lr 配置 |
| `src/log_classifier/training/weighted_trainer.py` | 集成 Focal Loss 和 FGM/PGD 对抗训练 |
| `src/log_classifier/training/__init__.py` | 更新导出 |
| `src/log_classifier/pipelines/hf_sequence_classification.py` | 传递新训练配置到 Trainer |
| `baselines/run_baseline_train.py` | 新增 CLI 参数支持所有增强选项 |
| `baselines/run_all_baselines_train.sh` | max_length=512, patience=5 |
| `baselines/requirements.txt` | 新增 4 个依赖 |
| `pyproject.toml` | 新增 4 个依赖 |

### 新增的文件

| 文件 | 用途 |
|------|------|
| `src/log_classifier/training/focal_loss.py` | Focal Loss 实现 |
| `src/log_classifier/training/adversarial.py` | FGM / PGD 对抗训练 |
| `src/log_classifier/data/augmentation.py` | EDA 数据增强 |
| `baselines/run_setfit_baseline.py` | SetFit 方法 |
| `baselines/run_embedding_baselines.py` | SBERT 嵌入 + LR/SVM/XGBoost/LightGBM |
| `baselines/run_ensemble.py` | Stacking 集成 |
| `baselines/run_enhanced_train.sh` | 增强 Transformer 训练（实验矩阵） |
| `baselines/run_all_new_baselines.sh` | 一键运行所有新增方法 |

---

## 8 运行指南

### 8.1 安装新依赖

```bash
pip install sentence-transformers setfit xgboost lightgbm
# 或
pip install -e .
```

### 8.2 快速运行所有新方法（无需 GPU）

```bash
bash baselines/run_all_new_baselines.sh
```

包含：SBERT 嵌入方法、SetFit、Stacking 集成，最终输出全局排行榜。

### 8.3 增强 Transformer 训练（需要 GPU）

```bash
bash baselines/run_enhanced_train.sh
```

包含：4 模型（BERT / DeBERTa / RoBERTa / DistilBERT）× 4 策略（baseline / focal / fgm / focal+fgm）+ with_meta 对比。

### 8.4 单独运行某个方法

```bash
# SetFit
python3 baselines/run_setfit_baseline.py

# SBERT 嵌入 + 全部分类器
python3 baselines/run_embedding_baselines.py --method all

# Stacking 集成
python3 baselines/run_ensemble.py

# 单模型增强训练（Focal Loss + FGM）
python3 baselines/run_baseline_train.py \
    --model_name bert-base-uncased \
    --max_length 512 \
    --use_focal_loss \
    --use_adversarial \
    --early_stopping_patience 5
```

### 8.5 原始 Baseline 重跑（使用新配置）

```bash
# ML 方法（不变）
bash baselines/run_all_ml_baselines.sh

# Transformer（已更新为 max_length=512, patience=5）
bash baselines/run_all_baselines_train.sh
```

---

## 9 baselines/ 目录索引

| 文件 | 类别 | 说明 |
|------|------|------|
| **脚本入口** | | |
| `run_all_new_baselines.sh` | 新增 | 一键运行所有新增方法 + 全局排行榜 |
| `run_enhanced_train.sh` | 新增 | 增强 Transformer 实验矩阵 |
| `run_all_baselines_train.sh` | 原有 | Transformer baseline 批量训练 |
| `run_all_ml_baselines.sh` | 原有 | ML baseline 批量运行 |
| `run_all_baselines.sh` | 原有 | 旧版零样本评估 |
| **Python 脚本** | | |
| `run_setfit_baseline.py` | Phase 2 | SetFit 小数据集分类 |
| `run_embedding_baselines.py` | Phase 2 | SBERT 嵌入 + LR/SVM/XGBoost/LightGBM |
| `run_ensemble.py` | Phase 4 | Stacking 集成 |
| `run_baseline_train.py` | 原有(已增强) | 单模型 Transformer 训练 |
| `run_ml_baselines.py` | 原有 | TF-IDF + FastText 方法 |
| `run_baseline.py` | 原有 | 旧版零样本评估 |
| **配置** | | |
| `requirements.txt` | 已更新 | Python 依赖 |
| `README.md` | 原有 | 原始 baseline 说明 |
