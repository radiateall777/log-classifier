# baselines/ — 日志分类实验脚本索引

本目录包含**所有实验阶段**的运行脚本与工具。结果输出到 `baseline_results/`，
目录说明见 [`../baseline_results/README.md`](../baseline_results/README.md)。

## 📁 文件一览

### Phase 0 · 传统 ML 方法

| 文件 | 作用 |
|---|---|
| [`run_ml_baselines.py`](./run_ml_baselines.py) | TF-IDF × {LR, NB, SVM} + FastText 的实现 |
| [`run_all_ml_baselines.sh`](./run_all_ml_baselines.sh) | 一键运行全部 ML baseline |

> 输出：`baseline_results/ml/*_results.json`，`summary_ml.json`

### Phase 2 · 新方法（SBERT embedding / SetFit）

| 文件 | 作用 |
|---|---|
| [`run_embedding_baselines.py`](./run_embedding_baselines.py) | SBERT 嵌入 + {LR, SVM, XGBoost, LightGBM} |
| [`run_setfit_baseline.py`](./run_setfit_baseline.py) | SetFit 对比学习微调 |
| [`run_all_new_baselines.sh`](./run_all_new_baselines.sh) | 一键跑 Phase 2 全部方法 + 汇总 |

### Phase 3 · 训练技巧增强（Focal / FGM）

| 文件 | 作用 |
|---|---|
| [`run_baseline_train.py`](./run_baseline_train.py) | 通用单模型 Transformer 训练入口（HF Trainer），支持全套 Phase A 技巧 |
| [`run_enhanced_train.sh`](./run_enhanced_train.sh) | 矩阵训练：BERT / RoBERTa × {baseline, focal, fgm, focal_fgm} |

> 输出：`baseline_results/phase3_enhanced/<backbone>_<strategy>/`

### Phase C · K-fold × 多 seed SOTA

| 文件 | 作用 |
|---|---|
| [`run_kfold_train.py`](./run_kfold_train.py) | K-fold × 多 seed 包装器，输出 OOF 概率 + 测试集概率 |
| [`run_sota.sh`](./run_sota.sh) | 矩阵入口（bert-base / roberta-base / deberta-v3-large × K-fold） |
| [`run_roberta_large.sh`](./run_roberta_large.sh) | 单独跑 roberta-large K-fold（替代 DeBERTa，后者在本 pipeline 上不收敛） |

> 输出：`baseline_results/<backbone>_sota/{kfold_summary.json, oof_probs.npy, test_probs.npy, ...}`
>
> ⚠️ **DeBERTa-v3 注意**：实测在本任务 + pipeline 下无法收敛（train loss 震荡于 1.6、grad_norm 爆到 40+、eval f1 稳定在 0.067）。
> 已改用 `roberta-large` 作为"强 backbone"补位。

### Phase D · Stacking 集成

| 文件 | 作用 |
|---|---|
| [`run_ensemble.py`](./run_ensemble.py) | Stacking：合并 ML OOF + Transformer OOF，训练 4 个元学习器（LR×2 / XGBoost / LightGBM）|

### 自动化 & 工具

| 文件 | 作用 |
|---|---|
| [`watch_run_phase_d.sh`](./watch_run_phase_d.sh) | 等 Phase C 全部完成 → 自动跑 Phase D → 归档 → 重建 README/leaderboard |
| [`regen_summary.py`](./regen_summary.py) | 扫描全部结果、重新生成 `baseline_results/summary_leaderboard.json` + `README.md` |

### 元数据

| 文件 | 作用 |
|---|---|
| `requirements.txt` | Python 依赖（与根目录 `pyproject.toml` 一致）|
| `.gitignore` | 忽略 `__pycache__` 等 |

## 🔗 推荐运行顺序

```bash
# 1. 传统 ML baseline（10 分钟）
bash baselines/run_all_ml_baselines.sh

# 2. Phase 2 新方法（SBERT / SetFit；可选）
bash baselines/run_all_new_baselines.sh

# 3. Phase 3 训练技巧矩阵（约 4 小时，单卡）
bash baselines/run_enhanced_train.sh

# 4. Phase C K-fold SOTA（bert + roberta × K=5 × seed=3）
CUDA_VISIBLE_DEVICES=5 bash baselines/run_sota.sh
# 如需在另一张 GPU 并行跑 roberta-large:
CUDA_VISIBLE_DEVICES=3 bash baselines/run_roberta_large.sh

# 5. 启动 Phase D 自动触发 watcher（等 Phase C 全部完成自动跑集成 + 重建汇总）
nohup bash baselines/watch_run_phase_d.sh > /dev/null 2>&1 &

# 6. 任意时刻手动刷新 baseline_results/{README.md, summary_leaderboard.json}
.venv/bin/python3 baselines/regen_summary.py
```

## 🧩 关键的 CLI 参数速查

### `run_baseline_train.py` / `run_kfold_train.py` 共享参数

```bash
# 基础
--model_name bert-base-uncased        # HuggingFace 模型 ID
--max_length 512                      # token 上限
--train_batch_size 16
--gradient_accumulation_steps 1       # 显存不够时调大
--learning_rate 2e-5
--num_train_epochs 20
--early_stopping_patience 5
--seed 42

# 混合精度
--bf16                                 # 推荐（Ampere+）；DeBERTa 必须
--no_fp16                              # 配合 bf16 使用

# 训练技巧（Phase A 全套）
--use_focal_loss --focal_loss_gamma 2.0
--use_adversarial --adversarial_method fgm --adversarial_epsilon 1.0
--use_rdrop --rdrop_alpha 1.0
--label_smoothing 0.1
--use_layerwise_lr_decay --layerwise_lr_decay_rate 0.9
--use_class_weights

# EDA 数据增强（定向增强"搜索算法"弱势类）
--use_eda --augment_target_classes 搜索算法 --num_aug_per_sample 3
```

### `run_kfold_train.py` 特有参数

```bash
--k_folds 5
--seeds 42 123 2024                   # 每折重复的训练 seed
--test_split_seed 42                  # 固定 test 集划分 seed（需与 Phase D 一致）
```

### `run_ensemble.py` 特有参数

```bash
--transformer_oof_dirs ./baseline_results/roberta_base_sota ...
--use_xgb --use_lgb                   # 启用 XGBoost / LightGBM 元学习器
--skip_ml                             # 不加 TF-IDF/FastText
--tag stacking_phase_d                # 输出文件前缀
```

## ⚙️ Python 依赖

核心依赖已在 `../pyproject.toml` 声明：

```
transformers>=5.5.0
torch>=2.11.0
datasets>=4.8.4
scikit-learn>=1.8.0
accelerate>=1.13.0
fasttext>=0.9.3
sentence-transformers>=3.0.0
setfit>=1.0.0
xgboost>=2.0.0
lightgbm>=4.0.0
sentencepiece              # DeBERTa 需要
protobuf
```

安装：
```bash
uv sync                    # 推荐（项目使用 uv）
# 或
pip install -e ..
```

## 📊 数据

- **数据文件**：`../data/random_samples.jsonl`（5000 条，每类 1000）
- **标签字段**：`label3`（5 类）
- **类别**：Java Spring 相关 / 代码补全 / 动态规划 / 排序算法 / 搜索算法
- **固定划分**：`../data/random_samples_splits.json`（Train 3999 / Dev 501 / Test 500，`seed=42`）

## 📈 当前最佳

见 [`../baseline_results/README.md`](../baseline_results/README.md) 顶部的排行榜。
本目录下所有脚本产出的结果都会被 `regen_summary.py` 自动聚合。
