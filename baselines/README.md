# baselines/ — 实验运行脚本

按 **Phase** 组织的日志分类实验入口。所有结果输出到 `../baseline_results/`，
排行榜见 [`../baseline_results/README.md`](../baseline_results/README.md)。

## 📁 目录结构

```
baselines/
├── _common.sh              # 公共 bash 工具（Python/颜色/计时/日志）
│
├── phase0_ml.sh            # Phase 0 · TF-IDF × {LR,SVM,NB} + FastText
├── phase2_embedding.sh     # Phase 2 · SBERT 嵌入 + SetFit（可选）
├── phase3_enhanced.sh      # Phase 3 · Transformer × {baseline,focal,fgm,focal_fgm} 矩阵
├── phase_c_kfold.sh        # Phase C · K-fold × 多 seed × 多 GPU 并行
├── phase_d_stacking.sh     # Phase D · Stacking 集成（软投票 + 元学习器）
├── watch_phase_d.sh        # Watcher: 等 Phase C 完成 → 自动 Phase D → 重建汇总
│
└── python/                 # Python 入口（由 shell 调用，也可独立运行）
    ├── train_single.py     # 单模型 Transformer 训练（HF Trainer）
    ├── train_kfold.py      # K-fold 并行调度器
    ├── _fold_worker.py     # 单折 worker（内部，仅 train_kfold 调用）
    ├── ml_baselines.py     # TF-IDF / FastText
    ├── embedding_baselines.py  # SBERT 嵌入 + LR/SVM/XGB/LGB
    ├── setfit_baseline.py  # SetFit 对比学习
    ├── ensemble.py         # Stacking 元学习器
    └── regen_summary.py    # 扫描结果 → 生成 leaderboard & README
```

## 🚀 快速开始

```bash
# 1. Phase 0 · 传统 ML（~1 分钟）
bash baselines/phase0_ml.sh

# 2. Phase 3 · 训练技巧矩阵（~4 小时/单卡）
CUDA_VISIBLE_DEVICES=5 bash baselines/phase3_enhanced.sh

# 3. Phase C · K-fold SOTA（多 GPU 并行）
GPUS="3 5" bash baselines/phase_c_kfold.sh
MODEL=roberta-large GPUS="3 4 6 7" BATCH=8 GRAD_ACC=2 bash baselines/phase_c_kfold.sh

# 4. Phase D · Stacking 集成（秒级）
bash baselines/phase_d_stacking.sh

# 任意时刻刷新 leaderboard
.venv/bin/python3 baselines/python/regen_summary.py
```

### 后台自动化

```bash
# 1) 后台启动多个 Phase C（不同 GPU 池）
nohup bash -c 'GPUS="3 5" bash baselines/phase_c_kfold.sh'            > /dev/null 2>&1 &
nohup bash -c 'MODEL=roberta-large GPUS="4 6 7" BATCH=8 GRAD_ACC=2 \
               TAG=roberta_large_sota bash baselines/phase_c_kfold.sh' > /dev/null 2>&1 &

# 2) Watcher 等 Phase C 全部完成 → 自动触发 Phase D + 归档 + 重建汇总
nohup bash baselines/watch_phase_d.sh > /dev/null 2>&1 &
```

## ⚙️ 通用环境变量

所有 phase 脚本都遵循以下 env var 惯例：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATA_PATH` | `./data/random_samples.jsonl` | 输入数据 |
| `SEED` | `42` | 全局随机 seed |
| `CUDA_VISIBLE_DEVICES` | `5` | GPU id（Phase 3） |
| `GPUS` | `3 5` | GPU 池（Phase C 多卡并行） |
| `MODEL` | `roberta-base` | Phase C backbone |
| `TAG` | `<model>_sota` | Phase C 输出目录名 |
| `BATCH` / `GRAD_ACC` | `16 / 1` | 有效 batch = `BATCH × GRAD_ACC` |
| `OOF_DIRS` | 两个 roberta sota 目录 | Phase D 输入 |

## 🧩 Python 入口直接运行

所有 `python/*.py` 都支持独立调用，无需通过 shell。示例：

```bash
# 单模型训练
.venv/bin/python3 baselines/python/train_single.py \
    --model_name bert-base-uncased \
    --data_path ./data/random_samples.jsonl \
    --output_dir ./baseline_results/bert_demo \
    --max_length 512 --num_train_epochs 20 \
    --use_focal_loss --use_adversarial --adversarial_method fgm

# K-fold 并行（替代 phase_c_kfold.sh）
.venv/bin/python3 baselines/python/train_kfold.py \
    --model_name roberta-large --gpus 3 4 6 7 \
    --output_dir ./baseline_results/roberta_large_sota \
    --k_folds 5 --seeds 42 123 2024 \
    --train_batch_size 8 --gradient_accumulation_steps 2 \
    --use_adversarial --use_rdrop --label_smoothing 0.1 \
    --use_layerwise_lr_decay --use_eda --augment_target_classes 搜索算法

# Stacking（可加 --skip_ml 只用 transformer OOF）
.venv/bin/python3 baselines/python/ensemble.py \
    --transformer_oof_dirs \
        ./baseline_results/phase_c_sota/roberta_base_sota \
        ./baseline_results/phase_c_sota/roberta_large_sota \
    --use_xgb --use_lgb
```

## 📊 数据与依赖

- 数据：`../data/random_samples.jsonl`（5000 条，5 类各 1000）
- 标签字段：`label3`；固定 test 划分：`test_split_seed=42, test_size=0.1`
- 依赖：见项目根 [`pyproject.toml`](../pyproject.toml)，安装 `uv sync` 或 `pip install -e ..`

## 📝 关键训练技巧（Phase A 全套）

所有 transformer 训练脚本共享以下 CLI 参数：

```bash
# 损失函数 & 对抗
--use_focal_loss --focal_loss_gamma 2.0
--use_adversarial --adversarial_method fgm --adversarial_epsilon 1.0
--use_rdrop --rdrop_alpha 1.0
--label_smoothing 0.1

# 优化器
--use_layerwise_lr_decay --layerwise_lr_decay_rate 0.9
--use_class_weights

# 数据增强（定向增强"搜索算法"弱势类）
--use_eda --augment_target_classes 搜索算法 --num_aug_per_sample 3

# 混合精度（Ampere+ 推荐 bf16，DeBERTa 必须）
--bf16 --no_fp16
```

## ⚠️ 已知问题

- **DeBERTa-v3** 在本 pipeline 上不收敛（train loss 震荡 1.6，eval f1 稳定 0.067），已改用 `roberta-large` 作为强 backbone。
- 训练完成后 transformer fold 权重**不保留**（`_scratch` 目录会被清理），只保存 OOF/test 概率的 `.npy`；Stacking 依赖这些 `.npy` 即可复现。如需对新样本推理，请使用 Phase 3 的 `model.safetensors`（本地保留，未推送 git）。

---

_此 README 记录脚本的运行方式；实验结果排行榜由 `python/regen_summary.py` 自动生成到 `../baseline_results/README.md`。_
