# baseline_results 目录导览

本目录按 **实验 Phase** 分层组织所有日志分类实验的结果。

## 当前最佳

> **⭐ Phase D Best (Stacking (stack_lr_c10))** — Test Macro F1 = **0.9256**

## 目录结构

```
baseline_results/
├── README.md                       本文件（自动生成）
├── summary_leaderboard.json        全局排行榜（所有实验按 test macro_f1 排序）
│
├── ml/                             Phase 0 · 传统 ML 方法
│   ├── summary_ml.json
│   ├── fasttext_results.json
│   └── tfidf_{lr,nb,svm}_results.json
│
├── phase3_enhanced/                Phase 3 · 训练技巧增强（Focal / FGM）
│   ├── summary_enhanced.json
│   └── <backbone>_<strategy>/       (model.safetensors + train_results.json)
│
├── phase_c_sota/                   Phase C · K-fold × 多 seed SOTA
│   └── <backbone>_sota/
│       ├── kfold_summary.json
│       ├── fold_logs/               每个 seed × fold 的训练日志
│       ├── oof_probs.npy / oof_labels.npy / oof_index.npy
│       └── test_probs.npy / test_labels.npy
│
└── phase_d_stacking/               Phase D · Stacking 集成（软投票 + 元学习器）
    ├── best.json                    全量元学习器排行 + 最佳 classification_report
    ├── soft_vote_results.json
    └── stack_{lr_c1,lr_c10,xgb,lgb}_results.json
```

## Top 10 排行榜

| # | Category | Method | Macro F1 | Accuracy |
|---|---|---|---:|---:|
| 🥇 | phase_d_stacking | ⭐ Phase D Best (Stacking (stack_lr_c10)) | 0.9256 | 0.9260 |
| 🥈 | phase_d_stacking | Stacking (stack_lr_c10) | 0.9256 | 0.9260 |
| 🥉 | phase_d_stacking | Stacking (stack_lr_c1) | 0.9236 | 0.9240 |
| 4 | phase_d_stacking | Soft Voting (average probs) | 0.9235 | 0.9240 |
| 5 | phase_d_stacking | Stacking (stack_xgb) | 0.9179 | 0.9180 |
| 6 | phase3_enhanced | bert-base-uncased + fgm | 0.9172 | 0.9180 |
| 7 | phase_d_stacking | Stacking (stack_lgb) | 0.9162 | 0.9160 |
| 8 | phase_c_sota | roberta-large (K=5×seed=3 ensemble) | 0.9156 | 0.9160 |
| 9 | phase3_enhanced | bert-base-uncased + focal_fgm | 0.9095 | 0.9100 |
| 10 | phase_c_sota | roberta-base (K=5×seed=3 ensemble) | 0.9050 | 0.9060 |

## 完整排行榜

| # | Category | Method | Macro F1 | Accuracy | Weighted F1 | Train (s) |
|---|---|---|---:|---:|---:|---:|
| 1 | phase_d_stacking | ⭐ Phase D Best (Stacking (stack_lr_c10)) | 0.9256 | 0.9260 | 0.9256 | — |
| 2 | phase_d_stacking | Stacking (stack_lr_c10) | 0.9256 | 0.9260 | 0.9256 | 0 |
| 3 | phase_d_stacking | Stacking (stack_lr_c1) | 0.9236 | 0.9240 | 0.9236 | 0 |
| 4 | phase_d_stacking | Soft Voting (average probs) | 0.9235 | 0.9240 | 0.9235 | — |
| 5 | phase_d_stacking | Stacking (stack_xgb) | 0.9179 | 0.9180 | 0.9179 | 2 |
| 6 | phase3_enhanced | bert-base-uncased + fgm | 0.9172 | 0.9180 | 0.9172 | 2960 |
| 7 | phase_d_stacking | Stacking (stack_lgb) | 0.9162 | 0.9160 | 0.9162 | 1 |
| 8 | phase_c_sota | roberta-large (K=5×seed=3 ensemble) | 0.9156 | 0.9160 | 0.9156 | 47640 |
| 9 | phase3_enhanced | bert-base-uncased + focal_fgm | 0.9095 | 0.9100 | 0.9095 | 1938 |
| 10 | phase_c_sota | roberta-base (K=5×seed=3 ensemble) | 0.9050 | 0.9060 | 0.9050 | 26031 |
| 11 | phase3_enhanced | bert-base-uncased + focal | 0.9012 | 0.9020 | 0.9012 | 920 |
| 12 | ml | FastText | 0.8948 | 0.8960 | 0.8948 | 6 |
| 13 | phase3_enhanced | roberta-base + fgm | 0.8927 | 0.8940 | 0.8927 | 1191 |
| 14 | phase3_enhanced | bert-base-uncased + baseline | 0.8912 | 0.8920 | 0.8912 | 697 |
| 15 | ml | TF-IDF + Linear SVM | 0.8907 | 0.8920 | 0.8907 | 1 |
| 16 | ml | TF-IDF + Logistic Regression | 0.8887 | 0.8900 | 0.8887 | 38 |
| 17 | phase3_enhanced | roberta-base + focal | 0.8797 | 0.8820 | 0.8797 | 485 |
| 18 | ml | TF-IDF + Naive Bayes | 0.8743 | 0.8760 | 0.8743 | 0 |
| 19 | phase3_enhanced | roberta-base + baseline | 0.8736 | 0.8740 | 0.8736 | 679 |

## 如何运行

```bash
# Phase 0 · ML baseline
bash baselines/phase0_ml.sh

# Phase 3 · Focal/FGM 增强矩阵
CUDA_VISIBLE_DEVICES=5 bash baselines/phase3_enhanced.sh

# Phase C · K-fold × 多 seed（多 GPU 并行，默认 roberta-base）
GPUS="3 5" bash baselines/phase_c_kfold.sh
# 切换 backbone：
MODEL=roberta-large GPUS="3 4 6 7" BATCH=8 GRAD_ACC=2 bash baselines/phase_c_kfold.sh

# Phase D · Stacking 集成（自动刷新 leaderboard）
bash baselines/phase_d_stacking.sh
```

## 复现历史结果

Phase 3 使用固定划分 `data/random_samples_splits.json` + `seed=42`（max_length=512, patience=5）。
Phase C 使用 `test_split_seed=42`（固定 test 集）+ 多训练 seed（`42 123 2024`）。

---

_此 README 由 `baselines/python/regen_summary.py` 自动生成，请勿手动编辑。_
