#!/bin/bash
# Phase C - 冲 0.95+ 的 SOTA 训练脚本
#
# 矩阵: 3 个强 backbone × 全套 Phase A 技巧 × K=5 × seed=3
# 预计单个 backbone 耗时:
#   - base 模型 (~120 M):  ~1.5 h (15 个模型 × 6 min)
#   - large 模型 (~350 M): ~5 h  (15 个模型 × 20 min)
#
# 输出: baseline_results/<backbone>_sota/
#   - kfold_summary.json
#   - oof_probs.npy / test_probs.npy (给 Phase D Stacking 用)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export TOKENIZERS_PARALLELISM=false

DATA_PATH="./data/random_samples.jsonl"
RESULTS_DIR="./baseline_results"

# ---- K-fold 配置 ----
K_FOLDS=5
SEEDS="42 123 2024"

# ---- 训练超参（Phase A 全套技巧） ----
MAX_LENGTH=512
NUM_EPOCHS=20
LR=2e-5
EARLY_STOP=4           # K-fold 内 val 小，早停放宽一点

# Phase A 技巧
COMMON_ARGS=(
    --max_length $MAX_LENGTH
    --num_train_epochs $NUM_EPOCHS
    --learning_rate $LR
    --early_stopping_patience $EARLY_STOP
    --use_adversarial --adversarial_method fgm --adversarial_epsilon 1.0
    --use_rdrop --rdrop_alpha 1.0
    --label_smoothing 0.1
    --use_layerwise_lr_decay --layerwise_lr_decay_rate 0.9
    --use_eda --augment_target_classes 搜索算法 --num_aug_per_sample 3
    --use_class_weights
    --k_folds $K_FOLDS
    --seeds $SEEDS
    --data_path $DATA_PATH
)

if [ -x ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

mkdir -p "$RESULTS_DIR"

# ------------------------------------------------------------------
# 实验矩阵
# ------------------------------------------------------------------
# 每项: "<model_id>|<batch>|<grad_accum>|<tag>"
# batch × grad_accum = 有效 batch（保持 16 等效）
# large 模型显存不够 → 减 batch + 加梯度累积
# ------------------------------------------------------------------

declare -a EXPERIMENTS=(
    "microsoft/deberta-v3-base|16|1|deberta_v3_base_sota"
    "roberta-base|16|1|roberta_base_sota"
    "microsoft/deberta-v3-large|8|2|deberta_v3_large_sota"
)

# 如果想跳过 large 模型（只跑快的 base），设置 SKIP_LARGE=1
if [ "${SKIP_LARGE:-0}" = "1" ]; then
    EXPERIMENTS=(
        "microsoft/deberta-v3-base|16|1|deberta_v3_base_sota"
        "roberta-base|16|1|roberta_base_sota"
    )
    echo -e "${YELLOW}[Info] SKIP_LARGE=1，跳过 large 模型${NC}"
fi

START_TIME=$(date +%s)
SUCCESS=0
FAIL=0

echo "==============================================================="
echo "Phase C: SOTA K-fold Transformer 训练"
echo "==============================================================="
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "K-fold:   $K_FOLDS"
echo "Seeds:    $SEEDS  （共 $(echo $SEEDS | wc -w) × $K_FOLDS 个模型/配置）"
echo "技巧:     FGM + R-Drop + Label Smoothing + Layerwise LR + EDA(搜索算法)"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================================================="

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r MODEL BATCH GRAD_ACC TAG <<< "$exp"
    OUT_DIR="${RESULTS_DIR}/${TAG}"

    # DeBERTa-v3 在 FP16 + layerwise-LR + AMP 组合下会触发
    # "Attempting to unscale FP16 gradients" 错误，必须用 bf16。
    # Ampere 架构 GPU（3090/A100）原生支持 bf16。
    PRECISION_ARGS=()
    if [[ "$MODEL" == *deberta* ]]; then
        PRECISION_ARGS=(--bf16 --no_fp16)
        echo -e "${BLUE}[Info] DeBERTa 自动启用 bf16${NC}"
    fi

    echo ""
    echo -e "${YELLOW}==============================================================${NC}"
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] SOTA: $MODEL  -> $OUT_DIR${NC}"
    echo -e "${YELLOW}  batch=$BATCH  grad_accum=$GRAD_ACC  (effective=${BATCH}×${GRAD_ACC})${NC}"
    echo -e "${YELLOW}==============================================================${NC}"

    $PYTHON baselines/run_kfold_train.py \
        --model_name "$MODEL" \
        --output_dir "$OUT_DIR" \
        --train_batch_size $BATCH \
        --gradient_accumulation_steps $GRAD_ACC \
        "${PRECISION_ARGS[@]}" \
        "${COMMON_ARGS[@]}"

    rc=$?
    if [ $rc -eq 0 ]; then
        echo -e "${GREEN}✓ $MODEL 完成${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}✗ $MODEL 失败 (rc=$rc)${NC}"
        FAIL=$((FAIL + 1))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "==============================================================="
echo -e "${GREEN}Phase C 完成${NC}"
echo "成功: $SUCCESS   失败: $FAIL"
echo "总耗时: ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================================================="
echo ""
echo "下一步：运行 Phase D Stacking 集成"
echo "  $PYTHON baselines/run_ensemble.py \\"
echo "      --transformer_oof_dirs \\"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r _ _ _ TAG <<< "$exp"
    echo "          ${RESULTS_DIR}/${TAG} \\"
done
echo "      --use_xgb --use_lgb"

# 各模型 ensemble macro_f1 预览
echo ""
echo "各 backbone K-fold 集成结果："
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r _ _ _ TAG <<< "$exp"
    SUM_PATH="${RESULTS_DIR}/${TAG}/kfold_summary.json"
    if [ -f "$SUM_PATH" ]; then
        $PYTHON -c "
import json
with open('$SUM_PATH') as f:
    s = json.load(f)
tm = s.get('test_ensemble_metrics', {})
print(f'  {s[\"model_name\"]:<40}  test_ensemble_macro_f1={tm.get(\"macro_f1\", 0):.4f}  acc={tm.get(\"accuracy\", 0):.4f}')
"
    fi
done
