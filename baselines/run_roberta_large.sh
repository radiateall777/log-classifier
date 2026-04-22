#!/bin/bash
# Phase C · RoBERTa-large K-fold × 多 seed
#
# 用 roberta-large 替代 DeBERTa-v3（后者在本 pipeline 上不收敛）
# RoBERTa-base 已验证 Phase A 全套配置稳定可行
#
# 用法:
#   CUDA_VISIBLE_DEVICES=3 bash baselines/run_roberta_large.sh
# 参数:
#   SKIP_EDA=1 可跳过 EDA 增强（若后续发现 large 模型 + EDA 过拟合）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export TOKENIZERS_PARALLELISM=false

if [ -x ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

MODEL="roberta-large"
TAG="roberta_large_sota"
OUT_DIR="./baseline_results/$TAG"

# large 模型约 355M 参数，需要减 batch + 梯度累积保持等效 batch=16
BATCH=8
GRAD_ACC=2

EDA_ARGS=(--use_eda --augment_target_classes 搜索算法 --num_aug_per_sample 3)
if [ "${SKIP_EDA:-0}" = "1" ]; then
    EDA_ARGS=()
fi

echo "=============================================================="
echo "Phase C · RoBERTa-large K-fold"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Model:     $MODEL"
echo "Output:    $OUT_DIR"
echo "Batch:     $BATCH × $GRAD_ACC (等效 16)"
echo "技巧:      FGM + R-Drop + Label Smoothing + Layerwise LR + EDA(搜索算法) + fp16"
echo "开始时间:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

rm -rf "$OUT_DIR" 2>/dev/null
mkdir -p "$OUT_DIR"

START_TIME=$(date +%s)

$PYTHON baselines/run_kfold_train.py \
    --model_name "$MODEL" \
    --output_dir "$OUT_DIR" \
    --train_batch_size $BATCH \
    --gradient_accumulation_steps $GRAD_ACC \
    --max_length 512 \
    --num_train_epochs 20 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --early_stopping_patience 4 \
    --use_adversarial --adversarial_method fgm --adversarial_epsilon 1.0 \
    --use_rdrop --rdrop_alpha 1.0 \
    --label_smoothing 0.1 \
    --use_layerwise_lr_decay --layerwise_lr_decay_rate 0.9 \
    --use_class_weights \
    --k_folds 5 \
    --seeds 42 123 2024 \
    --data_path ./data/random_samples.jsonl \
    "${EDA_ARGS[@]}"

rc=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================================="
if [ $rc -eq 0 ]; then
    echo "✓ $MODEL 完成"
else
    echo "✗ $MODEL 失败 (rc=$rc)"
fi
echo "耗时:      ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo "结束时间:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
