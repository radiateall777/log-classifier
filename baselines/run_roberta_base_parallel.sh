#!/bin/bash
# 2-GPU 并行跑 RoBERTa-base K-fold（Phase A 全套 + Fold 级并行）
# scratch 目录自动放到 /tmp，避免 home 配额超限
#
# 用法：
#   bash baselines/run_roberta_base_parallel.sh
#   GPUS="3 5" bash baselines/run_roberta_base_parallel.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM=false

GPUS="${GPUS:-3 5}"

if [ -x ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

OUT_DIR="./baseline_results/roberta_base_sota"
rm -rf "$OUT_DIR" 2>/dev/null
mkdir -p "$OUT_DIR"

echo "=============================================================="
echo "Phase C · RoBERTa-base K-fold（多 GPU 并行）"
echo "GPU 池:     $GPUS"
echo "开始时间:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

START_TIME=$(date +%s)

$PYTHON baselines/run_kfold_parallel.py \
    --model_name roberta-base \
    --output_dir "$OUT_DIR" \
    --gpus $GPUS \
    --k_folds 5 \
    --seeds 42 123 2024 \
    --data_path ./data/random_samples.jsonl \
    --text_mode user_assistant \
    --label_field label3 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_length 512 \
    --num_train_epochs 20 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --early_stopping_patience 4 \
    --use_adversarial --adversarial_method fgm --adversarial_epsilon 1.0 \
    --use_rdrop --rdrop_alpha 1.0 \
    --label_smoothing 0.1 \
    --use_layerwise_lr_decay --layerwise_lr_decay_rate 0.9 \
    --use_eda --augment_target_classes 搜索算法 --num_aug_per_sample 3 \
    --use_class_weights

rc=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================================="
if [ $rc -eq 0 ]; then
    echo "✓ RoBERTa-base 并行 K-fold 完成"
else
    echo "✗ 失败 (rc=$rc)"
fi
echo "耗时: ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo "=============================================================="
