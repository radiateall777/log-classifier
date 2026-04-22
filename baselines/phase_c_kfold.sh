#!/bin/bash
# Phase C · K-fold × 多 seed × 多 GPU 并行训练
#
# 统一参数化入口：通过环境变量切换 backbone / GPU 池。
#
# 用法：
#   # RoBERTa-base（默认，2-GPU 并行）
#   GPUS="3 5" bash baselines/phase_c_kfold.sh
#
#   # RoBERTa-large（4-GPU 并行 ~5.5h）
#   MODEL=roberta-large GPUS="3 4 6 7" BATCH=8 GRAD_ACC=2 \
#       TAG=roberta_large_sota bash baselines/phase_c_kfold.sh
#
#   # 跳过 EDA（large 模型若观察到过拟合）
#   SKIP_EDA=1 ... bash baselines/phase_c_kfold.sh
#
# 输出：baseline_results/<TAG>/
#   - kfold_summary.json
#   - oof_probs.npy / test_probs.npy（供 Phase D Stacking 用）

set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-roberta-base}"
TAG="${TAG:-${MODEL//\//_}_sota}"
GPUS="${GPUS:-3 5}"
BATCH="${BATCH:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
K_FOLDS="${K_FOLDS:-5}"
SEEDS="${SEEDS:-42 123 2024}"
MAX_LENGTH="${MAX_LENGTH:-512}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-2e-5}"
PATIENCE="${PATIENCE:-4}"
DATA_PATH="${DATA_PATH:-./data/random_samples.jsonl}"
OUT_DIR="${OUT_DIR:-./baseline_results/$TAG}"

EDA_ARGS=(--use_eda --augment_target_classes 搜索算法 --num_aug_per_sample 3)
[ "${SKIP_EDA:-0}" = "1" ] && EDA_ARGS=()

log_head "Phase C · K-fold Parallel Training"
log_info "Model:    $MODEL"
log_info "Tag:      $TAG"
log_info "GPUs:     $GPUS"
log_info "Batch:    $BATCH × GradAccum $GRAD_ACC（等效 $((BATCH * GRAD_ACC))）"
log_info "K-fold:   $K_FOLDS × seeds=($SEEDS)"
log_info "Output:   $OUT_DIR"
log_info "技巧:     FGM + R-Drop + Label Smoothing + Layerwise LR${SKIP_EDA:+（跳过 EDA）}${SKIP_EDA:-+ EDA(搜索算法)}"

rm -rf "$OUT_DIR" 2>/dev/null
mkdir -p "$OUT_DIR"
start_timer

"$PYTHON" "$PY_DIR/train_kfold.py" \
    --model_name "$MODEL" \
    --output_dir "$OUT_DIR" \
    --gpus $GPUS \
    --k_folds "$K_FOLDS" \
    --seeds $SEEDS \
    --data_path "$DATA_PATH" \
    --text_mode user_assistant --label_field label3 \
    --train_batch_size "$BATCH" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --max_length "$MAX_LENGTH" \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --warmup_ratio 0.1 \
    --early_stopping_patience "$PATIENCE" \
    --use_adversarial --adversarial_method fgm --adversarial_epsilon 1.0 \
    --use_rdrop --rdrop_alpha 1.0 \
    --label_smoothing 0.1 \
    --use_layerwise_lr_decay --layerwise_lr_decay_rate 0.9 \
    --use_class_weights \
    "${EDA_ARGS[@]}"

rc=$?
if [ $rc -eq 0 ]; then
    log_ok "Phase C · $MODEL 完成（耗时 $(elapsed)）"
else
    log_err "Phase C · $MODEL 失败（rc=$rc）"; exit $rc
fi
