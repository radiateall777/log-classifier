#!/bin/bash
# Phase 3 · Transformer 矩阵训练（BERT / RoBERTa × {baseline, focal, fgm, focal_fgm}）
#
# 用法：
#   CUDA_VISIBLE_DEVICES=5 bash baselines/phase3_enhanced.sh
#   MODELS="bert-base-uncased roberta-base" bash baselines/phase3_enhanced.sh
#
# 汇总由 `python/regen_summary.py` 统一生成（不再内嵌 Python）

set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

DATA_PATH="${DATA_PATH:-./data/random_samples.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-./baseline_results/phase3_enhanced}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-2e-5}"
SEED="${SEED:-42}"
PATIENCE="${PATIENCE:-5}"

DEFAULT_MODELS=(bert-base-uncased roberta-base)
DEFAULT_STRATEGIES=(baseline focal fgm focal_fgm)
read -r -a MODELS     <<< "${MODELS:-${DEFAULT_MODELS[@]}}"
read -r -a STRATEGIES <<< "${STRATEGIES:-${DEFAULT_STRATEGIES[@]}}"

log_head "Phase 3 · Enhanced Transformer Training"
log_info "模型:   ${MODELS[*]}"
log_info "策略:   ${STRATEGIES[*]}"
log_info "GPU:    $CUDA_VISIBLE_DEVICES"
mkdir -p "$RESULTS_DIR"
start_timer
SUCCESS=0; FAIL=0

for model_id in "${MODELS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        sanitized="${model_id//\//_}"
        output_dir="$RESULTS_DIR/${sanitized}_${strategy}"

        EXTRA_ARGS=()
        case $strategy in
            baseline)  ;;
            focal)     EXTRA_ARGS=(--use_focal_loss --focal_loss_gamma 2.0) ;;
            fgm)       EXTRA_ARGS=(--use_adversarial --adversarial_method fgm) ;;
            focal_fgm) EXTRA_ARGS=(--use_focal_loss --focal_loss_gamma 2.0 --use_adversarial --adversarial_method fgm) ;;
        esac

        log_head "$model_id [$strategy]"
        "$PYTHON" "$PY_DIR/train_single.py" \
            --model_name "$model_id" \
            --data_path "$DATA_PATH" \
            --output_dir "$output_dir" \
            --max_length "$MAX_LENGTH" \
            --train_batch_size "$BATCH_SIZE" \
            --num_train_epochs "$EPOCHS" \
            --learning_rate "$LR" \
            --seed "$SEED" \
            --label_field label3 --text_mode user_assistant \
            --use_class_weights \
            --early_stopping_patience "$PATIENCE" \
            --save_total_limit 2 \
            "${EXTRA_ARGS[@]}"

        if [ $? -eq 0 ]; then
            log_ok "$model_id [$strategy] 成功"
            SUCCESS=$((SUCCESS + 1))
        else
            log_err "$model_id [$strategy] 失败"
            FAIL=$((FAIL + 1))
        fi
    done
done

log_info "生成 summary_enhanced.json"
"$PYTHON" "$PY_DIR/regen_summary.py"

log_head "Phase 3 完成"
log_info "成功: $SUCCESS   失败: $FAIL   总耗时: $(elapsed)"
[ $FAIL -eq 0 ] || exit 1
