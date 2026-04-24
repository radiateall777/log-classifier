#!/bin/bash
# Phase 0 · 传统 ML baseline（TF-IDF × {LR, SVM, NB} + FastText）
#
# 用法：
#   bash baselines/phase0_ml.sh
#   DATA_PATH=./data/random_samples.jsonl SEED=42 bash baselines/phase0_ml.sh
#
# 输出：baseline_results/ml/*_results.json + summary_ml.json

set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATA_PATH="${DATA_PATH:-./data/random_samples.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./baseline_results}"
SEED="${SEED:-42}"

log_head "Phase 0 · ML Baseline"
log_info "Python:   $PYTHON"
log_info "数据路径: $DATA_PATH"
log_info "输出目录: $OUTPUT_DIR"

if [ ! -f "$DATA_PATH" ]; then
    log_err "数据文件不存在: $DATA_PATH"; exit 1
fi

mkdir -p "$OUTPUT_DIR"
start_timer

"$PYTHON" "$PY_DIR/ml_baselines.py" \
    --method all \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --label_field label3 \
    --text_mode user_assistant \
    --seed "$SEED"
rc=$?

if [ $rc -eq 0 ]; then
    log_ok "Phase 0 完成（耗时 $(elapsed)）"
else
    log_err "Phase 0 失败（exit=$rc）"; exit $rc
fi
