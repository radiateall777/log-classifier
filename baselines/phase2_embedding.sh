#!/bin/bash
# Phase 2 · SBERT 嵌入 baseline + SetFit（可选阶段，不在主排行榜）
#
# 用法：
#   bash baselines/phase2_embedding.sh
#   ENCODER=BAAI/bge-base-en-v1.5 bash baselines/phase2_embedding.sh

set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATA_PATH="${DATA_PATH:-./data/random_samples.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./baseline_results}"
SEED="${SEED:-42}"
ENCODER="${ENCODER:-sentence-transformers/all-mpnet-base-v2}"

log_head "Phase 2 · Embedding & SetFit"
log_info "Encoder:  $ENCODER"
mkdir -p "$OUTPUT_DIR"
start_timer

log_head "Phase 2.1 · SBERT Embedding Baselines"
"$PYTHON" "$PY_DIR/embedding_baselines.py" \
    --method all --encoder "$ENCODER" \
    --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR" --seed "$SEED"
rc_emb=$?

log_head "Phase 2.2 · SetFit"
"$PYTHON" "$PY_DIR/setfit_baseline.py" \
    --model_name "$ENCODER" \
    --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR" --seed "$SEED" \
    --num_epochs 1 --num_iterations 20
rc_setfit=$?

if [ $rc_emb -eq 0 ] && [ $rc_setfit -eq 0 ]; then
    log_ok "Phase 2 完成（耗时 $(elapsed)）"
else
    log_err "Phase 2 有失败：embedding=$rc_emb setfit=$rc_setfit"
    exit 1
fi
