#!/bin/bash
# Phase D · Stacking 集成（软投票 + 多元学习器）
#
# 用法：
#   bash baselines/phase_d_stacking.sh
#   OOF_DIRS="./baseline_results/phase_c_sota/roberta_base_sota ./baseline_results/phase_c_sota/roberta_large_sota" \
#       bash baselines/phase_d_stacking.sh
#
# 输出：baseline_results/phase_d_stacking/
#   - best.json
#   - stack_{lr_c1,lr_c10,xgb,lgb}_results.json
#   - soft_vote_results.json

set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

OOF_DIRS="${OOF_DIRS:-./baseline_results/phase_c_sota/roberta_base_sota ./baseline_results/phase_c_sota/roberta_large_sota}"
OUTPUT_DIR="${OUTPUT_DIR:-./baseline_results/phase_d_stacking}"
TAG="${TAG:-best}"

log_head "Phase D · Stacking Ensemble"
log_info "OOF 目录: $OOF_DIRS"
log_info "输出:     $OUTPUT_DIR"
start_timer

"$PYTHON" "$PY_DIR/ensemble.py" \
    --transformer_oof_dirs $OOF_DIRS \
    --output_dir "$OUTPUT_DIR" \
    --tag "$TAG" \
    --use_xgb --use_lgb

rc=$?
if [ $rc -eq 0 ]; then
    log_ok "Phase D 完成（耗时 $(elapsed)）"
    log_info "刷新 leaderboard..."
    "$PYTHON" "$PY_DIR/regen_summary.py"
else
    log_err "Phase D 失败（rc=$rc）"; exit $rc
fi
