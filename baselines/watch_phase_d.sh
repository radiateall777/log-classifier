#!/bin/bash
# Watcher: 等 Phase C 全部完成 → 自动跑 Phase D → 归档 → 重建 leaderboard
#
# 完成信号：每个 Phase C 训练结束会写入 <output_dir>/kfold_summary.json
#
# 用法：
#   nohup bash baselines/watch_phase_d.sh > /dev/null 2>&1 &
#   REQUIRED="./baseline_results/roberta_base_sota ./baseline_results/roberta_large_sota" \
#       bash baselines/watch_phase_d.sh

set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

REQUIRED="${REQUIRED:-./baseline_results/roberta_base_sota ./baseline_results/roberta_large_sota}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-48}"
POLL_SECONDS="${POLL_SECONDS:-120}"

mkdir -p output/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="output/logs/phase_d_auto_${TS}.log"
exec >> "$LOG" 2>&1

log_head "[$(date '+%F %T')] Phase D 自动触发 watcher"
log_info "等待以下目录出现 kfold_summary.json:"
for d in $REQUIRED; do echo "  - $d"; done

DEADLINE=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while true; do
    all_ok=1
    for d in $REQUIRED; do
        [ -f "$d/kfold_summary.json" ] || { all_ok=0; break; }
    done
    [ $all_ok -eq 1 ] && break
    [ $(date +%s) -gt $DEADLINE ] && { log_err "超过 ${MAX_WAIT_HOURS}h 未完成，退出"; exit 1; }
    sleep "$POLL_SECONDS"
done

log_ok "Phase C 全部完成，启动 Phase D"
"$PYTHON" "$PY_DIR/ensemble.py" \
    --transformer_oof_dirs $REQUIRED \
    --use_xgb --use_lgb --tag best
stack_rc=$?
log_info "Phase D 返回码: $stack_rc"

log_head "归档 Phase C 输出到 phase_c_sota/"
mkdir -p ./baseline_results/phase_c_sota
for d in $REQUIRED; do
    name=$(basename "$d")
    target="./baseline_results/phase_c_sota/$name"
    if [ -d "$d" ] && [ -f "$d/kfold_summary.json" ] && [ ! -d "$target" ]; then
        mv "$d" "$target" && log_info "$d → $target"
    fi
done

log_head "重新生成 leaderboard & README"
"$PYTHON" "$PY_DIR/regen_summary.py"

log_ok "[$(date '+%F %T')] 全流程完成；日志见 $LOG"
