#!/bin/bash
# 等所有 Phase C 训练完成后自动：
#   1. 跑 Phase D Stacking 集成
#   2. 把 *_sota/ 目录归档到 phase_c_sota/
#   3. 重新生成 summary_leaderboard.json 与 README.md
#
# 完成信号：每个训练结束时 run_kfold_train.py 会写入
# <output_dir>/kfold_summary.json，这个文件存在就代表训练完整结束。
#
# 用法：
#   nohup bash baselines/watch_run_phase_d.sh > /dev/null 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p output/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="output/logs/phase_d_auto_${TIMESTAMP}.log"
exec >> "$LOG" 2>&1

if [ -x ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

# 等待完成的目录（每个里有 kfold_summary.json 时视为完成）
# DeBERTa-v3 在本 pipeline 上不收敛，改用 roberta-base + roberta-large 两个 RoBERTa 变体
REQUIRED=(
    "./baseline_results/roberta_base_sota"
    "./baseline_results/roberta_large_sota"
)

echo "=============================================================="
echo "[$(date '+%F %T')] Phase D 自动触发 watcher"
echo "等待以下训练完成:"
for d in "${REQUIRED[@]}"; do echo "  - $d/kfold_summary.json"; done
echo "=============================================================="

# 1. 等所有三个 kfold_summary.json 都出现
MAX_WAIT_HOURS=48
WAIT_DEADLINE=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while true; do
    all_ok=1
    for d in "${REQUIRED[@]}"; do
        if [ ! -f "$d/kfold_summary.json" ]; then
            all_ok=0
            break
        fi
    done
    if [ $all_ok -eq 1 ]; then
        break
    fi
    if [ $(date +%s) -gt $WAIT_DEADLINE ]; then
        echo "[$(date '+%F %T')] 超过 ${MAX_WAIT_HOURS}h 仍未全部完成，退出"
        exit 1
    fi
    sleep 120
done

echo ""
echo "[$(date '+%F %T')] 所有 Phase C 训练已完成，开始 Phase D"

# 2. 跑 Phase D Stacking 集成
echo ""
echo "======= Phase D Stacking ======="
$PYTHON baselines/run_ensemble.py \
    --transformer_oof_dirs \
        ./baseline_results/roberta_base_sota \
        ./baseline_results/roberta_large_sota \
    --use_xgb --use_lgb \
    --tag stacking_phase_d
STACK_RC=$?
echo "[$(date '+%F %T')] Phase D Stacking 返回码: $STACK_RC"

# 3. 归档 *_sota/ 到 phase_c_sota/
echo ""
echo "======= 归档 Phase C 目录 ======="
mkdir -p ./baseline_results/phase_c_sota
for d in "${REQUIRED[@]}"; do
    name=$(basename "$d")
    if [ -d "$d" ] && [ -f "$d/kfold_summary.json" ]; then
        target="./baseline_results/phase_c_sota/$name"
        if [ -d "$target" ]; then
            echo "  [跳过] $target 已存在"
        else
            mv "$d" "$target"
            echo "  [归档] $d → $target"
        fi
    fi
done

# 4. 重新生成 summary_leaderboard.json 与 README.md
echo ""
echo "======= 重新生成 leaderboard & README ======="
$PYTHON baselines/regen_summary.py
REGEN_RC=$?
echo "[$(date '+%F %T')] 汇总重建返回码: $REGEN_RC"

echo ""
echo "=============================================================="
echo "[$(date '+%F %T')] 全流程完成"
echo "Phase D 日志见本文件；stacking 输出见 baseline_results/stacking_phase_d_*.json"
echo "=============================================================="
