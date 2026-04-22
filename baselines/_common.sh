#!/bin/bash
# baselines/_common.sh
# 所有 phase_*.sh 脚本共用的工具函数：Python 检测、颜色、计时、日志。
# 用法：
#   BASELINES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$BASELINES_DIR/_common.sh"
#
# 之后即可使用：$PYTHON / $PY_DIR / log_info / log_ok / log_warn / log_err / elapsed

# ------------- 定位项目根目录 -------------
if [ -z "${PROJECT_ROOT:-}" ]; then
    _this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$_this_dir/.." && pwd)"
fi
export PROJECT_ROOT

# Python 入口目录
export PY_DIR="$PROJECT_ROOT/baselines/python"

# ------------- Python 检测 -------------
if [ -x "$PROJECT_ROOT/.venv/bin/python3" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python3"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 > /dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi
export PYTHON

# ------------- 环境变量默认值 -------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ------------- 颜色 -------------
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' NC=''
fi
export RED GREEN YELLOW BLUE NC

# ------------- 日志工具 -------------
log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err()  { echo -e "${RED}[FAIL]${NC} $*" >&2; }
log_head() {
    echo -e "${YELLOW}==============================================${NC}"
    echo -e "${YELLOW}$*${NC}"
    echo -e "${YELLOW}==============================================${NC}"
}

# ------------- 计时 -------------
# 用法：
#   start_timer
#   ... 做一些事 ...
#   log_info "耗时: $(elapsed)"
start_timer() { export _TIMER_START=$(date +%s); }
elapsed() {
    local end=$(date +%s)
    local diff=$((end - ${_TIMER_START:-$end}))
    printf "%ds (%dm %ds)" "$diff" "$((diff / 60))" "$((diff % 60))"
}

# ------------- 快捷：切入项目根目录 -------------
cd "$PROJECT_ROOT"
