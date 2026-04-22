#!/bin/bash
# 运行所有 ML Baseline 方法的一键脚本
# 使用方法: bash baselines/run_all_ml_baselines.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

# 配置
DATA_PATH="./data/random_samples.jsonl"
OUTPUT_DIR="./baseline_results"
SEED=42

# 自动检测 Python：优先使用项目虚拟环境
if [ -x ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=============================================="
echo "日志分类 ML Baseline 实验"
echo "=============================================="
echo -e "Python:   ${BLUE}$PYTHON${NC} ($($PYTHON --version 2>&1))"
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# 预检：确认数据文件存在
if [ ! -f "$DATA_PATH" ]; then
    echo -e "${RED}错误: 数据文件不存在: $DATA_PATH${NC}"
    exit 1
fi

# 预检：确认核心依赖可导入
if ! $PYTHON -c "import sklearn" 2>/dev/null; then
    echo -e "${RED}错误: scikit-learn 未安装，请运行: pip install scikit-learn${NC}"
    exit 1
fi

START_TIME=$(date +%s)

# --method all 一次运行所有方法（TF-IDF 特征只构建一次，含 dev 集超参搜索）
echo -e "${YELLOW}==============================================${NC}"
echo -e "${YELLOW}运行所有 ML Baseline（all 模式）${NC}"
echo -e "${YELLOW}==============================================${NC}"
echo ""

$PYTHON baselines/run_ml_baselines.py \
    --method all \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --label_field "label3" \
    --text_mode "user_assistant" \
    --seed "$SEED"

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}所有 ML Baseline 运行完成!${NC}"
else
    echo -e "${RED}ML Baseline 运行过程中存在错误 (exit=$EXIT_CODE)${NC}"
fi
echo "总耗时: ${ELAPSED}s ($(($ELAPSED / 60))m $((ELAPSED % 60))s)"
echo "结果目录: $OUTPUT_DIR"
echo "=============================================="

exit $EXIT_CODE
