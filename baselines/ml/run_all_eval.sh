#!/bin/bash

# ==============================================================================
# 机器学习 Baseline 批量鲁棒性评估脚本
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_PATH="./data/random_samples.jsonl"
OUTPUT_DIR="./outputs/baselines/ml"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}错误: 找不到结果目录 $OUTPUT_DIR。请先运行 bash baselines/ml/run_all_train.sh${NC}"
    exit 1
fi

echo -e "${YELLOW}==============================================${NC}"
echo -e "${YELLOW}日志分类 机器学习 Baseline 测速与噪声鲁棒性分析${NC}"
echo -e "${YELLOW}==============================================${NC}"

# Find all ML model directories
mapfile -t MODEL_DIRS < <(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d)

if [ ${#MODEL_DIRS[@]} -eq 0 ]; then
    echo -e "${RED}错误: 在 $OUTPUT_DIR 中未找到方法目录！${NC}"
    exit 1
fi

for model_dir in "${MODEL_DIRS[@]}"; do
    echo -e "${YELLOW}==============================================${NC}"
    echo -e "${YELLOW}评估方法目录: $model_dir${NC}"
    echo -e "${YELLOW}==============================================${NC}"

    python3 baselines/ml/eval.py \
        --model_dir "$model_dir" \
        --data_path "$DATA_PATH"
    
    echo ""
done

# 生成评测汇总报告
python3 << 'EOF'
import json, os, glob

output_dir = "./outputs/baselines/ml"
result_files = glob.glob(os.path.join(output_dir, "*/noise_robustness_results.json"))

if not result_files:
    print("没有找到测试结果文件!")
    exit(0)

# 汇总各模型降级曲线
summary = {}
for f in result_files:
    model_name = os.path.basename(os.path.dirname(f))
    with open(f, "r", encoding="utf-8") as file:
        data = json.load(file)
        summary[model_name] = data

out_file = os.path.join(output_dir, "summary_eval_robustness.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n==============================================")
print(f"ML 批量评测完成! 鲁棒性汇总已保存至: {out_file}")
print("==============================================")
EOF
