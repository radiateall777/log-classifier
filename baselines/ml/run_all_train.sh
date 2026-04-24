#!/bin/bash

# ==============================================================================
# 机器学习 Baseline 批量训练脚本
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_PATH="./data/random_samples.jsonl"
OUTPUT_DIR="./outputs/baselines/ml"

declare -a ML_METHODS=(
    "tfidf_lr"
    "tfidf_svm"
    "tfidf_nb"
    "tfidf_xgb"
    "embed_lr"
    "embed_svm"
    "embed_xgb"
)

YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}==============================================${NC}"
echo -e "${YELLOW}日志分类 传统机器学习 Baseline 批量训练实验${NC}"
echo -e "${YELLOW}==============================================${NC}"
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "方法数量: ${#ML_METHODS[@]}"
echo ""

mkdir -p "$OUTPUT_DIR"

for method in "${ML_METHODS[@]}"; do
    echo -e "${YELLOW}==============================================${NC}"
    echo -e "${YELLOW}训练 ML Baseline: $method${NC}"
    echo -e "${YELLOW}==============================================${NC}"

    python3 baselines/ml/train.py \
        --method "$method" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR/$method"

    echo ""
done

# 生成汇总报告
python3 << 'EOF'
import json, os, glob

output_dir = "./outputs/baselines/ml"
result_files = glob.glob(os.path.join(output_dir, "*/*_train_results.json"))

if not result_files:
    print("没有找到训练结果文件!")
    exit(0)

results = []
for f in result_files:
    with open(f, "r", encoding="utf-8") as file:
        results.append(json.load(file))

# 按 Test Macro F1 降序排序
results.sort(key=lambda x: x["test_metrics"]["macro_f1"], reverse=True)

out_file = os.path.join(output_dir, "summary_train.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n==============================================")
print(f"ML 训练完成! 所有结果汇总已保存至: {out_file}")
print("==============================================")
for r in results:
    m = r["model_name"]
    f1 = r["test_metrics"]["macro_f1"]
    acc = r["test_metrics"]["accuracy"]
    val_f1 = r["dev_macro_f1"]
    print(f"{m:<12} | Test F1: {f1:.4f} | Acc: {acc:.4f} | Dev F1: {val_f1:.4f}")
EOF
