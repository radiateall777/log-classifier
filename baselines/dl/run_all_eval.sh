#!/bin/bash
# 运行所有模型噪声鲁棒性与吞吐量评估的一键脚本
# 使用方法: bash baselines/run_all_eval.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

export HF_ENDPOINT="https://hf-mirror.com"

# 配置
DATA_PATH="./data/random_samples.jsonl"
OUTPUT_DIR="./outputs/baselines/dl"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo "日志分类已微调模型 噪音鲁棒性与吞吐量评估"
echo "=============================================="
echo "数据路径: $DATA_PATH"
echo "模型目录: $OUTPUT_DIR"
echo ""

START_TIME=$(date +%s)

# 寻找所有存在 best_model.pt 的微调后模型目录
MODEL_DIRS=$(find "$OUTPUT_DIR" -maxdepth 1 -mindepth 1 -type d)

if [ -z "$MODEL_DIRS" ]; then
    echo -e "${RED}没有找到任何微调过的模型目录（在 $OUTPUT_DIR 下）。请先运行 bash baselines/run_all_train.sh${NC}"
    exit 1
fi

for model_dir in $MODEL_DIRS; do
    MODEL_NAME=$(basename "$model_dir")
    
    echo -e "${YELLOW}==============================================${NC}"
    echo -e "${YELLOW}开始评估模型: $MODEL_NAME${NC}"
    echo -e "${YELLOW}目录: $model_dir${NC}"
    echo -e "${YELLOW}==============================================${NC}"

    python3 baselines/dl/eval.py \
        --model_dir "$model_dir" \
        --data_path "$DATA_PATH"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $MODEL_NAME 评估成功${NC}"
    else
        echo -e "${RED}✗ $MODEL_NAME 评估失败${NC}"
    fi
    echo ""
done

# 生成总体汇总表格
echo "=============================================="
echo "生成汇总评估报告..."
echo "=============================================="

python3 << 'EOF'
import json, os, glob

output_dir = "./outputs/baselines/dl"
result_files = glob.glob(os.path.join(output_dir, "*/noise_robustness_results.json"))

if not result_files:
    print("没有找到评估结果文件!")
    exit(1)

summary_data = {}
for f in result_files:
    model_name = os.path.basename(os.path.dirname(f))
    with open(f, encoding="utf-8") as fp:
        try:
            r = json.load(fp)
            summary_data[model_name] = r
        except Exception as e:
            print(f"解析 {f} 失败: {e}")

print(f"\n={'='*80}")
print("评估结果汇总")
print("="*81)
print(f"{'Model':<35} {'Noise Ratio':<15} {'Macro F1':<10} {'Throughput(s/s)':<15}")
print(f"{'-'*81}")

for model_name, results in sorted(summary_data.items()):
    for r in results:
        ratio = r.get("noise_ratio", 0)
        f1 = r.get("macro_f1", 0)
        th = r.get("throughput_samples_per_sec", 0)
        print(f"{model_name:<35} {ratio*100:>5.1f}%          {f1:<10.4f} {th:<15.2f}")
    print(f"{'-'*81}")

summary_path = os.path.join(output_dir, "summary_eval_robustness.json")
with open(summary_path, "w", encoding="utf-8") as out:
    json.dump(summary_data, out, ensure_ascii=False, indent=2)

print(f"\n所有汇总数据已保存为 json: {summary_path}")
EOF

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo -e "${GREEN}所有配置评估完成!${NC}"
echo "总耗时: $ELAPSED 秒 ($(($ELAPSED / 60)) 分钟)"
echo "=============================================="
