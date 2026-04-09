#!/bin/bash
# 运行所有Baseline模型的一键脚本
# 使用方法: bash run_all_baselines.sh

set -e  # 遇到错误立即退出

# 设置HuggingFace镜像 (国内加速)
export HF_ENDPOINT="https://hf-mirror.com"

# 配置
DATA_PATH="./data/random_samples.jsonl"
OUTPUT_DIR="./baseline_results"
MAX_LENGTH=256
BATCH_SIZE=16

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 定义Baseline模型列表 (2022-2024年主流文本/代码分类模型)
# 格式: "模型名称:HuggingFace模型ID"
declare -a BASELINES=(
    "BERT:bert-base-uncased"
    "RoBERTa:roberta-base"
    "DeBERTa-v3:microsoft/deberta-v3-base"
    "ERNIE-2.0:nghuyong/ernie-2.0-base-en"
    "MacBERT:hfl/chinese-macbert-base"
    "XLNet:xlnet-base-cased"
    "ALBERT:albert-base-v2"
    "ELECTRA:google/electra-base-discriminator"
    "CodeBERT:microsoft/codebert-base"
    "GraphCodeBERT:microsoft/graphcodebert-base"
    "UnixCoder:Microsoft/unixcoder-base"
)

# 输出颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "日志分类多分类 Baseline 对比实验"
echo "=============================================="
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "模型数量: ${#BASELINES[@]}"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 运行每个baseline
for baseline in "${BASELINES[@]}"; do
    # 解析模型名称和ID
    NAME="${baseline%%:*}"
    MODEL_ID="${baseline##*:}"

    echo -e "${YELLOW}==============================================${NC}"
    echo -e "${YELLOW}运行 Baseline: $NAME${NC}"
    echo -e "${YELLOW}模型ID: $MODEL_ID${NC}"
    echo -e "${YELLOW}==============================================${NC}"

    # 检查模型是否已下载/缓存
    CACHE_DIR="$HOME/.cache/huggingface"
    MODEL_CACHE="$CACHE_DIR/hub/models--${MODEL_ID//\//--}"
    if [ -d "$MODEL_CACHE" ]; then
        echo -e "${GREEN}模型已缓存: $MODEL_ID${NC}"
    else
        echo -e "${YELLOW}模型将首次下载: $MODEL_ID${NC}"
    fi

    # 运行baseline
    python baselines/run_baseline.py \
        --model "$MODEL_ID" \
        --max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --label_field "label3" \
        --text_mode "user_assistant" \
        --seed 42

    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $NAME 运行成功${NC}"
    else
        echo -e "${RED}✗ $NAME 运行失败${NC}"
    fi

    echo ""
done

# 生成汇总报告
echo "=============================================="
echo "生成汇总报告..."
echo "=============================================="

python << 'EOF'
import json
import os
import glob
from collections import OrderedDict

output_dir = "./baseline_results"
result_files = glob.glob(os.path.join(output_dir, "*_results.json"))

if not result_files:
    print("没有找到结果文件!")
    exit(1)

results = []
for f in result_files:
    with open(f, "r", encoding="utf-8") as fp:
        r = json.load(fp)
        results.append(r)

# 按Macro F1排序
results.sort(key=lambda x: x["metrics"]["macro_f1"], reverse=True)

# 生成汇总表格
print("\n" + "="*100)
print("Baseline 对比实验结果汇总")
print("="*100)
print(f"{'Model':<30} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Precision':<12} {'Recall':<12} {'Time(s)':<10}")
print("-"*100)

summary_data = []
for r in results:
    model_name = r["model_name"]
    metrics = r["metrics"]
    elapsed = r.get("elapsed_seconds", 0)
    print(f"{model_name:<30} {metrics['accuracy']:<12.4f} {metrics['macro_f1']:<12.4f} {metrics['weighted_f1']:<12.4f} {metrics['macro_precision']:<12.4f} {metrics['macro_recall']:<12.4f} {elapsed:<10.2f}")
    summary_data.append({
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "elapsed_seconds": elapsed,
        "throughput": r.get("throughput_samples_per_second", 0),
    })

print("="*100)
print(f"\n最佳模型 (Macro F1): {results[0]['model_name']} - {results[0]['metrics']['macro_f1']:.4f}")

# 保存汇总结果
summary_path = os.path.join(output_dir, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, ensure_ascii=False, indent=2)

print(f"\n汇总结果已保存: {summary_path}")
EOF

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo -e "${GREEN}所有Baseline运行完成!${NC}"
echo "总耗时: $ELAPSED 秒"
echo "=============================================="
