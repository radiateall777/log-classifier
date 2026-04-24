#!/bin/bash
# 运行所有 Baseline 模型训练的一键脚本
# 使用方法: bash baselines/run_all_baselines_train.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 设置 HuggingFace 镜像（国内加速）
export HF_ENDPOINT="https://hf-mirror.com"

# 配置
DATA_PATH="./data/random_samples.jsonl"
OUTPUT_DIR="./baseline_results"
MAX_LENGTH=256
TRAIN_BATCH_SIZE=16
NUM_EPOCHS=5
LR=2e-5
SEED=42

mkdir -p "$OUTPUT_DIR"

# Baseline 模型列表
declare -a BASELINES=(
    "bert-base-uncased"
    "roberta-base"
    "microsoft/deberta-v3-base"
    "nghuyong/ernie-2.0-base-en"
    "hfl/chinese-macbert-base"
    "xlnet-base-cased"
    "albert-base-v2"
    "google/electra-base-discriminator"
    "microsoft/codebert-base"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo "日志分类 Baseline 训练实验"
echo "=============================================="
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "模型数量: ${#BASELINES[@]}"
echo ""

START_TIME=$(date +%s)

for model_id in "${BASELINES[@]}"; do
    MODEL_NAME="${model_id##*/}"  # 取最后一段作简短名

    echo -e "${YELLOW}==============================================${NC}"
    echo -e "${YELLOW}训练 Baseline: $model_id${NC}"
    echo -e "${YELLOW}==============================================${NC}"

    python3 baselines/run_baseline_train.py \
        --model_name "$model_id" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --max_length $MAX_LENGTH \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --num_train_epochs $NUM_EPOCHS \
        --learning_rate $LR \
        --seed $SEED \
        --label_field "label3" \
        --text_mode "user_assistant" \
        --use_class_weights \
        --early_stopping_patience 2 \
        --save_total_limit 2

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $model_id 训练成功${NC}"
    else
        echo -e "${RED}✗ $model_id 训练失败，跳过${NC}"
    fi
    echo ""
done

# 生成汇总报告
echo "=============================================="
echo "生成汇总报告..."
echo "=============================================="

python3 << 'EOF'
import json, os, glob

output_dir = "./baseline_results"
result_files = glob.glob(os.path.join(output_dir, "*_train_results.json"))

if not result_files:
    print("没有找到训练结果文件!")
    exit(1)

results = []
for f in result_files:
    with open(f, encoding="utf-8") as fp:
        r = json.load(fp)
        results.append(r)

results.sort(key=lambda x: x["test_metrics"]["macro_f1"], reverse=True)

print("\n" + "="*110)
print("Baseline 训练结果汇总")
print("="*110)
print(f"{'Model':<35} {'Accuracy':<10} {'Macro F1':<10} {'Wgt F1':<10} {'Precision':<10} {'Recall':<10} {'Time(s)':<10}")
print("-"*110)

summary_data = []
for r in results:
    m = r["model_name"]
    t = r["test_metrics"]
    e = r.get("elapsed_seconds", 0)
    print(f"{m:<35} {t['accuracy']:<10.4f} {t['macro_f1']:<10.4f} {t['weighted_f1']:<10.4f} {t['precision']:<10.4f} {t['recall']:<10.4f} {e:<10.2f}")
    summary_data.append({
        "model": m,
        "accuracy": t["accuracy"],
        "macro_f1": t["macro_f1"],
        "weighted_f1": t["weighted_f1"],
        "precision": t["precision"],
        "recall": t["recall"],
        "elapsed_seconds": e,
    })

print("="*110)
best = results[0]
print(f"\n最佳模型 (Macro F1): {best['model_name']} — {best['test_metrics']['macro_f1']:.4f}")

summary_path = os.path.join(output_dir, "summary_train.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, ensure_ascii=False, indent=2)

print(f"汇总结果已保存: {summary_path}")
EOF

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo -e "${GREEN}所有 Baseline 训练完成!${NC}"
echo "总耗时: $ELAPSED 秒 ($(($ELAPSED / 60)) 分钟)"
echo "=============================================="
