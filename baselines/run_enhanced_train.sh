#!/bin/bash
# 增强版 Transformer 训练脚本
# 包含: max_length=512, patience=5, focal loss, FGM 对抗训练
# 使用方法: bash baselines/run_enhanced_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=5

DATA_PATH="./data/random_samples.jsonl"
RESULTS_DIR="./baseline_results"
MAX_LENGTH=512
TRAIN_BATCH_SIZE=16
NUM_EPOCHS=30
LR=2e-5
SEED=42

if [ -x ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================
# 实验矩阵：模型 x 训练策略
# ============================================================

declare -a MODELS=(
    "bert-base-uncased"
    "microsoft/deberta-v3-base"
    "roberta-base"
    "distilbert-base-uncased"
)

declare -a STRATEGIES=(
    "baseline"
    "focal"
    "fgm"
    "focal_fgm"
)

echo "=============================================="
echo "增强版 Transformer 训练实验"
echo "=============================================="
echo -e "Python:   ${BLUE}$PYTHON${NC}"
echo "策略:     ${STRATEGIES[*]}"
echo "模型:     ${MODELS[*]}"
echo ""

mkdir -p "$RESULTS_DIR"
START_TIME=$(date +%s)
SUCCESS=0
FAIL=0

for model_id in "${MODELS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        sanitized="${model_id//\//_}"
        output_dir="${RESULTS_DIR}/${sanitized}_${strategy}"

        EXTRA_ARGS=""
        case $strategy in
            "baseline")
                ;;
            "focal")
                EXTRA_ARGS="--use_focal_loss --focal_loss_gamma 2.0"
                ;;
            "fgm")
                EXTRA_ARGS="--use_adversarial --adversarial_method fgm"
                ;;
            "focal_fgm")
                EXTRA_ARGS="--use_focal_loss --focal_loss_gamma 2.0 --use_adversarial --adversarial_method fgm"
                ;;
        esac

        echo -e "${YELLOW}==============================================${NC}"
        echo -e "${YELLOW}训练: $model_id [$strategy]${NC}"
        echo -e "${YELLOW}==============================================${NC}"

        $PYTHON baselines/run_baseline_train.py \
            --model_name "$model_id" \
            --data_path "$DATA_PATH" \
            --output_dir "$output_dir" \
            --max_length $MAX_LENGTH \
            --train_batch_size $TRAIN_BATCH_SIZE \
            --num_train_epochs $NUM_EPOCHS \
            --learning_rate $LR \
            --seed $SEED \
            --label_field "label3" \
            --text_mode "user_assistant" \
            --use_class_weights \
            --early_stopping_patience 5 \
            --save_total_limit 2 \
            $EXTRA_ARGS

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ $model_id [$strategy] 训练成功${NC}"
            SUCCESS=$((SUCCESS + 1))
        else
            echo -e "${RED}✗ $model_id [$strategy] 训练失败${NC}"
            FAIL=$((FAIL + 1))
        fi
        echo ""
    done
done

# with_meta 文本模式对比
echo -e "${YELLOW}==============================================${NC}"
echo -e "${YELLOW}text_mode=with_meta 对比实验${NC}"
echo -e "${YELLOW}==============================================${NC}"

for model_id in "bert-base-uncased" "microsoft/deberta-v3-base"; do
    sanitized="${model_id//\//_}"
    output_dir="${RESULTS_DIR}/${sanitized}_with_meta"

    echo -e "${YELLOW}训练: $model_id [with_meta]${NC}"

    $PYTHON baselines/run_baseline_train.py \
        --model_name "$model_id" \
        --data_path "$DATA_PATH" \
        --output_dir "$output_dir" \
        --max_length $MAX_LENGTH \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --num_train_epochs $NUM_EPOCHS \
        --learning_rate $LR \
        --seed $SEED \
        --label_field "label3" \
        --text_mode "with_meta" \
        --use_class_weights \
        --early_stopping_patience 5 \
        --save_total_limit 2

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $model_id [with_meta] 训练成功${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}✗ $model_id [with_meta] 训练失败${NC}"
        FAIL=$((FAIL + 1))
    fi
done

# 汇总
$PYTHON << 'EOF'
import json, os, glob

output_dir = "./baseline_results"
result_files = glob.glob(os.path.join(output_dir, "**/*_train_results.json"), recursive=True)

if not result_files:
    print("没有找到训练结果文件!")
    exit(0)

results = []
for f in result_files:
    with open(f, encoding="utf-8") as fp:
        r = json.load(fp)
        r["_source_file"] = os.path.relpath(f, output_dir)
        results.append(r)

results.sort(key=lambda x: x["test_metrics"]["macro_f1"], reverse=True)

print("\n" + "=" * 120)
print("增强训练结果汇总（所有实验）")
print("=" * 120)
print(f"{'Source':<50} {'Model':<35} {'Macro F1':<10} {'Accuracy':<10} {'Time(s)':<10}")
print("-" * 120)

for r in results:
    m = r["model_name"]
    t = r["test_metrics"]
    e = r.get("train_elapsed_seconds", 0)
    src = r.get("_source_file", "")
    print(f"{src:<50} {m:<35} {t['macro_f1']:<10.4f} {t['accuracy']:<10.4f} {e:<10.1f}")

print("=" * 120)
best = results[0]
print(f"\n最佳: {best.get('_source_file', '')} — macro_f1={best['test_metrics']['macro_f1']:.4f}")

summary_path = os.path.join(output_dir, "summary_enhanced.json")
summary = [{
    "source": r.get("_source_file", ""),
    "model": r["model_name"],
    "macro_f1": r["test_metrics"]["macro_f1"],
    "accuracy": r["test_metrics"]["accuracy"],
    "weighted_f1": r["test_metrics"]["weighted_f1"],
    "train_elapsed_seconds": r.get("train_elapsed_seconds", 0),
} for r in results]

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"汇总已保存: {summary_path}")
EOF

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo -e "${GREEN}增强训练完成!${NC}"
echo "成功: $SUCCESS, 失败: $FAIL"
echo "总耗时: ${ELAPSED}s ($(($ELAPSED / 60))m $((ELAPSED % 60))s)"
echo "=============================================="
