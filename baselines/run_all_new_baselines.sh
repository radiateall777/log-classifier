#!/bin/bash
# 运行所有新增方法的一键脚本
# Phase 2: SetFit, Embedding baselines, XGBoost/LightGBM via embeddings
# Phase 4: Stacking ensemble
# 使用方法: bash baselines/run_all_new_baselines.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export HF_ENDPOINT="https://hf-mirror.com"

DATA_PATH="./data/random_samples.jsonl"
OUTPUT_DIR="./baseline_results"
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

echo "=============================================="
echo "新增方法 Baseline 实验"
echo "=============================================="
echo -e "Python: ${BLUE}$PYTHON${NC}"
echo ""

mkdir -p "$OUTPUT_DIR"
START_TIME=$(date +%s)
SUCCESS=0
FAIL=0

# ============================================================
# 1. Sentence-Transformer Embedding Baselines
# ============================================================
echo -e "${YELLOW}==============================================${NC}"
echo -e "${YELLOW}Phase 2.3: Sentence-Transformer Embedding Baselines${NC}"
echo -e "${YELLOW}==============================================${NC}"

$PYTHON baselines/run_embedding_baselines.py \
    --method all \
    --encoder "sentence-transformers/all-mpnet-base-v2" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Embedding baselines 完成${NC}"
    SUCCESS=$((SUCCESS + 1))
else
    echo -e "${RED}✗ Embedding baselines 失败${NC}"
    FAIL=$((FAIL + 1))
fi

# ============================================================
# 2. SetFit
# ============================================================
echo -e "${YELLOW}==============================================${NC}"
echo -e "${YELLOW}Phase 2.1: SetFit Baselines${NC}"
echo -e "${YELLOW}==============================================${NC}"

for ENCODER in "sentence-transformers/all-mpnet-base-v2"; do
    echo -e "${YELLOW}SetFit: $ENCODER${NC}"
    $PYTHON baselines/run_setfit_baseline.py \
        --model_name "$ENCODER" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --seed "$SEED" \
        --num_epochs 1 \
        --num_iterations 20

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ SetFit ($ENCODER) 完成${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}✗ SetFit ($ENCODER) 失败${NC}"
        FAIL=$((FAIL + 1))
    fi
done

# ============================================================
# 3. Stacking Ensemble
# ============================================================
echo -e "${YELLOW}==============================================${NC}"
echo -e "${YELLOW}Phase 4.1: Stacking Ensemble${NC}"
echo -e "${YELLOW}==============================================${NC}"

$PYTHON baselines/run_ensemble.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Stacking ensemble 完成${NC}"
    SUCCESS=$((SUCCESS + 1))
else
    echo -e "${RED}✗ Stacking ensemble 失败${NC}"
    FAIL=$((FAIL + 1))
fi

# ============================================================
# 汇总所有结果
# ============================================================
echo -e "${YELLOW}==============================================${NC}"
echo -e "${YELLOW}生成全局汇总报告${NC}"
echo -e "${YELLOW}==============================================${NC}"

$PYTHON << 'PYEOF'
import json, os, glob

output_dir = "./baseline_results"
all_results = []

# ML baselines
for pattern in ["tfidf_*_results.json", "fasttext_results.json"]:
    for f in glob.glob(os.path.join(output_dir, pattern)):
        with open(f) as fp:
            r = json.load(fp)
            all_results.append({
                "category": "ML",
                "method": r.get("method_name", r.get("method", "")),
                "macro_f1": r["metrics"]["macro_f1"],
                "accuracy": r["metrics"]["accuracy"],
                "source": os.path.basename(f),
            })

# Embedding baselines
for pattern in ["sbert_*_results.json"]:
    for f in glob.glob(os.path.join(output_dir, pattern)):
        with open(f) as fp:
            r = json.load(fp)
            all_results.append({
                "category": "Embedding",
                "method": r.get("method_name", r.get("method", "")),
                "macro_f1": r["metrics"]["macro_f1"],
                "accuracy": r["metrics"]["accuracy"],
                "source": os.path.basename(f),
            })

# SetFit
for f in glob.glob(os.path.join(output_dir, "setfit_*_results.json")):
    with open(f) as fp:
        r = json.load(fp)
        all_results.append({
            "category": "SetFit",
            "method": r.get("method_name", r.get("method", "")),
            "macro_f1": r["metrics"]["macro_f1"],
            "accuracy": r["metrics"]["accuracy"],
            "source": os.path.basename(f),
        })

# Stacking
for f in glob.glob(os.path.join(output_dir, "stacking_*_results.json")):
    with open(f) as fp:
        r = json.load(fp)
        all_results.append({
            "category": "Ensemble",
            "method": r.get("method_name", r.get("method", "")),
            "macro_f1": r["metrics"]["macro_f1"],
            "accuracy": r["metrics"]["accuracy"],
            "source": os.path.basename(f),
        })

# Transformer baselines
for f in glob.glob(os.path.join(output_dir, "**/*_train_results.json"), recursive=True):
    with open(f) as fp:
        r = json.load(fp)
        all_results.append({
            "category": "Transformer",
            "method": r["model_name"] + " (" + os.path.basename(os.path.dirname(f)) + ")",
            "macro_f1": r["test_metrics"]["macro_f1"],
            "accuracy": r["test_metrics"]["accuracy"],
            "source": os.path.relpath(f, output_dir),
        })

if not all_results:
    print("没有找到任何结果!")
    exit(0)

all_results.sort(key=lambda x: x["macro_f1"], reverse=True)

print("\n" + "=" * 100)
print("全局方法排行榜 (macro_f1)")
print("=" * 100)
print(f"{'Rank':<6} {'Category':<12} {'Method':<50} {'Macro F1':<10} {'Accuracy':<10}")
print("-" * 100)

for i, r in enumerate(all_results, 1):
    print(f"{i:<6} {r['category']:<12} {r['method']:<50} {r['macro_f1']:<10.4f} {r['accuracy']:<10.4f}")

print("=" * 100)

summary_path = os.path.join(output_dir, "summary_all.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n全局汇总已保存: {summary_path}")
PYEOF

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=============================================="
echo -e "${GREEN}所有新增方法实验完成!${NC}"
echo "成功: $SUCCESS, 失败: $FAIL"
echo "总耗时: ${ELAPSED}s ($(($ELAPSED / 60))m $((ELAPSED % 60))s)"
echo "=============================================="
