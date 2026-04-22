"""重新生成 baseline_results/summary_leaderboard.json 与 README.md。

扫描所有 Phase 目录，聚合 test 指标，输出按 macro_f1 降序的全局排行榜。

可在以下时机调用：
  - 新 Phase C / Phase D 完成后
  - 手动整理结果后

用法::
    python3 baselines/regen_summary.py
"""

import glob
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BASE = os.path.join(os.path.dirname(__file__), "..", "baseline_results")
BASE = os.path.abspath(BASE)


# ------------------------------------------------------------------
# 各 Phase 收集器
# ------------------------------------------------------------------

def _collect_ml() -> List[Dict[str, Any]]:
    path = os.path.join(BASE, "ml", "summary_ml.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        rows.append({
            "category": "ml",
            "method": r["method"],
            "method_name": r["method_name"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "train_elapsed_seconds": r.get("train_time_seconds"),
            "throughput": r.get("throughput"),
        })
    return rows


def _collect_phase3() -> List[Dict[str, Any]]:
    phase3_dir = os.path.join(BASE, "phase3_enhanced")
    if not os.path.isdir(phase3_dir):
        return []

    rows = []
    for sub in sorted(os.listdir(phase3_dir)):
        d = os.path.join(phase3_dir, sub)
        if not os.path.isdir(d):
            continue
        jfiles = glob.glob(os.path.join(d, "*_train_results.json"))
        if not jfiles:
            continue
        with open(jfiles[0], "r", encoding="utf-8") as f:
            r = json.load(f)
        # 解析 backbone + strategy
        backbone, strategy = sub, "unknown"
        for strat in ("focal_fgm", "baseline", "focal", "fgm"):
            if sub.endswith("_" + strat):
                backbone = sub[:-(len(strat) + 1)]
                strategy = strat
                break
        tm = r["test_metrics"]
        rows.append({
            "category": "phase3_enhanced",
            "method": sub,
            "method_name": f"{backbone} + {strategy}",
            "accuracy": tm["accuracy"],
            "macro_f1": tm["macro_f1"],
            "weighted_f1": tm["weighted_f1"],
            "train_elapsed_seconds": r.get("train_elapsed_seconds"),
            "throughput": r.get("test_throughput_samples_per_sec"),
            "_extras": {
                "backbone": backbone,
                "strategy": strategy,
                "val_macro_f1": r["val_metrics"].get("eval_macro_f1"),
                "best_epoch": r["val_metrics"].get("epoch"),
            },
        })
    return rows


def _collect_phase_c() -> List[Dict[str, Any]]:
    """Phase C K-fold 结果，包括尚未迁入 phase_c_sota/ 的 *_sota 目录。"""
    rows = []
    candidates = [
        os.path.join(BASE, "phase_c_sota"),
        BASE,
    ]
    seen = set()
    for root in candidates:
        if not os.path.isdir(root):
            continue
        for sub in sorted(os.listdir(root)):
            if not sub.endswith("_sota"):
                continue
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                continue
            summary_path = os.path.join(d, "kfold_summary.json")
            if not os.path.isfile(summary_path):
                continue
            if sub in seen:
                continue
            seen.add(sub)
            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            tm = s.get("test_ensemble_metrics", {})
            rows.append({
                "category": "phase_c_sota",
                "method": sub,
                "method_name": f"{s['model_name']} (K={s.get('k_folds')}×seed={len(s.get('seeds', []))} ensemble)",
                "accuracy": tm.get("accuracy"),
                "macro_f1": tm.get("macro_f1"),
                "weighted_f1": tm.get("weighted_f1"),
                "train_elapsed_seconds": s.get("total_elapsed_seconds"),
                "throughput": None,
                "_extras": {
                    "num_models": s.get("num_models"),
                    "oof_macro_f1": s.get("oof_metrics", {}).get("macro_f1"),
                    "model_name": s.get("model_name"),
                },
            })
    return rows


def _collect_stacking() -> List[Dict[str, Any]]:
    """Phase D 集成结果（软投票 + 多元学习器 + best）。

    扫描 `phase_d_stacking/` 目录：
      - `<method>_results.json`：每个元学习器的独立结果
      - `best.json`：全局最佳（同时还列入 leaderboard 内置排行榜）

    也兼容旧版本顶层的 `stacking_*_results.json` 与 `stacking_phase_d_best.json`。
    """
    rows = []
    seen_methods = set()

    # 新位置：phase_d_stacking/
    stacking_dir = os.path.join(BASE, "phase_d_stacking")

    # 候选：单个元学习器结果文件
    candidate_files = []
    if os.path.isdir(stacking_dir):
        for name in ("stack_*_results.json", "soft_vote*_results.json", "stacking_*_results.json"):
            candidate_files.extend(glob.glob(os.path.join(stacking_dir, name)))
    # 向后兼容：顶层目录的旧文件
    candidate_files.extend(glob.glob(os.path.join(BASE, "stacking_*_results.json")))

    for p in sorted(set(candidate_files)):
        # best.json 走下面专门的分支
        if os.path.basename(p) in ("best.json", "stacking_phase_d_best.json"):
            continue
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        m = r.get("metrics", {})
        method = r.get("method", os.path.basename(p).replace("_results.json", ""))
        if method in seen_methods:
            continue
        seen_methods.add(method)
        rows.append({
            "category": "phase_d_stacking",
            "method": method,
            "method_name": r.get("method_name", method),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "weighted_f1": m.get("weighted_f1"),
            "train_elapsed_seconds": r.get("train_time_seconds"),
            "throughput": r.get("throughput_samples_per_second"),
            "_extras": {
                "base_models": r.get("base_models"),
                "stacked_features": r.get("stacked_features"),
            },
        })

    # best 单独列一条（优先新路径）
    best_candidates = [
        os.path.join(stacking_dir, "best.json"),
        os.path.join(BASE, "stacking_phase_d_best.json"),
    ]
    for best_path in best_candidates:
        if not os.path.exists(best_path):
            continue
        with open(best_path, "r", encoding="utf-8") as f:
            b = json.load(f)
        best = b.get("best", {})
        m = best.get("metrics", {})
        if m:
            rows.append({
                "category": "phase_d_stacking",
                "method": "stacking_best",
                "method_name": f"⭐ Phase D Best ({best.get('method_name','?')})",
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "weighted_f1": m.get("weighted_f1"),
                "train_elapsed_seconds": None,
                "throughput": None,
                "_extras": {"base_models": b.get("base_models")},
            })
        break
    return rows


# ------------------------------------------------------------------
# 输出
# ------------------------------------------------------------------

def _format_number(x, precision: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{precision}f}"
    return str(x)


def _sort_key(r: Dict[str, Any]):
    # 同分时把 stacking_best（带 ⭐）排在最前，便于 README "当前最佳" 取首行
    return (
        r["macro_f1"] is None,
        -(r["macro_f1"] or 0),
        0 if r.get("method") == "stacking_best" else 1,
    )


def _write_leaderboard(rows: List[Dict[str, Any]]) -> str:
    rows = sorted(rows, key=_sort_key)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    path = os.path.join(BASE, "summary_leaderboard.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    return path


def _render_readme(rows: List[Dict[str, Any]]) -> str:
    sorted_rows = sorted(rows, key=_sort_key)

    # Top-10
    top_lines = ["| # | Category | Method | Macro F1 | Accuracy |",
                 "|---|---|---|---:|---:|"]
    for i, r in enumerate(sorted_rows[:10], 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, str(i))
        top_lines.append(
            f"| {medal} | {r['category']} | {r['method_name']} | "
            f"{_format_number(r['macro_f1'])} | {_format_number(r['accuracy'])} |"
        )

    # Full table (macro_f1 desc)
    full_lines = ["| # | Category | Method | Macro F1 | Accuracy | Weighted F1 | Train (s) |",
                  "|---|---|---|---:|---:|---:|---:|"]
    for i, r in enumerate(sorted_rows, 1):
        te = r.get("train_elapsed_seconds")
        te_str = f"{te:.0f}" if isinstance(te, (int, float)) else "—"
        full_lines.append(
            f"| {i} | {r['category']} | {r['method_name']} | "
            f"{_format_number(r['macro_f1'])} | {_format_number(r['accuracy'])} | "
            f"{_format_number(r['weighted_f1'])} | {te_str} |"
        )

    best = sorted_rows[0] if sorted_rows else None
    best_name = best["method_name"] if best else "N/A"
    best_f1 = _format_number(best["macro_f1"]) if best else "N/A"

    template = f"""# baseline_results 目录导览

本目录按 **实验 Phase** 分层组织所有日志分类实验的结果。

## 当前最佳

> **{best_name}** — Test Macro F1 = **{best_f1}**

## 目录结构

```
baseline_results/
├── README.md                       本文件（自动生成）
├── summary_leaderboard.json        全局排行榜（所有实验按 test macro_f1 排序）
│
├── ml/                             Phase 0 · 传统 ML 方法
│   ├── summary_ml.json
│   ├── fasttext_results.json
│   └── tfidf_{{lr,nb,svm}}_results.json
│
├── phase3_enhanced/                Phase 3 · 训练技巧增强（Focal / FGM）
│   ├── summary_enhanced.json
│   └── <backbone>_<strategy>/       (model.safetensors + train_results.json)
│
├── phase_c_sota/                   Phase C · K-fold × 多 seed SOTA
│   └── <backbone>_sota/
│       ├── kfold_summary.json
│       ├── fold_logs/               每个 seed × fold 的训练日志
│       ├── oof_probs.npy / oof_labels.npy / oof_index.npy
│       └── test_probs.npy / test_labels.npy
│
└── phase_d_stacking/               Phase D · Stacking 集成（软投票 + 元学习器）
    ├── best.json                    全量元学习器排行 + 最佳 classification_report
    ├── soft_vote_results.json
    └── stack_{{lr_c1,lr_c10,xgb,lgb}}_results.json
```

## Top 10 排行榜

{chr(10).join(top_lines)}

## 完整排行榜

{chr(10).join(full_lines)}

## 如何运行

```bash
# Phase 0 · ML baseline
bash baselines/run_all_ml_baselines.sh

# Phase 3 · Focal/FGM 增强（推荐配置：max_length=512, patience=5）
bash baselines/run_enhanced_train.sh

# Phase C · K-fold × 多 seed SOTA
CUDA_VISIBLE_DEVICES=5 bash baselines/run_sota.sh
# 仅跑 base 模型：
SKIP_LARGE=1 bash baselines/run_sota.sh

# Phase D · Stacking 集成（默认输出到 baseline_results/phase_d_stacking/）
.venv/bin/python3 baselines/run_ensemble.py \\
    --transformer_oof_dirs \\
        ./baseline_results/phase_c_sota/roberta_base_sota \\
        ./baseline_results/phase_c_sota/roberta_large_sota \\
    --use_xgb --use_lgb
```

## 复现历史结果

Phase 3 使用固定划分 `data/random_samples_splits.json` + `seed=42`（max_length=512, patience=5）。
Phase C 使用 `test_split_seed=42`（固定 test 集）+ 多训练 seed（`42 123 2024`）。

---

_此 README 由 `baselines/regen_summary.py` 自动生成，请勿手动编辑。_
"""
    path = os.path.join(BASE, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(template)
    return path


def _render_phase3_summary(phase3_rows: List[Dict[str, Any]]) -> None:
    """保持 phase3_enhanced/summary_enhanced.json 最新。"""
    if not phase3_rows:
        return
    out = []
    for r in sorted(phase3_rows, key=lambda r: -(r["macro_f1"] or 0)):
        extras = r.get("_extras", {})
        out.append({
            "experiment": r["method"],
            "backbone": extras.get("backbone"),
            "strategy": extras.get("strategy"),
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "val_macro_f1": extras.get("val_macro_f1"),
            "best_epoch": extras.get("best_epoch"),
            "train_elapsed_seconds": r["train_elapsed_seconds"],
            "test_throughput_samples_per_sec": r["throughput"],
        })
    path = os.path.join(BASE, "phase3_enhanced", "summary_enhanced.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

def main():
    print(f"扫描 {BASE} ...")
    rows = []
    rows.extend(_collect_ml())
    phase3 = _collect_phase3()
    rows.extend(phase3)
    rows.extend(_collect_phase_c())
    rows.extend(_collect_stacking())

    print(f"共收集 {len(rows)} 条实验结果")

    _render_phase3_summary(phase3)
    lb_path = _write_leaderboard(rows)
    rm_path = _render_readme(rows)

    print(f"已更新: {lb_path}")
    print(f"已更新: {rm_path}")
    sorted_rows = sorted(rows, key=lambda r: -(r["macro_f1"] or 0))
    if sorted_rows:
        print(f"\nTop 5:")
        for i, r in enumerate(sorted_rows[:5], 1):
            print(f"  {i}. [{r['category']}] {r['method_name']}  macro_f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
