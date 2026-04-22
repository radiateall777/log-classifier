"""Stacking Ensemble 2.0 —— 支持 Transformer K-fold OOF。

核心流程：
1. 对 ML 基模（TF-IDF×{LR, SVM, NB}, FastText），用 cross_val_predict 生成
   train/dev 全量的 **无泄漏 OOF 概率**；在 test 上各自 fit(train+dev) 后预测。
2. 读取 Phase B 产出的 transformer OOF（训练池上的概率）+ test 概率（K×S 平均）。
3. 对齐数据顺序后拼接成 stacking 特征矩阵 [N_trainval, Σp] / [N_test, Σp]。
4. 训练多个元学习器（LR C=1/10、XGBoost、LightGBM）。
5. 选最佳元学习器，在 test 上评估并保存所有模型的结果。

用法::
    python3 baselines/run_ensemble.py \\
        --transformer_oof_dirs \\
            ./baseline_results/phase_c_sota/roberta_base_sota \\
            ./baseline_results/phase_c_sota/roberta_large_sota \\
        --use_xgb --use_lgb

默认输出目录：./baseline_results/phase_d_stacking/
  - best.json                    (全量元学习器排行 + 最佳 classification_report)
  - <method>_results.json        (每个元学习器的独立结果)
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from log_classifier.data.preprocess import (
    assign_label_ids,
    build_label_maps,
    build_samples,
    filter_rare_classes,
    load_json_data,
)


# ------------------------------------------------------------------
# 指标
# ------------------------------------------------------------------

def evaluate_predictions(y_true, y_pred, id2label: Dict[int, str]) -> Dict[str, Any]:
    all_label_ids = sorted(id2label.keys())
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=all_label_ids, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0)),
    }
    report = classification_report(
        y_true, y_pred,
        labels=all_label_ids,
        target_names=[id2label[i] for i in all_label_ids],
        digits=4, zero_division=0, output_dict=True,
    )
    return {"metrics": metrics, "classification_report": report}


# ------------------------------------------------------------------
# 数据划分：与 Phase B 对齐
# ------------------------------------------------------------------

def _split_fixed_test(
    samples: List[Dict[str, Any]],
    test_size: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    labels = [x["labels"] for x in samples]
    indices = list(range(len(samples)))
    return train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels,
    )


# ------------------------------------------------------------------
# ML 基模 OOF + test 预测
# ------------------------------------------------------------------

def _safe_probs(clf, X, num_classes: int) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    if hasattr(clf, "decision_function"):
        df = clf.decision_function(X)
        if df.ndim == 1:
            df = np.column_stack([-df, df])
        e = np.exp(df - df.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    preds = clf.predict(X)
    return np.eye(num_classes)[preds]


def collect_ml_oof(
    trainval_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    num_classes: int,
    k_folds: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """用 cross_val_predict 对 train+val 生成 OOF 概率，test 上 fit 后预测。

    返回：
        ml_oof_probs:  [N_trainval, num_models × num_classes]
        ml_test_probs: [N_test,     num_models × num_classes]
        names:         模型名列表
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.calibration import CalibratedClassifierCV

    trainval_texts = [s["text"] for s in trainval_samples]
    test_texts = [s["text"] for s in test_samples]
    y_trainval = np.array([s["labels"] for s in trainval_samples])

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
    X_trainval = vec.fit_transform(trainval_texts)
    X_test = vec.transform(test_texts)

    # 说明：为了在 OOF / 在 test 上都能拿到概率，SVM 用 CalibratedClassifierCV 包装
    base_models = [
        ("tfidf_lr", _LR(C=10.0, max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=seed)),
        ("tfidf_svm", CalibratedClassifierCV(
            LinearSVC(C=10.0, max_iter=5000, class_weight="balanced", random_state=seed, dual="auto"),
            cv=3, method="sigmoid",
        )),
        ("tfidf_nb", MultinomialNB(alpha=0.01)),
    ]

    oof_probs_list = []
    test_probs_list = []
    names: List[str] = []

    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for name, clf in base_models:
        print(f"  [ML OOF] {name} ...")
        oof = cross_val_predict(clf, X_trainval, y_trainval, cv=splitter, method="predict_proba", n_jobs=-1)
        clf.fit(X_trainval, y_trainval)
        t_probs = _safe_probs(clf, X_test, num_classes)
        oof_probs_list.append(oof)
        test_probs_list.append(t_probs)
        names.append(name)

    # FastText（若可用）
    try:
        import fasttext
        import tempfile

        print("  [ML OOF] fasttext ...")
        ft_oof = np.zeros((len(trainval_samples), num_classes), dtype=np.float32)

        # FastText 需要自己做 K-fold 循环
        for tr_idx, val_idx in splitter.split(np.zeros(len(trainval_samples)), y_trainval):
            with tempfile.TemporaryDirectory() as tmpdir:
                train_path = os.path.join(tmpdir, "train.txt")
                with open(train_path, "w", encoding="utf-8") as f:
                    for i in tr_idx:
                        s = trainval_samples[i]
                        text = s["text"].replace("\n", " ").replace("\r", " ")
                        f.write(f"__label__{s['labels']} {text}\n")
                model = fasttext.train_supervised(
                    input=train_path, epoch=25, lr=0.8,
                    wordNgrams=2, dim=100, loss="softmax",
                    thread=os.cpu_count() or 4, verbose=0,
                )
                val_texts = [trainval_samples[i]["text"].replace("\n", " ").replace("\r", " ") for i in val_idx]
                preds = model.predict(val_texts, k=num_classes)
                for row, (labels, probs) in enumerate(zip(preds[0], preds[1])):
                    pm = np.zeros(num_classes, dtype=np.float32)
                    for lab, pr in zip(labels, probs):
                        lid = int(lab.replace("__label__", ""))
                        if 0 <= lid < num_classes:
                            pm[lid] = pr
                    ft_oof[val_idx[row]] = pm

        # 在全部 trainval 上 fit，用于 test
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.txt")
            with open(train_path, "w", encoding="utf-8") as f:
                for s in trainval_samples:
                    text = s["text"].replace("\n", " ").replace("\r", " ")
                    f.write(f"__label__{s['labels']} {text}\n")
            ft_model = fasttext.train_supervised(
                input=train_path, epoch=25, lr=0.8,
                wordNgrams=2, dim=100, loss="softmax",
                thread=os.cpu_count() or 4, verbose=0,
            )
            clean_test_texts = [t.replace("\n", " ").replace("\r", " ") for t in test_texts]
            preds = ft_model.predict(clean_test_texts, k=num_classes)
            ft_test = np.zeros((len(test_texts), num_classes), dtype=np.float32)
            for row, (labels, probs) in enumerate(zip(preds[0], preds[1])):
                for lab, pr in zip(labels, probs):
                    lid = int(lab.replace("__label__", ""))
                    if 0 <= lid < num_classes:
                        ft_test[row, lid] = pr

        oof_probs_list.append(ft_oof)
        test_probs_list.append(ft_test)
        names.append("fasttext")
    except Exception as e:
        print(f"  FastText 跳过: {e}")

    return np.hstack(oof_probs_list), np.hstack(test_probs_list), names


# ------------------------------------------------------------------
# Transformer OOF：加载并与 trainval 对齐
# ------------------------------------------------------------------

def _load_transformer_oof(oof_dir: str) -> Dict[str, Any]:
    oof_probs = np.load(os.path.join(oof_dir, "oof_probs.npy"))
    oof_labels = np.load(os.path.join(oof_dir, "oof_labels.npy"))
    oof_index = np.load(os.path.join(oof_dir, "oof_index.npy"))
    test_probs = np.load(os.path.join(oof_dir, "test_probs.npy"))
    test_labels = np.load(os.path.join(oof_dir, "test_labels.npy"))
    summary_path = os.path.join(oof_dir, "kfold_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return {
        "oof_probs": oof_probs,
        "oof_labels": oof_labels,
        "oof_index": oof_index,
        "test_probs": test_probs,
        "test_labels": test_labels,
        "summary": summary,
        "model_name": summary.get("model_name", os.path.basename(oof_dir)),
    }


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stacking Ensemble 2.0")
    parser.add_argument("--data_path", type=str, default="./data/random_samples.jsonl")
    parser.add_argument("--output_dir", type=str, default="./baseline_results/phase_d_stacking")
    parser.add_argument("--label_field", type=str, default="label3")
    parser.add_argument("--text_mode", type=str, default="user_assistant")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--test_split_seed", type=int, default=42,
                        help="固定 test 划分 seed（必须与 Phase B 一致）")
    parser.add_argument("--oof_seed", type=int, default=42,
                        help="ML OOF 的 K-fold seed")
    parser.add_argument("--oof_k_folds", type=int, default=5)
    parser.add_argument("--transformer_oof_dirs", type=str, nargs="*", default=None,
                        help="Phase B 输出目录列表（包含 oof_probs.npy）")
    parser.add_argument("--skip_ml", action="store_true", default=False,
                        help="只用 transformer OOF（不加 TF-IDF/FastText）")
    parser.add_argument("--use_xgb", action="store_true", default=False)
    parser.add_argument("--use_lgb", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default="best",
                        help="best 文件名前缀（最终写入 <output_dir>/<tag>.json）")
    args = parser.parse_args()

    np.random.seed(args.oof_seed)

    # 1. 准备数据（与 Phase B 相同的 test 划分）
    raw_data = load_json_data(args.data_path)
    samples = build_samples(raw_data, args.label_field, args.text_mode)
    samples = filter_rare_classes(samples, min_count=2)
    label_list, label2id, id2label = build_label_maps(samples)
    assign_label_ids(samples, label2id)

    trainval_idx, test_idx = _split_fixed_test(samples, args.test_size, args.test_split_seed)
    trainval_samples = [samples[i] for i in trainval_idx]
    test_samples = [samples[i] for i in test_idx]
    y_trainval = np.array([s["labels"] for s in trainval_samples])
    y_test = np.array([s["labels"] for s in test_samples])
    num_classes = len(label_list)

    print(f"Train+Val: {len(trainval_samples)} | Test: {len(test_samples)} | Classes: {num_classes}")

    # 2. ML OOF（可选）
    feat_blocks_train: List[np.ndarray] = []
    feat_blocks_test: List[np.ndarray] = []
    block_names: List[str] = []

    if not args.skip_ml:
        print("\n==== 构建 ML OOF ====")
        ml_oof, ml_test, ml_names = collect_ml_oof(
            trainval_samples, test_samples,
            num_classes=num_classes,
            k_folds=args.oof_k_folds,
            seed=args.oof_seed,
        )
        # 一致性检查
        assert ml_oof.shape[0] == len(trainval_samples)
        assert ml_test.shape[0] == len(test_samples)
        feat_blocks_train.append(ml_oof)
        feat_blocks_test.append(ml_test)
        block_names.extend(ml_names)
    else:
        print("\n==== 跳过 ML OOF ====")

    # 3. Transformer OOF
    if args.transformer_oof_dirs:
        print("\n==== 加载 Transformer OOF ====")
        for td in args.transformer_oof_dirs:
            if not os.path.exists(os.path.join(td, "oof_probs.npy")):
                print(f"  [跳过] {td}: 缺少 oof_probs.npy")
                continue
            data = _load_transformer_oof(td)

            # 对齐到本脚本的 trainval 顺序
            # Phase B 的 oof_index 是「指向原 samples 列表」的索引
            # 本脚本也用同一个 split，所以 oof_index 应当就是 trainval_idx（顺序可能不同）
            expected_set = set(trainval_idx)
            if set(data["oof_index"].tolist()) != expected_set:
                raise RuntimeError(
                    f"Transformer OOF 的样本集与本脚本的 trainval 不一致！\n"
                    f"目录: {td}\n"
                    f"请确保 --test_size / --test_split_seed 与 Phase B 完全相同。"
                )

            # 按 trainval_idx 的顺序重排 transformer OOF
            pos_map = {i: pos for pos, i in enumerate(data["oof_index"].tolist())}
            reorder = np.array([pos_map[i] for i in trainval_idx], dtype=np.int64)
            aligned_oof = data["oof_probs"][reorder]

            # test 顺序：Phase B 和 本脚本都用同一个 test_split_seed → 顺序一致
            # 双重检查：标签是否一致
            if not np.array_equal(data["test_labels"], y_test):
                raise RuntimeError(
                    f"Transformer test_labels 顺序与本脚本不一致：{td}\n"
                    f"请检查 --test_split_seed 与 --test_size。"
                )

            feat_blocks_train.append(aligned_oof)
            feat_blocks_test.append(data["test_probs"])
            block_names.append(data["model_name"])

            # 打印单模型集成效果，作为参考
            single_preds = np.argmax(data["test_probs"], axis=-1)
            single_f1 = f1_score(y_test, single_preds, average="macro", zero_division=0)
            print(f"  [加载] {data['model_name']:<50} test_ensemble_macro_f1={single_f1:.4f}")

    if not feat_blocks_train:
        raise ValueError("没有任何基模特征！请至少提供 --transformer_oof_dirs 或不加 --skip_ml")

    X_train = np.hstack(feat_blocks_train)
    X_test = np.hstack(feat_blocks_test)
    print(f"\nStacking 特征矩阵: train={X_train.shape}  test={X_test.shape}  "
          f"({len(block_names)} 个基模: {block_names})")

    # 4. 软投票基线（不训元学习器，直接概率平均）
    print("\n==== 软投票（soft voting）基线 ====")
    # 把每个基模的概率块取平均（而非拼接）
    avg_probs_test = np.zeros((len(test_samples), num_classes), dtype=np.float32)
    for block in feat_blocks_test:
        avg_probs_test += block.reshape(len(test_samples), -1, num_classes).mean(axis=1)
    avg_probs_test /= len(feat_blocks_test)
    soft_vote_preds = np.argmax(avg_probs_test, axis=-1)
    soft_vote_eval = evaluate_predictions(y_test, soft_vote_preds, id2label)
    print(f"  soft_vote test macro_f1 = {soft_vote_eval['metrics']['macro_f1']:.4f}")

    # 5. 元学习器
    meta_configs: List[Tuple[str, Any]] = [
        ("stack_lr_c1",  LogisticRegression(C=1.0,  max_iter=2000, random_state=args.oof_seed)),
        ("stack_lr_c10", LogisticRegression(C=10.0, max_iter=2000, random_state=args.oof_seed)),
    ]
    if args.use_xgb:
        try:
            from xgboost import XGBClassifier
            meta_configs.append(("stack_xgb", XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9,
                random_state=args.oof_seed, eval_metric="mlogloss", verbosity=0,
            )))
        except ImportError:
            print("  [Warn] 未安装 xgboost，跳过")
    if args.use_lgb:
        try:
            from lightgbm import LGBMClassifier
            meta_configs.append(("stack_lgb", LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9,
                random_state=args.oof_seed, verbose=-1,
            )))
        except ImportError:
            print("  [Warn] 未安装 lightgbm，跳过")

    all_results: List[Dict[str, Any]] = []
    all_results.append({
        "method": "soft_vote",
        "method_name": "Soft Voting (average probs)",
        "base_models": block_names,
        **soft_vote_eval,
    })

    print("\n==== 训练元学习器 ====")
    for name, clf in meta_configs:
        print(f"\n  [Meta: {name}]")
        t0 = time.time()
        clf.fit(X_train, y_trainval)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = clf.predict(X_test)
        infer_time = time.time() - t0
        throughput = len(y_test) / infer_time if infer_time > 0 else float("inf")

        ev = evaluate_predictions(y_test, y_pred, id2label)
        ev["method"] = name
        ev["method_name"] = f"Stacking ({name})"
        ev["base_models"] = block_names
        ev["stacked_features"] = int(X_train.shape[1])
        ev["train_time_seconds"] = round(train_time, 3)
        ev["infer_time_seconds"] = round(infer_time, 3)
        ev["throughput_samples_per_second"] = round(throughput, 2)
        ev["test_samples"] = int(len(y_test))
        ev["num_labels"] = int(num_classes)

        print(f"    test  acc={ev['metrics']['accuracy']:.4f}  macro_f1={ev['metrics']['macro_f1']:.4f}")
        all_results.append(ev)

    # 6. 汇总保存
    all_results.sort(key=lambda r: r["metrics"]["macro_f1"], reverse=True)
    best = all_results[0]
    print("\n" + "=" * 70)
    print(f"最佳集成: {best['method_name']}  macro_f1 = {best['metrics']['macro_f1']:.4f}")
    print("=" * 70)
    print("\n[排行榜]")
    for r in all_results:
        m = r["metrics"]
        print(f"  {r['method_name']:<40}  macro_f1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # 最佳
    best_path = os.path.join(out_dir, f"{args.tag}_best.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_models": block_names,
            "stacked_features": int(X_train.shape[1]),
            "leaderboard": [
                {
                    "method": r["method"],
                    "method_name": r["method_name"],
                    "metrics": r["metrics"],
                }
                for r in all_results
            ],
            "best": {
                "method": best["method"],
                "method_name": best["method_name"],
                "metrics": best["metrics"],
                "classification_report": best["classification_report"],
            },
        }, f, ensure_ascii=False, indent=2)

    # 每个元学习器单独保存
    for r in all_results:
        p = os.path.join(out_dir, f"{r['method']}_results.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {best_path} 及各元学习器 results.json")


if __name__ == "__main__":
    main()
