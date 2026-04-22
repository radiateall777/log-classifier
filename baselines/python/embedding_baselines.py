"""Sentence-Transformer embedding + traditional classifier baselines.

Generates dense embeddings via pretrained Sentence-Transformers,
then trains SVM / LR / XGBoost / LightGBM on top.

Usage::
    python3 baselines/python/embedding_baselines.py --method all
    python3 baselines/python/embedding_baselines.py --method sbert_svm
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from log_classifier.data.preprocess import (
    assign_label_ids,
    build_label_maps,
    build_samples,
    filter_rare_classes,
    load_json_data,
    split_dataset,
)


METHOD_REGISTRY = {
    "sbert_lr": "SBERT + Logistic Regression",
    "sbert_svm": "SBERT + SVM (RBF)",
    "sbert_xgb": "SBERT + XGBoost",
    "sbert_lgbm": "SBERT + LightGBM",
}


def evaluate_predictions(y_true, y_pred, id2label):
    all_label_ids = sorted(id2label.keys())
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=all_label_ids, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0),
    }
    report = classification_report(
        y_true, y_pred,
        labels=all_label_ids,
        target_names=[id2label[i] for i in all_label_ids],
        digits=4, zero_division=0, output_dict=True,
    )
    return {"metrics": metrics, "classification_report": report}


def encode_texts(texts: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"  加载编码器: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  编码 {len(texts)} 条文本...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    print(f"  嵌入维度: {embeddings.shape[1]}")
    return embeddings


def run_sbert_lr(X_train, y_train, X_dev, y_dev, X_test, y_test, id2label, seed):
    print("\n[SBERT + Logistic Regression]")
    param_grid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}

    best_f1, best_c, best_clf, best_time = -1.0, 1.0, None, 0.0
    for c in param_grid["C"]:
        clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed)
        t0 = time.time()
        clf.fit(X_train, y_train)
        t_train = time.time() - t0
        dev_pred = clf.predict(X_dev)
        dev_f1 = f1_score(y_dev, dev_pred, average="macro", zero_division=0)
        print(f"    C={c}: dev macro_f1={dev_f1:.4f} ({t_train:.2f}s)")
        if dev_f1 > best_f1:
            best_f1, best_c, best_clf, best_time = dev_f1, c, clf, t_train

    print(f"  最佳: C={best_c}, dev macro_f1={best_f1:.4f}")
    t0 = time.time()
    y_pred = best_clf.predict(X_test).tolist()
    infer_time = time.time() - t0
    return y_pred, {"C": best_c}, best_f1, best_time, infer_time


def run_sbert_svm(X_train, y_train, X_dev, y_dev, X_test, y_test, id2label, seed):
    print("\n[SBERT + SVM (RBF)]")
    param_grid = {"C": [0.1, 1.0, 10.0, 100.0]}

    best_f1, best_c, best_clf, best_time = -1.0, 1.0, None, 0.0
    for c in param_grid["C"]:
        clf = SVC(C=c, kernel="rbf", class_weight="balanced", random_state=seed)
        t0 = time.time()
        clf.fit(X_train, y_train)
        t_train = time.time() - t0
        dev_pred = clf.predict(X_dev)
        dev_f1 = f1_score(y_dev, dev_pred, average="macro", zero_division=0)
        print(f"    C={c}: dev macro_f1={dev_f1:.4f} ({t_train:.2f}s)")
        if dev_f1 > best_f1:
            best_f1, best_c, best_clf, best_time = dev_f1, c, clf, t_train

    print(f"  最佳: C={best_c}, dev macro_f1={best_f1:.4f}")
    t0 = time.time()
    y_pred = best_clf.predict(X_test).tolist()
    infer_time = time.time() - t0
    return y_pred, {"C": best_c}, best_f1, best_time, infer_time


def run_sbert_xgb(X_train, y_train, X_dev, y_dev, X_test, y_test, id2label, seed):
    print("\n[SBERT + XGBoost]")
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  [跳过] xgboost 未安装")
        return None, None, None, None, None

    configs = [
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.1},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.1},
    ]

    best_f1, best_params, best_clf, best_time = -1.0, None, None, 0.0
    for cfg in configs:
        clf = XGBClassifier(
            **cfg, random_state=seed, use_label_encoder=False,
            eval_metric="mlogloss", verbosity=0, n_jobs=-1,
        )
        t0 = time.time()
        clf.fit(X_train, y_train)
        t_train = time.time() - t0
        dev_pred = clf.predict(X_dev)
        dev_f1 = f1_score(y_dev, dev_pred, average="macro", zero_division=0)
        print(f"    {cfg}: dev macro_f1={dev_f1:.4f} ({t_train:.2f}s)")
        if dev_f1 > best_f1:
            best_f1, best_params, best_clf, best_time = dev_f1, cfg, clf, t_train

    print(f"  最佳: {best_params}, dev macro_f1={best_f1:.4f}")
    t0 = time.time()
    y_pred = best_clf.predict(X_test).tolist()
    infer_time = time.time() - t0
    return y_pred, best_params, best_f1, best_time, infer_time


def run_sbert_lgbm(X_train, y_train, X_dev, y_dev, X_test, y_test, id2label, seed):
    print("\n[SBERT + LightGBM]")
    try:
        import lightgbm as lgb
    except ImportError:
        print("  [跳过] lightgbm 未安装")
        return None, None, None, None, None

    configs = [
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "num_leaves": 31},
        {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05, "num_leaves": 63},
        {"n_estimators": 300, "max_depth": -1, "learning_rate": 0.1, "num_leaves": 127},
    ]

    best_f1, best_params, best_clf, best_time = -1.0, None, None, 0.0
    for cfg in configs:
        clf = lgb.LGBMClassifier(
            **cfg, random_state=seed, class_weight="balanced",
            verbosity=-1, n_jobs=-1,
        )
        t0 = time.time()
        clf.fit(X_train, y_train)
        t_train = time.time() - t0
        dev_pred = clf.predict(X_dev)
        dev_f1 = f1_score(y_dev, dev_pred, average="macro", zero_division=0)
        print(f"    {cfg}: dev macro_f1={dev_f1:.4f} ({t_train:.2f}s)")
        if dev_f1 > best_f1:
            best_f1, best_params, best_clf, best_time = dev_f1, cfg, clf, t_train

    print(f"  最佳: {best_params}, dev macro_f1={best_f1:.4f}")
    t0 = time.time()
    y_pred = best_clf.predict(X_test).tolist()
    infer_time = time.time() - t0
    return y_pred, best_params, best_f1, best_time, infer_time


RUNNERS = {
    "sbert_lr": run_sbert_lr,
    "sbert_svm": run_sbert_svm,
    "sbert_xgb": run_sbert_xgb,
    "sbert_lgbm": run_sbert_lgbm,
}


def main():
    parser = argparse.ArgumentParser(description="Sentence-Transformer Embedding Baselines")
    parser.add_argument("--method", type=str, default="all",
                        choices=list(METHOD_REGISTRY.keys()) + ["all"])
    parser.add_argument("--encoder", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--data_path", type=str, default="./data/random_samples.jsonl")
    parser.add_argument("--output_dir", type=str, default="./baseline_results")
    parser.add_argument("--label_field", type=str, default="label3")
    parser.add_argument("--text_mode", type=str, default="user_assistant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    np.random.seed(args.seed)

    raw_data = load_json_data(args.data_path)
    samples = build_samples(raw_data, args.label_field, args.text_mode)
    samples = filter_rare_classes(samples, min_count=2)
    label_list, label2id, id2label = build_label_maps(samples)
    assign_label_ids(samples, label2id)
    train_data, dev_data, test_data = split_dataset(
        samples, seed=args.seed, test_size=0.1, dev_size=0.1,
    )

    print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

    all_texts = [s["text"] for s in train_data + dev_data + test_data]
    embeddings = encode_texts(all_texts, args.encoder, batch_size=args.batch_size)

    n_train = len(train_data)
    n_dev = len(dev_data)
    X_train = embeddings[:n_train]
    X_dev = embeddings[n_train:n_train + n_dev]
    X_test = embeddings[n_train + n_dev:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)

    y_train = [s["labels"] for s in train_data]
    y_dev = [s["labels"] for s in dev_data]
    y_test = [s["labels"] for s in test_data]

    methods = list(METHOD_REGISTRY.keys()) if args.method == "all" else [args.method]

    all_results = []
    for method in methods:
        runner = RUNNERS[method]
        y_pred, best_params, dev_f1, train_time, infer_time = runner(
            X_train, y_train, X_dev, y_dev, X_test, y_test, id2label, args.seed,
        )
        if y_pred is None:
            continue

        throughput = len(y_test) / infer_time if infer_time > 0 else float("inf")
        results = evaluate_predictions(y_test, y_pred, id2label)
        results["method"] = method
        results["method_name"] = METHOD_REGISTRY[method]
        results["encoder"] = args.encoder
        results["train_time_seconds"] = train_time
        results["infer_time_seconds"] = infer_time
        results["throughput_samples_per_second"] = throughput
        results["test_samples"] = len(y_test)
        results["num_labels"] = len(id2label)
        results["best_params"] = best_params
        results["dev_macro_f1"] = dev_f1
        results["embedding_dim"] = X_train.shape[1]

        m = results["metrics"]
        print(f"\n{'=' * 60}")
        print(f"{METHOD_REGISTRY[method]} (encoder: {args.encoder})")
        print(f"{'=' * 60}")
        print(f"Accuracy:        {m['accuracy']:.4f}")
        print(f"Macro F1:        {m['macro_f1']:.4f}")
        print(f"Weighted F1:     {m['weighted_f1']:.4f}")

        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, f"{method}_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {path}")
        all_results.append(results)

    if len(all_results) > 1:
        print(f"\n{'=' * 90}")
        print("Embedding Baseline 汇总")
        print(f"{'=' * 90}")
        all_results.sort(key=lambda x: x["metrics"]["macro_f1"], reverse=True)
        for r in all_results:
            m = r["metrics"]
            print(f"  {r['method_name']:<35} macro_f1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}")

        summary_path = os.path.join(args.output_dir, "summary_embedding.json")
        summary = [{
            "method": r["method"], "method_name": r["method_name"],
            "encoder": r["encoder"],
            "accuracy": r["metrics"]["accuracy"],
            "macro_f1": r["metrics"]["macro_f1"],
            "dev_macro_f1": r.get("dev_macro_f1", 0),
            "best_params": r.get("best_params", {}),
        } for r in all_results]
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
