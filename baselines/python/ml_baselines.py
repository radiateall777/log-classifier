"""机器学习 Baseline 方法

包含四种经典 ML 文本分类方法：
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM (LinearSVC)
3. TF-IDF + Naive Bayes (MultinomialNB)
4. FastText

用法::
    # 运行单个方法
    python3 baselines/python/ml_baselines.py --method tfidf_lr
    python3 baselines/python/ml_baselines.py --method tfidf_svm
    python3 baselines/python/ml_baselines.py --method tfidf_nb
    python3 baselines/python/ml_baselines.py --method fasttext

    # 运行所有 ML 方法
    python3 baselines/python/ml_baselines.py --method all

    # 一键脚本
    bash baselines/phase0_ml.sh
"""

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# =============================================================================
# 数据加载（与主项目 & run_baseline.py 保持一致）
# =============================================================================

def load_json_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    return data


def flatten_messages(
    messages: List[Dict[str, str]], text_mode: str, item: Dict[str, Any]
) -> str:
    user_texts, assistant_texts = [], []
    for msg in messages:
        role = msg.get("role", "").strip()
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            user_texts.append(content)
        elif role == "assistant":
            assistant_texts.append(content)

    user_text = " ".join(user_texts).strip()
    assistant_text = " ".join(assistant_texts).strip()

    if text_mode == "user_only":
        return f"user: {user_text}"
    if text_mode == "assistant_only":
        return f"assistant: {assistant_text}"
    if text_mode == "user_assistant":
        return f"user: {user_text} assistant: {assistant_text}"
    if text_mode == "with_meta":
        language = str(item.get("language", "")).strip()
        dataset_name = str(item.get("dataset", "")).strip()
        return (
            f"language: {language} dataset: {dataset_name} "
            f"user: {user_text} assistant: {assistant_text}"
        )
    raise ValueError(f"不支持的 text_mode: {text_mode}")


def build_samples(
    raw_data: List[Dict[str, Any]], label_field: str, text_mode: str
) -> List[Dict[str, Any]]:
    samples = []
    for item in raw_data:
        messages = item.get("messages", [])
        if not isinstance(messages, list):
            continue
        label = item.get(label_field, None)
        if label is None:
            continue
        text = flatten_messages(messages, text_mode, item)
        if not text:
            continue
        samples.append({
            "id": item.get("id", None),
            "text": text,
            "label_text": str(label).strip(),
        })
    return samples


def build_label_maps(samples: List[Dict[str, Any]]):
    label_list = sorted({x["label_text"] for x in samples})
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_list, label2id, id2label


def assign_label_ids(
    samples: List[Dict[str, Any]], label2id: Dict[str, int]
) -> None:
    for x in samples:
        x["labels"] = label2id[x["label_text"]]


def split_dataset(
    samples: List[Dict[str, Any]],
    seed: int,
    test_size: float,
    dev_size: float,
):
    labels = [x["labels"] for x in samples]
    indices = list(range(len(samples)))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    relative_dev_size = dev_size / (1.0 - test_size)

    train_idx, dev_idx = train_test_split(
        train_val_idx,
        test_size=relative_dev_size,
        random_state=seed,
        stratify=train_val_labels,
    )

    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in dev_idx],
        [samples[i] for i in test_idx],
    )


# =============================================================================
# 评估函数（显式传入 labels，避免标签缺失时的隐式错误）
# =============================================================================

def evaluate_predictions(
    y_true: List[int], y_pred: List[int], id2label: Dict[int, str]
) -> Dict[str, Any]:
    all_label_ids = sorted(id2label.keys())
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro",
            labels=all_label_ids, zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted",
            labels=all_label_ids, zero_division=0,
        ),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro",
            labels=all_label_ids, zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro",
            labels=all_label_ids, zero_division=0,
        ),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=all_label_ids,
        target_names=[id2label[i] for i in all_label_ids],
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    return {"metrics": metrics, "classification_report": report}


# =============================================================================
# TF-IDF 特征提取（全局构建一次）
# =============================================================================

def build_tfidf_features(
    train_texts: List[str],
    dev_texts: List[str],
    test_texts: List[str],
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
) -> Tuple[Any, Any, Any, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        dtype=np.float32,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_dev = vectorizer.transform(dev_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_dev, X_test, vectorizer


# =============================================================================
# TF-IDF 分类器工厂 & 超参网格
# =============================================================================

TFIDF_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "tfidf_lr": {"C": [0.01, 0.1, 1.0, 10.0]},
    "tfidf_svm": {"C": [0.01, 0.1, 1.0, 10.0]},
    "tfidf_nb": {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
}


def _make_tfidf_classifier(method: str, seed: int, **kwargs: Any) -> Any:
    if method == "tfidf_lr":
        return LogisticRegression(
            max_iter=1000,
            C=kwargs.get("C", 1.0),
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )
    if method == "tfidf_svm":
        return LinearSVC(
            C=kwargs.get("C", 1.0),
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
            dual="auto",
        )
    if method == "tfidf_nb":
        return MultinomialNB(alpha=kwargs.get("alpha", 1.0))
    raise ValueError(f"不支持的 TF-IDF 方法: {method}")


# =============================================================================
# 统一的 TF-IDF 分类（含 dev 集超参搜索）
# =============================================================================

def run_tfidf_method(
    method: str,
    X_train: Any,
    y_train: List[int],
    X_dev: Any,
    y_dev: List[int],
    X_test: Any,
    y_test: List[int],
    id2label: Dict[int, str],
    config: "MLBaselineConfig",
) -> Dict[str, Any]:
    method_name = METHOD_REGISTRY[method]
    print(f"\n[{method_name}]")
    print(f"  特征维度: {X_train.shape[1]}")

    param_grid = TFIDF_PARAM_GRIDS[method]
    param_name = list(param_grid.keys())[0]
    param_values = param_grid[param_name]

    print(f"  在 dev 集上搜索超参 {param_name}: {param_values}")
    best_f1 = -1.0
    best_val = param_values[0]
    best_clf = None
    best_train_time = 0.0

    for val in param_values:
        clf = _make_tfidf_classifier(method, config.seed, **{param_name: val})
        t0 = time.time()
        clf.fit(X_train, y_train)
        t_train = time.time() - t0

        dev_pred = clf.predict(X_dev)
        dev_f1 = f1_score(y_dev, dev_pred, average="macro", zero_division=0)
        print(f"    {param_name}={val}: dev macro_f1={dev_f1:.4f} ({t_train:.2f}s)")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_val = val
            best_clf = clf
            best_train_time = t_train

    print(f"  最佳: {param_name}={best_val}, dev macro_f1={best_f1:.4f}")

    print("  测试集推理...")
    t0 = time.time()
    y_pred = best_clf.predict(X_test).tolist()
    infer_time = time.time() - t0
    throughput = len(y_test) / infer_time if infer_time > 0 else float("inf")

    results = evaluate_predictions(y_test, y_pred, id2label)
    results["method"] = method
    results["method_name"] = method_name
    results["train_time_seconds"] = best_train_time
    results["infer_time_seconds"] = infer_time
    results["throughput_samples_per_second"] = throughput
    results["tfidf_features"] = X_train.shape[1]
    results["test_samples"] = len(y_test)
    results["num_labels"] = len(id2label)
    results["best_params"] = {param_name: best_val}
    results["dev_macro_f1"] = best_f1

    return results


# =============================================================================
# FastText（使用整数 ID 做标签，避免空格/下划线映射 bug）
# =============================================================================

def _write_fasttext_file(samples: List[Dict], filepath: str) -> None:
    """写入 FastText 格式: __label__<int_id> <text>"""
    with open(filepath, "w", encoding="utf-8") as f:
        for s in samples:
            label_id = s["labels"]
            text = s["text"].replace("\n", " ").replace("\r", " ")
            f.write(f"__label__{label_id} {text}\n")


def run_fasttext(
    train_data: List[Dict],
    dev_data: List[Dict],
    test_data: List[Dict],
    id2label: Dict[int, str],
    config: "MLBaselineConfig",
) -> Dict[str, Any]:
    try:
        import fasttext
    except ImportError:
        print("  [错误] 未安装 fasttext，请运行: pip install fasttext-wheel")
        raise

    print("\n[FastText]")

    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.txt")
        dev_path = os.path.join(tmpdir, "dev.txt")
        test_path = os.path.join(tmpdir, "test.txt")

        print("  准备 FastText 格式数据...")
        _write_fasttext_file(train_data, train_path)
        _write_fasttext_file(dev_data, dev_path)
        _write_fasttext_file(test_data, test_path)

        lr_candidates = [0.1, 0.3, 0.5, 0.8]
        epoch_candidates = [15, 25, 50]

        print(f"  在 dev 集上搜索超参: lr={lr_candidates}, epoch={epoch_candidates}")
        best_f1 = -1.0
        best_lr = config.fasttext_lr
        best_epoch = config.fasttext_epoch
        best_model = None
        best_train_time = 0.0

        y_dev = [s["labels"] for s in dev_data]

        for lr in lr_candidates:
            for epoch in epoch_candidates:
                t0 = time.time()
                model = fasttext.train_supervised(
                    input=train_path,
                    epoch=epoch,
                    lr=lr,
                    wordNgrams=config.fasttext_word_ngrams,
                    dim=config.fasttext_dim,
                    loss="softmax",
                    thread=os.cpu_count() or 4,
                    verbose=0,
                )
                t_train = time.time() - t0

                dev_texts = [
                    s["text"].replace("\n", " ").replace("\r", " ")
                    for s in dev_data
                ]
                predictions = model.predict(dev_texts)
                dev_pred = _parse_fasttext_predictions(predictions, len(id2label))
                dev_f1 = f1_score(y_dev, dev_pred, average="macro", zero_division=0)
                print(
                    f"    lr={lr}, epoch={epoch}: "
                    f"dev macro_f1={dev_f1:.4f} ({t_train:.2f}s)"
                )

                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    best_lr = lr
                    best_epoch = epoch
                    best_model = model
                    best_train_time = t_train

        print(
            f"  最佳: lr={best_lr}, epoch={best_epoch}, "
            f"dev macro_f1={best_f1:.4f}"
        )

        print("  测试集推理...")
        test_texts = [
            s["text"].replace("\n", " ").replace("\r", " ") for s in test_data
        ]
        y_test = [s["labels"] for s in test_data]

        t0 = time.time()
        predictions = best_model.predict(test_texts)
        infer_time = time.time() - t0
        throughput = (
            len(test_data) / infer_time if infer_time > 0 else float("inf")
        )

        y_pred = _parse_fasttext_predictions(predictions, len(id2label))

    results = evaluate_predictions(y_test, y_pred, id2label)
    results["method"] = "fasttext"
    results["method_name"] = "FastText"
    results["train_time_seconds"] = best_train_time
    results["infer_time_seconds"] = infer_time
    results["throughput_samples_per_second"] = throughput
    results["fasttext_params"] = {
        "lr": best_lr,
        "epoch": best_epoch,
        "wordNgrams": config.fasttext_word_ngrams,
        "dim": config.fasttext_dim,
    }
    results["test_samples"] = len(test_data)
    results["num_labels"] = len(id2label)
    results["best_params"] = {"lr": best_lr, "epoch": best_epoch}
    results["dev_macro_f1"] = best_f1

    return results


def _parse_fasttext_predictions(
    predictions: Tuple, num_labels: int
) -> List[int]:
    """从 FastText 预测结果中解析整数标签 ID"""
    y_pred = []
    unmapped = 0
    for pred_labels in predictions[0]:
        raw = pred_labels[0].replace("__label__", "")
        try:
            label_id = int(raw)
            if 0 <= label_id < num_labels:
                y_pred.append(label_id)
            else:
                y_pred.append(0)
                unmapped += 1
        except ValueError:
            y_pred.append(0)
            unmapped += 1
    if unmapped > 0:
        print(f"  [警告] {unmapped} 个样本的 FastText 预测标签无法映射")
    return y_pred


# =============================================================================
# 配置 & 入口
# =============================================================================

METHOD_REGISTRY = {
    "tfidf_lr": "TF-IDF + Logistic Regression",
    "tfidf_svm": "TF-IDF + Linear SVM",
    "tfidf_nb": "TF-IDF + Naive Bayes",
    "fasttext": "FastText",
}

TFIDF_METHODS = {"tfidf_lr", "tfidf_svm", "tfidf_nb"}


@dataclass
class MLBaselineConfig:
    method: str = "all"
    data_path: str = "./data/random_samples.jsonl"
    text_mode: str = "user_assistant"
    label_field: str = "label3"
    test_size: float = 0.1
    dev_size: float = 0.1
    seed: int = 42
    output_dir: str = "./baseline_results"

    # TF-IDF 参数
    max_features: int = 50000
    ngram_range: Tuple[int, int] = (1, 2)

    # FastText 参数
    fasttext_epoch: int = 25
    fasttext_lr: float = 0.5
    fasttext_word_ngrams: int = 2
    fasttext_dim: int = 100


def _prepare_data(config: MLBaselineConfig):
    """加载并划分数据，返回 train/dev/test + 标签映射"""
    np.random.seed(config.seed)

    print(f"加载数据: {config.data_path}")
    raw_data = load_json_data(config.data_path)
    samples = build_samples(raw_data, config.label_field, config.text_mode)

    if len(samples) == 0:
        raise ValueError("没有有效样本!")

    print(f"总样本数: {len(samples)}")

    label_list, label2id, id2label = build_label_maps(samples)
    assign_label_ids(samples, label2id)

    print(f"标签数: {len(label_list)}")
    print(f"标签列表: {label_list}")

    train_data, dev_data, test_data = split_dataset(
        samples,
        seed=config.seed,
        test_size=config.test_size,
        dev_size=config.dev_size,
    )

    print(
        f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}"
    )

    return train_data, dev_data, test_data, label_list, label2id, id2label


def _print_results(results: Dict[str, Any]) -> None:
    m = results["metrics"]
    print(f"\n{'=' * 60}")
    print(f"方法: {results['method_name']}")
    print(f"{'=' * 60}")
    if "best_params" in results:
        print(f"最佳超参:      {results['best_params']}")
    if "dev_macro_f1" in results:
        print(f"Dev Macro F1:    {results['dev_macro_f1']:.4f}")
    print(f"Accuracy:        {m['accuracy']:.4f}")
    print(f"Macro F1:        {m['macro_f1']:.4f}")
    print(f"Weighted F1:     {m['weighted_f1']:.4f}")
    print(f"Macro Precision: {m['macro_precision']:.4f}")
    print(f"Macro Recall:    {m['macro_recall']:.4f}")
    print(
        f"训练耗时: {results.get('train_time_seconds', 0):.2f}s, "
        f"推理吞吐: {results.get('throughput_samples_per_second', 0):.2f} samples/s"
    )


def _save_results(results: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    method = results["method"]
    path = os.path.join(output_dir, f"{method}_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="ML Baseline 方法评估")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=list(METHOD_REGISTRY.keys()) + ["all"],
        help="选择方法: tfidf_lr / tfidf_svm / tfidf_nb / fasttext / all",
    )
    parser.add_argument("--data_path", type=str, default="./data/random_samples.jsonl")
    parser.add_argument("--output_dir", type=str, default="./baseline_results")
    parser.add_argument("--label_field", type=str, default="label3")
    parser.add_argument("--text_mode", type=str, default="user_assistant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument(
        "--ngram_min", type=int, default=1, help="TF-IDF n-gram 下界"
    )
    parser.add_argument(
        "--ngram_max", type=int, default=2, help="TF-IDF n-gram 上界"
    )
    parser.add_argument("--fasttext_epoch", type=int, default=25)
    parser.add_argument("--fasttext_lr", type=float, default=0.5)
    parser.add_argument("--fasttext_word_ngrams", type=int, default=2)
    parser.add_argument("--fasttext_dim", type=int, default=100)

    args = parser.parse_args()

    config = MLBaselineConfig(
        method=args.method,
        data_path=args.data_path,
        output_dir=args.output_dir,
        label_field=args.label_field,
        text_mode=args.text_mode,
        seed=args.seed,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        fasttext_epoch=args.fasttext_epoch,
        fasttext_lr=args.fasttext_lr,
        fasttext_word_ngrams=args.fasttext_word_ngrams,
        fasttext_dim=args.fasttext_dim,
    )

    train_data, dev_data, test_data, label_list, label2id, id2label = (
        _prepare_data(config)
    )

    methods = (
        list(METHOD_REGISTRY.keys()) if config.method == "all" else [config.method]
    )

    # TF-IDF 特征只构建一次，所有 TF-IDF 方法共用
    tfidf_needed = any(m in TFIDF_METHODS for m in methods)
    X_train = X_dev = X_test = None
    if tfidf_needed:
        print("\n构建 TF-IDF 特征（所有 TF-IDF 方法共用）...")
        train_texts = [s["text"] for s in train_data]
        dev_texts = [s["text"] for s in dev_data]
        test_texts = [s["text"] for s in test_data]
        X_train, X_dev, X_test, _vectorizer = build_tfidf_features(
            train_texts, dev_texts, test_texts,
            max_features=config.max_features,
            ngram_range=config.ngram_range,
        )
        print(f"TF-IDF 特征维度: {X_train.shape[1]}")

    y_train = [s["labels"] for s in train_data]
    y_dev = [s["labels"] for s in dev_data]
    y_test = [s["labels"] for s in test_data]

    all_results = []
    for method in methods:
        print(f"\n{'#' * 60}")
        print(f"# 运行: {METHOD_REGISTRY[method]}")
        print(f"{'#' * 60}")

        try:
            if method in TFIDF_METHODS:
                results = run_tfidf_method(
                    method,
                    X_train, y_train,
                    X_dev, y_dev,
                    X_test, y_test,
                    id2label, config,
                )
            elif method == "fasttext":
                results = run_fasttext(
                    train_data, dev_data, test_data, id2label, config,
                )
            else:
                raise ValueError(
                    f"不支持的方法: {method}，可选: {list(METHOD_REGISTRY.keys())}"
                )

            _print_results(results)
            path = _save_results(results, config.output_dir)
            print(f"结果已保存: {path}")
            all_results.append(results)
        except Exception as e:
            print(f"[错误] {METHOD_REGISTRY[method]} 运行失败: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) > 1:
        print(f"\n{'=' * 100}")
        print("ML Baseline 汇总")
        print(f"{'=' * 100}")
        print(
            f"{'Method':<35} {'Accuracy':<10} {'Macro F1':<10} "
            f"{'Wgt F1':<10} {'Precision':<10} {'Recall':<10} "
            f"{'Dev F1':<10} {'Train(s)':<10}"
        )
        print("-" * 100)

        all_results.sort(key=lambda x: x["metrics"]["macro_f1"], reverse=True)
        summary_data = []
        for r in all_results:
            m = r["metrics"]
            t = r.get("train_time_seconds", 0)
            dev_f1 = r.get("dev_macro_f1", 0)
            print(
                f"{r['method_name']:<35} {m['accuracy']:<10.4f} "
                f"{m['macro_f1']:<10.4f} {m['weighted_f1']:<10.4f} "
                f"{m['macro_precision']:<10.4f} {m['macro_recall']:<10.4f} "
                f"{dev_f1:<10.4f} {t:<10.2f}"
            )
            summary_data.append({
                "method": r["method"],
                "method_name": r["method_name"],
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"],
                "dev_macro_f1": dev_f1,
                "best_params": r.get("best_params", {}),
                "train_time_seconds": t,
                "throughput": r.get("throughput_samples_per_second", 0),
            })

        print(f"{'=' * 100}")
        best = all_results[0]
        print(
            f"最佳方法 (Macro F1): {best['method_name']} — "
            f"{best['metrics']['macro_f1']:.4f}"
        )

        summary_path = os.path.join(config.output_dir, "summary_ml.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"ML 汇总结果已保存: {summary_path}")

    print("\n评估完成!")


if __name__ == "__main__":
    main()
