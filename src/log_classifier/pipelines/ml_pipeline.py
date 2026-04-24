"""机器学习的训练与验证 Pipeline"""

import os
import json
import time
import pickle
from typing import Dict, Any, List, Optional
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from log_classifier.config import DataConfig, TrainConfig
from log_classifier.models.baseline.ml_models import build_tfidf_vectorizer, build_ml_classifier, ML_PARAM_GRIDS, EmbeddingVectorizer
from log_classifier.data.preprocess import (
    load_json_data, build_samples, filter_rare_classes, build_label_maps,
    assign_label_ids, split_dataset
)


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, id2label: Dict[int, str]) -> Dict[str, Any]:
    all_label_ids = sorted(id2label.keys())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=all_label_ids, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=all_label_ids, zero_division=0),
    }


def _load_fixed_splits(data_cfg: DataConfig, train_cfg: TrainConfig) -> Optional[Dict[str, Any]]:
    # 与 DL 复用同样的 test_split 读取逻辑以保证一致性
    splits_path = data_cfg.data_path.replace(".jsonl", "_splits.json").replace(".json", "_splits.json")
    if os.path.exists(splits_path):
        try:
            with open(splits_path, "r", encoding="utf-8") as f:
                fixed = json.load(f)
            
            # 由于历史实现的原因，我们还需要严格比对设定参数是否吻合
            if fixed.get("config", {}).get("test_size") == data_cfg.test_size and \
               fixed.get("config", {}).get("seed") == train_cfg.seed:
                return fixed
        except Exception:
            pass
    return None

def run_ml_pipeline(
    method: str,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    base_output_dir: str,
) -> Dict[str, Any]:
    """
    ML 的独立微调 Pipeline
    负责使用 TF-IDF，执行超参搜索（在 Dev 上），使用最佳超参训练，在 Test 上推理，并落盘。
    """
    # 1. 准备数据
    fixed = _load_fixed_splits(data_cfg, train_cfg)
    if fixed is not None:
        train_data = fixed["train"]
        dev_data = fixed["dev"]
        test_data = fixed["test"]
        label_list = fixed.get("label_list")
        label2id = fixed.get("label2id")
        id2label = {int(k): v for k, v in fixed.get("id2label", {}).items()} # JSON 读出的 key 是字符串

        for s in train_data + dev_data + test_data:
            if "labels" not in s:
                s["labels"] = label2id[s["label_text"]]
    else:
        raw_data = load_json_data(data_cfg.data_path)
        samples = build_samples(raw_data, data_cfg.label_field, data_cfg.text_mode)
        samples = filter_rare_classes(samples, min_count=data_cfg.min_class_count)
        label_list, label2id, id2label = build_label_maps(samples)
        assign_label_ids(samples, label2id)
        train_data, dev_data, test_data = split_dataset(
            samples, seed=train_cfg.seed, test_size=data_cfg.test_size, dev_size=data_cfg.dev_size
        )

    # 提取特征文本
    train_texts = [s["text"] for s in train_data]
    dev_texts = [s["text"] for s in dev_data]
    test_texts = [s["text"] for s in test_data]
    
    y_train = np.array([s["labels"] for s in train_data])
    y_dev = np.array([s["labels"] for s in dev_data])
    y_test = np.array([s["labels"] for s in test_data])

    if method.startswith("embed_"):
        print("\n========== 稠密 Embedding 特征抽取 ==========")
        vectorizer = EmbeddingVectorizer()  # BAAI/bge-small-en-v1.5
    else:
        print("\n========== TF-IDF 特征抽取 ==========")
        vectorizer = build_tfidf_vectorizer()
        
    X_train = vectorizer.fit_transform(train_texts)
    X_dev = vectorizer.transform(dev_texts)
    X_test = vectorizer.transform(test_texts)
    print(f"Feature Vector Shape: {X_train.shape}")

    # 2. 超参数遍历（网格搜索）
    print(f"\n========== {method} 超参搜索 (Dev 集评测) ==========")
    param_grid = ML_PARAM_GRIDS.get(method, {})
    
    # 简单的网格搜索构造器
    from itertools import product
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    combinations = [dict(zip(keys, v)) for v in product(*value_lists)]
    
    if not combinations:
        combinations = [{}]

    best_dev_f1 = -1.0
    best_params = None
    best_clf = None
    best_train_time = 0.0

    for kwargs in combinations:
        clf = build_ml_classifier(method, train_cfg.seed, **kwargs)
        t0 = time.time()
        clf.fit(X_train, y_train)
        t_train = time.time() - t0
        
        dev_preds = clf.predict(X_dev)
        dev_f1 = f1_score(y_dev, dev_preds, average="macro", zero_division=0)
        
        print(f"参数: {kwargs} => Dev Macro F1: {dev_f1:.4f} (耗时: {t_train:.2fs})")
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_params = kwargs
            best_clf = clf
            best_train_time = t_train
    
    print(f"\n★ 最佳超参: {best_params} (Dev Macro F1: {best_dev_f1:.4f})")

    # 3. 在 Test 上验证吞吐量和精度
    print(f"\n========== {method} 测试集推理 ==========")
    t_start = time.perf_counter()
    test_preds = best_clf.predict(X_test)
    t_elapsed = time.perf_counter() - t_start
    throughput = len(y_test) / t_elapsed if t_elapsed > 0 else float("inf")

    test_metrics = _evaluate_predictions(y_test, test_preds, id2label)
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Throughput: {throughput:.2f} samples/s")

    # 4. 落盘保存 Pipeline (Vectorizer + Classifier)
    output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "model.pkl")
    vectorizer_path = os.path.join(output_dir, "vectorizer.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_clf, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    # 保存标签映射以备推理文件映射不一致
    labels_path = os.path.join(output_dir, "label_mappings.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False)
        
    print(f"\n模型权重和特征抽取器已保存到: {output_dir}")

    # 构造一致的结果 Dict
    result = {
        "model_name": method,
        "label_list": label_list,
        "label2id": label2id,
        "id2label": id2label,
        "train_samples": len(train_data),
        "dev_samples": len(dev_data),
        "test_samples": len(test_data),
        "num_labels": len(label_list),
        "dev_macro_f1": round(best_dev_f1, 4),
        "best_params": best_params,
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "train_time_seconds": round(best_train_time, 2),
        "test_throughput_samples_per_sec": round(throughput, 2),
        "model_path": model_path,
        "vectorizer_path": vectorizer_path,
    }
    return result
