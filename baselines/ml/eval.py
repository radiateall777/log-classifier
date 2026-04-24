"""测试 ML Baseline 在不同噪声下的准确率与吞吐量，与 DL 侧保持一致。"""

import os
import sys
import time
import json
import random
import pickle
import argparse
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sklearn.metrics import accuracy_score, f1_score
from log_classifier.config import DataConfig, TrainConfig
from log_classifier.pipelines.ml_pipeline import _load_fixed_splits
from log_classifier.models.baseline.ml_models import EmbeddingVectorizer  # needed for pickle.load


def _inject_noise(text: str, noise_ratio: float, rng: random.Random) -> str:
    """简单的单词级噪音注入。对于 ML，由于没有 huggingface tokenizer，我们按照空格进行词级替换。"""
    if noise_ratio <= 0.0:
        return text
    
    words = text.split()
    if not words:
        return text
        
    n_mask = int(len(words) * noise_ratio)
    mask_indices = set(rng.sample(range(len(words)), min(n_mask, len(words))))
    
    noisy_words = []
    for i, w in enumerate(words):
        if i in mask_indices:
            noisy_words.append("[UNK]")
        else:
            noisy_words.append(w)
            
    return " ".join(noisy_words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="ML 模型的目录 (需包含 model.pkl 和 vectorizer.pkl)")
    parser.add_argument("--data_path", default="./data/random_samples.jsonl")
    parser.add_argument("--noise_ratios", default="0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--throughput_rounds", type=int, default=3)
    args = parser.parse_args()

    noise_ratios = [float(x.strip()) for x in args.noise_ratios.split(",")]
    rng = random.Random(args.seed)

    print("============================================================")
    print("开始噪声鲁棒性与吞吐量测试 (ML模型)")
    print(f"Model Dir : {args.model_dir}")
    print("============================================================\n")

    # 1. 尝试加载训练时采用的相同划分
    data_cfg = DataConfig(data_path=args.data_path)
    train_cfg = TrainConfig(seed=args.seed)
    splits = _load_fixed_splits(data_cfg, train_cfg)
    
    if splits is None:
        print(f"ERROR: 找不到数据划分文件或参数不匹配。请验证训练环境。")
        return

    test_data = splits["test"]
    labels_path = os.path.join(args.model_dir, "label_mappings.json")
    with open(labels_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)
        label2id = mappings["label2id"]
        
    for s in test_data:
        if "labels" not in s:
            s["labels"] = label2id[s["label_text"]]

    # 2. 加载 Vectorizer 和 Model
    with open(os.path.join(args.model_dir, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(args.model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    print(f"已加载特征抽取器与模型。测试集大小: {len(test_data)}")

    # 3. 循环注入测试
    results_list = []
    print("\n============================================================")
    print("噪声比例\tAccuracy\tMacro F1\tWgt F1\tThroughput(s/s)")
    print("------------------------------------------------------------")

    for ratio in noise_ratios:
        noisy_texts = [_inject_noise(s["text"], ratio, rng) for s in test_data]
        y_true = [s["labels"] for s in test_data]
        
        # 吞吐量循环测算
        total_elapsed = 0.0
        for r in range(args.throughput_rounds):
            start_t = time.perf_counter()
            X_test = vectorizer.transform(noisy_texts)
            y_pred = model.predict(X_test)
            elapsed = time.perf_counter() - start_t
            total_elapsed += elapsed
            
        throughput = len(noisy_texts) / (total_elapsed / args.throughput_rounds)
        
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        wgt_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        print(f"{ratio*100:4.1f}%\t\t{acc:.4f}\t\t{macro_f1:.4f}\t\t{wgt_f1:.4f}\t{throughput:.2f}")

        results_list.append({
            "noise_ratio": ratio,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": wgt_f1,
            "throughput_samples_per_sec": throughput
        })

    out_file = os.path.join(args.model_dir, "noise_robustness_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
        
    print(f"\n测试结果已保存至: {out_file}")

if __name__ == "__main__":
    main()
