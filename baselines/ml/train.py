"""统一的 ML 训练驱动脚本"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from log_classifier.config import DataConfig, TrainConfig
from log_classifier.pipelines.ml_pipeline import run_ml_pipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["tfidf_lr", "tfidf_svm", "tfidf_nb", "tfidf_xgb", "embed_lr", "embed_svm", "embed_xgb"], help="ML模型名称")
    p.add_argument("--data_path", default="./data/random_samples.jsonl")
    p.add_argument("--output_dir", default=None, help="结果保存目录")
    p.add_argument("--text_mode", default="user_assistant")
    p.add_argument("--label_field", default="label3")
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--dev_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./outputs/baselines/ml/{args.method}"

    data_cfg = DataConfig(
        data_path=args.data_path,
        text_mode=args.text_mode,
        label_field=args.label_field,
        test_size=args.test_size,
        dev_size=args.dev_size,
    )
    train_cfg = TrainConfig(seed=args.seed)

    result = run_ml_pipeline(args.method, data_cfg, train_cfg, args.output_dir)

    result_path = os.path.join(
        args.output_dir,
        f"{args.method}_train_results.json",
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n训练结果已保存: {result_path}")

if __name__ == "__main__":
    main()
