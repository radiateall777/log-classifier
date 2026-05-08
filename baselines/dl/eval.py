"""测试不同 UNK token 缺失率下的准确率与吞吐量。

噪声定义：
    先对文本进行 tokenizer；
    然后在 input_ids 上按指定缺失率，将非特殊、非 PAD 的真实内容 token
    替换为 tokenizer.unk_token_id；
    attention_mask 保持不变。

这是真正的 input-id-level UNK replacement robustness，
不是 decode/retokenize 后的文本删除鲁棒性。

使用方法:
    python baselines/eval.py --model_dir ./baseline_results/bert-base-uncased
"""

import os
import time
import json
import argparse
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score


def build_special_token_mask(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """构建特殊 token mask。

    special tokens 包括：
        [PAD] / [CLS] / [SEP] / [UNK] / <s> / </s> / <pad> 等。

    Args:
        input_ids: [B, L]
        tokenizer: HuggingFace tokenizer

    Returns:
        special_mask: [B, L], True 表示该位置是特殊 token。
    """
    special_ids = set(tokenizer.all_special_ids)

    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        special_mask |= input_ids.eq(int(token_id))

    return special_mask


def apply_unk_token_noise_fixed_ratio(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    noise_ratio: float,
    min_keep_tokens: int = 1,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """在 input_ids 上按固定比例将内容 token 替换为 UNK。

    与旧 baseline 中的 _inject_noise 不同：
        - 不 decode 回字符串；
        - 不重新 tokenizer；
        - 不删除 UNK；
        - 不改变 attention_mask；
        - 不改变 token 位置。

    每条样本会替换：
        int(num_valid_content_tokens * noise_ratio)
    个 token。

    Args:
        input_ids:
            Tensor [B, L]
        attention_mask:
            Tensor [B, L]
        tokenizer:
            HuggingFace tokenizer
        noise_ratio:
            token 缺失率，范围 [0, 1]
        min_keep_tokens:
            每条样本至少保留的内容 token 数
        generator:
            可选 torch.Generator，用于可复现随机采样

    Returns:
        noisy_input_ids:
            Tensor [B, L]
        noise_mask:
            Bool Tensor [B, L]，True 表示该位置被替换为 UNK。
    """
    if tokenizer.unk_token_id is None:
        raise ValueError(
            "tokenizer.unk_token_id is None. "
            "UNK replacement robustness requires a tokenizer with unk_token_id."
        )

    if not 0.0 <= float(noise_ratio) <= 1.0:
        raise ValueError(f"noise_ratio must be in [0, 1], got {noise_ratio}")

    noisy_input_ids = input_ids.clone()
    device = input_ids.device

    special_mask = build_special_token_mask(input_ids, tokenizer).to(device)
    valid_mask = attention_mask.bool() & (~special_mask)

    noise_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    batch_size = input_ids.size(0)

    for i in range(batch_size):
        valid_positions = valid_mask[i].nonzero(as_tuple=True)[0]
        num_valid = valid_positions.numel()

        if num_valid == 0:
            continue

        if num_valid <= min_keep_tokens:
            continue

        max_maskable = max(0, num_valid - int(min_keep_tokens))
        n_mask = int(num_valid * float(noise_ratio))
        n_mask = min(n_mask, max_maskable)

        if n_mask <= 0:
            continue

        perm = torch.randperm(
            num_valid,
            device=device,
            generator=generator,
        )
        selected_positions = valid_positions[perm[:n_mask]]
        noise_mask[i, selected_positions] = True

    noisy_input_ids[noise_mask] = int(tokenizer.unk_token_id)

    return noisy_input_ids, noise_mask


def predict_with_noise(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    noise_ratio: float | None = None,
    min_keep_tokens: int = 1,
    seed: int = 42,
) -> List[int]:
    """对 clean 或 UNK-noisy 输入进行预测。"""
    model.eval()
    preds = []

    generator = None
    if noise_ratio is not None and noise_ratio > 0.0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            if noise_ratio is not None and noise_ratio > 0.0:
                noisy_input_ids, _ = apply_unk_token_noise_fixed_ratio(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=tokenizer,
                    noise_ratio=float(noise_ratio),
                    min_keep_tokens=int(min_keep_tokens),
                    generator=generator,
                )
                inputs["input_ids"] = noisy_input_ids

            logits = model(**inputs).logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(batch_preds.tolist())

    return preds


def measure_throughput(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    noise_ratio: float | None = None,
    min_keep_tokens: int = 1,
    seed: int = 42,
    rounds: int = 3,
) -> float:
    """测量端到端吞吐量。

    当前吞吐量包含：
        tokenizer + optional UNK noise injection + model forward

    这样更接近实际 noisy evaluation 的端到端代价。
    """
    print("\nWarm up (1 round)...")
    model.eval()

    generator = None
    if noise_ratio is not None and noise_ratio > 0.0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            if noise_ratio is not None and noise_ratio > 0.0:
                noisy_input_ids, _ = apply_unk_token_noise_fixed_ratio(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    tokenizer=tokenizer,
                    noise_ratio=float(noise_ratio),
                    min_keep_tokens=int(min_keep_tokens),
                    generator=generator,
                )
                inputs["input_ids"] = noisy_input_ids

            _ = model(**inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"\nStart measuring throughput ({rounds} rounds)...")
    total_elapsed = 0.0

    with torch.no_grad():
        for r in range(rounds):
            # 每一轮使用不同但可复现的随机噪声。
            round_generator = None
            if noise_ratio is not None and noise_ratio > 0.0:
                round_generator = torch.Generator(device=device)
                round_generator.manual_seed(int(seed) + 1000 + r)

            start_t = time.perf_counter()

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                inputs = tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                if noise_ratio is not None and noise_ratio > 0.0:
                    noisy_input_ids, _ = apply_unk_token_noise_fixed_ratio(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        tokenizer=tokenizer,
                        noise_ratio=float(noise_ratio),
                        min_keep_tokens=int(min_keep_tokens),
                        generator=round_generator,
                    )
                    inputs["input_ids"] = noisy_input_ids

                _ = model(**inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_t
            total_elapsed += elapsed

            samples_per_sec = len(texts) / elapsed
            print(f"Round {r + 1}: {samples_per_sec:.2f} samples/sec")

    avg_elapsed = total_elapsed / rounds
    throughput = len(texts) / avg_elapsed

    print(f"Average Throughput: {throughput:.2f} samples/sec")
    return throughput


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def average_noise_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "throughput_samples_per_sec": None,
        }

    return {
        "accuracy": float(np.mean([r["accuracy"] for r in results])),
        "macro_f1": float(np.mean([r["macro_f1"] for r in results])),
        "weighted_f1": float(np.mean([r["weighted_f1"] for r in results])),
        "throughput_samples_per_sec": float(
            np.mean([r["throughput_samples_per_sec"] for r in results])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="HF 模型目录，需包含 config.json / pytorch_model.bin / tokenizer 等",
    )
    parser.add_argument(
        "--data_path",
        default="./data/random_samples.jsonl",
        help="固定划分数据文件，需包含 test split",
    )
    parser.add_argument("--label_field", default="label3")
    parser.add_argument("--text_mode", default="user_assistant")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--noise_ratios",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="逗号分隔的 token 缺失率列表",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--throughput_rounds",
        type=int,
        default=3,
        help="吞吐量测试轮数",
    )
    parser.add_argument(
        "--min_keep_tokens",
        type=int,
        default=1,
        help="每条样本至少保留的真实内容 token 数",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="结果保存路径。若不指定，则保存到 model_dir/noise_robustness_results.json",
    )

    args = parser.parse_args()

    noise_ratios = [float(x.strip()) for x in args.noise_ratios.split(",") if x.strip()]

    print("============================================================")
    print("开始 input-id-level UNK token 缺失鲁棒性与吞吐量测试")
    print(f"Model Dir      : {args.model_dir}")
    print(f"Data Path      : {args.data_path}")
    print(f"Noise Ratios   : {noise_ratios}")
    print(f"Max Length     : {args.max_length}")
    print(f"Batch Size     : {args.batch_size}")
    print(f"Min Keep Tokens: {args.min_keep_tokens}")
    print("============================================================\n")

    if not os.path.exists(args.data_path):
        print(
            f"ERROR: 找不到数据划分文件 {args.data_path}。"
            "请先运行完整训练获取固定 test 划分，以保证测试对齐。"
        )
        return

    with open(args.data_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_data = splits["test"]
    label_list = splits.get("label_list", [])
    label2id = splits.get("label2id", {})
    id2label = splits.get("id2label", {v: k for k, v in label2id.items()})

    for s in test_data:
        if "labels" not in s:
            if "label_text" in s:
                s["labels"] = label2id[s["label_text"]]
            elif args.label_field in s:
                s["labels"] = label2id[s[args.label_field]]
            else:
                raise KeyError(
                    f"样本中找不到 labels / label_text / {args.label_field}: {s.keys()}"
                )

    texts = [s["text"] for s in test_data]
    labels = [int(s["labels"]) for s in test_data]

    print(f"加载测试集完成: {len(test_data)} 条样本")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    if tokenizer.unk_token_id is None:
        raise ValueError(
            f"Tokenizer from {args.model_dir} has no unk_token_id. "
            "Cannot run UNK replacement robustness evaluation."
        )

    print(f"UNK token   : {tokenizer.unk_token}")
    print(f"UNK token id: {tokenizer.unk_token_id}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    best_weights_path = os.path.join(args.model_dir, "best_model.pt")
    if os.path.exists(best_weights_path):
        print(f"检测到 best_model.pt，加载最优权重: {best_weights_path}")
        model.load_state_dict(
            torch.load(best_weights_path, map_location=device, weights_only=True)
        )
    else:
        print("未检测到 best_model.pt，使用默认 checkpoint。")

    results = {
        "model_dir": args.model_dir,
        "data_path": args.data_path,
        "noise_type": "input_id_level_unk_replacement_fixed_ratio",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "min_keep_tokens": args.min_keep_tokens,
        "seed": args.seed,
        "clean": None,
        "by_noise_ratio": [],
        "average_all_noise_ratios": None,
    }

    print("\n============================================================")
    print("Clean Evaluation")
    print("------------------------------------------------------------")

    clean_preds = predict_with_noise(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        noise_ratio=None,
        min_keep_tokens=args.min_keep_tokens,
        seed=args.seed,
    )
    clean_metrics = compute_metrics(labels, clean_preds)

    clean_throughput = measure_throughput(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        noise_ratio=None,
        min_keep_tokens=args.min_keep_tokens,
        seed=args.seed,
        rounds=args.throughput_rounds,
    )

    clean_metrics["throughput_samples_per_sec"] = float(clean_throughput)
    results["clean"] = clean_metrics

    print(
        f"Clean\tAcc={clean_metrics['accuracy']:.4f}\t"
        f"MacroF1={clean_metrics['macro_f1']:.4f}\t"
        f"WgtF1={clean_metrics['weighted_f1']:.4f}\t"
        f"Throughput={clean_metrics['throughput_samples_per_sec']:.2f}"
    )

    print("\n============================================================")
    print("Noise Ratio\tAccuracy\tMacro F1\tWgt F1\tThroughput(s/s)")
    print("------------------------------------------------------------")

    noise_results = []

    for ratio in noise_ratios:
        ratio_seed = args.seed + int(ratio * 1000)

        preds = predict_with_noise(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            noise_ratio=ratio,
            min_keep_tokens=args.min_keep_tokens,
            seed=ratio_seed,
        )

        metrics = compute_metrics(labels, preds)

        throughput = measure_throughput(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            noise_ratio=ratio,
            min_keep_tokens=args.min_keep_tokens,
            seed=ratio_seed,
            rounds=args.throughput_rounds,
        )

        result_item = {
            "noise_ratio": float(ratio),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "throughput_samples_per_sec": float(throughput),
        }

        results["by_noise_ratio"].append(result_item)
        noise_results.append(result_item)

        print(
            f"{ratio * 100:4.1f}%\t\t"
            f"{result_item['accuracy']:.4f}\t\t"
            f"{result_item['macro_f1']:.4f}\t\t"
            f"{result_item['weighted_f1']:.4f}\t"
            f"{result_item['throughput_samples_per_sec']:.2f}"
        )

    results["average_all_noise_ratios"] = average_noise_results(noise_results)

    print("------------------------------------------------------------")
    avg = results["average_all_noise_ratios"]
    print(
        f"AVG\t\t{avg['accuracy']:.4f}\t\t"
        f"{avg['macro_f1']:.4f}\t\t"
        f"{avg['weighted_f1']:.4f}\t"
        f"{avg['throughput_samples_per_sec']:.2f}"
    )
    print("============================================================\n")

    out_file = args.output_file
    if out_file is None:
        out_file = os.path.join(args.model_dir, "noise_robustness_results.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"测试结果已保存至: {out_file}")


if __name__ == "__main__":
    main()