"""测试在不同噪声情况下的准确率与吞吐量。

噪声指使用 UNK代替 text 文本中掉的 token。

使用方法:
    python baselines/eval.py --model_dir ./baseline_results/bert-base-uncased
"""

import os
import time
import json
import random
import argparse
from typing import Dict, Any, List

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score


def _inject_noise(text: str, tokenizer, noise_ratio: float, rng: random.Random) -> str:
    """将文本 tokenize 后按 noise_ratio 随机替换 token 为 [UNK]，再 decode 回字符串。"""
    if noise_ratio <= 0.0:
        return text

    # 使用 encode 获取 token IDs，不加特殊 token
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    if not token_ids:
        return text

    n_mask = int(len(token_ids) * noise_ratio)
    
    # 防止因随机取样导致超过列表长度的问题
    mask_indices = rng.sample(range(len(token_ids)), min(n_mask, len(token_ids)))
    
    for idx in mask_indices:
        token_ids[idx] = tokenizer.unk_token_id

    # 还原为带噪声的字符串
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def measure_throughput(model, tokenizer, texts: List[str], max_length: int, batch_size: int, device: torch.device, rounds: int = 3) -> float:
    """测量多次吞吐量，返回平均每秒样本数。"""
    print("\nWarm up (1 round)...")
    model.eval()
    
    # 预先tokenize所有数据并转为dataset/dataloader，避免计入tokenize的时间
    # (或者也可以在测速里包含tokenize的时间，这里选择包含tokenize，因为端到端推理都会带tokenize)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
            _ = model(**inputs)

    print(f"\nStart measuring throughput ({rounds} rounds)...")
    total_elapsed = 0.0
    with torch.no_grad():
        for r in range(rounds):
            start_t = time.perf_counter()
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
                _ = model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_t
            total_elapsed += elapsed
            print(f"Round {r+1}: {len(texts) / elapsed:.2f} samples/sec")

    avg_elapsed = total_elapsed / rounds
    throughput = len(texts) / avg_elapsed
    print(f"Average Throughput: {throughput:.2f} samples/sec")
    return throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="HF 模型目录 (需含 config.json, *model.pt / pytorch_model.bin, tokenizer 等)")
    parser.add_argument("--data_path", default="./data/random_samples.jsonl")
    parser.add_argument("--label_field", default="label3")
    parser.add_argument("--text_mode", default="user_assistant")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--noise_ratios", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9", help="逗号分隔的噪声比例列表")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--throughput_rounds", type=int, default=3, help="吞吐量测试轮数")
    args = parser.parse_args()

    noise_ratios = [float(x.strip()) for x in args.noise_ratios.split(",")]
    rng = random.Random(args.seed)

    print("============================================================")
    print("开始噪声鲁棒性与吞吐量测试")
    print(f"Model Dir : {args.model_dir}")
    print(f"Data Path : {args.data_path}")
    print(f"Noise Ratios: {noise_ratios}")
    print("============================================================\n")

    # 1. 尝试加载保存的固定划分数据获取 test 集
    splits_path = args.data_path
    if not os.path.exists(splits_path):
        print(f"ERROR: 找不到数据划分文件 {splits_path}。请先运行完整的训练获取固定的 test 划分以保证测试对齐。")
        return

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    test_data = splits["test"]
    label_list = splits.get("label_list", [])
    label2id = splits.get("label2id", {})
    id2label = splits.get("id2label", {v: k for k, v in label2id.items()})

    # 补全 labels
    for s in test_data:
        if "labels" not in s:
            s["labels"] = label2id[s["label_text"]]
            
    print(f"加载测试集完成: {len(test_data)} 条样本")

    # 2. 加载模型与 Tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # 如果有最优权重则加载
    best_weights_path = os.path.join(args.model_dir, "best_model.pt")
    if os.path.exists(best_weights_path):
        print(f"检测到 best_model.pt，加载最优权重: {best_weights_path}")
        model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    else:
        print("未检测到 best_model.pt，使用默认 checkpoint。")

    # 3. 循环各种噪声比例进行测试
    results_list = []
    
    print("\n============================================================")
    print("噪声比例\tAccuracy\tMacro F1\tWgt F1\tThroughput(s/s)")
    print("------------------------------------------------------------")

    for ratio in noise_ratios:
        # 添加噪声
        noisy_texts = []
        for s in test_data:
            noisy_text = _inject_noise(s["text"], tokenizer, ratio, rng)
            noisy_texts.append(noisy_text)
            
        labels = [s["labels"] for s in test_data]
        
        # 测试精度
        preds = []
        with torch.no_grad():
            for i in range(0, len(noisy_texts), args.batch_size):
                batch_text = noisy_texts[i:i+args.batch_size]
                inputs = tokenizer(batch_text, max_length=args.max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
                logits = model(**inputs).logits
                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                preds.extend(batch_preds.tolist())

        # 计算指标
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        wgt_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        
        # 4. 吞吐量测试 (只在当前 ratio 测一轮)
        throughput = measure_throughput(model, tokenizer, noisy_texts, args.max_length, args.batch_size, device, args.throughput_rounds)
        
        print(f"{ratio*100:4.1f}%\t\t{acc:.4f}\t\t{macro_f1:.4f}\t\t{wgt_f1:.4f}\t{throughput:.2f}")

        results_list.append({
            "noise_ratio": ratio,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": wgt_f1,
            "throughput_samples_per_sec": throughput
        })

    print("============================================================\n")

    # 5. 保存结果
    out_file = os.path.join(args.model_dir, "noise_robustness_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
        
    print(f"测试结果已保存至: {out_file}")


if __name__ == "__main__":
    main()
