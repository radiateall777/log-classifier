"""日志分类多分类任务的Baseline对比方法

支持2022-2024年间主流的Transformer-based文本分类模型作为Baseline对比。

使用方法:
    python baselines/run_baseline.py --model bert-base-uncased
    bash baselines/run_all_baselines.sh
"""

import os
# 设置HuggingFace镜像 (国内加速)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


# =============================================================================
# 数据加载 (与训练框架保持一致的数据处理逻辑)
# =============================================================================

def load_json_data(path: str) -> List[Dict[str, Any]]:
    """加载JSON或JSONL格式数据"""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    return data


def flatten_messages(messages: List[Dict[str, str]], text_mode: str, item: Dict[str, Any]) -> str:
    """将对话消息展平为单段文本"""
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
        return f"language: {language} dataset: {dataset_name} user: {user_text} assistant: {assistant_text}"
    raise ValueError(f"不支持的 text_mode: {text_mode}")


def build_samples(raw_data: List[Dict[str, Any]], label_field: str, text_mode: str) -> List[Dict[str, Any]]:
    """构建样本列表"""
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
    """构建标签映射"""
    label_list = sorted({x["label_text"] for x in samples})
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_list, label2id, id2label


def assign_label_ids(samples: List[Dict[str, Any]], label2id: Dict[str, int]) -> None:
    """为样本分配标签ID"""
    for x in samples:
        x["labels"] = label2id[x["label_text"]]


def split_dataset(samples: List[Dict[str, Any]], seed: int, test_size: float, dev_size: float):
    """划分训练/验证/测试集"""
    from sklearn.model_selection import train_test_split
    labels = [x["labels"] for x in samples]
    indices = list(range(len(samples)))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    relative_dev_size = dev_size / (1.0 - test_size)

    train_idx, dev_idx = train_test_split(
        train_val_idx, test_size=relative_dev_size, random_state=seed, stratify=train_val_labels
    )

    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in dev_idx],
        [samples[i] for i in test_idx],
    )


# =============================================================================
# 数据集类
# =============================================================================

class TextClassificationDataset(Dataset):
    """文本分类数据集"""

    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


class SimpleTextDataset(Dataset):
    """简化的文本数据集，用于pipeline推理"""

    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"text": self.texts[idx], "label": self.labels[idx]}


# =============================================================================
# 评估函数
# =============================================================================

def evaluate_predictions(y_true: List[int], y_pred: List[int], id2label: Dict[int, str]) -> Dict[str, Any]:
    """计算评估指标"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    report = classification_report(
        y_true, y_pred,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4, zero_division=0, output_dict=True
    )

    return {"metrics": metrics, "classification_report": report}


def run_inference(
    model_name: str,
    samples: List[Dict[str, Any]],
    id2label: Dict[int, str],
    max_length: int = 256,
    batch_size: int = 16,
) -> tuple:
    """使用指定模型运行推理"""
    print(f"\n{'='*60}")
    print(f"加载模型: {model_name}")
    print(f"{'='*60}")

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(id2label)

    # 使用CPU推理
    device = torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    model.to(device)
    model.eval()

    # 准备数据
    texts = [s["text"] for s in samples]
    labels = [s["labels"] for s in samples]

    all_preds = []

    # 分批推理
    print(f"开始推理 (样本数: {len(samples)}, 批大小: {batch_size})...")
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # batch_labels 未使用，移除避免 lint 警告
            _ = labels[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 推理
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

            all_preds.extend(preds)

            # 进度
            if (i + batch_size) % 200 == 0 or i + batch_size >= len(texts):
                print(f"  进度: {min(i+batch_size, len(texts))}/{len(texts)}")

    elapsed = time.time() - start_time
    throughput = len(samples) / elapsed if elapsed > 0 else float("inf")

    print(f"\n推理完成! 耗时: {elapsed:.2f}s, 吞吐量: {throughput:.2f} samples/s")

    return labels, all_preds, elapsed, throughput


def run_pipeline_inference(
    model_name: str,
    samples: List[Dict[str, Any]],
    id2label: Dict[int, str],
    max_length: int = 256,
) -> tuple:
    """使用HuggingFace pipeline运行推理 (更简单但可能更慢)"""
    print(f"\n{'='*60}")
    print(f"加载模型 (Pipeline): {model_name}")
    print(f"{'='*60}")

    device = 0 if torch.cuda.is_available() else -1  # -1表示CPU

    # 优先使用已下载的本地模型路径，避免重复下载
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    resolved_model = model_name
    for entry in os.listdir(cache_dir):
        if model_name.replace("/", "--") in entry:
            resolved_model = os.path.join(cache_dir, entry)
            break

    classifier = pipeline(
        "text-classification",
        model=resolved_model,
        tokenizer=resolved_model,
        device=device,
        max_length=max_length,
        truncation=True,
        top_k=None,  # 返回所有标签的分数，方便解析
    )

    texts = [s["text"] for s in samples]
    labels = [s["labels"] for s in samples]

    print(f"开始推理 (样本数: {len(samples)})...")
    start_time = time.time()

    # 批量处理
    predictions = classifier(texts, batch_size=8)

    elapsed = time.time() - start_time
    throughput = len(samples) / elapsed if elapsed > 0 else float("inf")

    # 解析预测结果：pipeline 可能返回整数索引 (如 "LABEL_0") 或原始标签名
    label2id_in_model = {v: k for k, v in id2label.items()}
    all_preds_ids = []
    for pred_batch in predictions:
        if isinstance(pred_batch, list) and len(pred_batch) > 0:
            # 取最高分结果
            best = max(pred_batch, key=lambda x: x["score"])
            label_str = best["label"]
            # 尝试直接映射为 id；若模型使用 LABEL_N 格式则按顺序取最大值
            if label_str in label2id_in_model:
                all_preds_ids.append(label2id_in_model[label_str])
            else:
                # fallback: 取最高分对应的 index
                sorted_preds = sorted(pred_batch, key=lambda x: x["score"], reverse=True)
                top_idx = int(sorted_preds[0]["label"].split("_")[-1]) if "_" in str(sorted_preds[0]["label"]) else 0
                all_preds_ids.append(top_idx)
        else:
            all_preds_ids.append(0)

    print(f"\n推理完成! 耗时: {elapsed:.2f}s, 吞吐量: {throughput:.2f} samples/s")

    return labels, all_preds_ids, elapsed, throughput


# =============================================================================
# 主函数
# =============================================================================

@dataclass
class BaselineConfig:
    """Baseline配置"""
    model_name: str
    max_length: int = 256
    batch_size: int = 16
    data_path: str = "./data/random_samples.jsonl"
    text_mode: str = "user_assistant"
    label_field: str = "label3"
    test_size: float = 0.1
    dev_size: float = 0.1
    seed: int = 42


def run_single_baseline(config: BaselineConfig, output_dir: str = "./baseline_results") -> Dict[str, Any]:
    """运行单个baseline评估"""

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 加载数据
    print(f"\n加载数据: {config.data_path}")
    raw_data = load_json_data(config.data_path)
    samples = build_samples(raw_data, config.label_field, config.text_mode)

    if len(samples) == 0:
        raise ValueError("没有有效样本!")

    print(f"总样本数: {len(samples)}")

    # 构建标签映射
    label_list, label2id, id2label = build_label_maps(samples)
    assign_label_ids(samples, label2id)

    print(f"标签数: {len(label_list)}")
    print(f"标签列表: {label_list}")

    # 划分数据集
    train_data, dev_data, test_data = split_dataset(
        samples, seed=config.seed,
        test_size=config.test_size, dev_size=config.dev_size
    )

    print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

    # 保存划分后的数据
    os.makedirs(output_dir, exist_ok=True)
    split_path = os.path.join(output_dir, f"{config.model_name.replace('/', '_')}_splits.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({
            "train": train_data,
            "dev": dev_data,
            "test": test_data,
            "label_list": label_list,
            "label2id": label2id,
            "id2label": id2label,
        }, f, ensure_ascii=False, indent=2)

    # 使用测试集评估
    print(f"\n使用测试集 ({len(test_data)} 样本) 进行评估...")

    try:
        y_true, y_pred, elapsed, throughput = run_inference(
            config.model_name,
            test_data,
            id2label,
            max_length=config.max_length,
            batch_size=config.batch_size,
        )
    except Exception as e:
        print(f"推理出错，尝试使用pipeline模式: {e}")
        y_true, y_pred, elapsed, throughput = run_pipeline_inference(
            config.model_name,
            test_data,
            id2label,
            max_length=config.max_length,
        )

    # 计算指标
    results = evaluate_predictions(y_true, y_pred, id2label)
    results["elapsed_seconds"] = elapsed
    results["throughput_samples_per_second"] = throughput
    results["model_name"] = config.model_name
    results["test_samples"] = len(test_data)
    results["num_labels"] = len(label_list)

    # 保存结果
    result_path = os.path.join(output_dir, f"{config.model_name.replace('/', '_')}_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"Baseline: {config.model_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Macro F1: {results['metrics']['macro_f1']:.4f}")
    print(f"Weighted F1: {results['metrics']['weighted_f1']:.4f}")
    print(f"Macro Precision: {results['metrics']['macro_precision']:.4f}")
    print(f"Macro Recall: {results['metrics']['macro_recall']:.4f}")
    print(f"耗时: {elapsed:.2f}s, 吞吐量: {throughput:.2f} samples/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="运行Baseline评估")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="模型名称 (HuggingFace)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批处理大小")
    parser.add_argument("--data_path", type=str, default="./data/random_samples.jsonl",
                        help="数据路径")
    parser.add_argument("--output_dir", type=str, default="./baseline_results",
                        help="输出目录")
    parser.add_argument("--label_field", type=str, default="label3",
                        help="标签字段")
    parser.add_argument("--text_mode", type=str, default="user_assistant",
                        help="文本模式")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    config = BaselineConfig(
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        data_path=args.data_path,
        label_field=args.label_field,
        text_mode=args.text_mode,
        seed=args.seed,
    )

    results = run_single_baseline(config, args.output_dir)

    print("\n评估完成!")
    return results


if __name__ == "__main__":
    main()
