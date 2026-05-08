import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from log_classifier.teacher.augment import TextCodeAugmenter
from log_classifier.teacher.data import ClassificationDataset, read_dataset
from log_classifier.teacher.model import CodeBERTClassifier
from log_classifier.teacher.train_stage1 import evaluate
from log_classifier.teacher.utils import get_device, load_label_mapping, load_yaml, save_json, set_seed


def collate_fn_eval(batch, tokenizer, max_length):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return {
        "ids": [item["id"] for item in batch],
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml(args.config)
    
    set_seed(config.get("seed", 42))
    device = get_device()

    checkpoint_dir = config["checkpoint_dir"]
    id2label, label2id = load_label_mapping(os.path.join(checkpoint_dir, "label_mapping.json"))
    num_labels = len(label2id)
    labels_list = list(id2label.keys())

    print("Loading data...")
    test_data = read_dataset(config.get("data_path", "data/random_samples_splits.json"), split="test")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    
    print("Initializing model...")
    model = CodeBERTClassifier(
        model_name=config.get("model_name", "microsoft/codebert-base"),
        num_labels=num_labels
    )
    # the encoder config should ideally be loaded from the checkpoint, but codebert-base works since its weights are restored
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"), map_location="cpu"))
    model.to(device)
    model.eval()

    # 1. Clean evaluation
    print("Evaluating on clean test set...")
    clean_dataset = ClassificationDataset(test_data, label2id)
    clean_loader = DataLoader(
        clean_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn_eval(b, tokenizer, config["max_length"])
    )
    clean_metrics = evaluate(model, clean_loader, device, labels_list, id2label)
    
    report = {
        "clean": clean_metrics
    }

    # 2. Robust evaluation
    if config.get("robust_eval", True):
        print("Evaluating on robust permutations...")
        augmenter = TextCodeAugmenter(seed=config.get("seed", 42))
        
        robust_types = [
            ("remove_role_markers", augmenter.remove_role_markers),
            ("remove_markdown_noise", augmenter.remove_markdown_noise),
            ("rename_code_variables", augmenter.rename_code_variables),
            ("remove_code_comments", augmenter.remove_code_comments),
            ("truncate_assistant_explanation", augmenter.truncate_assistant_explanation),
            ("add_harmless_noise", augmenter.add_harmless_noise),
            ("cn_en_term_swap", augmenter.cn_en_term_swap)
        ]
        
        robust_reports = {}
        robust_macro_f1s = []
        robust_accs = []
        
        for name, aug_func in robust_types:
            print(f"  -> Testing {name}...")
            # Perturb the test set
            perturbed_data = []
            for item in test_data:
                p_item = item.copy()
                p_item["text"] = augmenter._safe_augment(aug_func, item["text"])
                perturbed_data.append(p_item)
                
            p_dataset = ClassificationDataset(perturbed_data, label2id)
            p_loader = DataLoader(
                p_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                collate_fn=lambda b: collate_fn_eval(b, tokenizer, config["max_length"])
            )
            
            p_metrics = evaluate(model, p_loader, device, labels_list, id2label)
            robust_reports[name] = p_metrics
            robust_macro_f1s.append(p_metrics["macro_f1"])
            robust_accs.append(p_metrics["accuracy"])
            
        report["robust_by_type"] = robust_reports
        report["robust_average"] = {
            "accuracy": sum(robust_accs) / len(robust_accs),
            "macro_f1": sum(robust_macro_f1s) / len(robust_macro_f1s)
        }

    output_report = config["output_report"]
    save_json(report, output_report)
    print(f"Evaluation report saved to {output_report}")

if __name__ == "__main__":
    main()
