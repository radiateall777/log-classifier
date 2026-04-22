"""多 GPU 并行 K-fold × 多 seed 调度器。

核心思想：K-fold × 多 seed 产生 K*S 个**独立**训练任务（互不依赖），
按 GPU 数量分发给 N 个 subprocess，每个 subprocess 通过 `CUDA_VISIBLE_DEVICES`
独占一张 GPU，跑 `run_fold_worker.py`。

加速近线性：4 GPU 并行约 3.5-4× 提速，对精度没有任何影响（每折与单 GPU
版本完全一致）。

用法：
    python3 baselines/run_kfold_parallel.py \\
        --model_name roberta-large \\
        --output_dir ./baseline_results/roberta_large_sota \\
        --gpus 3 4 6 7 \\
        --k_folds 5 --seeds 42 123 2024 \\
        --use_adversarial --use_rdrop --label_smoothing 0.1 \\
        --use_layerwise_lr_decay --use_eda --augment_target_classes 搜索算法
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

from log_classifier.data.preprocess import (
    assign_label_ids, build_label_maps, build_samples,
    filter_rare_classes, load_json_data,
)


WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "run_fold_worker.py")


# ------------------------------------------------------------------
# CLI（与 run_kfold_train.py 对齐 + 新增 --gpus / --max_parallel）
# ------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多 GPU 并行 K-fold 训练")

    # 数据
    p.add_argument("--data_path", default="./data/random_samples.jsonl")
    p.add_argument("--text_mode", default="user_assistant")
    p.add_argument("--label_field", default="label3")
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--test_split_seed", type=int, default=42)
    p.add_argument("--min_class_count", type=int, default=2)

    # K-fold
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])

    # 模型
    p.add_argument("--model_name", required=True)
    p.add_argument("--max_length", type=int, default=512)

    # 并行
    p.add_argument("--gpus", type=int, nargs="+", required=True,
                   help="可用的物理 GPU id 列表，例如 --gpus 3 4 6 7")
    p.add_argument("--max_parallel", type=int, default=None,
                   help="最大并发 fold 数，默认等于 GPU 数")

    # 输出
    p.add_argument("--output_dir", required=True)
    p.add_argument("--scratch_root", default=None,
                   help="Trainer 临时 checkpoint 根目录。默认 /tmp/<tag>_kfold_scratch "
                        "（独立文件系统，避免占用 home 配额）")
    p.add_argument("--keep_fold_outputs", action="store_true", default=False,
                   help="保留每折 worker 产物（调试用），默认仅保留最终汇总")

    # 训练超参（将透传给 worker）
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=20)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--early_stopping_patience", type=int, default=4)
    p.add_argument("--use_class_weights", action="store_true", default=True)
    p.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--no_fp16", dest="fp16_default", action="store_false", default=True)

    # Phase A 技巧
    p.add_argument("--use_focal_loss", action="store_true", default=False)
    p.add_argument("--focal_loss_gamma", type=float, default=2.0)
    p.add_argument("--use_adversarial", action="store_true", default=False)
    p.add_argument("--adversarial_method", type=str, default="fgm", choices=["fgm", "pgd"])
    p.add_argument("--adversarial_epsilon", type=float, default=1.0)
    p.add_argument("--use_layerwise_lr_decay", action="store_true", default=False)
    p.add_argument("--layerwise_lr_decay_rate", type=float, default=0.95)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--use_rdrop", action="store_true", default=False)
    p.add_argument("--rdrop_alpha", type=float, default=1.0)
    p.add_argument("--use_eda", action="store_true", default=False)
    p.add_argument("--augment_target_classes", type=str, nargs="*", default=None)
    p.add_argument("--num_aug_per_sample", type=int, default=2)

    return p


# ------------------------------------------------------------------
# 工具
# ------------------------------------------------------------------

def _build_worker_common_args(args) -> List[str]:
    """把训练/数据/技巧参数转成 worker CLI list。"""
    cli = [
        "--data_path", args.data_path,
        "--text_mode", args.text_mode,
        "--label_field", args.label_field,
        "--test_size", str(args.test_size),
        "--test_split_seed", str(args.test_split_seed),
        "--min_class_count", str(args.min_class_count),
        "--model_name", args.model_name,
        "--max_length", str(args.max_length),
        "--kfold_k", str(args.k_folds),
        "--train_batch_size", str(args.train_batch_size),
        "--eval_batch_size", str(args.eval_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--num_train_epochs", str(args.num_train_epochs),
        "--warmup_ratio", str(args.warmup_ratio),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--focal_loss_gamma", str(args.focal_loss_gamma),
        "--adversarial_method", str(args.adversarial_method),
        "--adversarial_epsilon", str(args.adversarial_epsilon),
        "--layerwise_lr_decay_rate", str(args.layerwise_lr_decay_rate),
        "--label_smoothing", str(args.label_smoothing),
        "--rdrop_alpha", str(args.rdrop_alpha),
        "--num_aug_per_sample", str(args.num_aug_per_sample),
    ]
    if args.use_class_weights:
        cli += ["--use_class_weights"]
    else:
        cli += ["--no_class_weights"]
    if args.bf16:
        cli += ["--bf16"]
    if not args.fp16_default:
        cli += ["--no_fp16"]
    if args.use_focal_loss:
        cli += ["--use_focal_loss"]
    if args.use_adversarial:
        cli += ["--use_adversarial"]
    if args.use_layerwise_lr_decay:
        cli += ["--use_layerwise_lr_decay"]
    if args.use_rdrop:
        cli += ["--use_rdrop"]
    if args.use_eda:
        cli += ["--use_eda"]
    if args.augment_target_classes:
        cli += ["--augment_target_classes"] + list(args.augment_target_classes)
    return cli


def _run_one_fold(
    gpu_id: int,
    seed: int,
    fold_idx: int,
    common_cli: List[str],
    output_dir: str,
    log_path: str,
    scratch_root: str,
) -> Dict[str, Any]:
    """在指定 GPU 上跑单折 worker。

    scratch_root 默认使用 /tmp 下的独立目录，避免占用 home 目录的磁盘配额
    （每折 Trainer 的 optimizer state + model safetensors 临时占 ~3 GB）。
    """
    tag = f"seed{seed}_fold{fold_idx}"
    scratch_dir = os.path.join(scratch_root, tag)
    result_path = os.path.join(scratch_dir, "result.npz")

    cmd = (
        [sys.executable, WORKER_SCRIPT]
        + common_cli
        + [
            "--seed", str(seed),
            "--fold_idx", str(fold_idx),
            "--output_path", result_path,
            "--scratch_dir", scratch_dir,
        ]
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"   # 实时 flush，便于监控
    env["TQDM_MININTERVAL"] = "10"   # 降低 tqdm 刷新频率，减小日志体积

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.run(
            cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True,
        )
    elapsed = time.time() - t0

    info = {
        "seed": seed, "fold_idx": fold_idx, "gpu_id": gpu_id,
        "elapsed_seconds": elapsed, "returncode": proc.returncode,
        "result_path": result_path if proc.returncode == 0 else None,
        "log_path": log_path,
    }
    return info


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def main():
    args = _build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.scratch_root:
        scratch_root = args.scratch_root
    else:
        # 用 /tmp 下的独立目录，名字带 PID 与 tag 防冲突
        model_tag = args.model_name.replace("/", "_")
        scratch_root = f"/tmp/{model_tag}_kfold_scratch_pid{os.getpid()}"
    os.makedirs(scratch_root, exist_ok=True)
    print(f"Scratch root (Trainer checkpoint 临时目录): {scratch_root}")

    logs_dir = os.path.join(args.output_dir, "fold_logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 1. 生成任务列表
    tasks: List[Tuple[int, int]] = [
        (seed, fold_idx)
        for seed in args.seeds
        for fold_idx in range(args.k_folds)
    ]
    n_tasks = len(tasks)
    n_gpus = len(args.gpus)
    max_parallel = args.max_parallel or n_gpus

    print(f"=" * 70)
    print(f"多 GPU K-fold 并行调度")
    print(f"=" * 70)
    print(f"Model:       {args.model_name}")
    print(f"Output:      {args.output_dir}")
    print(f"K-fold:      {args.k_folds}")
    print(f"Seeds:       {args.seeds}")
    print(f"任务数:      {n_tasks}")
    print(f"GPU 池:      {args.gpus}")
    print(f"最大并发:    {max_parallel}")
    print(f"开始时间:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 70, flush=True)

    # 2. 预计算 trainval 池 + test 划分（供汇总用；worker 内部会再做一次一致的划分）
    raw_data = load_json_data(args.data_path)
    samples = build_samples(raw_data, args.label_field, args.text_mode)
    samples = filter_rare_classes(samples, min_count=args.min_class_count)
    label_list, label2id, id2label = build_label_maps(samples)
    assign_label_ids(samples, label2id)

    labels_all = [x["labels"] for x in samples]
    indices = list(range(len(samples)))
    trainval_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.test_split_seed,
        stratify=labels_all,
    )
    trainval_samples = [samples[i] for i in trainval_idx]
    test_samples = [samples[i] for i in test_idx]
    trainval_labels = np.array([s["labels"] for s in trainval_samples])
    test_labels = np.array([s["labels"] for s in test_samples])
    num_classes = len(label_list)

    print(f"TrainVal pool: {len(trainval_samples)}  Test: {len(test_samples)}  "
          f"Classes: {num_classes}", flush=True)

    # 3. 并发跑
    common_cli = _build_worker_common_args(args)

    # 用 round-robin 分配 GPU：任务 i 去 gpus[i % n_gpus]
    # （未来可以做更智能的负载均衡）
    from queue import Queue
    gpu_queue: Queue = Queue()
    for g in args.gpus:
        gpu_queue.put(g)

    def _submit_task(task):
        seed, fold_idx = task
        gpu_id = gpu_queue.get()
        log_name = f"seed{seed}_fold{fold_idx}_gpu{gpu_id}.log"
        log_path = os.path.join(logs_dir, log_name)
        try:
            info = _run_one_fold(
                gpu_id, seed, fold_idx, common_cli,
                args.output_dir, log_path, scratch_root,
            )
        finally:
            gpu_queue.put(gpu_id)
        return info

    per_fold_results: List[Dict[str, Any]] = []
    global_t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(_submit_task, t): t for t in tasks}
        done_count = 0
        for fut in as_completed(futures):
            info = fut.result()
            done_count += 1
            elapsed = info["elapsed_seconds"]
            status = "✓" if info["returncode"] == 0 else "✗"
            print(f"  [{done_count:2d}/{n_tasks}] {status} "
                  f"seed={info['seed']} fold={info['fold_idx']} "
                  f"GPU={info['gpu_id']} 耗时={elapsed:.0f}s  "
                  f"(log: {os.path.basename(info['log_path'])})", flush=True)
            per_fold_results.append(info)

    # 4. 聚合结果
    oof_probs = np.zeros((len(trainval_samples), num_classes), dtype=np.float32)
    oof_seed_count = np.zeros(len(trainval_samples), dtype=np.int32)
    test_probs_accum = np.zeros((len(test_samples), num_classes), dtype=np.float32)
    test_model_count = 0
    per_run_summaries: List[Dict[str, Any]] = []

    failed = [r for r in per_fold_results if r["returncode"] != 0]

    for r in per_fold_results:
        if r["returncode"] != 0:
            continue
        data = np.load(r["result_path"])
        val_idx = data["val_indices"]
        val_probs = data["val_probs"]
        test_probs = data["test_probs"]

        oof_probs[val_idx] += val_probs
        oof_seed_count[val_idx] += 1
        test_probs_accum += test_probs
        test_model_count += 1

        per_run_summaries.append({
            "seed": int(data["seed"]),
            "fold": int(data["fold_idx"]) + 1,
            "val_macro_f1": float(data["val_macro_f1"]),
            "test_macro_f1": float(data["test_macro_f1"]),
            "test_accuracy": float(data["test_accuracy"]),
            "train_elapsed_seconds": float(data["train_elapsed_seconds"]),
            "gpu_id": r["gpu_id"],
        })

    if test_model_count == 0:
        print("\n[Error] 所有 fold 失败，无法生成汇总", flush=True)
        if failed:
            print("\n失败 fold 的日志:")
            for f in failed:
                print(f"  {f['log_path']}")
        sys.exit(1)

    if oof_seed_count.min() == 0:
        missing = int((oof_seed_count == 0).sum())
        print(f"\n[Warning] 有 {missing} 个 trainval 样本从未被评估（可能有折失败），"
              f"OOF 将在这些位置为 0 向量", flush=True)
        oof_seed_count = np.maximum(oof_seed_count, 1)

    oof_probs /= oof_seed_count[:, None]
    test_probs = test_probs_accum / test_model_count

    # 5. 评估
    oof_preds = np.argmax(oof_probs, axis=-1)
    test_preds = np.argmax(test_probs, axis=-1)

    oof_metrics = {
        "accuracy": float(accuracy_score(trainval_labels, oof_preds)),
        "macro_f1": float(f1_score(trainval_labels, oof_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(trainval_labels, oof_preds, average="weighted", zero_division=0)),
        "precision": float(precision_score(trainval_labels, oof_preds, average="macro", zero_division=0)),
        "recall": float(recall_score(trainval_labels, oof_preds, average="macro", zero_division=0)),
    }
    test_metrics = {
        "accuracy": float(accuracy_score(test_labels, test_preds)),
        "macro_f1": float(f1_score(test_labels, test_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test_labels, test_preds, average="weighted", zero_division=0)),
        "precision": float(precision_score(test_labels, test_preds, average="macro", zero_division=0)),
        "recall": float(recall_score(test_labels, test_preds, average="macro", zero_division=0)),
    }

    all_ids = list(range(num_classes))
    oof_report = classification_report(
        trainval_labels, oof_preds, labels=all_ids,
        target_names=[id2label[i] for i in all_ids],
        digits=4, zero_division=0, output_dict=True,
    )
    test_report = classification_report(
        test_labels, test_preds, labels=all_ids,
        target_names=[id2label[i] for i in all_ids],
        digits=4, zero_division=0, output_dict=True,
    )

    # 6. 落盘
    np.save(os.path.join(args.output_dir, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(args.output_dir, "oof_labels.npy"), trainval_labels.astype(np.int64))
    np.save(os.path.join(args.output_dir, "oof_index.npy"), np.array(trainval_idx, dtype=np.int64))
    np.save(os.path.join(args.output_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(args.output_dir, "test_labels.npy"), test_labels.astype(np.int64))

    total_elapsed = time.time() - global_t0
    summary = {
        "model_name": args.model_name,
        "k_folds": args.k_folds,
        "seeds": args.seeds,
        "num_models": test_model_count,
        "num_failed": len(failed),
        "gpus": args.gpus,
        "max_parallel": max_parallel,
        "label_list": label_list,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "config": {
            "max_length": args.max_length,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "bf16": args.bf16,
            "use_focal_loss": args.use_focal_loss,
            "use_adversarial": args.use_adversarial,
            "adversarial_method": args.adversarial_method,
            "use_layerwise_lr_decay": args.use_layerwise_lr_decay,
            "label_smoothing": args.label_smoothing,
            "use_rdrop": args.use_rdrop,
            "rdrop_alpha": args.rdrop_alpha,
            "use_eda": args.use_eda,
            "augment_target_classes": args.augment_target_classes,
        },
        "oof_metrics": oof_metrics,
        "oof_classification_report": oof_report,
        "test_ensemble_metrics": test_metrics,
        "test_ensemble_classification_report": test_report,
        "per_run_summaries": sorted(per_run_summaries, key=lambda r: (r["seed"], r["fold"])),
        "total_elapsed_seconds": round(total_elapsed, 2),
    }

    summary_path = os.path.join(args.output_dir, "kfold_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 清理 /tmp 下的 scratch（checkpoint + optimizer state 很大，必须清理）
    if not args.keep_fold_outputs:
        shutil.rmtree(scratch_root, ignore_errors=True)
        print(f"已清理 scratch: {scratch_root}")

    # 打印总结
    per_test_f1 = [r["test_macro_f1"] for r in per_run_summaries]
    print("\n" + "=" * 70)
    print("K-fold 集成总结")
    print("=" * 70)
    print(f"[OOF]           acc={oof_metrics['accuracy']:.4f}  "
          f"macro_f1={oof_metrics['macro_f1']:.4f}")
    print(f"[Test ensemble] acc={test_metrics['accuracy']:.4f}  "
          f"macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"单模型 test_macro_f1 统计：mean={np.mean(per_test_f1):.4f} "
          f"std={np.std(per_test_f1):.4f}  min={np.min(per_test_f1):.4f}  "
          f"max={np.max(per_test_f1):.4f}")
    print(f"成功 fold：{test_model_count}/{n_tasks}（失败 {len(failed)}）")
    print(f"总耗时：{total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"结果已写入: {args.output_dir}")


if __name__ == "__main__":
    main()
