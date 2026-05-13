import argparse
import glob
import json
import os
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_name_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path)) or os.path.basename(path)


def get_clean_accuracy(data: dict[str, Any]) -> float | None:
    if isinstance(data.get("summary"), dict) and data["summary"].get("clean_accuracy") is not None:
        return float(data["summary"]["clean_accuracy"])
    if isinstance(data.get("clean"), dict) and data["clean"].get("accuracy") is not None:
        return float(data["clean"]["accuracy"])
    if isinstance(data.get("metrics"), dict) and data["metrics"].get("accuracy") is not None:
        return float(data["metrics"]["accuracy"])
    if data.get("accuracy") is not None:
        return float(data["accuracy"])
    return None


def get_robust_average_accuracy(data: dict[str, Any]) -> float | None:
    if isinstance(data.get("summary"), dict) and data["summary"].get("robust_average_accuracy") is not None:
        return float(data["summary"]["robust_average_accuracy"])
    if isinstance(data.get("robust_average"), dict) and data["robust_average"].get("accuracy") is not None:
        return float(data["robust_average"]["accuracy"])
    if isinstance(data.get("average_all_noise_ratios"), dict):
        avg = data["average_all_noise_ratios"]
        if avg.get("accuracy") is not None:
            return float(avg["accuracy"])
    if isinstance(data.get("unk_noise_average_0.1_0.9"), dict):
        avg = data["unk_noise_average_0.1_0.9"]
        if avg.get("accuracy") is not None:
            return float(avg["accuracy"])
    return None


def get_throughput(data: dict[str, Any]) -> float | None:
    if isinstance(data.get("summary"), dict) and data["summary"].get("throughput_samples_per_sec") is not None:
        return float(data["summary"]["throughput_samples_per_sec"])
    if isinstance(data.get("clean"), dict):
        clean = data["clean"]
        if clean.get("throughput_samples_per_sec") is not None:
            return float(clean["throughput_samples_per_sec"])
        if clean.get("throughput_samples_per_second") is not None:
            return float(clean["throughput_samples_per_second"])
    if data.get("throughput_samples_per_sec") is not None:
        return float(data["throughput_samples_per_sec"])
    if data.get("throughput_samples_per_second") is not None:
        return float(data["throughput_samples_per_second"])
    if isinstance(data.get("metrics"), dict) and data["metrics"].get("throughput_samples_per_second") is not None:
        return float(data["metrics"]["throughput_samples_per_second"])
    return None


def classify_source(path: str, data: dict[str, Any]) -> str:
    if "eval_quality_report.json" in path:
        return "ours"
    if "noise_robustness_results.json" in path:
        return "baseline_robustness"
    if "baseline_results" in path or data.get("method_name") or data.get("method"):
        return "baseline"
    return "other"


def collect_records(search_roots: list[str]) -> list[dict[str, Any]]:
    patterns = [
        "**/eval_quality_report.json",
        "**/noise_robustness_results.json",
        "**/*_results.json",
    ]

    seen_paths = set()
    records = []

    for root in search_roots:
        for pattern in patterns:
            for path in glob.glob(os.path.join(root, pattern), recursive=True):
                norm_path = os.path.normpath(path)
                if norm_path in seen_paths:
                    continue
                seen_paths.add(norm_path)
                try:
                    data = load_json(norm_path)
                except Exception:
                    continue

                record = {
                    "name": data.get("method_name") or data.get("method") or safe_name_from_path(norm_path),
                    "path": norm_path,
                    "source": classify_source(norm_path, data),
                    "clean_accuracy": get_clean_accuracy(data),
                    "robust_accuracy": get_robust_average_accuracy(data),
                    "throughput": get_throughput(data),
                }
                records.append(record)

    return records


def choose_baseline(records: list[dict[str, Any]], baseline_name: str | None) -> dict[str, Any] | None:
    baseline_candidates = [r for r in records if r["source"] in {"baseline", "baseline_robustness"}]
    if baseline_name:
        baseline_name_lower = baseline_name.lower()
        for record in baseline_candidates:
            if baseline_name_lower in str(record["name"]).lower() or baseline_name_lower in str(record["path"]).lower():
                return record
        return None

    if not baseline_candidates:
        return None

    baseline_candidates.sort(
        key=lambda r: (
            r["robust_accuracy"] is None,
            -(r["robust_accuracy"] or -1.0),
            -(r["clean_accuracy"] or -1.0),
        )
    )
    return baseline_candidates[0]


def build_summary(records, baseline_record, accuracy_target, robust_lift_target, throughput_lift_target):
    summary_rows = []
    baseline_clean = None if baseline_record is None else baseline_record["clean_accuracy"]
    baseline_robust = None if baseline_record is None else baseline_record["robust_accuracy"]
    baseline_throughput = None if baseline_record is None else baseline_record["throughput"]

    for record in records:
        clean_accuracy = record["clean_accuracy"]
        robust_accuracy = record["robust_accuracy"]
        throughput = record["throughput"]

        robust_lift = None
        throughput_lift = None
        robust_pass = None
        throughput_pass = None

        if robust_accuracy is not None and baseline_robust is not None:
            robust_lift = robust_accuracy - baseline_robust
            robust_pass = robust_lift >= robust_lift_target

        if throughput is not None and baseline_throughput is not None and baseline_throughput > 0:
            throughput_lift = throughput / baseline_throughput - 1.0
            throughput_pass = throughput_lift >= throughput_lift_target

        summary_rows.append(
            {
                "name": record["name"],
                "source": record["source"],
                "path": record["path"],
                "clean_accuracy": clean_accuracy,
                "robust_accuracy": robust_accuracy,
                "throughput_samples_per_sec": throughput,
                "meets_accuracy_90": clean_accuracy is not None and clean_accuracy >= accuracy_target,
                "robust_lift_vs_baseline": robust_lift,
                "meets_robust_plus_10pct": robust_pass,
                "throughput_lift_vs_baseline": throughput_lift,
                "meets_throughput_plus_20pct": throughput_pass,
            }
        )

    return {
        "targets": {
            "clean_accuracy": accuracy_target,
            "robust_lift_absolute": robust_lift_target,
            "throughput_lift_ratio": throughput_lift_target,
        },
        "baseline_reference": baseline_record,
        "results": summary_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search_roots",
        nargs="+",
        default=["baselines", "outputs"],
        help="Directories to scan for result JSON files.",
    )
    parser.add_argument(
        "--baseline_name",
        type=str,
        default=None,
        help="Optional baseline model name substring to use as the comparison reference.",
    )
    parser.add_argument(
        "--accuracy_target",
        type=float,
        default=0.90,
        help="Target clean accuracy.",
    )
    parser.add_argument(
        "--robust_lift_target",
        type=float,
        default=0.10,
        help="Required absolute robust accuracy lift over the baseline.",
    )
    parser.add_argument(
        "--throughput_lift_target",
        type=float,
        default=0.20,
        help="Required throughput lift ratio over the baseline.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/summary/core_metrics_summary.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    records = collect_records(args.search_roots)
    baseline_record = choose_baseline(records, args.baseline_name)
    summary = build_summary(
        records=records,
        baseline_record=baseline_record,
        accuracy_target=float(args.accuracy_target),
        robust_lift_target=float(args.robust_lift_target),
        throughput_lift_target=float(args.throughput_lift_target),
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Found {len(records)} result files")
    if baseline_record is None:
        print("Baseline reference: <not found>")
    else:
        print(
            "Baseline reference: "
            f"{baseline_record['name']} | robust={baseline_record['robust_accuracy']} | "
            f"throughput={baseline_record['throughput']}"
        )
    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
