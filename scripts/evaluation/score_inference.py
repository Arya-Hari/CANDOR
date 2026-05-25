#!/usr/bin/env python3
"""
Score exported CANDOR inference artifacts locally.
"""
import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Make direct execution from the repo root resolve `scripts.*` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.utils import MetricsComputer, determine_confusion_cell, normalize_text
from scripts.evaluation.wandb_utils import finish_wandb_run, init_wandb_run, log_wandb_artifact, log_wandb_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("candor_scoring.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> Dict:
    rows = []
    metadata = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "metadata":
                metadata = row.get("metadata", {})
            else:
                rows.append(row)
    return {"metadata": metadata, "rows": rows}


def score_rows(rows: List[Dict]) -> List[Dict]:
    scored = []
    for row in rows:
        if row.get("majority_answer") and not row.get("model_answer"):
            row["model_answer"] = row["majority_answer"]

        if row.get("per_run") and isinstance(row.get("per_run"), list):
            per_run = row["per_run"]
            confidences = [run.get("verbalized_confidence") for run in per_run if run.get("verbalized_confidence") is not None]
            if confidences and row.get("verbalized_conf_avg") is None:
                row["verbalized_conf_avg"] = round(sum(confidences) / len(confidences), 2)
                row["verbalized_conf"] = row["verbalized_conf_avg"]
            row["per_run_json"] = json.dumps(per_run, ensure_ascii=False)

        if row.get("confidence_label") not in {"confident", "uncertain"}:
            row["confidence_label"] = "confident" if float(row.get("consistency_score", 0.0)) >= 0.8 else "uncertain"

        is_correct = normalize_text(row.get("model_answer", "")) == normalize_text(row.get("ground_truth", ""))
        is_confident = row.get("confidence_label") == "confident" or bool(row.get("is_confident", False))
        is_confident = is_confident and bool(row.get("is_valid", False))
        row["is_correct"] = is_correct
        row["confusion_cell"] = determine_confusion_cell(is_confident, is_correct)
        scored.append(row)
    return scored


def write_csv(rows: List[Dict], output_path: Path) -> None:
    if not rows:
        logger.warning(f"No rows to write to {output_path}")
        return
    serializable_rows = []
    for row in rows:
        serialized = dict(row)
        if isinstance(serialized.get("per_run"), list):
            serialized["per_run"] = json.dumps(serialized["per_run"], ensure_ascii=False)
        serializable_rows.append(serialized)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable_rows[0].keys()))
        writer.writeheader()
        writer.writerows(serializable_rows)


def build_summary(rows: List[Dict], metadata: Dict) -> Dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["subset"])].append(row)

    summary = {"metadata": metadata}
    for (model_name, subset_name), subset_rows in grouped.items():
        summary.setdefault(model_name, {})[subset_name] = MetricsComputer.compute_bcs(subset_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw CANDOR inference artifacts locally")
    parser.add_argument("--input", required=True, help="Path to raw JSONL artifacts from run_inference.py")
    parser.add_argument("--results-dir", default="results/scored")
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = init_wandb_run(
        run_name=f"score-{input_path.stem}",
        job_type="scoring",
        tags=["scoring"],
        config={
            "input": str(input_path),
            "results_dir": args.results_dir,
            "output_name": args.output_name,
        },
    )

    try:
        artifact = load_jsonl(str(input_path))
        rows = artifact["rows"]
        scored_rows = score_rows(rows)
        summary = build_summary(scored_rows, artifact["metadata"])

        output_name = args.output_name or f"scored_{input_path.stem}.csv"
        scored_csv_path = output_dir / output_name
        summary_path = output_dir / "summary.json"

        write_csv(scored_rows, scored_csv_path)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        if wandb_run is not None:
            log_wandb_metrics(wandb_run, summary, prefix="scoring")
            log_wandb_artifact(
                wandb_run,
                scored_csv_path,
                artifact_name=f"candor-scored-{input_path.stem}",
                artifact_type="scored-results",
                metadata={"input": str(input_path)},
            )
            log_wandb_artifact(
                wandb_run,
                summary_path,
                artifact_name=f"candor-summary-{input_path.stem}",
                artifact_type="summary",
                metadata={"input": str(input_path)},
            )
            wandb_run.summary["scored_csv"] = str(scored_csv_path)
            wandb_run.summary["summary_json"] = str(summary_path)

        logger.info(f"Saved scored CSV to {scored_csv_path}")
        logger.info(f"Saved summary JSON to {summary_path}")
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()