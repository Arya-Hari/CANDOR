#!/usr/bin/env python3
"""Run the CANDOR chain-of-thought boundary calibration experiment end to end."""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in minimal environments
    def load_dotenv():
        return None

# Make direct execution from the repo root resolve `scripts.*` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

load_dotenv()

from scripts.evaluation.utils import (  # noqa: E402
    MetricsComputer,
    build_answer_prompt,
    build_confidence_prompt,
    classify_response,
    determine_confusion_cell,
    extract_final_answer,
    is_valid_answer,
    normalize_text,
    parse_verbalized_confidence,
)
from scripts.inference.closed_source import get_closed_source_model  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("candor_EBP_experiment.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


TASK_FILES = {
    "longtail": "data/long_tailed.csv",
    "near_true": "data/near_true.csv",
    "anchor_induced": "data/anchor_induced.csv",
}

TASK_VARIANTS = {
    "longtail": "default",
    "near_true": "default",
    "anchor_induced": ["grounded", "ungrounded"],
}

DEFAULT_MODELS = ["Llama-3.3-70B", "GPT-4.1", "Gemini-2.5-pro", "DeepSeek-v3.2"]
DEFAULT_CONDITIONS = ["baseline", "zero_shot_EBP", "boundary_aware_EBP"]
RAW_SUBSET_STEMS = {
    "longtail": ["longtail", "long_tail"],
    "near_true": ["near_true"],
    "anchor_induced": ["anchor_induced", "mixed_fact"],
}
SAMPLE_FILE_STEMS = {
    "longtail": "long_tail",
    "near_true": "near_true",
    "anchor_induced": "mixed_fact",
}


def safe_slug(value: str) -> str:
    slug = []
    for char in value.lower():
        if char.isalnum():
            slug.append(char)
        elif slug and slug[-1] != "-":
            slug.append("-")
    return "".join(slug).strip("-") or "item"


def read_rows(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl_rows(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("type") == "metadata":
                continue
            rows.append(row)
    return rows


def get_question(row: Dict, task_name: str, variant: str = "default") -> str:
    if task_name == "anchor_induced" and variant == "ungrounded":
        return row.get("ungrounded_question", row.get("question", ""))
    if task_name == "anchor_induced":
        return row.get("grounded_question", row.get("question", ""))
    return row.get("question", "")


def get_ground_truth(row: Dict, task_name: str) -> str:
    if task_name == "longtail":
        return row.get("answer", "")
    if task_name == "near_true":
        return row.get("true_value", "")
    if task_name == "anchor_induced":
        return row.get("ground_truth", "")
    return ""


def sample_csv_path_for_task(task_name: str, sample_size: int, sample_csv_dir: Path) -> Path:
    return sample_csv_dir / f"sample_{SAMPLE_FILE_STEMS[task_name]}_{sample_size}.csv"


def task_question_variants(task_name: str) -> List[str]:
    variants = TASK_VARIANTS[task_name]
    if isinstance(variants, list):
        return variants
    return [variants]


def baseline_model_name_variants(model_name: str) -> List[str]:
    variants = [model_name]
    if ":" in model_name:
        variants.append(model_name.replace(":", "_"))
    if model_name.startswith("us."):
        variants.append(model_name.removeprefix("us."))
        if ":" in model_name:
            variants.append(model_name.removeprefix("us.").replace(":", "_"))
    # Preserve order while removing duplicates.
    seen = set()
    unique_variants = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            unique_variants.append(variant)
    return unique_variants


def find_baseline_file(task_name: str, model_name: str, baseline_dir: Path) -> Path:
    stems = RAW_SUBSET_STEMS[task_name]
    candidates = []
    for stem in stems:
        for model_variant in baseline_model_name_variants(model_name):
            candidates.extend(sorted(baseline_dir.glob(f"**/raw_{stem}_{model_variant}.jsonl")))
            candidates.extend(sorted(baseline_dir.glob(f"raw_{stem}_{model_variant}.jsonl")))
            candidates.extend(sorted(baseline_dir.glob(f"**/raw_{stem}_{model_variant}_*.jsonl")))
            candidates.extend(sorted(baseline_dir.glob(f"raw_{stem}_{model_variant}_*.jsonl")))
    if not candidates:
        raise FileNotFoundError(f"Could not find baseline raw JSONL for task={task_name}, model={model_name} under {baseline_dir}")
    for candidate in candidates:
        path_text = str(candidate)
        if any(stem in path_text for stem in stems):
            return candidate
    return candidates[0]


def raw_jsonl_output_path(task_name: str, model_name: str, condition: str) -> Path:
    stem = RAW_SUBSET_STEMS[task_name][0]
    model_file_slug = model_name.replace(":", "_")
    condition_slug = safe_slug(condition)
    return Path("results/inference_results") / SAMPLE_FILE_STEMS[task_name] / f"raw_{stem}_{model_file_slug}_{condition_slug}.jsonl"


def build_reused_baseline_record(
    model_name: str,
    task_name: str,
    row_index: int,
    row: Dict,
    variant: str,
    raw_record: Dict,
) -> Dict:
    question = get_question(row, task_name, variant=variant)
    ground_truth = get_ground_truth(row, task_name)
    majority_answer = raw_record.get("majority_answer", raw_record.get("model_answer", ""))
    consistency_score = float(raw_record.get("consistency_score", 0.0) or 0.0)
    verbalized_conf = raw_record.get("verbalized_conf_avg", raw_record.get("verbalized_conf"))
    if verbalized_conf in {None, ""}:
        verbalized_conf = None
    elif isinstance(verbalized_conf, str):
        try:
            verbalized_conf = float(verbalized_conf)
        except ValueError:
            verbalized_conf = None

    is_confident = bool(raw_record.get("is_confident", consistency_score >= 0.8 and is_valid_answer(majority_answer)))
    is_valid = bool(raw_record.get("is_valid", is_valid_answer(majority_answer)))
    is_refusal = bool(raw_record.get("is_refusal", False))
    is_correct = normalize_text(majority_answer) == normalize_text(ground_truth)
    confusion_cell = determine_confusion_cell(is_confident, is_correct)
    vcd = None
    if verbalized_conf is not None:
        vcd = round(abs(float(verbalized_conf) - (consistency_score * 100.0)), 2)

    return {
        "model": model_name,
        "task": task_name,
        "prompt_condition": "baseline",
        "row_index": row_index,
        "question_variant": variant if task_name == "anchor_induced" else None,
        "question": question,
        "ground_truth": ground_truth,
        "majority_answer": majority_answer,
        "model_answer": majority_answer,
        "full_answer": raw_record.get("full_answer", ""),
        "final_answer_samples": raw_record.get("answer_samples", raw_record.get("final_answer_samples", "")),
        "stripped_answer": raw_record.get("stripped_answer", majority_answer),
        "consistency_score": consistency_score,
        "confidence_label": raw_record.get("confidence_label", "confident" if is_confident else "uncertain"),
        "is_confident": is_confident,
        "verbalized_conf": verbalized_conf,
        "verbalized_conf_avg": verbalized_conf,
        "vcd": vcd,
        "is_correct": is_correct,
        "is_valid": is_valid,
        "is_refusal": is_refusal,
        "malformed": bool(raw_record.get("malformed", not is_valid)),
        "confusion_cell": confusion_cell,
        "per_run_json": raw_record.get("per_run_json", ""),
    }


def get_numeric(row: Dict, *keys: str) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def quantile_bucket(value: Optional[float], sorted_values: List[float], n_buckets: int = 5) -> str:
    if value is None or not sorted_values:
        return "bucket-unknown"
    rank = bisect_right(sorted_values, value)
    percentile = rank / max(len(sorted_values), 1)
    bucket = min(n_buckets - 1, int(percentile * n_buckets))
    return f"bucket-{bucket}"


def anchor_induced_gap(row: Dict) -> Optional[float]:
    head = get_numeric(row, "head_sitelinks")
    tail = get_numeric(row, "tail_sitelinks")
    if head is None or tail is None:
        return None
    return math.log1p(head) - math.log1p(tail)


def build_group_key(task_name: str, row: Dict, row_index: int, numeric_reference: Dict[str, List[float]]) -> Tuple:
    if task_name == "longtail":
        page_views = get_numeric(row, "page_views")
        return (
            row.get("relation", "unknown"),
            quantile_bucket(page_views, numeric_reference.get("page_views", []), n_buckets=5),
        )
    if task_name == "near_true":
        return (
            row.get("perturbation_type", "unknown"),
            row.get("fact_type", "unknown"),
        )
    if task_name == "anchor_induced":
        gap = anchor_induced_gap(row)
        return (
            row.get("domain", "unknown"),
            quantile_bucket(gap, numeric_reference.get("mixed_gap", []), n_buckets=5),
        )
    return (task_name, row_index)


def stratified_sample(rows: List[Dict], task_name: str, sample_size: int, seed: int) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
    if sample_size >= len(rows):
        return list(rows), {}

    numeric_reference = {}
    if task_name == "longtail":
        numeric_reference["page_views"] = sorted(
            value for value in (get_numeric(row, "page_views") for row in rows) if value is not None
        )
    elif task_name == "anchor_induced":
        numeric_reference["mixed_gap"] = sorted(
            value for value in (anchor_induced_gap(row) for row in rows) if value is not None
        )

    grouped: Dict[Tuple, List[Tuple[int, Dict]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        key = build_group_key(task_name, row, row_index, numeric_reference)
        grouped[key].append((row_index, row))

    total_rows = len(rows)
    desired = {key: sample_size * len(items) / total_rows for key, items in grouped.items()}
    selected_counts = {}
    remainders = []
    for key, items in grouped.items():
        target = desired[key]
        base = min(len(items), math.floor(target))
        selected_counts[key] = base
        remainders.append((target - base, len(items) - base, key))

    assigned = sum(selected_counts.values())
    remaining = sample_size - assigned
    rng = random.Random(seed)

    if remaining > 0:
        remainders.sort(key=lambda item: (item[0], item[1], str(item[2])), reverse=True)
        index = 0
        while remaining > 0 and remainders:
            _, capacity, key = remainders[index % len(remainders)]
            if selected_counts[key] < len(grouped[key]):
                selected_counts[key] += 1
                remaining -= 1
            index += 1
            if index > len(remainders) * 5 and remaining > 0:
                # Fall back to any group with spare capacity if rounding left us with a hard-to-fill tail.
                for _, capacity, fallback_key in remainders:
                    if selected_counts[fallback_key] < len(grouped[fallback_key]):
                        selected_counts[fallback_key] += 1
                        remaining -= 1
                        break
                else:
                    break

    sampled_rows: List[Tuple[int, Dict]] = []
    allocation_manifest: Dict[str, Dict[str, int]] = {}
    for key, items in grouped.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        take = selected_counts.get(key, 0)
        allocation_manifest[str(key)] = {"available": len(items), "selected": take}
        sampled_rows.extend(shuffled[:take])

    sampled_rows.sort(key=lambda pair: pair[0])
    return [row for _, row in sampled_rows], allocation_manifest


def average_optional(values: List[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 2)


def most_common_answer(answers: List[str]) -> Tuple[str, int]:
    counts = Counter(answers)
    if not counts:
        return "", 0
    answer, count = counts.most_common(1)[0]
    return answer, count


def build_run_record(run_number: int, raw_answer: str, final_answer: str, confidence_raw: str, confidence_value: Optional[int]) -> Dict:
    response_type, flags = classify_response(final_answer)
    return {
        "run": run_number,
        "raw_answer_text": raw_answer,
        "final_answer_text": final_answer,
        "answer_text": final_answer,
        "stripped_answer": final_answer.strip(),
        "response_type": response_type,
        "is_refusal": flags.get("is_refusal", False),
        "is_deflection": flags.get("is_deflection", False),
        "is_malformed": flags.get("is_malformed", False),
        "verbalized_confidence": confidence_value,
        "verbalized_confidence_raw": confidence_raw,
    }


def compute_row_metrics(rows: List[Dict]) -> Dict:
    metrics = MetricsComputer.compute_bcs(rows)
    total = metrics.get("n", 0) or 0
    accuracy = round((metrics.get("confident_correct", 0) + metrics.get("uncertain_correct", 0)) / total, 4) if total else 0.0
    cw_rate = round(metrics.get("confident_wrong", 0) / total, 4) if total else 0.0
    uw_rate = round(metrics.get("uncertain_wrong", 0) / total, 4) if total else 0.0
    vcd_values = [row.get("vcd") for row in rows if row.get("vcd") is not None]
    vcd = round(sum(vcd_values) / len(vcd_values), 4) if vcd_values else None
    metrics.update(
        {
            "accuracy": accuracy,
            "cw_rate": cw_rate,
            "uw_rate": uw_rate,
            "vcd": vcd,
        }
    )
    return metrics


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_delta(baseline_rows: List[Dict], comparison_rows: List[Dict], seed: int, samples: int = 1000) -> Dict:
    if not baseline_rows or not comparison_rows:
        return {"bcs": None, "cw_rate": None, "uw_rate": None, "vcd": None}

    if len(baseline_rows) != len(comparison_rows):
        raise ValueError("Bootstrap comparison requires aligned row counts")

    rng = random.Random(seed)
    n = len(baseline_rows)
    indices = list(range(n))
    deltas: Dict[str, List[float]] = {"bcs": [], "cw_rate": [], "uw_rate": [], "vcd": []}

    for _ in range(samples):
        draw = [rng.choice(indices) for _ in indices]
        baseline_sample = [baseline_rows[i] for i in draw]
        comparison_sample = [comparison_rows[i] for i in draw]
        baseline_metrics = compute_row_metrics(baseline_sample)
        comparison_metrics = compute_row_metrics(comparison_sample)
        for key in deltas:
            base_value = baseline_metrics.get(key)
            comp_value = comparison_metrics.get(key)
            if base_value is None or comp_value is None:
                continue
            deltas[key].append(comp_value - base_value)

    summary = {}
    for key, values in deltas.items():
        if not values:
            summary[key] = {"delta": None, "ci95": [None, None]}
            continue
        mean_delta = round(sum(values) / len(values), 4)
        summary[key] = {
            "delta": mean_delta,
            "ci95": [round(percentile(values, 0.025), 4), round(percentile(values, 0.975), 4)],
        }
    return summary


def validate_environment(models: List[str]) -> None:
    requested = set(models)
    needs_openai = any(model in requested for model in {"GPT-4.1"})
    needs_gemini = any(model in requested for model in {"Gemini-2.5-pro"})
    needs_deepseek = any(model in requested for model in {"DeepSeek-v3.2", "DeepSeek-Bedrock"})
    needs_bedrock = any(model in requested for model in {"Llama-3.3-70B"})

    if needs_openai and not (os.getenv("OPENAI_API_KEY") or (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))):
        logger.warning("GPT-4.1 requires OPENAI_API_KEY or AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT")
    if needs_gemini and not (os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT_ID") or os.getenv("PROJECT_ID") or os.getenv("GCLOUD_PROJECT")):
        logger.warning("Gemini-2.5-pro requires Google Cloud ADC/project configuration")
    if needs_deepseek and not (
        (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    ):
        logger.warning("DeepSeek Bedrock requires AWS credentials or AWS_BEARER_TOKEN_BEDROCK")
    if needs_bedrock and not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")) and not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        logger.warning("Llama-3.3-70B requires AWS Bedrock credentials or AWS_BEARER_TOKEN_BEDROCK")


def evaluate_question(
    model,
    model_name: str,
    task_name: str,
    row: Dict,
    row_index: int,
    variant: str,
    condition: str,
    temperature: float,
    samples_per_question: int,
    confidence_temperature: float,
    max_tokens: int,
) -> Dict:
    question = get_question(row, task_name, variant=variant)
    ground_truth = get_ground_truth(row, task_name)
    if not question or not ground_truth:
        raise ValueError(f"Missing question or ground truth for task={task_name}, row_index={row_index}")

    run_records: List[Dict] = []
    raw_answers: List[str] = []
    final_answers: List[str] = []
    normalized_answers: List[str] = []
    confidences: List[Optional[int]] = []

    for run_number in range(1, samples_per_question + 1):
        prompt = build_answer_prompt(question, subset_name=task_name, variant=variant, prompt_condition=condition)
        raw_answer = model.infer(prompt, temperature=temperature, max_tokens=max_tokens)
        final_answer = extract_final_answer(raw_answer)
        confidence_raw = ""
        confidence_value: Optional[int] = None

        if is_valid_answer(final_answer):
            confidence_prompt = build_confidence_prompt(question, final_answer)
            confidence_raw = model.infer(confidence_prompt, temperature=confidence_temperature, max_tokens=8)
            confidence_value = parse_verbalized_confidence(confidence_raw)

        run_record = build_run_record(run_number, raw_answer, final_answer, confidence_raw, confidence_value)
        run_records.append(run_record)
        raw_answers.append(raw_answer)
        final_answers.append(final_answer)
        normalized_answers.append(normalize_text(final_answer))
        confidences.append(confidence_value)

    majority_norm, majority_count = most_common_answer(normalized_answers)
    majority_index = next((idx for idx, answer in enumerate(normalized_answers) if answer == majority_norm), 0)
    majority_answer = final_answers[majority_index] if final_answers else ""

    consistency_score = round(majority_count / samples_per_question, 2) if samples_per_question else 0.0
    confidence_label = "confident" if consistency_score >= 0.8 else "uncertain"
    verbalized_conf_avg = average_optional([float(conf) if conf is not None else None for conf in confidences])
    is_valid = is_valid_answer(majority_answer)
    is_refusal = any(record["is_refusal"] for record in run_records)
    is_correct = normalize_text(majority_answer) == normalize_text(ground_truth)
    is_confident = confidence_label == "confident" and is_valid
    confusion_cell = determine_confusion_cell(is_confident, is_correct)
    vcd = None
    if verbalized_conf_avg is not None:
        vcd = round(abs(verbalized_conf_avg - (consistency_score * 100.0)), 2)

    result = {
        "model": model_name,
        "task": task_name,
        "prompt_condition": condition,
        "row_index": row_index,
        "question_variant": variant if task_name == "anchor_induced" else None,
        "question": question,
        "ground_truth": ground_truth,
        "majority_answer": majority_answer,
        "model_answer": majority_answer,
        "full_answer": "|".join(raw_answers),
        "final_answer_samples": "|".join(final_answers),
        "stripped_answer": majority_answer,
        "consistency_score": consistency_score,
        "confidence_label": confidence_label,
        "is_confident": is_confident,
        "verbalized_conf": verbalized_conf_avg,
        "verbalized_conf_avg": verbalized_conf_avg,
        "vcd": vcd,
        "is_correct": is_correct,
        "is_valid": is_valid,
        "is_refusal": is_refusal,
        "malformed": not is_valid,
        "confusion_cell": confusion_cell,
        "per_run_json": json.dumps(run_records, ensure_ascii=False),
    }

    result.update(compute_row_metrics([result]))
    return result


def write_raw_jsonl(metadata: Dict, rows: List[Dict], output_path: Path) -> None:
    """Write inference_results-style raw JSONL: a metadata line then one JSON per record.

    The function normalizes keys so that records include `subset` and `per_run`.
    """
    if not rows:
        logger.warning("No rows to save to %s", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write metadata header
    with open(output_path, "w", encoding="utf-8") as handle:
        header = {"type": "metadata", "metadata": metadata}
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for row in rows:
            out = dict(row)
            # normalize `task` -> `subset` if present
            if out.get("task") and not out.get("subset"):
                out["subset"] = out.pop("task")
            # ensure per_run is a list
            if "per_run" not in out:
                per_run_json = out.get("per_run_json") or out.get("per_run_json_string") or out.get("per_runs")
                if isinstance(per_run_json, str) and per_run_json:
                    try:
                        out["per_run"] = json.loads(per_run_json)
                    except Exception:
                        out["per_run"] = []
                else:
                    out["per_run"] = out.get("per_run") or []
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def run_experiment(args) -> Dict:
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "baseline" not in args.conditions:
        raise ValueError("The EBP experiment requires the baseline condition so deltas can be computed")

    summary: Dict = {
        "metadata": {
            "sample_size": args.sample_size,
            "samples_per_question": args.samples_per_question,
            "temperature": args.temperature,
            "confidence_temperature": args.confidence_temperature,
            "seed": args.seed,
            "tasks": args.tasks,
            "models": args.models,
            "conditions": args.conditions,
        }
    }

    for task_name in args.tasks:
        task_path = Path(TASK_FILES[task_name])
        if args.sample_csv_dir:
            sample_path = sample_csv_path_for_task(task_name, args.sample_size, Path(args.sample_csv_dir))
            if not sample_path.exists():
                raise FileNotFoundError(f"Sample CSV not found: {sample_path}")
            sampled_rows = read_rows(sample_path)
            allocation_manifest = {"source": str(sample_path), "selected": len(sampled_rows)}
            task_source_path = sample_path
        else:
            task_rows = read_rows(task_path)
            sampled_rows, allocation_manifest = stratified_sample(task_rows, task_name, args.sample_size, args.seed)
            task_source_path = task_path

        task_summary = {
            "source_path": str(task_source_path),
            "sample_size": len(sampled_rows),
            "allocation_manifest": allocation_manifest,
            "models": {},
        }

        logger.info("Loaded %s rows for task %s from %s", len(sampled_rows), task_name, task_source_path)

        for model_name in args.models:
            logger.info("Loading model %s", model_name)
            model = get_closed_source_model(model_name)
            model.load()

            try:
                model_summary = {
                    "conditions": {},
                    "deltas": {},
                }
                condition_rows: Dict[str, List[Dict]] = {}
                baseline_rows: Optional[List[Dict]] = None

                if args.reuse_baseline:
                    baseline_path = find_baseline_file(task_name, model_name, Path(args.baseline_input_dir))
                    raw_records = load_jsonl_rows(baseline_path)
                    raw_by_key = {}
                    for record in raw_records:
                        record_variant = record.get("question_variant", "default")
                        if task_name == "anchor_induced" and record_variant in {"", "default", None}:
                            record_variant = "grounded"
                        record_question = get_question(record, task_name, variant=record_variant)
                        if record_question:
                            raw_by_key[(record_variant, record_question)] = record

                    baseline_rows = []
                    missing_questions = []
                    for row_index, row in enumerate(sampled_rows):
                        for variant in task_question_variants(task_name):
                            question = get_question(row, task_name, variant=variant)
                            raw_record = raw_by_key.get((variant, question))
                            if raw_record is None:
                                missing_questions.append(f"{variant}: {question}")
                                continue
                            baseline_rows.append(
                                build_reused_baseline_record(model_name, task_name, row_index, row, variant, raw_record)
                            )
                    if missing_questions:
                        raise ValueError(
                            f"Baseline reuse is missing {len(missing_questions)} sampled questions for {task_name}/{model_name}: {missing_questions[:5]}"
                        )
                    condition_rows["baseline"] = baseline_rows
                    model_summary["conditions"]["baseline"] = compute_row_metrics(baseline_rows)
                    # Write inference_results-style raw JSONL for judge ingestion.
                    try:
                        inference_path = raw_jsonl_output_path(task_name, model_name, "baseline")
                        subset_stem = RAW_SUBSET_STEMS[task_name][0]
                        metadata = {
                            "model": model_name,
                            "model_type": "closed_source",
                            "subset": subset_stem,
                            "subset_path": TASK_FILES[task_name],
                            "generated_at": datetime.utcnow().isoformat(),
                            "seed": args.seed,
                            "batch_size": 1,
                            "samples_per_question": args.samples_per_question,
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens,
                            "device": "auto",
                            "dtype": "float16",
                            "quantization": None,
                        }
                        write_raw_jsonl(metadata, baseline_rows, inference_path)
                        logger.info("Wrote inference raw JSONL to %s", inference_path)
                    except Exception:
                        logger.exception("Failed to write inference_results JSONL for baseline")

                for condition in args.conditions:
                    if condition == "baseline" and args.reuse_baseline:
                        continue
                    logger.info("Evaluating %s on %s with condition=%s", model_name, task_name, condition)
                    evaluated_rows = []
                    for row_index, row in enumerate(sampled_rows):
                        for variant in task_question_variants(task_name):
                            evaluated_rows.append(
                                evaluate_question(
                                    model=model,
                                    model_name=model_name,
                                    task_name=task_name,
                                    row=row,
                                    row_index=row_index,
                                    variant=variant,
                                    condition=condition,
                                    temperature=args.temperature,
                                    samples_per_question=args.samples_per_question,
                                    confidence_temperature=args.confidence_temperature,
                                    max_tokens=args.max_tokens,
                                )
                            )

                    condition_rows[condition] = evaluated_rows
                    model_summary["conditions"][condition] = compute_row_metrics(evaluated_rows)

                    # Write inference_results-style raw JSONL for this condition.
                    try:
                        inference_path = raw_jsonl_output_path(task_name, model_name, condition)
                        subset_stem = RAW_SUBSET_STEMS[task_name][0]
                        metadata = {
                            "model": model_name,
                            "model_type": "closed_source",
                            "subset": subset_stem,
                            "subset_path": TASK_FILES[task_name],
                            "generated_at": datetime.utcnow().isoformat(),
                            "seed": args.seed,
                            "batch_size": 1,
                            "samples_per_question": args.samples_per_question,
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens,
                            "device": "auto",
                            "dtype": "float16",
                            "quantization": None,
                        }
                        write_raw_jsonl(metadata, evaluated_rows, inference_path)
                        logger.info("Wrote inference raw JSONL to %s", inference_path)
                    except Exception:
                        logger.exception("Failed to write inference_results JSONL for condition %s", condition)

                if baseline_rows is None:
                    baseline_rows = condition_rows["baseline"]
                for condition in args.conditions:
                    if condition == "baseline":
                        continue
                    model_summary["deltas"][condition] = bootstrap_delta(
                        baseline_rows,
                        condition_rows[condition],
                        seed=args.seed,
                        samples=args.bootstrap_samples,
                    )

                task_summary["models"][model_name] = model_summary
            finally:
                model.unload()

        summary[task_name] = task_summary

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.info("Saved summary to %s", summary_path)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CANDOR EBP boundary-calibration experiment")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Closed-source models to evaluate",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["longtail", "near_true", "anchor_induced"],
        choices=["longtail", "near_true", "anchor_induced"],
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        choices=DEFAULT_CONDITIONS,
        help="Prompt conditions to evaluate",
    )
    parser.add_argument("--sample-size", type=int, default=200, help="Stratified questions per task")
    parser.add_argument("--samples-per-question", type=int, default=5, help="Repeated answer samples per question")
    parser.add_argument("--temperature", type=float, default=0.0, help="Answer sampling temperature")
    parser.add_argument("--confidence-temperature", type=float, default=0.0, help="Confidence follow-up temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for the answer generation step")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap resamples for delta CIs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results-dir", default="results/EBP_experiment", help="Directory for experiment outputs")
    parser.add_argument(
        "--reuse-baseline",
        action="store_true",
        help="Reuse baseline from existing raw JSONL outputs instead of rerunning baseline inference",
    )
    parser.add_argument(
        "--baseline-input-dir",
        default="results/inference_results",
        help="Directory that contains raw baseline JSONL files for reuse",
    )
    parser.add_argument(
        "--sample-csv-dir",
        default="",
        help="Directory containing fixed sample CSVs such as sample_long_tail_200.csv",
    )
    parser.add_argument(
        "--mixed-facts-variant",
        choices=["grounded", "ungrounded"],
        default=None,
        help="Deprecated: anchor_induced now always evaluates both grounded and ungrounded questions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    validate_environment(args.models)
    run_experiment(args)


if __name__ == "__main__":
    main()