#!/usr/bin/env python3
"""
Run batched CANDOR inference and export raw artifacts for later local scoring.
"""
import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Make direct execution from the repo root resolve `scripts.*` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

from scripts.evaluation.utils import build_answer_prompt, build_confidence_prompt, classify_response, is_valid_answer, parse_verbalized_confidence
from scripts.evaluation.wandb_utils import finish_wandb_run, init_wandb_run, log_wandb_artifact, log_wandb_metrics
from scripts.inference.closed_source import get_closed_source_model
from scripts.inference.open_source import get_open_source_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("candor_inference.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


FIELD_MAPPING = {
    "outdated": {"question": "question", "ground_truth": "new_value"},
    "longtail": {"question": "question", "ground_truth": "answer"},
    "anchor_induced": {
        "question": "grounded_question",
        "question_ungrounded": "ungrounded_question",
        "ground_truth": "ground_truth",
    },
    "near_true": {"question": "question", "ground_truth": "true_value"},
    "counterfactual": {"question": "question", "ground_truth": None},
    "non_existant": {"question": "question", "ground_truth": None},
    "head_tail_rarity": {"question": "question", "ground_truth": "answer"},
}


SUBSET_ALIASES = {
    "outdated": "outdated",
    "longtail": "longtail",
    "long_tailed": "longtail",
    "anchor_induced": "anchor_induced",
    "mixed-facts": "anchor_induced",
    "near_true": "near_true",
    "near-true": "near_true",
    "counterfactual": "counterfactual",
    "counterfactual-final": "counterfactual",
    "non_existant": "non_existant",
    "non-existant": "non_existant",
    "nonexistent": "non_existant",
    "non-existent": "non_existant",
    "non-existant-final": "non_existant",
    "head_tail_rarity": "head_tail_rarity",
    "head-tail-rarity": "head_tail_rarity",
    "head-tail-rarity-final": "head_tail_rarity",
}


def canonical_subset_name(subset_name: str) -> str:
    return SUBSET_ALIASES.get(subset_name, subset_name)


MODEL_NAME_ALIASES = {
    # Replacement requested: route prior Qwen3.5-9B runs to Bedrock Llama 3.3 70B.
    "Qwen3.5-9B": "meta.llama3-3-70b-instruct-v1:0",
}


BEDROCK_MODELS = {
    "meta.llama3-3-70b-instruct-v1:0",
}


def canonical_model_name(model_name: str) -> str:
    return MODEL_NAME_ALIASES.get(model_name, model_name)


def load_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def chunked(values: List[Dict], size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def get_question(row: Dict, subset_name: str, variant: str = "default") -> str:
    subset_name = canonical_subset_name(subset_name)
    mapping = FIELD_MAPPING[subset_name]
    if subset_name == "anchor_induced" and variant == "ungrounded":
        return row.get(mapping["question_ungrounded"], "")
    return row.get(mapping["question"], "")


def get_ground_truth(row: Dict, subset_name: str) -> str:
    subset_name = canonical_subset_name(subset_name)
    gt_key = FIELD_MAPPING[subset_name].get("ground_truth")
    if not gt_key:
        return ""
    return row.get(gt_key, "")


def load_model(model_name: str, model_type: str, device: str, dtype: str, quantization: Optional[str]):
    resolved_model_name = canonical_model_name(model_name)
    if resolved_model_name in BEDROCK_MODELS:
        if model_type == "open_source":
            logger.info("Model '%s' is a Bedrock model; routing through closed_source adapter.", resolved_model_name)
        return get_closed_source_model(resolved_model_name)

    if model_type == "open_source":
        return get_open_source_model(resolved_model_name, device=device, dtype=dtype, quantization=quantization)
    return get_closed_source_model(resolved_model_name)


def run_model_batch(model, prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
    # Run prompts sequentially to avoid batch-level complexity and OOMs.
    outputs: List[str] = []
    for prompt_index, prompt in enumerate(prompts, start=1):
        try:
            if hasattr(model, "infer"):
                outputs.append(model.infer(prompt, temperature=temperature, max_tokens=max_tokens))
            elif hasattr(model, "infer_batch"):
                # fallback to batch API with a single-item list
                outputs.append(model.infer_batch([prompt], temperature=temperature, max_tokens=max_tokens)[0])
            else:
                raise RuntimeError("Model has no infer or infer_batch method")
        except Exception:
            logger.exception("Prompt %s failed; continuing with an empty response", prompt_index)
            outputs.append("")
    return outputs


def majority_vote(response_types: List[str], samples: int) -> tuple[str, int, float, bool]:
    valid_types = [response_type for response_type in response_types if response_type]
    if not valid_types:
        return "", 0, 0.0, False

    counts = Counter(valid_types)
    majority_type, majority_count = counts.most_common(1)[0]
    consistency_score = round(majority_count / samples, 2)
    is_confident = consistency_score >= 0.8
    return majority_type, majority_count, consistency_score, is_confident


def average_confidence(confidences: List[Optional[int]]) -> Optional[float]:
    values = [confidence for confidence in confidences if confidence is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def representative_answer(response_type: str, run_records: List[Dict]) -> str:
    for run_record in run_records:
        if run_record.get("response_type") == response_type:
            return run_record.get("answer_text", "")
    return run_records[0].get("answer_text", "") if run_records else ""


def example_identity(example: Dict) -> tuple:
    """Build a stable key for matching exported rows back to source examples."""
    return (
        example.get("subset"),
        example.get("variant", "default"),
        example.get("row_index"),
        example.get("question", ""),
        example.get("ground_truth", ""),
    )


def load_existing_artifact(artifact_path: Path) -> tuple[Dict, set[tuple], int]:
    """Load prior exports so inference can resume from the last completed row."""
    if not artifact_path.exists() or artifact_path.stat().st_size == 0:
        return {}, set(), 0

    metadata: Dict = {}
    processed_keys: set[tuple] = set()
    processed_rows = 0

    with open(artifact_path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "metadata":
                metadata = row.get("metadata", {})
                continue

            processed_rows += 1
            processed_keys.add(example_identity(row))

    return metadata, processed_keys, processed_rows


def build_examples(rows: List[Dict], subset_name: str) -> List[Dict]:
    subset_name = canonical_subset_name(subset_name)
    examples = []
    for row_index, row in enumerate(rows):
        if subset_name == "anchor_induced":
            for variant in ["default", "ungrounded"]:
                question = get_question(row, subset_name, variant)
                if question:
                    examples.append(
                        {
                            "row_index": row_index,
                            "variant": variant,
                            "question": question,
                            "ground_truth": get_ground_truth(row, subset_name),
                            "row": row,
                        }
                    )
        else:
            question = get_question(row, subset_name)
            if question:
                examples.append(
                    {
                        "row_index": row_index,
                        "variant": "default",
                        "question": question,
                        "ground_truth": get_ground_truth(row, subset_name),
                        "row": row,
                    }
                )
    return examples


def build_batch_request_lines(examples: List[Dict], subset_name: str, samples: int) -> List[Dict]:
    """Build batch-ready lines that preserve the existing 5-run structure."""
    batch_lines: List[Dict] = []
    for example in examples:
        for run in range(1, samples + 1):
            batch_lines.append(
                {
                    "example_id": f"{subset_name}:{example['row_index']}:{example['variant']}",
                    "row_index": example["row_index"],
                    "subset": subset_name,
                    "question_variant": example["variant"],
                    "run": run,
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": build_answer_prompt(example["question"], subset_name, example["variant"]) }],
                            }
                        ],
                        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 50},
                    },
                }
            )
    return batch_lines


def export_inference(model_name: str, model_type: str, subset_name: str, subset_path: str, output_dir: str, batch_size: int, samples: int, temperature: float, max_tokens: int, device: str, dtype: str, quantization: Optional[str], limit: Optional[int], seed: int, model=None) -> Path:
    subset_name = canonical_subset_name(subset_name)
    rows = load_rows(subset_path)
    if limit:
        rows = rows[:limit]

    examples = build_examples(rows, subset_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifact_path = output_path / f"raw_{subset_name}_{model_name}.jsonl"
    existing_metadata, processed_keys, processed_rows = load_existing_artifact(artifact_path)
    if existing_metadata and existing_metadata.get("model") not in {None, model_name}:
        logger.warning(
            "Existing artifact %s was created for model %s; appending new rows anyway.",
            artifact_path,
            existing_metadata.get("model"),
        )
    if processed_rows:
        logger.info("Resuming %s from %s existing rows in %s", model_name, processed_rows, artifact_path)
        if processed_rows < len(examples):
            examples = examples[processed_rows:]
        else:
            logger.info(
                "Artifact %s already contains all %s examples for %s/%s",
                artifact_path,
                len(examples),
                subset_name,
                model_name,
            )
            examples = []
    wandb_run = init_wandb_run(
        run_name=f"infer-{model_name}-{subset_name}",
        job_type="inference-export",
        tags=[model_name, subset_name, model_type],
        config={
            "model": model_name,
            "model_type": model_type,
            "subset": subset_name,
            "subset_path": subset_path,
            "batch_size": batch_size,
            "samples_per_question": samples,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "device": device,
            "dtype": dtype,
            "quantization": quantization,
            "limit": limit,
            "seed": seed,
        },
    )
    metadata = {
        "model": model_name,
        "model_type": model_type,
        "subset": subset_name,
        "subset_path": subset_path,
        "generated_at": datetime.now().isoformat(),
        "seed": seed,
        "batch_size": batch_size,
        "samples_per_question": samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "device": device,
        "dtype": dtype,
        "quantization": quantization,
    }

    if model is None:
        logger.info(f"Loading {model_name} for batched export")
        model = load_model(model_name, model_type, device, dtype, quantization)
    if hasattr(model, "loaded") and not model.loaded:
        model.load()

    try:
        write_mode = "a" if artifact_path.exists() and artifact_path.stat().st_size > 0 else "w"
        written_examples = 0
        skipped_examples = 0
        with open(artifact_path, write_mode, encoding="utf-8") as handle:
            if write_mode == "w":
                # Write metadata header for a fresh export.
                handle.write(json.dumps({"type": "metadata", "metadata": metadata}, ensure_ascii=False) + "\n")

            # Process examples sequentially (no batching)
            for example in examples:
                if example_identity(example) in processed_keys:
                    logger.info(
                        "Skipping already exported example row_index=%s variant=%s for %s/%s",
                        example.get("row_index"),
                        example.get("variant", "default"),
                        subset_name,
                        model_name,
                    )
                    skipped_examples += 1
                    continue

                prompts = [build_answer_prompt(example["question"], subset_name, example["variant"]) for _ in range(samples)]
                answer_outputs = run_model_batch(model, prompts, temperature=temperature, max_tokens=max_tokens)

                run_records: List[Dict] = []
                response_types: List[str] = []
                refusal_count = 0
                malformed_count = 0
                deflection_count = 0

                for run_index, sample_answer in enumerate(answer_outputs):
                    response_type, flags = classify_response(sample_answer)
                    response_types.append(response_type)
                    refusal_count += int(flags.get("is_refusal", False))
                    malformed_count += int(flags.get("is_malformed", False))
                    deflection_count += int(flags.get("is_deflection", False))
                    run_records.append(
                        {
                            "run": run_index + 1,
                            "answer_text": sample_answer,
                            "stripped_answer": sample_answer.lower().strip(),
                            "response_type": response_type,
                            "is_refusal": flags.get("is_refusal", False),
                            "is_deflection": flags.get("is_deflection", False),
                            "is_malformed": flags.get("is_malformed", False),
                            "verbalized_confidence": None,
                            "verbalized_confidence_raw": "",
                        }
                    )

                majority_type, majority_count, consistency_score, is_confident = majority_vote(response_types, samples)
                model_answer = representative_answer(majority_type, run_records)
                is_valid = not any(r["is_malformed"] for r in run_records) and not any(r["is_refusal"] for r in run_records)

                # Build and run confidence prompts for non-empty answers
                confidence_prompts: List[str] = []
                for run_record in run_records:
                    if run_record["answer_text"].strip():
                        confidence_prompts.append(build_confidence_prompt(example["question"], run_record["answer_text"].strip()))

                confidence_outputs = run_model_batch(model, confidence_prompts, temperature=0.0, max_tokens=5) if confidence_prompts else []

                for idx, raw_conf in enumerate(confidence_outputs):
                    parsed = parse_verbalized_confidence(raw_conf)
                    run_records[idx]["verbalized_confidence_raw"] = raw_conf
                    run_records[idx]["verbalized_confidence"] = parsed

                per_run = run_records
                confidences: List[Optional[int]] = [r.get("verbalized_confidence") for r in per_run]
                raw_confidences: List[str] = [r.get("verbalized_confidence_raw", "") for r in per_run]

                row = {
                    "model": model_name,
                    "subset": subset_name,
                    "question_variant": example["variant"],
                    "question": example["question"],
                    "ground_truth": example["ground_truth"],
                    "per_run": per_run,
                    "majority_answer": model_answer,
                    "majority_response_type": majority_type,
                    "majority_count": majority_count,
                    "model_answer": model_answer,
                    "answer_samples": "|".join(record["stripped_answer"] for record in per_run),
                    "full_answer": "|".join(record["answer_text"] for record in per_run),
                    "stripped_answer": model_answer,
                    "consistency_score": consistency_score,
                    "confidence_label": "confident" if is_confident else "uncertain",
                    "is_confident": is_confident,
                    "refusal_count": refusal_count,
                    "refusal_rate": round(refusal_count / samples, 2),
                    "malformed_count": malformed_count,
                    "malformed_rate": round(malformed_count / samples, 2),
                    "deflection_count": deflection_count,
                    "deflection_rate": round(deflection_count / samples, 2),
                    "verbalized_conf_raw": "|".join(raw_confidences),
                    "verbalized_conf_avg": average_confidence([c for c in confidences if c is not None]),
                    "verbalized_conf": average_confidence([c for c in confidences if c is not None]),
                    "per_run_json": json.dumps(per_run, ensure_ascii=False),
                    "is_valid": is_valid,
                    "is_refusal": majority_type == "refusal",
                    "malformed": majority_type == "malformed" or not is_valid,
                }

                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                written_examples += 1

        if wandb_run is not None:
            log_wandb_metrics(
                wandb_run,
                {
                    "exported_examples": written_examples,
                    "skipped_examples": skipped_examples,
                    "samples_per_question": samples,
                    "batch_size": 1,
                },
                prefix="inference",
            )
            log_wandb_artifact(
                wandb_run,
                artifact_path,
                artifact_name=f"candor-raw-{subset_name}-{model_name}",
                artifact_type="raw-inference",
                metadata=metadata,
            )
            wandb_run.summary["raw_artifact"] = str(artifact_path)
            wandb_run.summary["subset"] = subset_name
    finally:
        if wandb_run is not None:
            finish_wandb_run(wandb_run)


def _default_subset_path(subset_name: str) -> str:
    subset_name = canonical_subset_name(subset_name)
    base = Path("data")
    mapping = {
        "outdated": base / "outdated_facts.csv",
        "longtail": base / "long_tailed.csv",
        "anchor_induced": base / "anchor_induced.csv",
        "near_true": base / "near_true.csv",
        "counterfactual": base / "counterfactual.csv",
        "non_existant": base / "non-existant.csv",
        "head_tail_rarity": base / "head-tail-rarity.csv",
    }
    path = mapping.get(subset_name)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="Export raw model inference for CANDOR datasets")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--model-type", choices=["open_source", "closed_source"], default="open_source")
    parser.add_argument("--subsets", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results/inference_results")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    logger.info(f"Arguments: models={args.models} subsets={args.subsets} device={args.device} dtype={args.dtype}")

    # Pre-validate subsets and models to provide early, visible errors
    for model_name in args.models:
        resolved_model_name = canonical_model_name(model_name)
        try:
            # instantiate model wrapper (do not call .load()) to surface unknown-model errors early
            try:
                model = load_model(resolved_model_name, args.model_type, args.device, args.dtype, args.quantization)
            except Exception as e:
                logger.error(f"Model resolution failed for '{model_name}' (resolved: '{resolved_model_name}'): {e}")
                continue

            logger.info(f"Loading model once for all subsets: {resolved_model_name}")
            model.load()

            try:
                for subset in args.subsets:
                    canonical_subset = canonical_subset_name(subset)
                    subset_path = _default_subset_path(canonical_subset)
                    if not Path(subset_path).exists():
                        logger.error(f"Subset file not found for '{subset}': {subset_path} - skipping")
                        continue

                    logger.info(f"Starting export: model={resolved_model_name} subset={canonical_subset} file={subset_path}")
                    try:
                        export_inference(
                            model_name=resolved_model_name,
                            model_type=args.model_type,
                            subset_name=canonical_subset,
                            subset_path=subset_path,
                            output_dir=args.output_dir,
                            batch_size=args.batch_size,
                            samples=args.samples,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            device=args.device,
                            dtype=args.dtype,
                            quantization=args.quantization,
                            limit=args.limit,
                            seed=args.seed,
                            model=model,
                        )
                    except Exception:
                        logger.exception(f"Failed exporting model={model_name} subset={subset}")
            finally:
                if hasattr(model, "unload"):
                    model.unload()
        except Exception:
            logger.exception(f"Unexpected error while preparing model={model_name}")


if __name__ == "__main__":
    main()