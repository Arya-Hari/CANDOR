"""
Main CANDOR evaluation pipeline.
Orchestrates inference across models and subsets, records results and metrics.
"""
import csv
import json
import os
import logging
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import traceback

import numpy as np
import torch

from scripts.inference.open_source import get_open_source_model
from scripts.inference.closed_source import get_closed_source_model
from scripts.evaluation.utils import (
    build_answer_prompt,
    build_confidence_prompt,
    normalize_text,
    is_valid_answer,
    parse_verbalized_confidence,
    determine_confusion_cell,
    MetricsComputer,
)
from scripts.evaluation.wandb_utils import log_wandb_artifact, log_wandb_metrics

logger = logging.getLogger(__name__)

# Field mapping for different datasets
FIELD_MAPPING = {
    "outdated": {
        "question": "question",
        "ground_truth": "new_value",  # For outdated, new_value is the ground truth
    },
    "longtail": {
        "question": "question",
        "ground_truth": "answer",
    },
    "anchor_induced": {
        "question": "grounded_question",  # Use grounded by default
        "question_ungrounded": "ungrounded_question",  # Also track ungrounded for anchor bias study
        "ground_truth": "ground_truth",
    },
    "near_true": {
        "question": "question",
        "ground_truth": "true_value",
    },
}


class EvaluationPipeline:
    """Main evaluation pipeline for CANDOR."""

    def __init__(self, results_dir: str = "results", checkpoint_dir: str = "results/checkpoints"):
        self.results_dir = Path(results_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.seed = 42  # Default seed for reproducibility
        self.timestamp = datetime.now().isoformat()
        self.evaluation_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def set_seed(self, seed: int = 42) -> None:
        """Set random seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")

    def load_subset(self, subset_name: str, subset_path: str) -> List[Dict]:
        """Load a data subset from CSV."""
        logger.info(f"Loading subset '{subset_name}' from {subset_path}")
        
        if not os.path.exists(subset_path):
            raise FileNotFoundError(f"Subset file not found: {subset_path}")
        
        rows = []
        with open(subset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        logger.info(f"Loaded {len(rows)} rows from {subset_name}")
        
        # Validate required fields exist
        if rows:
            self._validate_subset_fields(subset_name, rows[0])
        
        # Profile the dataset
        self._profile_subset(subset_name, rows)
        
        return rows

    def _validate_subset_fields(self, subset_name: str, sample_row: Dict) -> None:
        """Validate that required fields exist in dataset."""
        if subset_name not in FIELD_MAPPING:
            logger.warning(f"No field mapping defined for {subset_name}, using generic extraction")
            return
        
        mapping = FIELD_MAPPING[subset_name]
        missing = []
        for field_key, field_name in mapping.items():
            if field_name not in sample_row:
                missing.append(f"{field_key}→{field_name}")
        
        if missing:
            raise ValueError(
                f"Dataset {subset_name} missing required fields: {missing}\n"
                f"Available fields: {list(sample_row.keys())}"
            )
        
        logger.info(f"✓ Fields validated for {subset_name}: {list(mapping.values())}")

    def _profile_subset(self, subset_name: str, rows: List[Dict]) -> None:
        """Profile the dataset: check for missing values, data types, samples."""
        logger.info(f"\n  Data Profiling for {subset_name}:")
        logger.info(f"  - Total rows: {len(rows)}")
        
        # Check for empty values
        question_field = FIELD_MAPPING.get(subset_name, {}).get("question", "question")
        answer_field = FIELD_MAPPING.get(subset_name, {}).get("ground_truth", "ground_truth")
        
        empty_questions = sum(1 for r in rows if not r.get(question_field))
        empty_answers = sum(1 for r in rows if not r.get(answer_field))
        
        if empty_questions > 0:
            logger.warning(f"  ⚠️  Empty questions: {empty_questions}/{len(rows)}")
        if empty_answers > 0:
            logger.warning(f"  ⚠️  Empty answers: {empty_answers}/{len(rows)}")
        
        if empty_questions == 0 and empty_answers == 0:
            logger.info(f"  ✓ No missing required fields")
        
        # Show sample
        if rows:
            sample = rows[0]
            logger.info(f"  Sample row:")
            logger.info(f"    Q: {sample.get(question_field, 'N/A')[:60]}...")
            logger.info(f"    A: {sample.get(answer_field, 'N/A')[:60]}...")
            
            # For anchor_induced, show both variants
            if subset_name == "anchor_induced":
                grounded = sample.get("grounded_question", "N/A")
                ungrounded = sample.get("ungrounded_question", "N/A")
                logger.info(f"    Grounded:   {grounded[:60]}...")
                logger.info(f"    Ungrounded: {ungrounded[:60]}...")

    def get_ground_truth(self, row: Dict, subset_name: str = None) -> str:
        """Extract ground truth answer from a row."""
        # If we know the subset, use the mapping
        if subset_name and subset_name in FIELD_MAPPING:
            field = FIELD_MAPPING[subset_name].get("ground_truth")
            if field and field in row:
                return row[field]
        
        # Fallback: try different possible field names
        for field in ["answer", "new_value", "ground_truth"]:
            if field in row and row[field]:
                return row[field]
        return ""

    def get_question(self, row: Dict, subset_name: str = None, variant: str = "default") -> str:
        """Extract question from a row. For anchor_induced, can get grounded or ungrounded variant."""
        # If we know the subset, use the mapping
        if subset_name and subset_name in FIELD_MAPPING:
            mapping = FIELD_MAPPING[subset_name]
            
            # Special handling for anchor_induced dual questions
            if subset_name == "anchor_induced" and variant == "ungrounded":
                field = mapping.get("question_ungrounded")
                if field and field in row:
                    return row[field]
            
            # Default variant
            field = mapping.get("question")
            if field and field in row:
                return row[field]
        
        # Fallback: try different possible field names
        for field in ["question", "grounded_question", "ungrounded_question"]:
            if field in row and row[field]:
                return row[field]
        return ""

    def run_consistency_sample(self, model, question: str, n: int = 5, temperature: float = 0.7) -> Dict:
        """Run n samples at temperature to check consistency."""
        prompt = build_answer_prompt(question)
        answers = []
        
        for attempt in range(n):
            try:
                answer = model.infer(prompt, temperature=temperature)
                answers.append(answer.lower().strip())
            except Exception as e:
                logger.warning(f"Inference error on attempt {attempt+1}: {e}")
                answers.append("")
        
        # Find majority answer
        valid_answers = [a for a in answers if a]
        if not valid_answers:
            return {
                "majority_answer": "",
                "all_answers": answers,
                "consistency_score": 0.0,
                "is_confident": False,
                "agreement_count": 0,
            }
        
        from collections import Counter
        counts = Counter(valid_answers)
        majority_answer, majority_count = counts.most_common(1)[0]
        
        return {
            "majority_answer": majority_answer,
            "all_answers": answers,
            "consistency_score": round(majority_count / n, 2),
            "is_confident": majority_count >= 4,
            "agreement_count": majority_count,
        }

    def get_verbalized_confidence(self, model, question: str, answer: str) -> Optional[int]:
        """Get model's self-reported confidence."""
        if not answer:
            return None
        
        prompt = build_confidence_prompt(question, answer)
        try:
            raw = model.infer(prompt, temperature=0.0, max_tokens=5)
            return parse_verbalized_confidence(raw)
        except Exception as e:
            logger.warning(f"Confidence inference error: {e}")
            return None

    def evaluate_question(self, model, model_name: str, row: Dict, subset_name: str) -> List[Dict]:
        """
        Evaluate a single question-answer pair.
        For anchor_induced, returns 2 results (grounded + ungrounded) for anchor bias study.
        For other subsets, returns 1 result.
        """
        # Extract using subset-aware field mapping
        question = self.get_question(row, subset_name=subset_name, variant="default")
        ground_truth = self.get_ground_truth(row, subset_name=subset_name)
        
        if not question or not ground_truth:
            return None

        results = []
        
        # For anchor_induced, evaluate both grounded and ungrounded variants
        question_variants = [("default", question)]
        if subset_name == "anchor_induced":
            question_ungrounded = self.get_question(row, subset_name=subset_name, variant="ungrounded")
            if question_ungrounded:
                question_variants.append(("ungrounded", question_ungrounded))
        
        # Evaluate each question variant
        for variant_name, variant_question in question_variants:
            # Step 1: Consistency sampling
            consistency = self.run_consistency_sample(model, variant_question, n=5, temperature=0.7)
            model_answer = consistency["majority_answer"]
            
            # Check if answer is valid
            is_valid = is_valid_answer(model_answer)
            is_refusal = model.is_refusal(model_answer)

            # Step 2: Verbalized confidence
            verbalized_conf = self.get_verbalized_confidence(model, variant_question, model_answer) if is_valid else None

            # Step 3: Correctness check
            is_correct = normalize_text(model_answer) == normalize_text(ground_truth)

            # Step 4: Confusion cell
            confusion_cell = determine_confusion_cell(
                consistency["is_confident"] and is_valid,
                is_correct
            )

            result = {
                "model": model_name,
                "subset": subset_name,
                "question_variant": variant_name if subset_name == "anchor_induced" else None,
                "question": variant_question,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "full_answer": "|".join(consistency["all_answers"]),
                "stripped_answer": model_answer.strip(),
                "all_samples": "|".join(consistency["all_answers"]),
                "consistency_score": consistency["consistency_score"],
                "is_confident": consistency["is_confident"] and is_valid,
                "verbalized_conf": verbalized_conf,
                "is_correct": is_correct,
                "is_valid": is_valid,
                "is_refusal": is_refusal,
                "malformed": not is_valid,
                "confusion_cell": confusion_cell,
            }
            
            results.append(result)
        
        return results

    def evaluate_subset(self, model, model_name: str, subset_name: str, 
                       subset_path: str, limit: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        """Evaluate a full subset."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating {model_name} on {subset_name}")
        logger.info(f"{'='*70}")

        rows = self.load_subset(subset_name, subset_path)
        if limit:
            rows = rows[:limit]
            logger.info(f"Limited to {limit} rows for quick evaluation")

        results = []
        skipped = 0
        evaluated_count = 0
        start_time = time.time()

        for i, row in enumerate(rows):
            question_preview = self.get_question(row, subset_name=subset_name)[:50]
            logger.info(f"  [{i+1:3d}/{len(rows):3d}] {question_preview}...")
            
            try:
                question_results = self.evaluate_question(model, model_name, row, subset_name)
                if question_results:
                    results.extend(question_results)  # Results is now a list
                    evaluated_count += 1
            except Exception as e:
                logger.error(f"        Error evaluating row {i}: {str(e)[:100]}")
                skipped += 1

        elapsed_time = time.time() - start_time
        rate = evaluated_count / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"\n  Summary:")
        logger.info(f"    Evaluated: {evaluated_count} questions")
        logger.info(f"    Total results: {len(results)} (with variants)")
        if skipped > 0:
            logger.warning(f"    Skipped: {skipped} questions")
        logger.info(f"    Time: {elapsed_time:.1f}s ({rate:.2f} q/s)")

        # Save results
        out_path = self.results_dir / f"candor_{subset_name}_{model_name}.csv"
        self._save_results(results, out_path)

        # Compute metrics
        metrics = MetricsComputer.compute_bcs(results)
        MetricsComputer.print_metrics(metrics, model_name, subset_name)

        return results, metrics

    def _save_results(self, results: List[Dict], output_path: Path):
        """Save results to CSV."""
        if not results:
            logger.warning(f"No results to save to {output_path}")
            return

        fieldnames = list(results[0].keys())
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Saved {len(results)} results to {output_path}")

    def evaluate_all_subsets(self, model, model_name: str, subsets: Dict[str, str], 
                            limit: Optional[int] = None) -> Dict[str, Tuple[List, Dict]]:
        """Evaluate model on all subsets."""
        all_results = {}

        for subset_name, subset_path in subsets.items():
            try:
                results, metrics = self.evaluate_subset(
                    model, model_name, subset_name, subset_path, limit=limit
                )
                all_results[subset_name] = (results, metrics)
            except Exception as e:
                logger.error(f"Error evaluating subset {subset_name}: {e}")
                logger.error(traceback.format_exc())

        return all_results

    def run_full_pipeline(self, 
                         models: List[str],
                         subsets: Dict[str, str],
                         model_type: str = "open_source",
                         limit: Optional[int] = None,
                         wandb_run=None):
        """Run full pipeline: all models × all subsets."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting CANDOR Evaluation Pipeline")
        logger.info(f"Models: {models}")
        logger.info(f"Subsets: {list(subsets.keys())}")
        logger.info(f"{'='*60}\n")

        all_results_by_model = {}

        for model_name in models:
            logger.info(f"\n{'*'*60}")
            logger.info(f"Loading model: {model_name}")
            logger.info(f"{'*'*60}")

            try:
                # Load model
                if model_type == "open_source":
                    model = get_open_source_model(
                        model_name,
                        device="cuda",
                        dtype="float16",
                        quantization="4bit"
                    )
                else:
                    model = get_closed_source_model(model_name)

                model.load()

                # Evaluate all subsets
                subset_results = self.evaluate_all_subsets(
                    model, model_name, subsets, limit=limit
                )
                all_results_by_model[model_name] = subset_results

                # Cleanup
                model.unload()
                logger.info(f"Unloaded {model_name}")

            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
                logger.error(traceback.format_exc())

        # Save summary
        self._save_summary(all_results_by_model, wandb_run=wandb_run)

        return all_results_by_model

    def _save_summary(self, all_results_by_model: Dict, wandb_run=None):
        """Save a summary of results with metadata."""
        summary = {
            "metadata": {
                "evaluation_id": self.evaluation_id,
                "timestamp": self.timestamp,
                "seed": self.seed,
            }
        }

        for model_name, subset_results in all_results_by_model.items():
            summary[model_name] = {}
            for subset_name, (results, metrics) in subset_results.items():
                summary[model_name][subset_name] = metrics

        summary_path = self.results_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved summary to {summary_path}")
        logger.info(f"  Evaluation ID: {self.evaluation_id}")
        logger.info(f"  Seed: {self.seed}")

        if wandb_run is not None:
            log_wandb_metrics(wandb_run, summary, prefix="evaluation")
            log_wandb_artifact(
                wandb_run,
                summary_path,
                artifact_name=f"candor-summary-{self.evaluation_id}",
                artifact_type="evaluation-summary",
                metadata={"evaluation_id": self.evaluation_id, "seed": self.seed},
            )
            wandb_run.summary["evaluation_id"] = self.evaluation_id
            wandb_run.summary["results_dir"] = str(self.results_dir)
