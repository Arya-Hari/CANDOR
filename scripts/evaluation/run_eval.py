#!/usr/bin/env python3
"""
CANDOR Evaluation Pipeline Runner
Run inference on datasets with multiple models and compute metrics.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import torch

# Make direct execution from the repo root resolve `scripts.*` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("candor_eval.log"),
        logging.StreamHandler(),
    ]
)

from scripts.evaluation.evaluate import EvaluationPipeline
from scripts.evaluation.wandb_utils import finish_wandb_run, init_wandb_run


def _first_env_value(*names: str):
    for name in names:
        value = os.getenv(name)
        if value:
            return value, name
    return None, None


def _azure_openai_configured() -> bool:
    api_key, _ = _first_env_value("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_APIP_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    return bool(api_key and endpoint)


def validate_config(model_type: str = "open_source", models=None) -> None:
    """Validate configuration before running."""
    logger = logging.getLogger(__name__)
    models = models or []
    
    logger.info("\n" + "="*70)
    logger.info("CONFIGURATION VALIDATION")
    logger.info("="*70)
    
    # Check inference device
    device = os.getenv("INFERENCE_DEVICE", "cuda").lower()
    if device not in ["cuda", "cpu"]:
        raise ValueError(f"Invalid INFERENCE_DEVICE: {device}. Must be 'cuda' or 'cpu'")
    
    if device == "cuda":
        if torch.cuda.is_available():
            logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️  INFERENCE_DEVICE=cuda but no GPU available!")
            logger.warning("   Falling back to CPU (10x slower)")
    else:
        logger.info("✓ Using CPU mode")
    
    # Check inference dtype
    dtype = os.getenv("INFERENCE_DTYPE", "float16")
    if dtype not in ["float16", "float32", "bfloat16"]:
        raise ValueError(f"Invalid INFERENCE_DTYPE: {dtype}")
    logger.info(f"✓ Inference dtype: {dtype}")
    
    # Check quantization
    quantization = os.getenv("INFERENCE_QUANTIZATION", "4bit")
    if quantization not in ["4bit", "8bit", "None"]:
        raise ValueError(f"Invalid INFERENCE_QUANTIZATION: {quantization}")
    logger.info(f"✓ Quantization: {quantization if quantization != 'None' else 'None (full precision)'}")
    
    # Check API keys for closed-source models
    if model_type == "closed_source":
        logger.info("\nChecking API keys for closed-source models:")
        requested_models = {model.lower() for model in models}
        azure_gpt_requested = "gpt-5.1" in requested_models
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            logger.info(f"✓ OPENAI_API_KEY configured")
        else:
            if azure_gpt_requested and _azure_openai_configured():
                logger.info("✓ GPT-5.1 will use Azure OpenAI configuration")
            else:
                logger.warning("⚠️  OPENAI_API_KEY not set (GPT-5.1 will fail unless Azure OpenAI is configured)")
        
        legacy_gemini_requested = "gemini-2.0-flash" in requested_models
        vertex_gemini_requested = "gemini-2.5-pro" in requested_models

        gemini_key = os.getenv("GEMINI_API_KEY") if legacy_gemini_requested else None
        if legacy_gemini_requested:
            if gemini_key:
                logger.info(f"✓ GEMINI_API_KEY configured")
            else:
                logger.warning("⚠️  GEMINI_API_KEY not set (Gemini-2.0-flash will fail)")

        vertex_gemini_ready = False
        if vertex_gemini_requested:
            project_id, project_env = _first_env_value(
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_PROJECT_ID",
                "PROJECT_ID",
                "GCLOUD_PROJECT",
            )
            if project_id:
                logger.info(f"✓ Google Cloud project configured via {project_env}: {project_id}")
                vertex_gemini_ready = True
            else:
                logger.warning(
                    "⚠️  Google Cloud project not set (Gemini-2.5-pro needs GOOGLE_CLOUD_PROJECT or an equivalent project env var)"
                )

            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path:
                if Path(credentials_path).expanduser().exists():
                    logger.info("✓ GOOGLE_APPLICATION_CREDENTIALS configured")
                else:
                    logger.warning(f"⚠️  GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {credentials_path}")
            else:
                logger.warning("⚠️  GOOGLE_APPLICATION_CREDENTIALS not set (default ADC may still work if already configured)")
        
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            logger.info(f"✓ DEEPSEEK_API_KEY configured")
        else:
            logger.warning("⚠️  DEEPSEEK_API_KEY not set (DeepSeek will fail)")
        
        if not (openai_key or deepseek_key or gemini_key or vertex_gemini_ready or _azure_openai_configured()):
            raise ValueError(
                "No closed-source credentials configured. Set OPENAI_API_KEY, AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT, DEEPSEEK_API_KEY, GEMINI_API_KEY, or Google Cloud ADC/project for Gemini-2.5-pro"
            )
    
    logger.info("\n✓ Configuration validated\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run CANDOR evaluation pipeline on datasets"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen2.5-7B-Instruct"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--model-type",
        choices=["open_source", "closed_source"],
        default="open_source",
        help="Type of models (open_source or closed_source)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["outdated"],
        help="Data subsets to evaluate on",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions per subset",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    wandb_run = init_wandb_run(
        run_name=f"eval-{args.model_type}",
        job_type="evaluation",
        tags=[args.model_type, "evaluation"],
        config={
            "models": args.models,
            "model_type": args.model_type,
            "subsets": args.subsets,
            "limit": args.limit,
            "results_dir": args.results_dir,
            "seed": args.seed,
        },
    )

    # Map subset names to file paths
    subset_map = {
        "outdated": "data/raw/outdated_facts.csv",
        "longtail": "data/raw/long_tailed.csv",
        "anchor_induced": "data/anchor_induced.csv",
        "near_true": "data/near_true.csv",
    }

    # Validate subsets
    subsets = {}
    for subset_name in args.subsets:
        if subset_name not in subset_map:
            raise ValueError(
                f"Unknown subset: {subset_name}. Available: {list(subset_map.keys())}"
            )
        subsets[subset_name] = subset_map[subset_name]

    # Validate configuration
    validate_config(model_type=args.model_type, models=args.models)

    # Initialize pipeline
    pipeline = EvaluationPipeline(results_dir=args.results_dir)
    pipeline.set_seed(args.seed)

    # Run pipeline
    try:
        results = pipeline.run_full_pipeline(
            models=args.models,
            subsets=subsets,
            model_type=args.model_type,
            limit=args.limit,
            wandb_run=wandb_run,
        )

        print("\n" + "="*60)
        print("CANDOR Evaluation Complete!")
        print(f"Results saved to: {args.results_dir}")
        print("="*60)
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
