"""Helpers for optional Weights & Biases experiment tracking."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


def _get_wandb_module():
    try:
        import wandb

        return wandb
    except ImportError:
        logger.warning("Weights & Biases is not installed; skipping W&B logging.")
        return None


def _flatten_numeric_metrics(data: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    for key, value in data.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_numeric_metrics(value, full_key))
        elif isinstance(value, bool):
            flattened[full_key] = float(value)
        elif isinstance(value, (int, float)):
            flattened[full_key] = float(value)
    return flattened


def init_wandb_run(
    *,
    project: str = "CANDOR",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[Iterable[str]] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
):
    """Initialize a W&B run if the package and API key are available."""
    if os.getenv("WANDB_ENABLE", "true").lower() in {"0", "false", "no", "off"}:
        logger.info("WANDB_ENABLE is disabled; skipping W&B initialization.")
        return None

    if not os.getenv("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY is not set; skipping W&B initialization.")
        return None

    wandb = _get_wandb_module()
    if wandb is None:
        return None

    mode = os.getenv("WANDB_MODE", "online")
    entity = os.getenv("WANDB_ENTITY") or None
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", project),
        entity=entity,
        name=run_name,
        config=config or {},
        tags=list(tags) if tags else None,
        group=group,
        job_type=job_type,
        mode=mode,
        reinit=True,
    )
    return run


def log_wandb_metrics(run: Any, metrics: Dict[str, Any], prefix: str = "") -> None:
    """Log flattened numeric metrics to W&B."""
    if run is None:
        return

    payload = _flatten_numeric_metrics(metrics, prefix=prefix)
    if payload:
        run.log(payload)


def log_wandb_artifact(
    run: Any,
    file_path: Path,
    *,
    artifact_name: str,
    artifact_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a file as a W&B artifact."""
    if run is None:
        return

    wandb = _get_wandb_module()
    if wandb is None:
        return

    artifact = wandb.Artifact(artifact_name, type=artifact_type, metadata=metadata or {})
    artifact.add_file(str(file_path))
    run.log_artifact(artifact)


def finish_wandb_run(run: Any) -> None:
    """Finish a W&B run if one was created."""
    if run is not None:
        run.finish()