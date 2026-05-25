# Usage

This repository is organized around three parts:

1. `data/` contains the benchmark datasets.
2. `scripts/` contains the code that runs inference and evaluates outputs.
3. `results/` and `EBP_results/` are kept only as reference for the evaluation code layout; generated outputs are not part of the public release.

The repo is meant to stay light: source code and datasets live in version control, while inference outputs, evaluation outputs, metrics, plots, and generation outputs should be treated as generated artifacts.

The current evaluation code still exposes a legacy `outdated` path for backward compatibility, but the bundled released datasets in this repo are the long-tail, mixed-facts, near-true, counterfactual, non-existant, and head-tail-rarity splits under `data/`.

## Setup

```bash
cp .env.example .env
pip install -r requirements.txt
```

Populate `.env` with the model API keys or cloud credentials you actually plan to use. The example file lists every environment variable read by the current scripts.

## Weights & Biases

W&B tracking is optional. It is used by the evaluation runners when enabled.

Login once from your shell:

```bash
wandb login
```

Or put your key in `.env`:

```bash
WANDB_API_KEY=your_key_here
WANDB_ENABLE=true
```

Useful knobs:

- `WANDB_ENABLE=false` disables all W&B logging.
- `WANDB_PROJECT` sets the project name.
- `WANDB_ENTITY` sets the team or user namespace.
- `WANDB_MODE=online` sends runs to Wandb; `offline` keeps local history.

## Inference Export

`scripts/evaluation/run_inference.py` exports raw model answers for later scoring.

Example:

```bash
python scripts/evaluation/run_inference.py \
  --models Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct \
  --subsets longtail anchor_induced near_true \
  --model-type open_source
```

Arguments:

- `--models`: one or more model names to run.
- `--model-type`: `open_source` or `closed_source`.
- `--subsets`: one or more subsets to export.
- `--batch-size`: sequential export chunk size, default `8`.
- `--samples`: repeated generations per question, default `5`.
- `--temperature`: answer sampling temperature, default `0.0`.
- `--max-tokens`: maximum answer tokens, default `50`.
- `--device`: inference device for open-source models, default `auto`.
- `--dtype`: model dtype, default `float16`.
- `--quantization`: quantization mode, default `None`.
- `--limit`: optional row cap per subset.
- `--output-dir`: output directory, default `results/inference_results`.
- `--seed`: random seed, default `0`.

Important environment variables:

- `INFERENCE_DEVICE`
- `INFERENCE_DTYPE`
- `INFERENCE_QUANTIZATION`
- `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT` or `GOOGLE_PROJECT_ID` or `PROJECT_ID` or `GCLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION` or `VERTEX_AI_LOCATION`
- `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, or `AWS_BEARER_TOKEN_BEDROCK`
- `BEDROCK_REGION` or `AWS_REGION`

## Evaluation

`scripts/evaluation/run_eval.py` runs the full pipeline and computes the main metrics.

Example:

```bash
python scripts/evaluation/run_eval.py \
  --models Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct \
  --subsets longtail anchor_induced near_true
```

Arguments:

- `--models`: models to evaluate, default `Qwen2.5-7B-Instruct`.
- `--model-type`: `open_source` or `closed_source`, default `open_source`.
- `--subsets`: subsets to score, default `outdated`.
- `--limit`: optional row cap per subset.
- `--results-dir`: output directory, default `results`.
- `--seed`: random seed, default `42`.

## EBP Boundary Experiment

`scripts/evaluation/run_EBP_experiment.py` runs the chain-of-thought boundary-calibration experiment.

Example:

```bash
python scripts/evaluation/run_EBP_experiment.py \
  --models Llama-3.3-70B GPT-4.1 Gemini-2.5-pro DeepSeek-v3.2 \
  --tasks longtail near_true anchor_induced \
  --conditions baseline zero_shot_EBP boundary_aware_EBP
```

Arguments:

- `--models`: closed-source models to evaluate.
- `--tasks`: `longtail`, `near_true`, and/or `anchor_induced`.
- `--conditions`: `baseline`, `zero_shot_EBP`, and/or `boundary_aware_EBP`.
- `--sample-size`: questions sampled per task, default `200`.
- `--samples-per-question`: repeated samples per question, default `5`.
- `--temperature`: answer sampling temperature, default `0.0`.
- `--confidence-temperature`: confidence follow-up temperature, default `0.0`.
- `--max-tokens`: answer token limit, default `512`.
- `--bootstrap-samples`: bootstrap resamples for confidence intervals, default `1000`.
- `--seed`: random seed, default `42`.
- `--results-dir`: output directory, default `results/EBP_experiment`.
- `--reuse-baseline`: reuse baseline raw JSONL outputs instead of rerunning them.
- `--baseline-input-dir`: directory containing raw baseline JSONL files.
- `--sample-csv-dir`: directory containing fixed sample CSVs.
- `--mixed-facts-variant`: deprecated compatibility flag for mixed-facts sampling.

## Environment Summary

The current code reads these environment variables:

- `OPENAI_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_APIP_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_PROJECT_ID`
- `PROJECT_ID`
- `GCLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `VERTEX_AI_LOCATION`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_BEARER_TOKEN_BEDROCK`
- `AWS_REGION`
- `BEDROCK_REGION`
- `SPARQL_ENDPOINT`
- `PAGEVIEW_MIN_INTERVAL`
- `INFERENCE_DEVICE`
- `INFERENCE_DTYPE`
- `INFERENCE_QUANTIZATION`
- `WANDB_ENABLE`
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `WANDB_MODE`
- `WANDB_API_KEY`
