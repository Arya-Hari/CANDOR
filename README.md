# CANDOR: Calibrated Answering across Knowledge Region-Boundaries


CANDOR is a benchmark for measuring model knowledge boundaries and calibration.

For setup, commands, and CLI arguments, see [USAGE.md](USAGE.md).

Optional WandB tracking:

```bash
wandb login
```

Or set `WANDB_API_KEY` in `.env`, then keep `WANDB_ENABLE=true` to log runs.

Quick start:

```bash
cp .env.example .env
pip install -r requirements.txt
```

## Liscence

CANDOR is released under CC BY-SA 4.0 in compliance with its Wikipedia and Wikidata sources. Evaluation code is released under MIT license. Model outputs from proprietary systems are included for research reproducibility subject to the respective providers' terms of use.