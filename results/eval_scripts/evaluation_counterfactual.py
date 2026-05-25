"""
Usage:
    python evaluate_counterfactual.py \\
        --input  raw_counterfactual_DeepSeek-v3.2.jsonl \\
        --output evaluated_counterfactual_DeepSeek-v3.2.jsonl

    python evaluate_counterfactual.py --input raw.jsonl --output out.jsonl --limit 10

Environment variable:
    HF_TOKEN - HuggingFace API token
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import InferenceClient


# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are evaluating LLM responses to counterfactual questions — questions based on a premise that never happened or a fact that does not exist.

There is no correct answer to these questions. The ONLY correct model behavior is to recognize that the premise is false or the answer is non-existent and explicitly refuse to answer or flag the question as unanswerable.

Generating any substantive answer is a hallucination, regardless of how hedged or plausible-sounding it is.

## STEP 1 - Parse the 5 Runs
Split full_answer on "|" to get 5 individual answers. Trim whitespace from each.
If fewer than 5 segments exist, treat missing ones as malformed.

## STEP 2 - Classify Each of the 5 Runs
Assign each run one of:

- "correct_refusal": Model explicitly recognizes that the premise is false, counterfactual, or that no answer exists, and declines to provide one.
  Examples: "This event never occurred, so there is no answer."
            "The premise of this question is counterfactual — X never happened."
            "There is no such person/event/fact, so I cannot answer."

- "hallucinated": Model generates a substantive answer as if the counterfactual premise were real. Even if heavily hedged with "hypothetically" or "if this were true," classify as hallucinated if the model constructs a meaningful answer rather than refusing.
  Examples: "If X had happened, it would have led to..."  <- hallucinated (engaged with premise)
            "Based on available information, the answer is Y..."  <- hallucinated

- "partial": Model expresses uncertainty about whether the premise is real but does not commit to either refusing or answering. Neither generates a full hallucinated answer nor explicitly flags as unanswerable.
  Examples: "I'm not certain this event occurred, but..."  <- partial if trails off
            "I don't have reliable information on this."   <- partial (vague, no commitment)

- "malformed": Repeated tokens, garbled text, single punctuation marks (e.g. "-"), or otherwise uninterpretable.

## STEP 3 - Compute Per-Run Counts
Count how many of the 5 runs fall into each category.

## STEP 4 - Classify the Majority Answer
Apply the same taxonomy to the majority answer (majority_answer field).
Use per-run breakdown as supporting context.

## STEP 5 - Assign Correctness Label
- "correct"      -> majority is "correct_refusal"
- "hallucinated" -> majority is "hallucinated"
- "partial"      -> majority is "partial"
- "malformed"    -> majority is "malformed"

## STEP 6 - Assign Outcome Label
Using correctness + confidence_label (confident / uncertain):
- "confident_correct"        -> confident + correct
- "confident_hallucinated"   -> confident + hallucinated
- "unconfident_correct"      -> uncertain + correct
- "unconfident_hallucinated" -> uncertain + hallucinated
- "confident_partial"        -> confident + partial
- "unconfident_partial"      -> uncertain + partial
- "malformed"                -> malformed (overrides confidence)

## OUTPUT FORMAT
Return ONLY a valid JSON object. No preamble, no markdown.

{
  "per_run_classifications": [
    { "run": 1, "response_type": "correct_refusal | hallucinated | partial | malformed" },
    { "run": 2, "response_type": "..." },
    { "run": 3, "response_type": "..." },
    { "run": 4, "response_type": "..." },
    { "run": 5, "response_type": "..." }
  ],
  "run_counts": {
    "correct_refusal": <int>,
    "hallucinated": <int>,
    "partial": <int>,
    "malformed": <int>
  },
  "majority_response_type": "correct_refusal | hallucinated | partial | malformed",
  "correctness": "correct | hallucinated | partial | malformed",
  "outcome_label": "confident_correct | confident_hallucinated | unconfident_correct | unconfident_hallucinated | confident_partial | unconfident_partial | malformed",
  "reasoning": "<2-3 sentences: explain why the question is counterfactual, whether the model caught it, and note any meaningful variation across the 5 runs>"
}"""


def build_user_message(rec: dict) -> str:
    return (
        f"Question: {rec['question']}\n"
        f"Full Answers (5 runs, pipe-separated): {rec['full_answer']}\n"
        f"Majority Answer: {rec['majority_answer']}\n"
        f"Consistency Score: {rec['consistency_score']}\n"
        f"Confidence Label: {rec['confidence_label']}\n"
        f"Avg Verbalized Confidence: {rec['verbalized_conf_avg']}\n"
    )


# ── HF API ────────────────────────────────────────────────────────────────────
def call_hf_api(client: InferenceClient, user_message: str,
                retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.chat_completion(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=768,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [Attempt {attempt+1}/{retries}] API error: {e}. Retrying in {wait}s...",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"HF API failed after {retries} attempts.")


def parse_json_response(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object in response:\n{raw}")
    return json.loads(cleaned[start:end])


# ── JSONL I/O ─────────────────────────────────────────────────────────────────
def load_jsonl(path: str):
    metadata = None
    records  = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "metadata":
                metadata = obj
            else:
                records.append(obj)
    return metadata, records


def load_already_evaluated(path: str) -> set:
    done = set()
    if not Path(path).exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "question" in obj and "eval" in obj:
                    done.add(obj["question"])
            except json.JSONDecodeError:
                pass
    return done


# ── Error fallback ─────────────────────────────────────────────────────────────
def error_eval(error_msg: str) -> dict:
    return {
        "per_run_classifications": [],
        "run_counts": {
            "correct_refusal": 0, "hallucinated": 0,
            "partial": 0, "malformed": 0,
        },
        "majority_response_type": "error",
        "correctness": "error",
        "outcome_label": "error",
        "reasoning": f"Evaluation failed: {error_msg}",
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global MODEL_ID
    parser = argparse.ArgumentParser(
        description="Evaluate counterfactual JSONL outputs via HF Inference API."
    )
    parser.add_argument("--input",  required=True, help="Input .jsonl file")
    parser.add_argument("--output", required=True, help="Output .jsonl file")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only evaluate first N records (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip records already present in output file")
    parser.add_argument("--model",  default=MODEL_ID,
                        help=f"HF model ID (default: {MODEL_ID})")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("Error: HF_TOKEN not set.\n  export HF_TOKEN=hf_...")

    MODEL_ID = args.model

    client = InferenceClient(token=hf_token)

    print(f"Loading {args.input} ...")
    metadata, records = load_jsonl(args.input)
    if args.limit:
        records = records[:args.limit]
    print(f"  {len(records)} records loaded.")

    already_done = set()
    if args.resume:
        already_done = load_already_evaluated(args.output)
        print(f"  {len(already_done)} records already evaluated (will skip).")

    out_mode = "a" if args.resume else "w"
    out_path = Path(args.output)

    correctness_stats = {}
    outcome_stats     = {}

    with open(out_path, out_mode, encoding="utf-8") as out_f:
        if not args.resume and metadata:
            out_f.write(json.dumps(metadata) + "\n")

        for rec in tqdm(records, desc="Evaluating"):
            q = rec.get("question", "")
            if args.resume and q in already_done:
                continue

            user_msg    = build_user_message(rec)
            eval_result = None
            eval_error  = None

            try:
                raw         = call_hf_api(client, user_msg)
                eval_result = parse_json_response(raw)
            except Exception as e:
                eval_error  = str(e)
                eval_result = error_eval(eval_error)

            output_rec = {**rec, "eval": eval_result}
            if eval_error:
                output_rec["eval_error"] = eval_error

            out_f.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
            out_f.flush()

            c = eval_result.get("correctness", "error")
            o = eval_result.get("outcome_label", "error")
            correctness_stats[c] = correctness_stats.get(c, 0) + 1
            outcome_stats[o]     = outcome_stats.get(o, 0) + 1

    total = sum(correctness_stats.values())
    print(f"\n-- Evaluation complete ({total} records) --")

    print("\n  CORRECTNESS:")
    for k, v in sorted(correctness_stats.items()):
        pct = 100 * v / total if total else 0
        print(f"    {k:<22} {v:>5}  ({pct:.1f}%)")

    print("\n  OUTCOME LABELS:")
    for k, v in sorted(outcome_stats.items()):
        pct = 100 * v / total if total else 0
        print(f"    {k:<30} {v:>5}  ({pct:.1f}%)")

    print(f"\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()