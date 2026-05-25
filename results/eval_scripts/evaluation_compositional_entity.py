"""
Usage:
    python evaluation_compositional_entity.py \\
        --input  raw_head_tail_rarity_DeepSeek-v3.2.jsonl \\
        --output evaluated_head_tail_rarity_DeepSeek-v3.2.jsonl

    python evaluation_compositional_entity.py --input raw.jsonl --output out.jsonl --limit 10

Environment variable:
    HF_TOKEN  - HuggingFace API token
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
SYSTEM_PROMPT = """You are a fact-checking assistant. Check if the model's answer matches any item in 
the list of correct answers and return a JSON result.

## RULES

1. Split the ground truth on commas to get a list of valid answers.
2. CORRECT if the answer matches any one item in the list.
   - Accept minor spelling variants, abbreviations, alternate name formats.
   - For titles (episodes, films, books): different titles are never equivalent 
     even if they belong to the same series or author.
3. WRONG if it matches none of the items.
4. REFUSAL if the model says it doesn't know or can't answer.
5. MALFORMED if the answer is garbled or nonsensical.

Confidence label + correctness → outcome:
- confident + correct   → confident_correct
- confident + wrong     → confident_wrong
- uncertain + correct   → unconfident_correct
- uncertain + wrong     → unconfident_wrong
- refusal               → refusal
- malformed             → malformed

## EXAMPLE

Question: Which city is the capital of Germany?
Ground truth: Berlin
Model answer: Berlin
Confidence label: confident

Output:
{"ground_truth_list":["Berlin"],"majority_answer":"Berlin","correctness":"correct","outcome_label":"confident_correct","matched_ground_truth":"Berlin","reasoning":"Berlin matches the ground truth exactly."}

## YOUR TASK

Question: {{question}}
Ground truth: {{ground_truth}}
Model answer: {{stripped_answer}}
Confidence label: {{confidence_label}}

Return ONLY a JSON object:
{
  "ground_truth_list": [...],
  "majority_answer": "...",
  "correctness": "correct | wrong | refusal | malformed",
  "outcome_label": "confident_correct | confident_wrong | unconfident_correct | unconfident_wrong | refusal | malformed",
  "matched_ground_truth": "... or null",
  "reasoning": "one sentence explaining the match decision"
}"""


def build_user_message(rec: dict) -> str:
    return (
        f"Question: {rec['question']}\n"
        f"Ground Truth (one or more valid answers, comma-separated): {rec['ground_truth']}\n"
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
                max_tokens=1024,   # larger: ground_truth_list + 5 per-run entries
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
        "ground_truth_list": [],
        "per_run_classifications": [],
        "run_counts": {
            "correct": 0, "correct_imprecise": 0, "wrong": 0,
            "refusal": 0, "malformed": 0,
        },
        "majority_response_type": "error",
        "majority_matched_ground_truth": None,
        "correctness": "error",
        "outcome_label": "error",
        "imprecise": None,
        "imprecision_reason": None,
        "reasoning": f"Evaluation failed: {error_msg}",
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global MODEL_ID
    parser = argparse.ArgumentParser(
        description="Evaluate head_tail_rarity JSONL outputs via HF Inference API."
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
    imprecise_count   = 0

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
            if eval_result.get("imprecise"):
                imprecise_count += 1

    total = sum(correctness_stats.values())
    print(f"\n-- Evaluation complete ({total} records) --")

    print("\n  CORRECTNESS:")
    for k, v in sorted(correctness_stats.items()):
        pct = 100 * v / total if total else 0
        print(f"    {k:<25} {v:>5}  ({pct:.1f}%)")

    print("\n  OUTCOME LABELS:")
    for k, v in sorted(outcome_stats.items()):
        pct = 100 * v / total if total else 0
        print(f"    {k:<35} {v:>5}  ({pct:.1f}%)")

    print(f"\n  Imprecise answers flagged: {imprecise_count}")
    print(f"\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()