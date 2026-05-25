"""
Usage:
    python evaluate_longtail.py \
        --input  raw_longtail_Gemma-2-9b-it.jsonl \
        --output evaluated_Gemma-2-9b-it.jsonl

    # Limit to first N records
    python evaluate_longtail.py --input raw.jsonl --output out.jsonl --limit 20

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
SYSTEM_PROMPT = """You are an expert evaluator assessing LLM outputs on factual questions about long-tail entities.

You will be given:
- A factual question
- The ground truth answer
- The majority answer (most frequent answer across 5 runs of the LLM)
- The consistency score (fraction of runs matching the majority answer)
- The binary confidence label: "confident" (consistency >= 0.8) or "uncertain"
- The average verbalized confidence score (0-100)

Your job is to carefully analyze the inputs and return a JSON object with the fields defined below.

1. CORRECTNESS - compare majority answer against ground truth:
   - "correct"           - matches ground truth or clearly valid equivalent
   - "correct_imprecise" - The majority answer is in the right direction but at the wrong level of granularity or specificity. Examples:
        a. Ground truth is a village, municipality, district or city, model answered with the country or state/region it belongs to. This is an important case, so pay attention
        b. Ground truth is a full name, model answered with just a last name
        c. Ground truth is a specific date, model answered with just the year
        d. The answer is a superset or parent category of the correct answer.
        For example
    In these cases the model is not wrong per se, but it is less precise than required.
   - "wrong"             - factually incorrect
   - "refusal"           - model refused to answer
   - "malformed"         - answer cannot be evaluated

IMPORTANT - If there is a lack of precision in the answer, but it is still in the right direction, label it as "correct_imprecise" and set imprecise=true in the output. Do NOT label it as "wrong" if it is factually correct but just less specific than the ground truth.

2. OUTCOME LABEL - from correctness + confidence:
   - "confident_correct"    - confident + (correct or correct_imprecise)
   - "confident_wrong"      - confident + wrong
   - "unconfident_correct"  - uncertain + (correct or correct_imprecise)
   - "unconfident_wrong"    - uncertain + wrong
   - "refusal"              - majority answer is a refusal
   - "malformed"            - majority answer is malformed

3. IMPRECISION - if correctness is "correct_imprecise", set imprecise=true and explain.

OUTPUT FORMAT: Return ONLY a valid JSON object - no explanation, no markdown, no preamble.

{
  "majority_answer_type": "normal" | "refusal" | "malformed",
  "correctness": "correct" | "correct_imprecise" | "wrong" | "refusal" | "malformed",
  "outcome_label": "confident_correct" | "confident_wrong" | "unconfident_correct" | "unconfident_wrong" | "refusal" | "malformed",
  "imprecise": true | false,
  "imprecision_reason": "<short explanation or null>",
  "reasoning": "<2-4 sentences explaining your correctness judgment>"
}"""


def build_user_message(rec: dict) -> str:
    return (
        f"Question: {rec['question']}\n"
        f"Ground Truth: {rec['ground_truth']}\n"
        f"Majority Answer: {rec['majority_answer']}\n"
        f"Consistency Score: {rec['consistency_score']}\n"
        f"Confidence Label: {rec['confidence_label']}\n"
        f"Average Verbalized Confidence: {rec['verbalized_conf_avg']}"
    )


# ── HF API call ─────────────────────────────────────────────────────────────
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
                max_tokens=512,
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


# ── Error eval record ─────────────────────────────────────────────────────────
ERROR_EVAL = {
    "majority_answer_type": "error",
    "correctness":          "error",
    "outcome_label":        "error",
    "imprecise":            None,
    "imprecision_reason":   None,
    "reasoning":            "Evaluation failed -- see eval_error field.",
}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global MODEL_ID

    parser = argparse.ArgumentParser(
        description="Evaluate longtail JSONL benchmark outputs via HF Inference API."
    )
    parser.add_argument("--input",  required=True, help="Input .jsonl file")
    parser.add_argument("--output", required=True, help="Output .jsonl file")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only evaluate the first N records (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip records already present in the output file")
    parser.add_argument("--model",  default=MODEL_ID,
                        help=f"HF model ID (default: {MODEL_ID})")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit(
            "Error: HF_TOKEN not set.\n"
            "  export HF_TOKEN=hf_..."
        )

    MODEL_ID = args.model

    client = InferenceClient(token=hf_token)

    print(f"Loading {args.input} ...")
    metadata, records = load_jsonl(args.input)
    if args.limit:
        records = records[:args.limit]
    print(f"  {len(records)} data records loaded.")

    already_done = set()
    if args.resume:
        already_done = load_already_evaluated(args.output)
        print(f"  {len(already_done)} records already evaluated (will skip).")

    out_mode = "a" if args.resume else "w"
    out_path = Path(args.output)
    with open(out_path, out_mode, encoding="utf-8") as out_f:
        if not args.resume and metadata:
            out_f.write(json.dumps(metadata) + "\n")

        stats = {"correct": 0, "correct_imprecise": 0, "wrong": 0,
                 "refusal": 0, "malformed": 0, "error": 0}

        for rec in tqdm(records, desc="Evaluating"):
            q = rec.get("question", "")

            if args.resume and q in already_done:
                continue

            user_msg = build_user_message(rec)
            eval_result = None
            eval_error  = None

            try:
                raw = call_hf_api(client, user_msg)
                eval_result = parse_json_response(raw)
            except Exception as e:
                eval_error  = str(e)
                eval_result = dict(ERROR_EVAL)
                eval_result["eval_error"] = eval_error

            output_rec = {**rec, "eval": eval_result}
            if eval_error:
                output_rec["eval_error"] = eval_error

            out_f.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
            out_f.flush() 

            label = eval_result.get("correctness", "error")
            if label in stats:
                stats[label] += 1
            else:
                stats["error"] += 1

    total = sum(stats.values())
    print(f"\n-- Evaluation complete ({total} records) --")
    for k, v in stats.items():
        pct = 100 * v / total if total else 0
        print(f"  {k:<20} {v:>5}  ({pct:.1f}%)")
    print(f"\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()