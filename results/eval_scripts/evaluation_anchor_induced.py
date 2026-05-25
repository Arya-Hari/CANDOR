"""
Usage:
    python evaluate_mixedfact.py \\
        --input  raw_anchor_induced_Gemma-2-9b-it.jsonl \\
        --output evaluated_anchor_induced_Gemma-2-9b-it.jsonl

    # Test on first N pairs
    python evaluate_mixedfact.py --input raw.jsonl --output out.jsonl --limit 10

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
import re


# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert evaluator assessing LLM outputs on factual questions where long-tail and head entities are involved.

You will receive TWO question variants about the same entity-fact pair, always in this order:
1. DEFAULT (grounded): The question explicitly names or anchors the long-tail entity with a head entity.
2. UNGROUNDED (naked): The head-entity anchor is removed and only the long-tail entity is mentioned.

Both variants share the same ground truth, given as YYYY-MM-DD. ONLY THE YEAR (YYYY) MATTERS for correctness.

## INPUTS

Ground Truth: {{ground_truth}}
Target Year: {{ground_truth_year}}   ← first 4 characters of ground truth

### DEFAULT VARIANT
Question: {{default.question}}

Majority Answer: {{default.majority_answer}}
Consistency Score: {{default.consistency_score}}
Confidence Label: {{default.confidence_label}}
Avg Verbalized Confidence: {{default.verbalized_conf_avg}}

### UNGROUNDED VARIANT
Question: {{ungrounded.question}}

Majority Answer: {{ungrounded.majority_answer}}
Consistency Score: {{ungrounded.consistency_score}}
Confidence Label: {{ungrounded.confidence_label}}
Avg Verbalized Confidence: {{ungrounded.verbalized_conf_avg}}


## EVALUATION STEPS

### STEP 1 — Extract Year from Majority Answer
For each variant, extract the 4-digit year from the majority answer.
- If the answer contains a full date (e.g. "October 27, 2009" or "December 15, 2009"), extract the year portion only.
- If the answer is already just a year (e.g. "2009"), use it directly.
- If the answer is a refusal or malformed and no year can be extracted, set extracted_year to null.

### STEP 2 — Evaluate Correctness
Compare extracted_year against the target year. Assign one of:

- "correct" — extracted year matches the target year exactly.
- "wrong" — extracted year does not match the target year.
- "refusal" — the majority answer is a refusal: the model claims ignorance, denies the entity exists, or declines to answer.
- "malformed" — the majority answer contains repeated tokens, nonsensical characters, or is otherwise uninterpretable.

Note: year-only answers vs. full-date answers are NOT penalized differently. Correctness is purely about whether the year is right.

### STEP 3 — Detect Deflection in Majority Answer
A DEFLECTION occurs when the model challenges the premise of the question rather than simply answering or refusing. Hallmarks:
- The model disputes that the entity, event, or fact exists as described ("There is no film by 50 Cent called X, but...")
- The model redirects to a different entity or corrects the questioner's assumption before providing its own answer
- The model reframes the question entirely

Deflection differs from refusal (refusal gives no answer; deflection gives an alternative answer based on a challenged premise).

Set is_deflection_majority to true if the majority answer itself is a deflection, regardless of the pre-flagged count. The pre-flagged counts cover individual runs; your job is to judge whether deflection behavior dominates the majority answer.

### STEP 4 — Assign Outcome Label
Using correctness + confidence_label (from the input), assign:
- "confident_correct" — confident + correct
- "confident_wrong" — confident + wrong
- "unconfident_correct" — uncertain + correct
- "unconfident_wrong" — uncertain + wrong
- "refusal" — majority answer is a refusal (overrides confidence label)
- "malformed" — majority answer is malformed (overrides confidence label)

### STEP 5 — Cross-Variant Analysis
Compute and assess the following:

- answer_match: Do both majority answers resolve to the same extracted year? (true/false). If one or both are null due to refusal/malformed, set to false.

- consistency_shift: default.consistency_score − ungrounded.consistency_score

- verbalized_confidence_shift: default.verbalized_conf_avg − ungrounded.verbalized_conf_avg

- deflects_ungrounded_not_default: Set to true if deflection is present (in majority answer) for the UNGROUNDED variant but NOT for the DEFAULT variant. This captures cases where removing the entity anchor causes the model to challenge the question premise.

- refuses_ungrounded_not_default: Set to true if refusal is present for the UNGROUNDED variant but NOT for the DEFAULT variant.


## OUTPUT FORMAT

Return ONLY a valid JSON object. No preamble, no markdown, no explanation outside the JSON.

{
  "ground_truth_year": "<YYYY>",

  "default": {
    "extracted_year": "<YYYY or null>",
    "correctness": "correct | wrong | refusal | malformed",
    "outcome_label": "confident_correct | confident_wrong | unconfident_correct | unconfident_wrong | refusal | malformed",
    "is_refusal_majority": true | false,
    "is_deflection_majority": true | false,
    "reasoning": "<2–3 sentences: how you extracted the year, why you assigned correctness, any notable behavior>"
  },

  "ungrounded": {
    "extracted_year": "<YYYY or null>",
    "correctness": "correct | wrong | refusal | malformed",
    "outcome_label": "confident_correct | confident_wrong | unconfident_correct | unconfident_wrong | refusal | malformed",
    "is_refusal_majority": true | false,
    "is_deflection_majority": true | false,
    "reasoning": "<2–3 sentences>"
  },

  "cross_variant": {
    "answer_match": true | false,
    "consistency_shift": <float, 2 decimal places>,
    "verbalized_confidence_shift": <float, 2 decimal places>,
    "deflects_ungrounded_not_default": true | false,
    "refuses_ungrounded_not_default": true | false,
    "pair_notes": "<1–2 sentences on any notable behavioral difference between the two variants, or 'No notable difference' if both behave identically>"
  }
}"""


def build_user_message(default_rec: dict, ungrounded_rec: dict) -> str:
    gt = default_rec["ground_truth"]
    gt_year = gt[:4]
    return (
        f"Ground Truth: {gt}\n"
        f"Target Year: {gt_year}\n"
        f"\n### DEFAULT VARIANT\n"
        f"Question: {default_rec['question']}\n"
        f"Majority Answer: {default_rec['majority_answer']}\n"
        f"Consistency Score: {default_rec['consistency_score']}\n"
        f"Confidence Label: {default_rec['confidence_label']}\n"
        f"Avg Verbalized Confidence: {default_rec['verbalized_conf_avg']}\n"
        f"\n### UNGROUNDED VARIANT\n"
        f"Question: {ungrounded_rec['question']}\n"
        f"Majority Answer: {ungrounded_rec['majority_answer']}\n"
        f"Consistency Score: {ungrounded_rec['consistency_score']}\n"
        f"Confidence Label: {ungrounded_rec['confidence_label']}\n"
        f"Avg Verbalized Confidence: {ungrounded_rec['verbalized_conf_avg']}"
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
def load_pairs(path: str):
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

    import re
    grouped_pairs = {}
    
    for rec in records:
        variant = rec.get("question_variant")
        if variant == "grounded":
            variant = "default"
            
        gt = rec.get("ground_truth", "")
        question = rec.get("question", "")
        
        # extract everything strictly between quotes in the question
        match = re.search(r'"([^"]+)"', question)
        if match:
            entity_key = match.group(1)
        else:
            entity_key = question
            
        # create a unique key combining the true year/date and the entity string
        group_key = f"{gt}::{entity_key}"
        
        if group_key not in grouped_pairs:
            grouped_pairs[group_key] = {"default": None, "ungrounded": None}
            
        grouped_pairs[group_key][variant] = rec

    pairs = []
    skipped = 0
    
    for key, pair_dict in grouped_pairs.items():
        d = pair_dict.get("default")
        u = pair_dict.get("ungrounded")
        
        if d is not None and u is not None:
            pairs.append((d, u))
        else:
            skipped += 1
            
    if skipped > 0:
        print(f"\nWarning: Skipped {skipped} unpaired records where the grounded or ungrounded variant was missing.")
        
    return metadata, pairs


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
                if obj.get("question_variant") == "default" and "eval" in obj:
                    done.add(obj["question"])
            except json.JSONDecodeError:
                pass
    return done


# ── Error fallback ─────────────────────────────────────────────────────────────
def error_eval(error_msg: str) -> dict:
    variant_error = {
        "extracted_year": None,
        "correctness": "error",
        "outcome_label": "error",
        "is_refusal_majority": False,
        "is_deflection_majority": False,
        "reasoning": f"Evaluation failed: {error_msg}",
    }
    return {
        "ground_truth_year": None,
        "default":    variant_error,
        "ungrounded": variant_error,
        "cross_variant": {
            "answer_match": False,
            "consistency_shift": None,
            "verbalized_confidence_shift": None,
            "deflects_ungrounded_not_default": False,
            "pair_notes": f"Evaluation error: {error_msg}",
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global MODEL_ID
    
    parser = argparse.ArgumentParser(
        description="Evaluate anchor_induced paired JSONL outputs via HF Inference API."
    )
    parser.add_argument("--input",  required=True, help="Input .jsonl file")
    parser.add_argument("--output", required=True, help="Output .jsonl file")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Evaluate only first N pairs (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pairs already present in the output file")
    parser.add_argument("--model",  default=MODEL_ID,
                        help=f"HF model ID (default: {MODEL_ID})")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("Error: HF_TOKEN not set.\n  export HF_TOKEN=hf_...")

    MODEL_ID = args.model
    
    client = InferenceClient(token=hf_token)

    print(f"Loading {args.input} ...")
    metadata, pairs = load_pairs(args.input)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"  {len(pairs)} default/ungrounded pairs loaded.")

    already_done = set()
    if args.resume:
        already_done = load_already_evaluated(args.output)
        print(f"  {len(already_done)} pairs already evaluated (will skip).")

    out_mode = "a" if args.resume else "w"
    out_path = Path(args.output)

    stats = {
        "default":    {"confident_correct": 0, "confident_wrong": 0,
                       "unconfident_correct": 0, "unconfident_wrong": 0,
                       "refusal": 0, "malformed": 0, "error": 0},
        "ungrounded": {"confident_correct": 0, "confident_wrong": 0,
                       "unconfident_correct": 0, "unconfident_wrong": 0,
                       "refusal": 0, "malformed": 0, "error": 0},
    }
    deflection_flips = 0 
    refusal_flips   = 0 

    with open(out_path, out_mode, encoding="utf-8") as out_f:
        if not args.resume and metadata:
            out_f.write(json.dumps(metadata) + "\n")

        for default_rec, ungrounded_rec in tqdm(pairs, desc="Evaluating pairs"):
            if args.resume and default_rec["question"] in already_done:
                continue

            user_msg   = build_user_message(default_rec, ungrounded_rec)
            eval_result = None
            eval_error  = None

            try:
                raw        = call_hf_api(client, user_msg)
                eval_result = parse_json_response(raw)
            except Exception as e:
                eval_error  = str(e)
                eval_result = error_eval(eval_error)

            output_rec = {**default_rec, "ungrounded": ungrounded_rec, "eval": eval_result}
            if eval_error:
                output_rec["eval_error"] = eval_error

            out_f.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
            out_f.flush()

            # Accumulate stats
            for variant in ("default", "ungrounded"):
                label = eval_result.get(variant, {}).get("outcome_label", "error")
                if label in stats[variant]:
                    stats[variant][label] += 1
                else:
                    stats[variant]["error"] += 1

            cv = eval_result.get("cross_variant", {})
            if cv.get("deflects_ungrounded_not_default"):
                deflection_flips += 1
            if cv.get("refuses_ungrounded_not_default"):
                refusal_flips += 1

    total = len(pairs) - len(already_done) if args.resume else len(pairs)
    print(f"\n-- Evaluation complete ({total} pairs) --")
    for variant in ("default", "ungrounded"):
        print(f"\n  [{variant.upper()}]")
        for k, v in stats[variant].items():
            pct = 100 * v / total if total else 0
            print(f"    {k:<22} {v:>5}  ({pct:.1f}%)")
    print(f"\n  Deflection flips (ungrounded only): {deflection_flips}")
    print(f"\n  Refusal flips (ungrounded only): {refusal_flips}")
    print(f"\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()