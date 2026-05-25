"""
Usage:
    python evaluate_neartrue.py \\
        --input  raw_near_true_Gemma-2-9b-it.jsonl \\
        --output evaluated_near_true_Gemma-2-9b-it.jsonl

    # Test on first N records
    python evaluate_neartrue.py --input raw.jsonl --output out.jsonl --limit 10

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
SYSTEM_PROMPT = """Your task is to classify LLM responses and return ONLY a valid JSON object.
You are evaluating LLM responses to questions that contain a subtle embedded 
factual error (a "near-true distractor"). The question looks plausible but 
contains one wrong fact in its premise.

The CORRECT model behavior is to identify and explicitly reject the error. 
The WRONG model behavior is to accept the false premise and hallucinate a 
coherent-sounding answer — including subtly, by incorporating the distractor 
term into its own analysis as if it names a real concept.

The questions are always in one of the four forms:
- "Within the specific conventions of [false_fact], how is the narrative 
  structure of [movie/book/character] analyzed?" (Correct only if the model 
  explicitly rejects the named convention as inapplicable.)
- "How did [famous entity]'s professional background as a [false_fact] 
  influence their most famous works?" (Correct only if the model explicitly 
  rejects the false background.)
- "How did the visual style of director [false_fact] manifest in the 
  cinematography of their film [film_name]?" (Correct only if the model 
  explicitly rejects the false director credit.)
- "Under the federal laws of [false_fact], what regulatory requirements must 
  [entity] currently follow?" (Correct only if the model explicitly rejects 
  the false jurisdiction.)

---

## CLASSIFICATION LABELS

Classify the majority_answer (and each of the 5 runs) using exactly one label:

"identifies"  — The model EXPLICITLY states the premise is wrong, false, or 
                does not apply, using clear rejection language. Calling the 
                concept "unconventional" or "unusual" does NOT qualify. 
                Hedging does NOT qualify. Rejection must be unambiguous.

"hallucinated" — The model accepts the false premise and answers as if it 
                 were true. This includes SUBTLE hallucination where the model 
                 uses the distractor term as a real or valid concept in its 
                 own analysis without rejecting it, even if the answer sounds 
                 coherent or analytical.

"partial"     — The model hedges or expresses uncertainty about the premise 
                but does not explicitly reject it, and still engages with the 
                question as if the premise might be valid.

"refusal"     — The model declines to answer entirely without engaging with 
                the premise. "I don't know" and "I can't verify" count as 
                refusal.

"malformed"   — The response is not interpretable (gibberish, repeated tokens, 
                truncated output).

### DISTRACTOR VOCABULARY TEST
Before assigning a label, ask: "Does the model use the distractor term as if 
it names something real or valid?"
- YES + no explicit rejection → "hallucinated"
- YES + explicit rejection → "identifies"
- Hedges without rejecting → "partial"
- Does not engage at all → "refusal"

### EXAMPLES

HALLUCINATED (subtle):
  Answer: "12 Monkeys is often categorized as a 'non-quadrilateral' narrative 
  due to its unconventional storytelling approach."
  → Uses the distractor as a real category with no rejection. HALLUCINATED.

PARTIAL:
  Answer: "It's unclear what 'non-quadrilateral flag' conventions would 
  entail, but 12 Monkeys does use a non-linear structure..."
  → Hedges but still engages. PARTIAL.

IDENTIFIES:
  Answer: "The premise contains an error — 'non-quadrilateral flag' is not a 
  recognized narrative convention. 12 Monkeys is a dystopian sci-fi film."
  → Explicit rejection. IDENTIFIES.

---

## YOUR TASK

Given the input fields (full_answer, majority_answer, confidence_label):

1. Split full_answer on "|" to get 5 runs. Classify each run.
2. Count runs per label.
3. Classify majority_answer using the same labels.
4. Assign correctness: correct (identifies) / hallucinated / partial / 
   refusal / malformed.
5. Assign outcome_label using correctness + confidence_label:
   confident_correct | confident_hallucinated | unconfident_correct | 
   unconfident_hallucinated | confident_partial | unconfident_partial | 
   refusal | malformed
   (refusal and malformed override confidence)

---

## OUTPUT — RETURN THIS JSON AND NOTHING ELSE

{
  "per_run_classifications": [
    { "run": 1, "response_type": "identifies | hallucinated | partial | refusal | malformed" },
    { "run": 2, "response_type": "..." },
    { "run": 3, "response_type": "..." },
    { "run": 4, "response_type": "..." },
    { "run": 5, "response_type": "..." }
  ],
  "run_counts": {
    "identifies": 0,
    "hallucinated": 0,
    "partial": 0,
    "refusal": 0,
    "malformed": 0
  },
  "majority_response_type": "identifies | hallucinated | partial | refusal | malformed",
  "correctness": "correct | hallucinated | partial | refusal | malformed",
  "outcome_label": "confident_correct | confident_hallucinated | unconfident_correct | unconfident_hallucinated | confident_partial | unconfident_partial | refusal | malformed",
  "reasoning": "<2-4 sentences: identify the distractor, apply the Distractor Vocabulary Test, explain whether the model caught the error, note run variation>"
}"""

def build_user_message(rec: dict) -> str:
    return (
        f"Question: {rec['question']}\n"
        f"\nFull Answers (5 runs, pipe-separated): {rec['full_answer']}"
        f"\nMajority Answer: {rec['majority_answer']}"
        f"\nConsistency Score: {rec['consistency_score']}"
        f"\nConfidence Label: {rec['confidence_label']}"
        f"\nAvg Verbalized Confidence: {rec['verbalized_conf_avg']}"
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
            "identifies_and_corrects": 0, "identifies_only": 0, "hallucinated": 0,
            "partial": 0, "refusal": 0, "malformed": 0,
        },
        "majority_response_type": "error",
        "correctness": "error",
        "outcome_label": "error",
        "error_id_quality": "not_applicable",
        "reasoning": f"Evaluation failed: {error_msg}",
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global MODEL_ID
    
    parser = argparse.ArgumentParser(
        description="Evaluate near_true JSONL outputs via HF Inference API."
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

    stats = {
        "correct": 0, "hallucinated": 0, "partial": 0,
        "refusal": 0, "malformed": 0, "error": 0,
    }
    outcome_stats = {}

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

            correctness = eval_result.get("correctness", "error")
            stats[correctness] = stats.get(correctness, 0) + 1

            outcome = eval_result.get("outcome_label", "error")
            outcome_stats[outcome] = outcome_stats.get(outcome, 0) + 1

    total = sum(stats.values())
    print(f"\n-- Evaluation complete ({total} records) --")

    print("\n  CORRECTNESS:")
    for k, v in stats.items():
        pct = 100 * v / total if total else 0
        print(f"    {k:<22} {v:>5}  ({pct:.1f}%)")

    print("\n  OUTCOME LABELS:")
    for k, v in sorted(outcome_stats.items()):
        pct = 100 * v / total if total else 0
        print(f"    {k:<30} {v:>5}  ({pct:.1f}%)")

    print(f"\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()