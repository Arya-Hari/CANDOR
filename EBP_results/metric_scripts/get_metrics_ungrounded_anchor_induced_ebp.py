import argparse
import csv
import json
import re
from pathlib import Path


YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def extract_year(text: str) -> str:
    match = YEAR_RE.search(str(text or ""))
    return match.group(0) if match else ""


def classify_raw_row(data: dict) -> str:
    response_type = str(data.get("majority_response_type", "")).lower()
    if data.get("malformed") or response_type == "malformed":
        return "malformed"
    if data.get("is_refusal") or response_type == "refusal":
        return "refusal"

    predicted_year = extract_year(
        data.get("majority_answer")
        or data.get("model_answer")
        or data.get("stripped_answer")
    )
    truth_year = extract_year(data.get("ground_truth"))
    is_confident = str(data.get("confidence_label", "")).lower() == "confident" or bool(
        data.get("is_confident")
    )

    if predicted_year and truth_year and predicted_year == truth_year:
        return "confident_correct" if is_confident else "unconfident_correct"
    return "confident_wrong" if is_confident else "unconfident_wrong"


def extract(input_dir: str, output_csv: str):
    results_dir = Path(input_dir)
    metrics = []
    matched_files = list(sorted(results_dir.glob("*.jsonl")))

    if not matched_files:
        raise FileNotFoundError(
            f"No JSONL files found in {results_dir}. Pass the directory that contains evaluated anchor_induced EBP JSONL files."
        )

    for file_path in matched_files:
        model_name = file_path.stem
        counts = {
            "ug_confident_correct": 0,
            "ug_confident_wrong": 0,
            "ug_unconfident_correct": 0,
            "ug_unconfident_wrong": 0,
            "ug_refusal": 0,
            "ug_malformed": 0,
        }

        with open(file_path, encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                if data.get("type") == "metadata":
                    continue
                if "ungrounded" not in data:
                    continue

                ug_outcome = (
                    data.get("eval", {})
                    .get("ungrounded", {})
                    .get("outcome_label", "")
                ) or classify_raw_row(data)

                if ug_outcome == "confident_correct":
                    counts["ug_confident_correct"] += 1
                elif ug_outcome == "confident_wrong":
                    counts["ug_confident_wrong"] += 1
                elif ug_outcome == "unconfident_correct":
                    counts["ug_unconfident_correct"] += 1
                elif ug_outcome == "unconfident_wrong":
                    counts["ug_unconfident_wrong"] += 1
                elif ug_outcome == "refusal":
                    counts["ug_refusal"] += 1
                elif ug_outcome == "malformed":
                    counts["ug_malformed"] += 1

        N = 2819
        cc = counts["ug_confident_correct"]
        cw = counts["ug_confident_wrong"]
        uc = counts["ug_unconfident_correct"]
        uw = counts["ug_unconfident_wrong"]
        ref = counts["ug_refusal"]
        acc = (cc + uc) / N * 100
        bcs = (cc + uw) / N - 2 * (cw / N)

        metrics.append(
            {
                "Model": model_name,
                "UG_CC": cc,
                "UG_CW": cw,
                "UG_UC": uc,
                "UG_UW": uw,
                "UG_Refusal": ref,
                "UG_Acc%": round(acc, 1),
                "UG_BCS": round(bcs, 3),
            }
        )
        print(
            f"{model_name}: CC={cc}, CW={cw}, UC={uc}, UW={uw}, "
            f"Ref={ref}, Acc%={acc:.1f}, BCS={bcs:.3f}"
        )

    fieldnames = ["Model", "UG_CC", "UG_CW", "UG_UC", "UG_UW", "UG_Refusal", "UG_Acc%", "UG_BCS"]
    with open(output_csv, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    print(f"\nSaved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="../evaluation_results/mixedfact_EBP/")
    parser.add_argument("--output-csv", default="../metrics/mixedfact_EBP_ungrounded_metrics.csv")
    args = parser.parse_args()
    extract(args.input_dir, args.output_csv)