import json
import csv
from pathlib import Path

results_dir = Path("../evaluation_results/anchor_induced")
output_csv = Path("../metrics/anchor_induced_cumulative_metrics.csv")

output_csv.parent.mkdir(parents=True, exist_ok=True)

metrics = []

for file_path in results_dir.glob("*.jsonl"):
    model_name = file_path.stem
    
    counts = {
        # Base Correctness/Confidence (using the default variant as base)
        "confident_correct": 0,
        "unconfident_correct": 0,
        "confident_wrong": 0,
        "unconfident_wrong": 0,
        
        # Turnarounds
        "deflection_turnarounds": 0,
        "refusal_turnarounds": 0,
        
        # Shift Accumulators (Summed across the file, will be averaged later)
        "total_consistency_shift": 0.0,
        "total_verbalized_confidence_shift": 0.0,
        "count_valid_shifts": 0
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # Skip the metadata row
            if data.get("type") == "metadata":
                continue
            
            if "ungrounded" in data:
                # 1. Base Correctness & Confidence (Using the default variant outcome)
                # (Mixed fact datasets usually have the default evaluation inside the main "eval" or inside the base dataset)
                # But here, we can check data.get("eval", {}) -> this usually corresponds to the default target
                
                # Check where 'default' eval is. It typically resides at data["eval"]["default"] or data["eval"]
                eval_data = data.get("eval", {})
                
                # Some candor mixed_fact pipelines nest default within eval, let's check
                default_eval = eval_data.get("default", eval_data)
                
                outcome = default_eval.get("outcome_label", "")
                if outcome == "confident_correct":
                    counts["confident_correct"] += 1
                elif outcome == "unconfident_correct":
                    counts["unconfident_correct"] += 1
                elif outcome == "confident_wrong":
                    counts["confident_wrong"] += 1
                elif outcome == "unconfident_wrong":
                    counts["unconfident_wrong"] += 1
                
                # 2. Extract Cross-Variant shifts and turnarounds
                cross = eval_data.get("cross_variant", {})
                if cross:
                    cons_shift = cross.get("consistency_shift")
                    verb_conf_shift = cross.get("verbalized_confidence_shift")
                    
                    if cons_shift is not None and verb_conf_shift is not None:
                        try:
                            counts["total_consistency_shift"] += float(cons_shift)
                            counts["total_verbalized_confidence_shift"] += float(verb_conf_shift)
                            counts["count_valid_shifts"] += 1
                        except (ValueError, TypeError):
                            # Skip values that are "N/A" or cannot be converted to float
                            pass
                    
                    if cross.get("deflects_ungrounded_not_default"):
                        counts["deflection_turnarounds"] += 1
                        
                    if cross.get("refuses_ungrounded_not_default"):
                        counts["refusal_turnarounds"] += 1

    # Calculate overall totals
    total_correct = counts["confident_correct"] + counts["unconfident_correct"]
    total_wrong = counts["confident_wrong"] + counts["unconfident_wrong"]
    
    # Calculate Averages for Shifts
    avg_consist_shift = 0.0
    avg_verb_conf_shift = 0.0
    if counts["count_valid_shifts"] > 0:
        avg_consist_shift = counts["total_consistency_shift"] / counts["count_valid_shifts"]
        avg_verb_conf_shift = counts["total_verbalized_confidence_shift"] / counts["count_valid_shifts"]

    metrics.append({
        "Model": model_name,
        "Total Correct (Default)": total_correct,
        "Total Wrong (Default)": total_wrong,
        "Confident Correct": counts["confident_correct"],
        "Confident Wrong": counts["confident_wrong"],
        "Unconfident Correct": counts["unconfident_correct"],
        "Unconfident Wrong": counts["unconfident_wrong"],
        "Avg Consistency Shift": round(avg_consist_shift, 4),
        "Avg Verb Conf Shift": round(avg_verb_conf_shift, 4),
        "Deflection Turnarounds": counts["deflection_turnarounds"],
        "Refusal Turnarounds": counts["refusal_turnarounds"]
    })

fieldnames = [
    "Model", 
    "Total Correct (Default)", 
    "Total Wrong (Default)", 
    "Confident Correct", 
    "Confident Wrong", 
    "Unconfident Correct", 
    "Unconfident Wrong", 
    "Avg Consistency Shift",
    "Avg Verb Conf Shift",
    "Deflection Turnarounds",
    "Refusal Turnarounds"
]

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in metrics:
        writer.writerow(row)

print(f"Mixed-Facts metrics successfully written to => {output_csv.absolute()}")