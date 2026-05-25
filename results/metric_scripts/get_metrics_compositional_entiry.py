import json
import csv
from pathlib import Path

results_dir = Path("../evaluation_results/head_tail_rarity")
output_csv = Path("../metrics/head_tail_rarity_cumulative_metrics.csv")

output_csv.parent.mkdir(parents=True, exist_ok=True)

metrics = []

for file_path in results_dir.glob("*.jsonl"):
    model_name = file_path.stem
    
    counts = {
        "total_correct": 0,
        "total_wrong": 0,
        "confident_correct": 0,
        "unconfident_correct": 0,
        "confident_wrong": 0,
        "unconfident_wrong": 0,
        "total_refusals": 0,
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # Skip the metadata row if present
            if data.get("type") == "metadata":
                continue
            
            # 1. Track Refusals (using the top-level flag)
            if data.get("is_refusal", False):
                counts["total_refusals"] += 1
                
            # 2. Extract Evaluations
            eval_data = data.get("eval", {})
            if not eval_data:
                continue
                
            correctness = eval_data.get("correctness")
            outcome = eval_data.get("outcome_label", "")
            
            # Correctness Counts
            if correctness == "correct":
                counts["total_correct"] += 1
            elif correctness == "wrong":
                counts["total_wrong"] += 1
                
            # Confidence Breakdown
            if outcome == "confident_correct":
                counts["confident_correct"] += 1
            elif outcome == "unconfident_correct":
                counts["unconfident_correct"] += 1
            elif outcome == "confident_wrong":
                counts["confident_wrong"] += 1
            elif outcome == "unconfident_wrong":
                counts["unconfident_wrong"] += 1

    metrics.append({
        "Model": model_name,
        "Total Correct": counts["total_correct"],
        "Total Wrong": counts["total_wrong"],
        "Confident Correct": counts["confident_correct"],
        "Unconfident Correct": counts["unconfident_correct"],
        "Confident Wrong": counts["confident_wrong"],
        "Unconfident Wrong": counts["unconfident_wrong"],
        "Total Refusals": counts["total_refusals"]
    })

fieldnames = [
    "Model",
    "Total Correct",
    "Total Wrong",
    "Confident Correct",
    "Unconfident Correct",
    "Confident Wrong",
    "Unconfident Wrong",
    "Total Refusals"
]

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in metrics:
        writer.writerow(row)

print(f"Head/Tail Rarity metrics successfully written to => {output_csv.absolute()}")