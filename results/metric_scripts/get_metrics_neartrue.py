import json
import csv
from pathlib import Path

results_dir = Path("../evaluation_results/near_true")
output_csv = Path("../metrics/near_true_cumulative_metrics.csv")

output_csv.parent.mkdir(parents=True, exist_ok=True)

metrics = []

for file_path in results_dir.glob("*.jsonl"):
    model_name = file_path.stem
    
    counts = {
        "confident_correct": 0,
        "unconfident_correct": 0,
        "confident_wrong": 0,
        "unconfident_wrong": 0,
        "refusals": 0,
        "partial_answers": 0,
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            if data.get("type") == "metadata":
                continue
                
            eval_data = data.get("eval", {})
            outcome = eval_data.get("outcome_label", "")
            correctness = eval_data.get("correctness", "")
            
            # 1. Map the outcome to our confidence buckets
            # Note: For near_true, "wrong" is usually stored as "hallucinated"
            if outcome == "confident_correct":
                counts["confident_correct"] += 1
            elif outcome == "unconfident_correct":
                counts["unconfident_correct"] += 1
            elif outcome in ["confident_hallucinated", "confident_wrong"]:
                counts["confident_wrong"] += 1
            elif outcome in ["unconfident_hallucinated", "unconfident_wrong"]:
                counts["unconfident_wrong"] += 1
            elif outcome == "refusal":
                counts["refusals"] += 1
                
            # 2. Track Partials and explicitly correct refusals fallback
            if correctness == "partial":
                counts["partial_answers"] += 1
            if correctness == "refusal" and outcome != "refusal":
                counts["refusals"] += 1

    # Calculate overall totals
    total_correct = counts["confident_correct"] + counts["unconfident_correct"]
    total_wrong = counts["confident_wrong"] + counts["unconfident_wrong"]

    metrics.append({
        "Model": model_name,
        "Total Correct": total_correct,
        "Total Wrong": total_wrong,
        "Partial Answers": counts["partial_answers"],
        "Refusals": counts["refusals"],
        "Confident Correct": counts["confident_correct"],
        "Confident Wrong": counts["confident_wrong"],
        "Unconfident Correct": counts["unconfident_correct"],
        "Unconfident Wrong": counts["unconfident_wrong"],
    })

fieldnames = [
    "Model", 
    "Total Correct", 
    "Total Wrong", 
    "Partial Answers",
    "Refusals",
    "Confident Correct", 
    "Confident Wrong", 
    "Unconfident Correct", 
    "Unconfident Wrong",
]

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in metrics:
        writer.writerow(row)

print(f"Near-True metrics successfully written to => {output_csv.absolute()}")