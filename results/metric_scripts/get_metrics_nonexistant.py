import json
import csv
from pathlib import Path

results_dir = Path("../evaluation_results/non_existant")
output_csv = Path("../metrics/non_existant_cumulative_metrics.csv")

output_csv.parent.mkdir(parents=True, exist_ok=True)

metrics = []

for file_path in results_dir.glob("*.jsonl"):
    model_name = file_path.stem
    
    counts = {
        "total_correct_refusal": 0,
        "total_hallucinated": 0,
        "total_malformed": 0,
        
        # Confidence Breakdown for Refusals (Correct)
        "confident_correct_refusal": 0,
        "unconfident_correct_refusal": 0,
        
        # Confidence Breakdown for Hallucinations (Wrong)
        "confident_hallucinated": 0,
        "unconfident_hallucinated": 0,
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # Skip the metadata row if present
            if data.get("type") == "metadata":
                continue
                
            # Extract Evaluations
            eval_data = data.get("eval", {})
            if not eval_data:
                continue
                
            correctness = eval_data.get("correctness")
            outcome = eval_data.get("outcome_label", "")
            
            # 1. Base Correctness Counts
            if correctness in ("correct_refusal", "correct") or eval_data.get("majority_response_type") == "correct_refusal":
                counts["total_correct_refusal"] += 1
            elif correctness == "hallucinated":
                counts["total_hallucinated"] += 1
            elif correctness == "malformed":
                counts["total_malformed"] += 1
                
            # 2. Confidence Breakdown
            if outcome == "confident_correct_refusal":
                counts["confident_correct_refusal"] += 1
            elif outcome == "unconfident_correct_refusal":
                counts["unconfident_correct_refusal"] += 1
            elif outcome == "confident_hallucinated":
                counts["confident_hallucinated"] += 1
            elif outcome == "unconfident_hallucinated":
                counts["unconfident_hallucinated"] += 1

    metrics.append({
        "Model": model_name,
        "Total Correct (Refused)": counts["total_correct_refusal"],
        "Total Hallucinated (Wrong)": counts["total_hallucinated"],
        "Total Malformed": counts["total_malformed"],
        "Confident Correct Refusal": counts["confident_correct_refusal"],
        "Unconfident Correct Refusal": counts["unconfident_correct_refusal"],
        "Confident Hallucinated": counts["confident_hallucinated"],
        "Unconfident Hallucinated": counts["unconfident_hallucinated"]
    })

fieldnames = [
    "Model",
    "Total Correct (Refused)",
    "Total Hallucinated (Wrong)",
    "Total Malformed",
    "Confident Correct Refusal",
    "Unconfident Correct Refusal",
    "Confident Hallucinated",
    "Unconfident Hallucinated",
]

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in metrics:
        writer.writerow(row)

print(f"Non-Existant metrics successfully written to => {output_csv.absolute()}")