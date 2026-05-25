import json
import csv
from pathlib import Path

results_dir = Path("../evaluation_results/long_tail")
output_csv = Path("../metrics/long_tail_cumulative_metrics.csv")

metrics = []

VERBALIZED_CONFIDENCE_THRESHOLD = 80.0
CONSISTENCY_UNCONFIDENT_THRESHOLD = 0.8

# Process each JSONL file in the directory
for file_path in results_dir.glob("*.jsonl"):
    model_name = file_path.stem
    
    counts = {
        "confident_correct": 0,
        "unconfident_correct": 0,
        "confident_wrong": 0,
        "unconfident_wrong": 0,
        "refusal": 0,
        "verb_confident_but_consistancy_unconfident": 0,
        "correct_imprecise": 0
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # Skip the initial metadata row
            if data.get("type") == "metadata":
                continue
                
            # 1. Base outcome counts
            eval_data = data.get("eval", {})
            outcome = eval_data.get("outcome_label")
            correctness = eval_data.get("correctness")
            
            if outcome in counts:
                counts[outcome] += 1
                
            # Count correct but imprecise classifications
            if correctness == "correct_imprecise":
                counts["correct_imprecise"] += 1
                
            # 2. Verbalized Confidence Confident vs Consistency Unconfident
            consistency = data.get("consistency_score", 1.0)
            avg_verb_conf = data.get("verbalized_conf_avg")
            
            # Coalesce None to 0.0
            if avg_verb_conf is None:
                avg_verb_conf = 0.0
                
            # Sometimes confidence_label is recorded directly in the JSON
            confidence_label = data.get("confidence_label")
            
            is_verb_confident = avg_verb_conf >= VERBALIZED_CONFIDENCE_THRESHOLD
            
            is_consistency_unconfident = (
                consistency < CONSISTENCY_UNCONFIDENT_THRESHOLD 
                or confidence_label == "unconfident"
                or confidence_label == "uncertain"
            )
            
            if is_verb_confident and is_consistency_unconfident:
                counts["verb_confident_but_consistancy_unconfident"] += 1
                
    # Calculate cumulative correctness
    total_correct = counts["confident_correct"] + counts["unconfident_correct"]
    total_wrong = counts["confident_wrong"] + counts["unconfident_wrong"]
    
    metrics.append({
        "Model": model_name,
        "Total Correct": total_correct,
        "Total Wrong": total_wrong,
        "Correct Imprecise": counts["correct_imprecise"],
        "Confident Correct": counts["confident_correct"],
        "Confident Wrong": counts["confident_wrong"],
        "Unconfident Correct": counts["unconfident_correct"],
        "Unconfident Wrong": counts["unconfident_wrong"],
        "Refusals": counts["refusal"],
        "Verb Confident but Consist Unconfident": counts["verb_confident_but_consistancy_unconfident"]
    })

fieldnames = [
    "Model", 
    "Total Correct", 
    "Total Wrong", 
    "Correct Imprecise",
    "Confident Correct", 
    "Confident Wrong", 
    "Unconfident Correct", 
    "Unconfident Wrong", 
    "Refusals",
    "Verb Confident but Consist Unconfident"
]

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in metrics:
        writer.writerow(row)

print(f"Metrics successfully written to => {output_csv.absolute()}")