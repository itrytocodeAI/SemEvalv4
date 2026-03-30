"""
ensemble_predictions.py — A fast utility script to ensemble all 5 seeds into a single SOTA 
predictions file for the SemEval-2026 leaderboard submission.

This is a post-processing script. It does NOT require model weights or neural network training.
It simply takes the generated `predictions.jsonl` files from the multiple seeds in the 
`checkpoints/` or `logs/` directory and averages them (Soft Voting for ST3, Mean for ST1).
"""

import sys
import json
import glob
from pathlib import Path
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config_v4 import OUTPUT_DIR

def ensemble_subtask_1(domain):
    """Average the V/A values for Regression."""
    print(f"Ensembling Subtask 1 for {domain.capitalize()}...")
    pattern = str(OUTPUT_DIR / f"st1_eng_{domain}_seed*_predictions.jsonl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"  No ST1 prediction files found for {domain}.")
        return
        
    print(f"  Found {len(files)} ST1 seed files.")
    
    # Structure: ID -> Aspect -> [V, A] values across seeds
    ensemble_data = defaultdict(lambda: defaultdict(list))
    
    for f_path in files:
        with open(f_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip(): continue
                obj = json.loads(line)
                i_id = obj["ID"]
                for asp_entry in obj.get("Aspect_VA", []):
                    aspect = asp_entry["Aspect"]
                    v, a = map(float, asp_entry["VA"].split("#"))
                    ensemble_data[i_id][aspect].append((v, a))
                    
    # Write ensembled output
    out_path = OUTPUT_DIR / f"st1_eng_{domain}_ensemble_predictions.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for i_id, aspects in ensemble_data.items():
            aspect_va_list = []
            for aspect, va_list in aspects.items():
                avg_v = sum(v for v, a in va_list) / len(va_list)
                avg_a = sum(a for v, a in va_list) / len(va_list)
                # Cap securely inside 1.0 to 9.0 bounds
                avg_v = max(1.0, min(9.0, avg_v))
                avg_a = max(1.0, min(9.0, avg_a))
                aspect_va_list.append({
                    "Aspect": aspect,
                    "VA": f"{avg_v:.4f}#{avg_a:.4f}"
                })
            obj = {
                "ID": i_id,
                "Language": "English",
                "Domain": domain.capitalize(),
                "Aspect_VA": aspect_va_list
            }
            fh.write(json.dumps(obj) + "\n")
            
    print(f"  Saved {out_path.name}")

if __name__ == "__main__":
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for dom in ["laptop", "restaurant"]:
        ensemble_subtask_1(dom)
        print()
    
    print("\nEnsembling complete. You can now submit these engineered files to SemEval!")
