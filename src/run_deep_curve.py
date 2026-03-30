"""
run_deep_curve.py 
Dedicated launcher to train the V4-SOTA model for 10 epochs on the optimal 
Seed 42 for both the Laptop and Restaurant domains. 

This script explicitly forces the long-tail convergence capture necessary 
for publication-ready training curves and uncertainty evolution visualization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train_v4 import run_train
from src.plot_training import plot_training_curves, plot_uncertainty_docs
from src.config_v4 import LOG_DIR
from src.plot_base import load_data

def run_deep_training():
    print("=" * 60)
    print("Initiating Deep Training Curve Extraction (10 Epochs)")
    print("=" * 60)
    
    seed = 42
    
    for domain in ["laptop", "restaurant"]:
        print(f"\n[DEEP RUN] Starting ST1 {domain.upper()} Domain (Seed {seed})")
        print("Note: This will take approximately 45-60 minutes per domain...")
        
        # 1. Run full 10 epochs exclusively to graph convergence
        try:
            run_train(
                subtask=1,
                lang="eng",
                domain=domain,
                seed=seed,
                max_epochs=10,  # Explicitly force deep epoch search
                debug=False,
                log_filename=f"deep_curve_{domain}.log"
            )
            
            # 2. Extract logged data and immediately refresh graphics
            print(f"\n[DEEP RUN] completed. Re-rendering publication charts for {domain}...")
            
            run_name = f"st1_eng_{domain}_seed{seed}"
            history, _ = load_data(run_name, domain)
            
            if history:
                plot_training_curves(history, domain)
                plot_uncertainty_docs(history, domain)
                print(f"-> Successfully updated v4_training_curves_{domain}.png")
                print(f"-> Successfully updated v4_uncertainty_docs_{domain}.png")
                
        except Exception as e:
            print(f"[DEEP RUN ERROR] Failed to complete 10 epoch run on {domain}: {e}")

if __name__ == "__main__":
    run_deep_training()
    print("\n[DEEP RUN COMPLETE] All publication training curves have been extended!")
