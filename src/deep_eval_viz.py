"""
deep_eval_viz.py
Reads the results from deep_eval_runner.py and selects the best seed to plot.
Generates final publication-ready training and uncertainty curves.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.plot_base import EMERALD_GREEN, OUTPUT_DIR, load_data
from src.plot_training import plot_training_curves, plot_uncertainty_docs
from src.config_v4 import LOG_DIR

def find_best_seed(csv_path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    # Select seed with lowest RMSE_VA
    best_row = df.loc[df["RMSE_VA"].idxmin()]
    return int(best_row["Seed"]), best_row["RMSE_VA"]

def generate_deep_plots():
    print("=== Generating Deep Performance Visuals from Best Seed ===")
    
    for domain in ["laptop", "restaurant"]:
        csv_path = LOG_DIR / f"deep_run_{domain}.csv"
        best_info = find_best_seed(csv_path)
        
        if best_info:
            seed, score = best_info
            print(f"Domain: {domain.upper()} | Best Seed: {seed} (RMSE: {score:.4f})")
            
            run_name = f"st1_eng_{domain}_seed{seed}"
            history, _ = load_data(run_name, domain)
            
            if history:
                plot_training_curves(history, domain)
                plot_uncertainty_docs(history, domain)
                print(f"Successfully updated charts for {domain} (10 Epochs).")
            else:
                print(f"Failed to load history for {run_name}.")
        else:
            print(f"No deep run results found for {domain}.")

if __name__ == "__main__":
    generate_deep_plots()
