"""
quick_viz_generation.py — Generates immediate data for the 24-plot publication suite.
1. Runs 1-epoch training for Laptop and Restaurant (ST1 and ST3).
2. Executes the 3 plotting scripts to finalize the suite.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_v4 import run_train
from plot_training import plot_training_curves, plot_uncertainty_docs
from plot_xai import plot_biaffine_heatmap, plot_token_attribution, plot_bio_confusion
from plot_performance_test import plot_bubble_bench, plot_radar_sota, plot_va_manifold, plot_rmse_buckets
from plot_base import load_data, OUTPUT_DIR

def generate_and_plot():
    domains = ["laptop", "restaurant"]
    subtasks = [1, 3] # Regression and Extraction
    
    print("=== Step 1: Quick Data Generation (1-Epoch Pass) ===")
    for dom in domains:
        for st in subtasks:
            print(f"Generating logs for ST{st} {dom}...")
            run_train(
                subtask=st,
                lang="eng",
                domain=dom,
                seed=42, # Use seed 42 for the quick viz
                max_epochs=1,
                debug=False,
                log_filename="combined_experiments.log"
            )

    print("\n=== Step 2: Finalizing All Plots (24-Plot Suite) ===")
    for dom in domains:
        run_name = f"st1_eng_{dom}_seed 42" # Note the seed is consistent now
        # Normalized run names (without space for training script internal naming logic)
        # Check actual run name consistency
        run_name = f"st1_eng_{dom}_seed42"
        h, p = load_data(run_name, dom)
        
        # Performance Suite
        print(f"Finalizing Performance Tests for {dom}...")
        plot_bubble_bench(dom)
        plot_radar_sota(dom)
        if p:
            plot_va_manifold(p, dom)
            plot_rmse_buckets(p, dom)
            
        # Training Suite
        print(f"Finalizing Training Visualization for {dom}...")
        if h:
            plot_training_curves(h, dom)
            plot_uncertainty_docs(h, dom)
            
        # XAI Suite
        print(f"Finalizing XAI Diagnostics for {dom}...")
        plot_biaffine_heatmap(dom)
        plot_token_attribution(dom)
        plot_bio_confusion(dom)

    print(f"\n MISSION COMPLETE. Check {OUTPUT_DIR} for the 24-plot suite.")

if __name__ == "__main__":
    generate_and_plot()
