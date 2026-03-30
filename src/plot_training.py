"""
plot_training.py — Handles convergence and uncertainty evolution.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_base import EMERALD_GREEN, OUTPUT_DIR, load_data

def plot_training_curves(history, domain):
    """Training Loss & Convergence."""
    if not history: return
    df = pd.DataFrame(history)
    
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Total Train Loss', color=EMERALD_GREEN, marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='Total Val Loss', color='gray', linestyle='--', marker='s')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"V4 Convergence - {domain.capitalize()}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_DIR / f"v4_training_curves_{domain}.png")
    plt.close()

def plot_uncertainty_docs(history, domain):
    """Plot σv and σa evolution."""
    if not history or 'sigma_v' not in history[0]: return
    df = pd.DataFrame(history)
    
    plt.figure(figsize=(8, 5))
    # Add minor jitter to prevent perfect visual overlap if they both remain at 1.0
    v_jitter = df['sigma_v'] + (np.random.rand(len(df)) * 0.002 - 0.001) if df['sigma_v'].nunique() == 1 else df['sigma_v']
    a_jitter = df['sigma_a'] + (np.random.rand(len(df)) * 0.002 - 0.001) if df['sigma_a'].nunique() == 1 else df['sigma_a']
    
    plt.plot(df['epoch'], v_jitter, color=EMERALD_GREEN, marker='o', label=r'Valence $\sigma_v$')
    plt.plot(df['epoch'], a_jitter, color='orange', marker='s', label=r'Arousal $\sigma_a$')
    
    if df['sigma_v'].nunique() <= 1 and len(df) <= 3:
        plt.ylim(0.0, 2.0) # Force perspective so 1.0 looks like baseline, not entire graph
        plt.annotate('UW_DELAY_EPOCHS=3\nWeighting activates Epoch > 3', xy=(len(df)/2, 1.1), ha='center')
        
    plt.xlabel("Epoch")
    plt.ylabel(r"Uncertainty Parameter $\sigma$")
    plt.title(f"Homoscedastic Uncertainty Evolution - {domain.capitalize()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"v4_uncertainty_docs_{domain}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    for dom in ["laptop", "restaurant"]:
        run_name = f"st1_eng_{dom}_seed42"
        h, _ = load_data(run_name, dom)
        if h:
            print(f"Plotting Training for {dom}...")
            plot_training_curves(h, dom)
            plot_uncertainty_docs(h, dom)
