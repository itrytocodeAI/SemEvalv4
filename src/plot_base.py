"""
plot_base.py — Shared utilities and styling for DimABSA V4 visualization.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Professional Paper Aesthetics
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_context("paper")
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11})

# Branding Colors
EMERALD_GREEN = "#50C878" # V4-SOTA
V3_COLOR = "#FF7F50"       # Coral
BASELINE_COLOR = "#A9A9A9" # Gray

# Paths
OUTPUT_DIR = Path("plots/v4_publication_suite")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("logs")

def load_data(run_name, domain):
    """Load history and validation pairs."""
    hist_path = LOG_DIR / f"{run_name}_history.json"
    pairs_path = LOG_DIR / f"{run_name}_val_pairs.json"
    
    history, pairs = [], []
    if hist_path.exists():
        with open(hist_path, "r") as f: history = json.load(f)
    if pairs_path.exists():
        with open(pairs_path, "r") as f: pairs = json.load(f)
    return history, pairs
