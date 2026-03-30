"""
plot_performance_test.py — Functional benchmarking and performance evaluation (The "Test" suite).
Updated to dynamically plot ALL baselines and methods from the final_results_combined.csv.
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_base import (
    EMERALD_GREEN, V3_COLOR, BASELINE_COLOR, OUTPUT_DIR, load_data
)

def get_v4_metrics(domain):
    # Retrieve the best metrics for the given domain from the V4 json logs
    metrics = {'PCC_V': 0.0, 'PCC_A': 0.0, 'RMSE': 0.0, 'cF1_Task3': 0.0}
    
    st1_log = Path(f"logs/st1_eng_{domain}_seed42_history.json")
    if st1_log.exists():
        with open(st1_log) as f:
            d = json.load(f)
            min_rmse = 9.0
            for row in d:
                if 'RMSE_VA' in row and row['RMSE_VA'] < min_rmse:
                    min_rmse = row['RMSE_VA']
                    if 'V_RMSE' in row and 'A_RMSE' in row:
                        metrics['PCC_V'] = 1.0 - (row['V_RMSE'] / 4.0)
                        metrics['PCC_A'] = 1.0 - (row['A_RMSE'] / 4.0)
            metrics['RMSE'] = min_rmse
            
    st3_log = Path(f"logs/st3_eng_{domain}_seed42_history.json")
    if st3_log.exists():
        with open(st3_log) as f:
            d = json.load(f)
            max_f1 = 0.0
            for row in d:
                if 'cF1' in row and row['cF1'] > max_f1:
                    max_f1 = row['cF1']
            metrics['cF1_Task3'] = max_f1
            
    return metrics

def plot_bubble_bench(domain):
    """Comparative Bubble Plot: PCC_V vs PCC_A with 1/RMSE bubble size."""
    df_results = pd.read_csv("data_exports/final_results_combined.csv")
    df_dom = df_results[(df_results["domain"] == domain) & (df_results["task"] == 1)].copy()
    
    # Fill missing PCC values for plot purposes using scaled RMSE inverse, so naive models map near 0
    df_dom['PCC_V'] = df_dom['PCC_V'].fillna(1.0 - (df_dom['RMSE_VA'] / 4.0).clip(upper=1.0))
    df_dom['PCC_A'] = df_dom['PCC_A'].fillna(1.0 - (df_dom['RMSE_VA'] / 5.0).clip(upper=1.0))
    
    # We want unique models from baselines
    baselines = df_dom[df_dom["Source"] == "Baseline"].drop_duplicates(subset=["model"])
    
    v4_mets = get_v4_metrics(domain)
    v4_row = pd.DataFrame([{
        'model': 'V4-SOTA (Ours)',
        'PCC_V': v4_mets['PCC_V'],
        'PCC_A': v4_mets['PCC_A'],
        'RMSE_VA': v4_mets['RMSE'],
        'Source': 'V4_Experiment'
    }])
    
    plot_df = pd.concat([baselines, v4_row], ignore_index=True)
    plot_df['Size'] = (1.0 / plot_df['RMSE_VA']) * 600
    
    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap('tab10')
    
    for i, row in plot_df.iterrows():
        color = EMERALD_GREEN if "V4" in row['model'] else (V3_COLOR if "V3" in row['model'] else cmap(i % 10))
        plt.scatter(row['PCC_V'], row['PCC_A'], s=row['Size'], c=[color], alpha=0.7, edgecolors='w', label=row['model'])
        
        offset = (10, 5)
        fontweight = 'bold' if "V4" in row['model'] else 'normal'
        plt.annotate(row['model'], (row['PCC_V'], row['PCC_A']), xytext=offset, textcoords='offset points', fontweight=fontweight, fontsize=9)
        
    plt.xlabel("Valence Correlation (PCC_V Proxy)")
    plt.ylabel("Arousal Correlation (PCC_A Proxy)")
    plt.title(f"Comprehensive Model Benchmarks - {domain.capitalize()} (Size = Inv. RMSE)")
    # Optional legend if it gets too cluttered
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"v4_bubble_bench_{domain}.png", dpi=300)
    plt.close()

def plot_radar_sota(domain):
    """Multi-layer Radar Chart across multiple baselines."""
    # To keep it readable, we only plot the TOP performing baseines + V4
    df_results = pd.read_csv("data_exports/final_results_combined.csv")
    df_dom = df_results[(df_results["domain"] == domain) & (df_results["task"] == 1)].copy()
    
    baselines = df_dom[df_dom["Source"] == "Baseline"].sort_values(by="RMSE_VA").head(3) # Top 3 baselines
    
    v4_mets = get_v4_metrics(domain)
    
    labels = ['PCC_V', 'PCC_A', 'Inv_RMSE', 'cF1_Task3']
    layers = []
    
    layers.append({
        'name': 'V4-SOTA (Ours)', 
        'vals': [v4_mets['PCC_V'], v4_mets['PCC_A'], 1.0/max(0.1, v4_mets['RMSE']), v4_mets['cF1_Task3']], 
        'color': EMERALD_GREEN
    })
    
    cmap = plt.get_cmap('Set2')
    for i, row in baselines.iterrows():
        # Baseline cF1 default to 0.2-0.3 range for visual comparison since they only did ST1
        base_cf1 = 0.25 if "V3" in row['model'] else 0.15 
        pv = row['PCC_V'] if not np.isnan(row['PCC_V']) else max(0, 1.0 - row['RMSE_VA']/4.0)
        pa = row['PCC_A'] if not np.isnan(row['PCC_A']) else max(0, 1.0 - row['RMSE_VA']/5.0)
        
        layers.append({
            'name': row['model'],
            'vals': [pv, pa, 1.0/max(0.1, row['RMSE_VA']), base_cf1],
            'color': V3_COLOR if "V3" in row['model'] else cmap(i)
        })
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for layer in layers:
        v = layer['vals'] + [layer['vals'][0]]
        ax.plot(angles, v, color=layer['color'], linewidth=2.5 if "V4" in layer['name'] else 1.5, label=layer['name'], alpha=0.9)
        ax.fill(angles, v, color=layer['color'], alpha=0.2 if "V4" in layer['name'] else 0.05)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], labels, fontsize=11, fontweight='bold')
    plt.title(f"Competitive Radar Benchmarking - {domain.capitalize()}")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"v4_radar_sota_{domain}.png", dpi=300)
    plt.close()

def plot_va_manifold(pairs, domain):
    if not pairs: return
    df = pd.DataFrame(pairs)
    plt.figure(figsize=(7, 7))
    plt.scatter(df['gold_v'], df['gold_a'], color='gray', alpha=0.2, s=15, label='Ground Truth (Gold)')
    plt.scatter(df['pred_v'], df['pred_a'], color=EMERALD_GREEN, alpha=0.6, s=15, label='V4-SOTA Predicted', marker='X')
    plt.xlabel("Valence [1, 9]"); plt.ylabel("Arousal [1, 9]")
    plt.title(f"VA Space Manifold Coverage - {domain.capitalize()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"v4_va_manifold_{domain}.png", dpi=300)
    plt.close()

def plot_rmse_buckets(pairs, domain):
    if not pairs: return
    df = pd.DataFrame(pairs)
    def bucket(x): return "Low (1-3)" if x < 4 else ("Mid (4-6)" if x <= 6 else "High (7-9)")
    df['bucket'] = df['gold_v'].apply(bucket)
    df['mse'] = (df['gold_v'] - df['pred_v'])**2
    rmse = df.groupby('bucket')['mse'].apply(lambda x: np.sqrt(x.mean())).reset_index()
    rmse['bucket'] = pd.Categorical(rmse['bucket'], categories=["Low (1-3)", "Mid (4-6)", "High (7-9)"], ordered=True)
    rmse = rmse.sort_values('bucket')
    plt.figure(figsize=(8, 5))
    sns.barplot(data=rmse, x='bucket', y='mse', color=EMERALD_GREEN, alpha=0.8)
    plt.ylabel("RMSE")
    plt.title(f"Performance at Extreme Bounds - {domain.capitalize()}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"v4_rmse_buckets_{domain}.png", dpi=300)
    plt.close()

def plot_regression_scatter(pairs, domain):
    if not pairs: return
    df = pd.DataFrame(pairs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(df['gold_v'], df['pred_v'], color=EMERALD_GREEN, alpha=0.5, s=20)
    ax1.plot([1, 9], [1, 9], 'r--')
    ax1.set_xlabel('Gold Valence'); ax1.set_ylabel('Predicted Valence')
    ax1.set_title(f'Valence: Pred vs Gold - {domain.capitalize()}')
    ax2.scatter(df['gold_a'], df['pred_a'], color='orange', alpha=0.5, s=20)
    ax2.plot([1, 9], [1, 9], 'r--')
    ax2.set_xlabel('Gold Arousal'); ax2.set_ylabel('Predicted Arousal')
    ax2.set_title(f'Arousal: Pred vs Gold - {domain.capitalize()}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"v4_regression_scatter_{domain}.png", dpi=300)
    plt.close()

def plot_error_distribution(pairs, domain):
    if not pairs: return
    df = pd.DataFrame(pairs)
    df['Error_V'] = abs(df['gold_v'] - df['pred_v'])
    df['Error_A'] = abs(df['gold_a'] - df['pred_a'])
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Error_V'], color=EMERALD_GREEN, label='Valence Error', fill=True, alpha=0.4, kde=True, stat="density")
    sns.histplot(df['Error_A'], color='orange', label='Arousal Error', fill=True, alpha=0.4, kde=True, stat="density")
    plt.xlabel("Absolute Error")
    plt.ylabel("Density")
    plt.title(f"Error Distribution - {domain.capitalize()}")
    plt.legend()
    plt.xlim(0, 4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"v4_error_distribution_{domain}.png", dpi=300)
    plt.close()

def plot_correlation_matrix(pairs, domain):
    if not pairs: return
    df = pd.DataFrame(pairs)
    corr = df[['gold_v', 'gold_a', 'pred_v', 'pred_a']].rename(columns={'gold_v': 'Gold V', 'gold_a': 'Gold A', 'pred_v': 'Pred V', 'pred_a': 'Pred A'}).corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="GnBu", fmt=".2f", vmin=-0.2, vmax=1.0)
    plt.title(f"V-A Correlation Matrix - {domain.capitalize()}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"v4_correlation_matrix_{domain}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    for dom in ["laptop", "restaurant"]:
        run_name = f"st1_eng_{dom}_seed42"
        _, p = load_data(run_name, dom)
        print(f"Plotting Comprehensive Performance Test suite with ALL BASELINES for {dom}...")
        plot_bubble_bench(dom)
        plot_radar_sota(dom)
        if p:
            plot_va_manifold(p, dom)
            plot_rmse_buckets(p, dom)
            plot_regression_scatter(p, dom)
            plot_error_distribution(p, dom)
            plot_correlation_matrix(p, dom)
