import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config_v4 import DATA_ROOT, LOG_DIR
from src.plot_base import EMERALD_GREEN

WORKSPACE = Path(r"d:\semeval_v5")
EXPORT_DIR = WORKSPACE / "data_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
EDA_PLOT_DIR = WORKSPACE / "plots" / "eda"
EDA_PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl_to_df(filepath, split):
    records = []
    if not filepath.exists():
        return pd.DataFrame()
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            obj = json.loads(line)
            text = obj.get("Text", "")
            id_ = obj.get("ID", f"unknown_{idx}")
            
            # Subtask 1
            if "Aspect_VA" in obj:
                for entry in obj["Aspect_VA"]:
                    v_str, a_str = entry["VA"].split("#")
                    records.append({
                        "ID": id_, "Split": split, "Text": text,
                        "Aspect": entry.get("Aspect", ""), "Opinion": "", "Category": "",
                        "Valence": float(v_str), "Arousal": float(a_str), "Task": "ST1"
                    })
            elif "Aspect" in obj: # ST1 Test
                for entry in obj["Aspect"]:
                    records.append({
                        "ID": id_, "Split": split, "Text": text,
                        "Aspect": entry, "Opinion": "", "Category": "",
                        "Valence": None, "Arousal": None, "Task": "ST1"
                    })
            
            # Subtask 2
            if "Triplet" in obj:
                for entry in obj["Triplet"]:
                    if "VA" in entry:
                        v_str, a_str = entry["VA"].split("#")
                    else:
                        v_str, a_str = None, None
                        
                    records.append({
                        "ID": id_, "Split": split, "Text": text,
                        "Aspect": entry.get("Aspect", ""), "Opinion": entry.get("Opinion", ""), "Category": "",
                        "Valence": float(v_str) if v_str else None, "Arousal": float(a_str) if a_str else None, "Task": "ST2"
                    })
                    
            # Subtask 3
            if "Quadruplet" in obj:
                for entry in obj["Quadruplet"]:
                    if "VA" in entry:
                        v_str, a_str = entry["VA"].split("#")
                    else:
                        v_str, a_str = None, None
                        
                    records.append({
                        "ID": id_, "Split": split, "Text": text,
                        "Aspect": entry.get("Aspect", ""), "Opinion": entry.get("Opinion", ""), "Category": entry.get("Category", ""),
                        "Valence": float(v_str) if v_str else None, "Arousal": float(a_str) if a_str else None, "Task": "ST3"
                    })
                    
    return pd.DataFrame(records)

def create_data_exports():
    print("Creating Training, Dev, and Test CSVs...")
    all_train = []
    all_dev = []
    all_test = []
    
    # Loop over subtasks, languages, and domains
    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith(".jsonl"):
                path = Path(root) / file
                if "train" in file:
                    all_train.append(load_jsonl_to_df(path, "train"))
                elif "dev" in file:
                    all_dev.append(load_jsonl_to_df(path, "dev"))
                elif "test" in file:
                    all_test.append(load_jsonl_to_df(path, "test"))
                    
    df_train = pd.concat([df for df in all_train if not df.empty], ignore_index=True) if all_train else pd.DataFrame()
    df_dev = pd.concat([df for df in all_dev if not df.empty], ignore_index=True) if all_dev else pd.DataFrame()
    df_test = pd.concat([df for df in all_test if not df.empty], ignore_index=True) if all_test else pd.DataFrame()
    
    if not df_train.empty: df_train.to_csv(EXPORT_DIR / "training.csv", index=False)
    if not df_dev.empty: df_dev.to_csv(EXPORT_DIR / "dev.csv", index=False)
    if not df_test.empty: df_test.to_csv(EXPORT_DIR / "test.csv", index=False)
    print("Data export complete.")
    
    # Combined for EDA
    df_all = pd.concat([df_train, df_dev, df_test], ignore_index=True)
    return df_all

def perform_eda(df):
    if df.empty:
        print("Empty DataFrame. Cannot perform EDA.")
        return
        
    print("Performing Exploratory Data Analysis...")
    
    # 1. Text Length
    df["Text_Length"] = df["Text"].apply(lambda x: len(str(x).split()))
    
    # EDA stats
    stats = []
    stats.append({"Metric": "Total Records", "Value": len(df)})
    stats.append({"Metric": "Unique Text Sentences", "Value": df["Text"].nunique()})
    stats.append({"Metric": "Mean Text Length (words)", "Value": df["Text_Length"].mean()})
    stats.append({"Metric": "Max Text Length (words)", "Value": df["Text_Length"].max()})
    stats.append({"Metric": "Mean Valence", "Value": df["Valence"].mean()})
    stats.append({"Metric": "Mean Arousal", "Value": df["Arousal"].mean()})
    stats.append({"Metric": "Records with NULL Aspect", "Value": (df["Aspect"] == "NULL").sum()})
    stats.append({"Metric": "Records with NULL Opinion", "Value": (df["Opinion"] == "NULL").sum()})
    
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(EXPORT_DIR / "eda_summary_statistics.csv", index=False)
    print("Saved eda_summary_statistics.csv")
    
    # --- Plots ---
    sns.set_theme(style="whitegrid")
    
    # 1. Text Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Text_Length"], bins=50, color=EMERALD_GREEN, kde=True)
    plt.title("Distribution of Text Lengths (Words)")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(EDA_PLOT_DIR / "text_length_distribution.png", dpi=300)
    plt.close()
    
    # 2. Valence vs Arousal Heatmap
    plt.figure(figsize=(8, 7))
    v_clean = df["Valence"].dropna()
    a_clean = df["Arousal"].dropna()
    if len(v_clean) > 0 and len(a_clean) > 0:
        plt.hist2d(v_clean, a_clean, bins=30, cmap="Greens")
        plt.colorbar(label="Count")
        plt.title("Valence vs Arousal Scatter Density")
        plt.xlabel("Valence [1-9]")
        plt.ylabel("Arousal [1-9]")
        plt.tight_layout()
        plt.savefig(EDA_PLOT_DIR / "valence_arousal_density.png", dpi=300)
    plt.close()
    
    # 3. Category Distribution
    plt.figure(figsize=(10, 6))
    valid_cats = df[(df["Category"].notna()) & (df["Category"] != "")]["Category"]
    if len(valid_cats) > 0:
        cat_counts = valid_cats.value_counts().head(20)
        sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="viridis")
        plt.title("Top 20 Categories in Subtask 3")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(EDA_PLOT_DIR / "category_distribution.png", dpi=300)
    plt.close()
    
    # 4. Task Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Task", data=df, palette="crest")
    plt.title("Distribution of Labels across Tasks")
    plt.tight_layout()
    plt.savefig(EDA_PLOT_DIR / "task_distribution.png", dpi=300)
    plt.close()

    print("EDA plots and stats generated.")

def consolidate_results():
    print("Consolidating final results...")
    
    # 1. Multi-seed and Ablation
    phase5_path = LOG_DIR / "results_phase5.csv"
    phase6_path = LOG_DIR / "results_phase6_ablation.csv"
    
    df_p5 = pd.DataFrame()
    if phase5_path.exists():
        df_p5 = pd.read_csv(phase5_path)
        df_p5.to_csv(EXPORT_DIR / "multiseed_results.csv", index=False)
        print("Saved multiseed_results.csv")
        
    df_p6 = pd.DataFrame()
    if phase6_path.exists():
        df_p6 = pd.read_csv(phase6_path)
        df_p6.to_csv(EXPORT_DIR / "ablation_results.csv", index=False)
        print("Saved ablation_results.csv")
        
    # 2. Final comprehensive
    baseline_path = LOG_DIR / "baseline_results.csv"
    df_base = pd.DataFrame()
    if baseline_path.exists():
        df_base = pd.read_csv(baseline_path)
        
    # Combine baseline with the models from phase5 & phase6
    final_dfs = []
    if not df_base.empty:
        # Standardize columns
        if "N" in df_base.columns: df_base = df_base.drop(columns=["N"])
        if "timestamp" in df_base.columns: df_base = df_base.drop(columns=["timestamp"])
        df_base["Source"] = "Baseline"
        final_dfs.append(df_base)
        
    if not df_p5.empty:
        df_p5["Source"] = "V4_Experiment"
        df_p5["model"] = "DimABSA_V4_Seed_" + df_p5["Seed"].astype(str)
        df_p5["domain"] = df_p5["Domain"]
        df_p5["task"] = df_p5["Subtask"]
        final_dfs.append(df_p5)

    if not df_p6.empty:
        df_p6["Source"] = "V4_Ablation"
        df_p6["model"] = "DimABSA_V4_" + df_p6["Ablation_Name"].astype(str) + "_Seed_" + df_p6["Seed"].astype(str)
        df_p6["domain"] = df_p6["Domain"]
        df_p6["task"] = df_p6["Subtask"]
        final_dfs.append(df_p6)
            
    if final_dfs:
        df_final = pd.concat(final_dfs, ignore_index=True)
        # Reorder to put model, domain, task first
        cols = ["Source", "model", "domain", "task", "RMSE_VA", "RMSE_norm", "PCC_V", "PCC_A", "V_RMSE", "A_RMSE", "cF1", "cP", "cR"]
        cols = [c for c in cols if c in df_final.columns] + [c for c in df_final.columns if c not in cols]
        df_final = df_final[cols]
        df_final.to_csv(EXPORT_DIR / "final_results_combined.csv", index=False)
        print("Saved final_results_combined.csv")
    
    print("Results consolidation complete.")

if __name__ == "__main__":
    df_all = create_data_exports()
    perform_eda(df_all)
    consolidate_results()
