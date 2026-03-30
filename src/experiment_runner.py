"""
experiment_runner.py
Automates Grid Evaluatons (Phase 5) and Ablation Studies (Phase 6).
Logs all metrics sequentially into CSV sheets.
"""

import os
import csv
import sys
from pathlib import Path
from datetime import datetime

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train_v4 import run_train
from src.predict_v4 import run_predict
from src.config_v4 import LOG_DIR
from src.utils import get_logger

logger = get_logger("experiment_runner", log_dir=LOG_DIR)

SEEDS = [42, 1337, 2024, 7, 100]
DOMAINS = ["laptop", "restaurant"]
LANGS = ["eng"]
SUBTASKS = [1, 3] # Train Subtask 1 (Regression) & 3 (Extraction)

def init_csv(path, headers):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

def append_csv(path, row_dict, headers):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row_dict.get(h, "") for h in headers])

MASTER_LOG_FILE = "combined_experiments.log"

def run_phase5():
    logger.info("=== Starting Phase 5: Multi-Seed Multi-Domain Grid Training ===")
    csv_path = LOG_DIR / "results_phase5.csv"
    headers = [
        "Timestamp", "Subtask", "Lang", "Domain", "Seed", "Epoch",
        "Train_Loss", "Val_Loss", "RMSE_VA", "V_RMSE", "A_RMSE", "V_Loss", "A_Loss",
        "cF1", "cP", "cR"
    ]
    init_csv(csv_path, headers)
    
    for subtask in SUBTASKS:
        for domain in DOMAINS:
            for lang in LANGS:
                for seed in SEEDS:
                    logger.info(f"Running Phase 5: ST{subtask} | {lang}_{domain} | Seed {seed}")
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        metrics = run_train(
                            subtask=subtask,
                            lang=lang,
                            domain=domain,
                            seed=seed,
                            max_epochs=3,    # Accelerated run defaults for full sequences
                            debug=DEBUG_MODE,
                            log_filename=MASTER_LOG_FILE
                        )
                        
                        # Generate inference
                        run_predict(subtask=subtask, lang=lang, domain=domain, seed=seed)
                        
                        row = {
                            "Timestamp": ts, "Subtask": subtask, "Lang": lang, "Domain": domain, "Seed": seed,
                            "Epoch": metrics.get("epoch", ""),
                            "Train_Loss": f"{metrics.get('train_loss', 0):.4f}",
                            "Val_Loss": f"{metrics.get('val_loss', 0):.4f}",
                            "RMSE_VA": f"{metrics.get('RMSE_VA', '')}",
                            "V_RMSE": f"{metrics.get('V_RMSE', '')}",
                            "A_RMSE": f"{metrics.get('A_RMSE', '')}",
                            "V_Loss": f"{metrics.get('V_Loss', '')}",
                            "A_Loss": f"{metrics.get('A_Loss', '')}",
                            "cF1": f"{metrics.get('cF1', '')}",
                            "cP": f"{metrics.get('cP', '')}",
                            "cR": f"{metrics.get('cR', '')}"
                        }
                        append_csv(csv_path, row, headers)
                    except Exception as e:
                        logger.error(f"Failed ST{subtask} {lang}_{domain} Seed {seed}: {e}")

def run_phase6():
    logger.info("=== Starting Phase 6: Ablation Study ===")
    csv_path = LOG_DIR / "results_phase6_ablation.csv"
    headers = [
        "Timestamp", "Ablation_Name", "Subtask", "Lang", "Domain", "Seed", 
        "Use_Ling", "Use_Hybrid", "Use_Focal", "Use_UW", "Epoch",
        "Train_Loss", "Val_Loss", "RMSE_VA", "V_RMSE", "A_RMSE", "V_Loss", "A_Loss",
        "cF1", "cP", "cR"
    ]
    init_csv(csv_path, headers)
    
    # Run ablation on English Restaurant data to compare directly
    ablations = [
        {"name": "1_FullModel",    "kw": {"use_ling_features": True,  "use_hybrid_loss": True,  "use_focal_loss": True,  "use_uncertainty_weight": True}},
        {"name": "2_NoLing",       "kw": {"use_ling_features": False, "use_hybrid_loss": True,  "use_focal_loss": True,  "use_uncertainty_weight": True}},
        {"name": "3_NoHybrid",     "kw": {"use_ling_features": True,  "use_hybrid_loss": False, "use_focal_loss": True,  "use_uncertainty_weight": True}},
        {"name": "4_NoFocal",      "kw": {"use_ling_features": True,  "use_hybrid_loss": True,  "use_focal_loss": False, "use_uncertainty_weight": True}},
        {"name": "5_NoUncertainty","kw": {"use_ling_features": True,  "use_hybrid_loss": True,  "use_focal_loss": True,  "use_uncertainty_weight": False}},
    ]
    
    domain = "restaurant"
    lang = "eng"
    ablation_seeds = [42, 1337] # Just 2 seeds for ablation to manage total execution time footprint
    
    for ab in ablations:
        for subtask in SUBTASKS:
            for seed in ablation_seeds:
                logger.info(f"Running Phase 6: {ab['name']} | ST{subtask} | {lang}_{domain} | Seed {seed}")
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    metrics = run_train(
                        subtask=subtask,
                        lang=lang,
                        domain=domain,
                        seed=seed,
                        max_epochs=2, # Accelerated
                        debug=DEBUG_MODE,
                        log_filename=MASTER_LOG_FILE,
                        **ab["kw"]
                    )
                    
                    row = {
                        "Timestamp": ts, "Ablation_Name": ab["name"], "Subtask": subtask, "Lang": lang, "Domain": domain, "Seed": seed,
                        "Use_Ling": ab["kw"].get("use_ling_features", True),
                        "Use_Hybrid": ab["kw"].get("use_hybrid_loss", True),
                        "Use_Focal": ab["kw"].get("use_focal_loss", True),
                        "Use_UW": ab["kw"].get("use_uncertainty_weight", True),
                        "Epoch": metrics.get("epoch", ""),
                        "Train_Loss": f"{metrics.get('train_loss', 0):.4f}",
                        "Val_Loss": f"{metrics.get('val_loss', 0):.4f}",
                        "RMSE_VA": f"{metrics.get('RMSE_VA', '')}",
                        "V_RMSE": f"{metrics.get('V_RMSE', '')}",
                        "A_RMSE": f"{metrics.get('A_RMSE', '')}",
                        "V_Loss": f"{metrics.get('V_Loss', '')}",
                        "A_Loss": f"{metrics.get('A_Loss', '')}",
                        "cF1": f"{metrics.get('cF1', '')}",
                        "cP": f"{metrics.get('cP', '')}",
                        "cR": f"{metrics.get('cR', '')}"
                    }
                    append_csv(csv_path, row, headers)
                except Exception as e:
                    logger.error(f"Failed Phase 6 {ab['name']} ST{subtask} {lang}_{domain} Seed {seed}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[5, 6, 0], default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    DEBUG_MODE = args.debug
    
    if args.phase in (0, 5):
        run_phase5()
    if args.phase in (0, 6):
        run_phase6()
