"""
deep_eval_runner.py 
Performs a systematic 5-seed sweep (10 epochs) for statistical stability analysis.
Output: logs/deep_run_laptop.csv and logs/deep_run_restaurant.csv.
"""

import os
import csv
import sys
from pathlib import Path
from datetime import datetime

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train_v4 import run_train
from src.config_v4 import LOG_DIR
from src.utils import get_logger

logger = get_logger("deep_eval_runner", log_dir=LOG_DIR)

SEEDS = [42, 1337, 2024, 7, 100]
DOMAINS = ["laptop", "restaurant"]
EPOCHS = 10

def init_csv(path, headers):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

def append_csv(path, row_dict, headers):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row_dict.get(h, "") for h in headers])

def run_sweep():
    headers = ["Seed", "Domain", "RMSE_VA", "V_RMSE", "A_RMSE", "Train_Loss", "Val_Loss", "Timestamp"]
    
    for domain in DOMAINS:
        csv_path = LOG_DIR / f"deep_run_{domain}.csv"
        init_csv(csv_path, headers)
        
        for seed in SEEDS:
            logger.info(f"== Deep Run: {domain.upper()} | Seed {seed} | 10 Epochs ==")
            try:
                metrics = run_train(
                    subtask=1,
                    lang="eng",
                    domain=domain,
                    seed=seed,
                    max_epochs=EPOCHS,
                    debug=False,
                    log_filename=f"deep_run_{domain}_seed{seed}.log"
                )
                
                row = {
                    "Seed": seed,
                    "Domain": domain,
                    "RMSE_VA": metrics.get("RMSE_VA", ""),
                    "V_RMSE": metrics.get("V_RMSE", ""),
                    "A_RMSE": metrics.get("A_RMSE", ""),
                    "Train_Loss": f"{metrics.get('train_loss', 0):.4f}",
                    "Val_Loss": f"{metrics.get('val_loss', 0):.4f}",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                append_csv(csv_path, row, headers)
                logger.info(f"Seed {seed} complete. RMSE_VA: {metrics.get('RMSE_VA')}")
                
            except Exception as e:
                logger.error(f"Failed Seed {seed} for {domain}: {e}")

if __name__ == "__main__":
    run_sweep()
