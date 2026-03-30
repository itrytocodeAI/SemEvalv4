"""
train_v4.py — DimABSA V4 Training Loop.

Features:
  - Mixed-precision (AMP)
  - Gradient Accumulation & Clipping
  - Delayed Uncertainty Weighting (starts epoch 4)
  - Evaluates ST1 via RMSE_VA, ST2/3 via overall loss + focal proxy metrics
  - Early stopping based on dev set metric
  - Extensive cache clearing for VRAM safety
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_v4 import (
    MODEL_NAME, DATA_ROOT, LOG_DIR, CHECKPOINT_DIR, BATCH_SIZE, MAX_LEN,
    GRAD_ACCUM_STEPS, LR_ENCODER, LR_HEAD, LR_LOGVAR, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, WARMUP_RATIO, GRAD_CLIP, UW_DELAY_EPOCHS,
    AMP_DTYPE, SUBTASK_KEY, va_denormalise
)
from src.utils import set_seed, vram_report, clear_cache, get_logger, rmse_va
from src.metrics_v4 import compute_grid_metrics
from src.data_loader_v4 import get_dataloader
from src.model_v4 import DimABSAV4Model

class HistoryTracker:
    def __init__(self):
        self.history = []
        
    def log(self, epoch, metrics):
        item = {"epoch": epoch}
        item.update({k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v for k, v in metrics.items()})
        self.history.append(item)
        
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

def run_train(
    subtask: int,
    lang: str = "eng",
    domain: str = "restaurant",
    seed: int = 42,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    debug: bool = False,
    use_ling_features: bool = True,
    use_hybrid_loss: bool = True,
    use_focal_loss: bool = True,
    use_uncertainty_weight: bool = True,
    log_filename: str | None = None
):
    set_seed(seed)
    
    # Paths
    run_name = f"st{subtask}_{lang}_{domain}_seed{seed}"
    log = get_logger(run_name, log_dir=LOG_DIR, log_filename=log_filename)
    ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    
    log.info(f"=== DimABSA V4 Training: Subtask {subtask} ({lang}_{domain}) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_type = torch.bfloat16 if AMP_DTYPE == "bfloat16" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    log.info(f"Device: {device}, AMP: {amp_type}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Dataloaders - use language and domain for specific train/dev splits
    st_path = DATA_ROOT / f"subtask_{subtask}" / lang
    train_file = st_path / "train" / f"{lang}_{domain}_train_alltasks.jsonl"
    
    dev_key_part = "task1" if subtask == 1 else "task2" if subtask == 2 else "task3"
    dev_file = st_path / f"{lang}_{domain}_dev_{dev_key_part}.jsonl"

    if not train_file.exists():
        # Fallback to base dir if train subfolder doesn't exist
        train_file = st_path / f"{lang}_{domain}_train_alltasks.jsonl"
    
    if not train_file.exists() or not dev_file.exists():
        log.error(f"Cannot find data files: \n  {train_file}\n  {dev_file}")
        return

    log.info(f"Loading train: {train_file}")
    train_dl = get_dataloader(train_file, subtask, tokenizer, batch_size, MAX_LEN, is_train=True)
    
    log.info(f"Loading dev: {dev_file}")
    dev_dl = get_dataloader(dev_file, subtask, tokenizer, batch_size, MAX_LEN, is_train=False)

    if debug:
        max_epochs = 2
        log.info("DEBUG mode: max_epochs = 2")

    model = DimABSAV4Model(
        use_ling_features=use_ling_features,
        use_hybrid_loss=use_hybrid_loss,
        use_focal_loss=use_focal_loss,
        use_uncertainty_weight=use_uncertainty_weight
    ).to(device)
    
    # Optimizer groups
    enc_params = []
    head_params = []
    logvar_params = []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "log_vars" in n:
            logvar_params.append(p)
        elif "encoder" in n:
            enc_params.append(p)
        else:
            head_params.append(p)
            
    optimizer = AdamW([
        {"params": enc_params, "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
        {"params": logvar_params, "lr": LR_LOGVAR, "weight_decay": 0.0}
    ])
    
    t_total = (len(train_dl) // GRAD_ACCUM_STEPS) * max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * WARMUP_RATIO), num_training_steps=t_total
    )
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_metric = float("inf")
    patience_cnt = 0
    tracker = HistoryTracker()
    val_pairs = []
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        use_uw = (epoch > UW_DELAY_EPOCHS)
        
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        
        for step, batch in enumerate(train_dl):
            vram_report(prefix=f"Epoch {epoch} Step {step} Before ") if step == 0 else None
            
            with torch.amp.autocast(device_type=device.type, dtype=amp_type, enabled=torch.cuda.is_available()):
                if subtask == 1:
                    pred_va = model.forward_regression(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["token_type_ids"].to(device),
                        batch["ling_features"].to(device),
                        batch["aspect_token_mask"].to(device)
                    )
                    loss_d = model.compute_regression_loss(
                        pred_va,
                        batch["valence"].to(device),
                        batch["arousal"].to(device),
                        batch["has_label"].to(device),
                        use_uw=use_uw
                    )
                else:
                    grid_l, cat_l = model.forward_extraction(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["ling_features"].to(device)
                    )
                    loss_d = model.compute_extraction_loss(
                        grid_l, cat_l,
                        batch["grid_labels"].to(device),
                        batch["category_ids"].to(device),
                        batch["va_mask"].to(device),
                        use_uw=use_uw
                    )
                    
                loss = loss_d["loss"] / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            train_loss += loss.item() * GRAD_ACCUM_STEPS
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1 == len(train_dl)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
            if debug and step > 5:
                break
                
        train_loss /= len(train_dl)
        dur = time.time() - t0
        
        # Eval
        model.eval()
        val_loss = 0.0
        val_v_loss = 0.0
        val_a_loss = 0.0
        all_pred_v, all_pred_a, all_gold_v, all_gold_a = [], [], [], []
        
        all_grid_p = []
        all_grid_g = []
        
        with torch.no_grad():
            for step, batch in enumerate(dev_dl):
                with torch.amp.autocast(device_type=device.type, dtype=amp_type, enabled=torch.cuda.is_available()):
                    if subtask == 1:
                        pred_va = model.forward_regression(
                            batch["input_ids"].to(device),
                            batch["attention_mask"].to(device),
                            batch["token_type_ids"].to(device),
                            batch["ling_features"].to(device),
                            batch["aspect_token_mask"].to(device)
                        )
                        ld = model.compute_regression_loss(
                            pred_va,
                            batch["valence"].to(device),
                            batch["arousal"].to(device),
                            batch["has_label"].to(device),
                            use_uw=False
                        )
                        val_loss += ld["loss"].item()
                        if "v_loss" in ld:
                            val_v_loss += ld["v_loss"].item()
                            val_a_loss += ld["a_loss"].item()
                        
                        valid = batch["has_label"].bool()
                        if valid.sum() > 0:
                            all_pred_v.extend(pred_va[valid, 0].float().cpu().numpy().tolist())
                            all_pred_a.extend(pred_va[valid, 1].float().cpu().numpy().tolist())
                            all_gold_v.extend(batch["valence"][valid].float().cpu().numpy().tolist())
                            all_gold_a.extend(batch["arousal"][valid].float().cpu().numpy().tolist())
                    else:
                        grid_l, cat_l = model.forward_extraction(
                            batch["input_ids"].to(device),
                            batch["attention_mask"].to(device),
                            batch["ling_features"].to(device)
                        )
                        ld = model.compute_extraction_loss(
                            grid_l, cat_l,
                            batch["grid_labels"].to(device),
                            batch["category_ids"].to(device),
                            batch["va_mask"].to(device),
                            use_uw=False
                        )
                        val_loss += ld["loss"].item()
                        
                        probs = torch.sigmoid(grid_l).float().cpu().numpy()
                        labels = batch["grid_labels"].float().cpu().numpy()
                        all_grid_p.append(probs)
                        all_grid_g.append(labels)
                        
                if debug and step > 5:
                    break
                    
        val_loss /= max(len(dev_dl), 1)
        val_v_loss /= max(len(dev_dl), 1)
        val_a_loss /= max(len(dev_dl), 1)
        
        # Metrics
        metric_val = val_loss
        metric_name = "Val Loss"
        metrics_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        
        if subtask == 1 and len(all_pred_v) > 0:
            # Denormalize to compute official RMSE
            pv = np.array([va_denormalise(x) for x in all_pred_v])
            pa = np.array([va_denormalise(x) for x in all_pred_a])
            gv = np.array([va_denormalise(x) for x in all_gold_v])
            ga = np.array([va_denormalise(x) for x in all_gold_a])
            
            rmse = rmse_va(pv, pa, gv, ga)
            metric_val = rmse
            metric_name = "RMSE_VA"
            
            # Individual dimension RSMEs implicitly scaled
            v_rmse = np.sqrt(np.mean((pv - gv)**2))
            a_rmse = np.sqrt(np.mean((pa - ga)**2))
            metrics_dict.update({
                "RMSE_VA": rmse, "V_RMSE": v_rmse, "A_RMSE": a_rmse, 
                "V_Loss": val_v_loss, "A_Loss": val_a_loss
            })
            
        elif subtask in (2, 3) and len(all_grid_p) > 0:
            pg = np.concatenate(all_grid_p, axis=0)
            gg = np.concatenate(all_grid_g, axis=0)
            cP, cR, cF1 = compute_grid_metrics(pg, gg, threshold=0.5)
            # Use negative continuous F1 for early stopping to mimic "lower is better" standard
            metric_val = -cF1 if cF1 > 0 else val_loss
            metric_name = "cF1"
            metrics_dict.update({
                "cF1": cF1, "cP": cP, "cR": cR
            })
            
        # Add uncertainty sigmas to history for all tasks
        if use_uncertainty_weight:
            metrics_dict.update({
                "sigma_v": torch.exp(model.log_vars[0]).item(),
                "sigma_a": torch.exp(model.log_vars[1]).item(),
            })
            if subtask in (2, 3):
                metrics_dict.update({
                    "sigma_ext": torch.exp(model.log_vars[2]).item(),
                })
            
        log.info(f"Ep {epoch:2d}/{max_epochs} | UW:{use_uw} | "
                 f"T-Loss: {train_loss:.4f} | V-Loss: {val_loss:.4f} | "
                 f"Dev {metric_name}: {metric_val if metric_val > 0 else -metric_val:.4f} | Time: {dur:.1f}s")
                 
        tracker.log(epoch, metrics_dict)
        
        if metric_val < best_metric:
            best_metric = metric_val
            best_metrics_dict = metrics_dict.copy()
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
            log.info(f"  --> Saved new best dev {metric_name}: {metric_val if metric_val > 0 else -metric_val:.4f}")
            
            # Save best val pairs for regression plots
            if subtask == 1 and len(all_pred_v) > 0:
                val_pairs = []
                for i in range(len(all_pred_v)):
                    val_pairs.append({
                        "gold_v": va_denormalise(all_gold_v[i]),
                        "gold_a": va_denormalise(all_gold_a[i]),
                        "pred_v": va_denormalise(all_pred_v[i]),
                        "pred_a": va_denormalise(all_pred_a[i])
                    })
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                log.info(f"Early stopping at epoch {epoch}")
                break
                
        clear_cache()
        
    # Final saves
    tracker.save(LOG_DIR / f"{run_name}_history.json")
    if val_pairs:
        with open(LOG_DIR / f"{run_name}_val_pairs.json", "w") as f:
            json.dump(val_pairs, f, indent=2)
            
    log.info("Training complete.")
    return best_metrics_dict if 'best_metrics_dict' in locals() else metrics_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", type=int, default=1)
    parser.add_argument("--lang", type=str, default="eng")
    parser.add_argument("--domain", type=str, default="restaurant")
    parser.add_argument("--debug", action="store_true")
    
    # Ablation toggles
    parser.add_argument("--no_ling", action="store_true", help="Disable linguistic features")
    parser.add_argument("--no_hybrid", action="store_true", help="Disable Huber+CCC hybrid loss")
    parser.add_argument("--no_focal", action="store_true", help="Disable focal loss")
    parser.add_argument("--no_uw", action="store_true", help="Disable homoscedastic uncertainty weighting")
    
    args = parser.parse_args()
    
    run_train(
        args.subtask, args.lang, args.domain, 
        debug=args.debug,
        use_ling_features=not args.no_ling,
        use_hybrid_loss=not args.no_hybrid,
        use_focal_loss=not args.no_focal,
        use_uncertainty_weight=not args.no_uw
    )
