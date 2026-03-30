"""
metrics_v4.py — Evaluation metrics for DimABSA V4.

Contains Logic for Grid Extraction Continuous Scoring (cP, cR, cF1)
and V_Loss / A_Loss split reporting.
"""

import numpy as np

def compute_grid_metrics(pred_grids: np.ndarray, gold_grids: np.ndarray, threshold: float = 0.5):
    """
    Computes Precision, Recall, and F1 over the 2D grid cells.
    To approximate continuous scoring (cP, cR, cF1), we weight true-positives 
    by their raw predicted logits/probabilities (soft F1).
    
    pred_grids: [N, L, L] float probabilities or logits (if sigmoid applied)
    gold_grids: [N, L, L] integers {0, 1, -100}
    """
    # Flatten and keep only valid cells
    valid = gold_grids != -100
    
    if not valid.any():
        return 0.0, 0.0, 0.0
        
    p_valid = pred_grids[valid]
    g_valid = gold_grids[valid]
    
    # Binary thresholding for traditional sets
    pred_bin = (p_valid > threshold).astype(np.float32)
    
    # Continuous weighting for matched cells
    # A true positive's "value" is its confidence if it's correct
    tp_continuous = np.sum((g_valid == 1) * p_valid * pred_bin)
    
    # Standard counts
    fp = np.sum((g_valid == 0) * pred_bin)
    fn = np.sum((g_valid == 1) * (1 - pred_bin))
    
    # Continuous Precision and Recall
    cP = tp_continuous / (tp_continuous + fp + 1e-9)
    cR = tp_continuous / (tp_continuous + fn + 1e-9)
    
    cF1 = 2 * (cP * cR) / (cP + cR + 1e-9)
    
    return float(cP), float(cR), float(cF1)
