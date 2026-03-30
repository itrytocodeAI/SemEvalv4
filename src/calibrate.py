"""
calibrate.py
Post-hoc Linear Calibration algorithm with 5-fold Cross-Validation logic.
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np

def calibrate_predictions(gold_va: np.ndarray, pred_va: np.ndarray, mode: str="linear") -> np.ndarray:
    """
    Fits a simple linear regression map: cal_pred = w * pred + b
    to reduce systematic bias in regression outputs on dev fold.
    """
    if mode != "linear":
        return pred_va
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    calibrated_preds = np.zeros_like(pred_va)
    
    # Needs two dims for regressor
    pv = pred_va.reshape(-1, 1)
    gv = gold_va.reshape(-1, 1)
    
    for train_index, val_index in kf.split(pv):
        pv_train, gv_train = pv[train_index], gv[train_index]
        pv_val = pv[val_index]
        
        reg = LinearRegression().fit(pv_train, gv_train)
        calibrated_preds[val_index] = reg.predict(pv_val).flatten()
        
    return np.clip(calibrated_preds, 1.0, 9.0)

if __name__ == "__main__":
    # verification map
    gold = np.array([5.0, 7.0, 3.0, 9.0, 1.0])
    pred = np.array([5.5, 7.5, 3.5, 9.5, 1.5])
    
    cal = calibrate_predictions(gold, pred)
    print("Gold:", gold)
    print("Pred:", pred)
    print("Calibrated:", cal)
