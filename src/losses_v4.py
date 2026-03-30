"""
losses_v4.py — DimABSA V4 loss functions.

  CCC loss       : Concordance Correlation Coefficient (1 - CCC)
  Focal loss     : Focal BCE for sparse 2D biaffine grid (alpha=0.25, gamma=2.0)
  Hybrid reg loss: alpha*Huber + (1-alpha)*CCC per dimension
  Uncertainty UW : LogSigma homoscedastic weighting (Kendall 2018)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. CCC Loss
# ─────────────────────────────────────────────────────────────────────────────

def ccc_loss(pred: torch.Tensor, gold: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Concordance Correlation Coefficient loss: returns 1 - CCC.

    Args:
        pred, gold: 1-D tensors of the same length (already in [-1, 1])
    Returns:
        scalar loss (lower = better alignment)
    """
    if pred.numel() < 2:
        return F.mse_loss(pred, gold)

    pred_mean = pred.mean()
    gold_mean = gold.mean()
    pred_var  = ((pred - pred_mean) ** 2).mean()
    gold_var  = ((gold - gold_mean) ** 2).mean()
    cov       = ((pred - pred_mean) * (gold - gold_mean)).mean()

    ccc = (2.0 * cov) / (pred_var + gold_var + (pred_mean - gold_mean) ** 2 + eps)
    return 1.0 - ccc


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hybrid Regression Loss  (Huber + CCC)
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_regression_loss(
    pred_va: torch.Tensor,   # [N, 2]  in (-1, 1)
    gold_va: torch.Tensor,   # [N, 2]  in [-1, 1]
    alpha: float = 0.5,      # weight for Huber; (1-alpha) for CCC
    huber_delta: float = 0.5,
) -> torch.Tensor:
    """
    Hybrid loss per dimension:
        L = alpha * Huber(pred, gold) + (1 - alpha) * CCC_loss(pred, gold)
    Applied independently to Valence and Arousal, then averaged.
    """
    huber = F.smooth_l1_loss(pred_va, gold_va, beta=huber_delta)

    pred_v, pred_a = pred_va[:, 0], pred_va[:, 1]
    gold_v, gold_a = gold_va[:, 0], gold_va[:, 1]

    ccc_v = ccc_loss(pred_v, gold_v)
    ccc_a = ccc_loss(pred_a, gold_a)
    ccc_combined = (ccc_v + ccc_a) / 2.0

    return alpha * huber + (1.0 - alpha) * ccc_combined


# ─────────────────────────────────────────────────────────────────────────────
# 3. Focal Loss  (2D biaffine grid)
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,        # [B, L, L]  raw grid logits
    labels: torch.Tensor,        # [B, L, L]  {0, 1, -100}
    alpha: float = 0.25,
    gamma: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Binary focal loss for the sparse 2D adjacency grid.

    Only computed on valid cells (label != -100).
    Positive (1) cells use weight alpha; negatives use (1-alpha).
    """
    valid = labels != ignore_index            # [B, L, L] bool
    if valid.sum() == 0:
        return logits.sum() * 0.0            # safe zero

    logits_v = logits[valid]                 # [N_valid]
    labels_v = labels[valid].float()         # [N_valid]

    bce    = F.binary_cross_entropy_with_logits(logits_v, labels_v, reduction="none")
    pt     = torch.exp(-bce)                 # probability of correct class
    alpha_t = alpha * labels_v + (1.0 - alpha) * (1.0 - labels_v)
    fl     = alpha_t * (1.0 - pt) ** gamma * bce
    return fl.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Uncertainty-weighted (LogSigma) multi-task loss
# ─────────────────────────────────────────────────────────────────────────────

def uncertainty_weighted_loss(
    task_losses: list[torch.Tensor],   # list of scalar losses, one per task
    log_vars: torch.Tensor,            # [n_tasks] learnable log-variance params
    use_uw: bool = True,
) -> torch.Tensor:
    """
    Homoscedastic uncertainty weighting (Kendall et al., 2018):
        L_total = sum_k [ exp(-log_var_k) * L_k  +  log_var_k ]

    If use_uw=False (first UW_DELAY_EPOCHS epochs), returns a simple sum.
    """
    assert len(task_losses) == log_vars.shape[0], "Mismatch in number of tasks"

    if not use_uw:
        return sum(task_losses)

    total = 0.0
    for k, loss_k in enumerate(task_losses):
        log_v = log_vars[k].clamp(-3.0, 3.0)
        total = total + torch.exp(-log_v) * loss_k + log_v
    return total


if __name__ == "__main__":
    torch.manual_seed(42)
    B, L = 2, 8

    # CCC
    p = torch.randn(16)
    g = torch.randn(16)
    print(f"CCC loss (random): {ccc_loss(p, g).item():.4f}  (expect ~1.0)")
    print(f"CCC loss (perfect): {ccc_loss(g, g).item():.6f}  (expect 0.0)")

    # Hybrid reg
    pva = torch.tanh(torch.randn(8, 2))
    gva = torch.tanh(torch.randn(8, 2))
    print(f"Hybrid reg loss: {hybrid_regression_loss(pva, gva).item():.4f}")

    # Focal
    logits = torch.randn(B, L, L)
    labels = torch.zeros(B, L, L, dtype=torch.long)
    labels[0, 2, 4] = 1;  labels[1, 1, 3] = 1
    labels[:, 0, :] = -100;  labels[:, :, 0] = -100   # special tokens
    print(f"Focal loss: {focal_loss(logits, labels).item():.4f}")

    # UW
    losses = [torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.2)]
    log_v = nn.Parameter(torch.zeros(3))
    print(f"UW loss (delayed): {uncertainty_weighted_loss(losses, log_v, use_uw=False).item():.4f}")
    print(f"UW loss (active):  {uncertainty_weighted_loss(losses, log_v, use_uw=True).item():.4f}")
    print("losses_v4.py OK")
