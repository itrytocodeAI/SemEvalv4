"""
utils.py — DimABSA V4 utility functions.

Covers:
  - Reproducibility (seeding)
  - VRAM reporting / cache clearing
  - Logging helpers
"""

import gc
import os
import random
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fully deterministic seeding for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Note: setting deterministic may slow CuDNN but improves reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── VRAM helpers ───────────────────────────────────────────────────────────────

def vram_report(prefix: str = "") -> dict:
    """
    Print and return a dict of current CUDA memory stats (in MB).

    Returns keys: allocated_mb, reserved_mb, free_mb, total_mb
    """
    if not torch.cuda.is_available():
        print(f"[VRAM] {prefix}No CUDA device available.")
        return {}

    alloc  = torch.cuda.memory_allocated() / 1e6
    reserv = torch.cuda.memory_reserved() / 1e6
    total  = torch.cuda.get_device_properties(0).total_memory / 1e6
    free   = total - reserv

    msg = (
        f"[VRAM] {prefix}"
        f"allocated={alloc:.0f}MB  reserved={reserv:.0f}MB  "
        f"free={free:.0f}MB / {total:.0f}MB"
    )
    print(msg)
    return dict(allocated_mb=alloc, reserved_mb=reserv, free_mb=free, total_mb=total)


def clear_cache() -> None:
    """Free unreferenced CUDA tensors and Python garbage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── Logging ────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_dir: str | Path | None = None, log_filename: str | None = None) -> logging.Logger:
    """
    Return a logger that writes to stdout and (optionally) to a file in log_dir.
    If log_filename is provided, it uses that exact name (appends).
    Otherwise, it creates a timestamped file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if log_filename:
            path = log_dir / log_filename
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"{name}_{ts}.log"
        
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Device helpers ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_amp_dtype() -> torch.dtype:
    """
    Pick the best AMP dtype for the available GPU.
    Ampere+ (sm_80+) supports bfloat16 natively.
    Fall back to float16 for older VRAM.
    """
    if not torch.cuda.is_available():
        return torch.float32
    cc = torch.cuda.get_device_capability()
    return torch.bfloat16 if cc[0] >= 8 else torch.float16


# ── Metric helpers ─────────────────────────────────────────────────────────────

def rmse_va(v_pred: np.ndarray, a_pred: np.ndarray,
            v_gold: np.ndarray, a_gold: np.ndarray) -> float:
    """
    Official SemEval-2026 DimABSA RMSE_VA:
        RMSE_VA = sqrt( sum((Vp-Vg)² + (Ap-Ag)²) / N )
    """
    n = len(v_pred)
    sq = (v_pred - v_gold) ** 2 + (a_pred - a_gold) ** 2
    return float(np.sqrt(sq.sum() / n))


def pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson Correlation Coefficient; returns 0.0 on degenerate input."""
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


if __name__ == "__main__":
    set_seed(42)
    print("set_seed OK")
    vram_report("test: ")
    clear_cache()
    print("clear_cache OK")
    log = get_logger("test", log_dir=None)
    log.info("Logger working!")
