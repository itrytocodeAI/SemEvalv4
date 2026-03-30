"""
config_v4.py — DimABSA V4 global configuration.

All paths, hyper-parameters, and data registries live here so
every other module imports a single authoritative source.
"""

import os
import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(
    r"C:\Users\ASUS\.gemini\antigravity\scratch\DimABSA2026\task-dataset\track_a"
)
WORKSPACE = Path(r"d:\semeval_v5")
CHECKPOINT_DIR = WORKSPACE / "checkpoints"
LOG_DIR = WORKSPACE / "logs"
PREDICTION_DIR = WORKSPACE / "predictions"

for _d in (CHECKPOINT_DIR, LOG_DIR, PREDICTION_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Backbone ───────────────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/deberta-v3-base"
HIDDEN_SIZE = 768          # DeBERTa-v3-base hidden dim
BIAFFINE_DIM = 256         # Project down BEFORE biaffine multiplication (VRAM safety)
LING_DIM = 8               # Linguistic feature vector dimension

# ── Training ───────────────────────────────────────────────────────────────────
MAX_LEN = 128
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2       # Effective batch = 8 × 2 = 16
LR_ENCODER = 2e-5
LR_HEAD = 5e-5
LR_LOGVAR = 1e-3           # Lowered from 5e-2 for stability with bfloat16
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 15
PATIENCE = 3               # Early stopping on dev RMSE_VA
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0
SEEDS = [42, 1337, 2024]

# Mixed-precision
AMP_DTYPE = "bfloat16"     # bf16 on Ampere+; fall back to fp16 if needed

# Uncertainty weighting delay (start after this many epochs)
UW_DELAY_EPOCHS = 3

# ── Focal Loss hyper-params (2D Biaffine grid) ─────────────────────────────────
FOCAL_ALPHA = 0.25         # Down-weight easy negatives
FOCAL_GAMMA = 2.0          # Focus on hard positives

# ── Normalisation constants ────────────────────────────────────────────────────
VA_MID = 5.0               # Centre of [1, 9] range
VA_HALF = 4.0              # Half-range

def va_normalise(val: float) -> float:
    """Map [1, 9] → [-1, 1].  normalised = (val - 5) / 4"""
    return (val - VA_MID) / VA_HALF

def va_denormalise(pred: float) -> float:
    """Map [-1, 1] → [1, 9].  original = pred * 4 + 5"""
    return pred * VA_HALF + VA_MID

# ── Category registry ──────────────────────────────────────────────────────────
# Scan all train files once to build a category list.
# This is kept here so config is self-contained and importable.

def _build_category_registry() -> dict:
    """
    Scans all *_train_alltasks.jsonl files, collects every unique
    ENTITY#ATTRIBUTE string, and returns:
        {category_str: int_index}
    """
    cats: set = set()
    for st in ["subtask_1", "subtask_2", "subtask_3"]:
        st_path = DATA_ROOT / st
        if not st_path.exists():
            continue
        for lang_dir in st_path.iterdir():
            if not lang_dir.is_dir():
                continue
            for fp in lang_dir.glob("*_train_alltasks.jsonl"):
                with open(fp, encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        for quad in obj.get("Quadruplet", []):
                            cat = quad.get("Category", "")
                            if cat and cat != "NULL":
                                cats.add(cat)
    # Stable sort for reproducibility
    sorted_cats = sorted(cats)
    return {cat: idx for idx, cat in enumerate(sorted_cats)}

# Build once at import time — takes < 2 s on SSD
try:
    CATEGORY2IDX: dict = _build_category_registry()
except Exception as _e:
    import warnings
    warnings.warn(f"[config_v4] Could not build category registry: {_e}")
    CATEGORY2IDX = {}

IDX2CATEGORY: dict = {v: k for k, v in CATEGORY2IDX.items()}
NUM_CATEGORIES: int = len(CATEGORY2IDX)

# ── Languages / domains in the dataset ────────────────────────────────────────
LANGUAGES = ["eng", "jpn", "rus", "tat", "ukr", "zho"]
DOMAINS = ["restaurant", "laptop", "hotel", "finance"]

# ── Grid label map ─────────────────────────────────────────────────────────────
GRID_IGNORE = -100          # Special tokens / padding
GRID_NEGATIVE = 0           # No aspect-opinion relation
GRID_POSITIVE = 1           # Aspect-span row × Opinion-span column

# ── Subtask file key names ─────────────────────────────────────────────────────
# Maps subtask_N → the JSON key used in gold dev/test files
SUBTASK_KEY = {
    1: "Aspect_VA",
    2: "Triplet",
    3: "Quadruplet",
}

if __name__ == "__main__":
    print(f"MODEL_NAME : {MODEL_NAME}")
    print(f"NUM_CATEGORIES : {NUM_CATEGORIES}")
    print(f"Sample categories : {list(CATEGORY2IDX.keys())[:10]}")
    print(f"VA_MID={VA_MID}, VA_HALF={VA_HALF}")
    print(f"va_normalise(1.0) = {va_normalise(1.0):.4f}  (expect -1.0)")
    print(f"va_normalise(5.0) = {va_normalise(5.0):.4f}  (expect  0.0)")
    print(f"va_normalise(9.0) = {va_normalise(9.0):.4f}  (expect +1.0)")
    print(f"va_denormalise(-1.0) = {va_denormalise(-1.0):.4f}  (expect 1.0)")
    print(f"va_denormalise( 0.0) = {va_denormalise(0.0):.4f}  (expect 5.0)")
    print(f"va_denormalise(+1.0) = {va_denormalise(1.0):.4f}  (expect 9.0)")
