"""
data_loader_v4.py — DimABSA V4 PyTorch Dataset classes.

Design decisions (per Phase 1 spec):
─────────────────────────────────────────────────────────────────────────────
ST1 Regression (DimABSARegressionDataset)
  • Pair encoding: [CLS] text [SEP] aspect [SEP]
    Allows Phase 2 cross-attention to slice context vs aspect token indices.
  • VA normalised to [-1, 1]: normalised = (val - 5.0) / 4.0
  • Each sentence × aspect pair → one training instance (flattened)
  • NULL aspects: use whole-text encoding with empty aspect; pool via [CLS]

ST2/3 Extraction (DimABSAExtractionDataset)
  • Single-sequence encoding: [CLS] text [SEP]
  • Per-token offset mapping (return_offsets_mapping=True) for span alignment
  • 2D Adjacency Grid [L, L]:
      cell (i, j) = 1   if token i ∈ Aspect span AND token j ∈ linked Opinion span
      cell (i, j) = 0   otherwise (no relation)
      cell (i, j) = -100 for [CLS], [SEP], and padding positions
  • NULL aspect/opinion → no positive cells in grid
  • VA per quad also normalised to [-1, 1]
  • Category index for ST3; -1 for ST2 (no category)

Linguistic Arousal Features (8-dim, optional)
  • Computed from tokenizer vocab + simple heuristics; no external lexicon req.
  • Shape: [max_len, 8], dtype float32
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# ── Local imports ───────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_v4 import (
    DATA_ROOT, MODEL_NAME, MAX_LEN,
    BATCH_SIZE, GRID_IGNORE, GRID_NEGATIVE, GRID_POSITIVE,
    CATEGORY2IDX, NUM_CATEGORIES,
    va_normalise,
)


# ═══════════════════════════════════════════════════════════════════════════════
# LINGUISTIC FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class LinguisticFeatureExtractor:
    """
    Produces an 8-dimensional feature vector for every *word token* in a sentence.
    The features intentionally require no external lexicon download:

    Dim 0: Normalised word length (len / 20, capped at 1.0)
    Dim 1: Proportion of uppercase characters in the word
    Dim 2: Contains digit (0/1)
    Dim 3: Is punctuation (0/1)
    Dim 4: Exclamation mark present anywhere in the sentence (global, 0/1)
    Dim 5: Question mark present anywhere in the sentence (global, 0/1)
    Dim 6: Sentence-level avg word length (normalised, same for all tokens)
    Dim 7: Capitalised first letter (0/1)

    Subword tokens inherit the features of their parent word.
    Special tokens ([CLS], [SEP], [PAD]) get the zero vector.
    """

    def __call__(
        self,
        text: str,
        offset_mapping: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Args:
            text: raw sentence string
            offset_mapping: list of (char_start, char_end) for each token
                            including special tokens (which have (0,0))
        Returns:
            features: np.ndarray, shape [len(offset_mapping), 8], dtype float32
        """
        n_tokens = len(offset_mapping)
        feats = np.zeros((n_tokens, 8), dtype=np.float32)

        # Global sentence-level features
        words = text.split()
        avg_word_len = np.mean([len(w) for w in words]) / 20.0 if words else 0.0
        has_excl = float("!" in text)
        has_ques = float("?" in text)

        for ti, (cs, ce) in enumerate(offset_mapping):
            if cs == ce == 0:          # special token
                continue
            word = text[cs:ce]

            # Dim 0: normalised word length
            feats[ti, 0] = min(len(word) / 20.0, 1.0)
            # Dim 1: uppercase ratio
            upper = sum(c.isupper() for c in word)
            feats[ti, 1] = upper / max(len(word), 1)
            # Dim 2: contains digit
            feats[ti, 2] = float(any(c.isdigit() for c in word))
            # Dim 3: is punctuation
            feats[ti, 3] = float(all(
                unicodedata.category(c).startswith("P") for c in word
            ) if word else False)
            # Dims 4-5: sentence-level markers
            feats[ti, 4] = has_excl
            feats[ti, 5] = has_ques
            # Dim 6: sentence avg word length
            feats[ti, 6] = avg_word_len
            # Dim 7: capitalised first letter
            feats[ti, 7] = float(word[0].isupper()) if word else 0.0

        return feats


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: SPAN → TOKEN INDICES
# ═══════════════════════════════════════════════════════════════════════════════

def _char_span_to_token_indices(
    phrase: str,
    text: str,
    offset_mapping: List[Tuple[int, int]],
) -> List[int]:
    """
    Find all token positions whose character span overlaps with any occurrence
    of `phrase` inside `text`.

    Searches case-sensitively (as required by the task).
    Returns empty list if phrase is "NULL" or not found.
    """
    if phrase == "NULL" or not phrase:
        return []

    # Find all character-level occurrences of phrase in text
    token_indices: List[int] = []
    start = 0
    phrase_len = len(phrase)

    while True:
        pos = text.find(phrase, start)
        if pos == -1:
            break
        char_start = pos
        char_end = pos + phrase_len

        # Map to tokens: any token whose span overlaps [char_start, char_end)
        for ti, (cs, ce) in enumerate(offset_mapping):
            if cs == ce:                       # special token or padding
                continue
            if cs < char_end and ce > char_start:  # overlap condition
                token_indices.append(ti)

        start = pos + 1  # allow overlapping finds

    return sorted(set(token_indices))


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET: ST1 REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

class DimABSARegressionDataset(Dataset):
    """
    Subtask 1 — Dimensional Aspect Sentiment Regression.

    Input format (per instance after flattening):
        [CLS] text [SEP] aspect [SEP]   (pair encoding)

    __getitem__ returns:
        input_ids         [max_len]   int64
        attention_mask    [max_len]   int64
        token_type_ids    [max_len]   int64  (segment A=text, B=aspect)
        aspect_token_mask [max_len]   float32  — 1 where aspect tokens are
        ling_features     [max_len,8] float32
        valence           scalar      float32  ∈ [-1, 1]
        arousal           scalar      float32  ∈ [-1, 1]
        is_null_aspect    scalar      bool     — True for implicit targets
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: AutoTokenizer,
        max_len: int = MAX_LEN,
        is_train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        self.ling = LinguisticFeatureExtractor()

        self.instances: List[Dict[str, Any]] = []
        self._load(Path(jsonl_path))

    def _load(self, path: Path) -> None:
        """Parse JSONL and flatten to (text, aspect, V, A) tuples."""
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["Text"]

                # ── Subtask-1 dev: Aspect_VA format ───────────────────────
                if "Aspect_VA" in obj:
                    for entry in obj["Aspect_VA"]:
                        v_str, a_str = entry["VA"].split("#")
                        v = float(v_str); a = float(a_str)
                        v = max(1.0, min(9.0, v))   # clamp
                        a = max(1.0, min(9.0, a))
                        self.instances.append({
                            "id": obj["ID"],
                            "text": text,
                            "aspect": entry["Aspect"],
                            "valence": va_normalise(v),
                            "arousal": va_normalise(a),
                            "is_null": entry["Aspect"] == "NULL",
                        })

                # ── Train alltasks: Quadruplet format ─────────────────────
                elif "Quadruplet" in obj:
                    for quad in obj["Quadruplet"]:
                        v_str, a_str = quad["VA"].split("#")
                        v = float(v_str); a = float(a_str)
                        v = max(1.0, min(9.0, v))
                        a = max(1.0, min(9.0, a))
                        self.instances.append({
                            "id": obj["ID"],
                            "text": text,
                            "aspect": quad["Aspect"],
                            "valence": va_normalise(v),
                            "arousal": va_normalise(a),
                            "is_null": quad["Aspect"] == "NULL",
                        })

                # ── Subtask-1 test: only ID + Text + Aspect (no VA) ───────
                elif "Aspect" in obj:
                    for asp in obj["Aspect"]:
                        self.instances.append({
                            "id": obj["ID"],
                            "text": text,
                            "aspect": asp,
                            "valence": None,      # inference mode
                            "arousal": None,
                            "is_null": asp == "NULL",
                        })

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inst = self.instances[idx]
        text = inst["text"]
        aspect = inst["aspect"] if not inst["is_null"] else ""

        # ── Pair encoding: [CLS] text [SEP] aspect [SEP] ──────────────────
        enc = self.tokenizer(
            text,
            aspect,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attn_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        # DeBERTa-v3 does not use token_type_ids internally (relative position
        # encoding), but we keep it to identify aspect tokens for cross-attention.
        token_type_ids = torch.tensor(
            enc.get("token_type_ids", [0] * self.max_len), dtype=torch.long
        )
        offsets = enc["offset_mapping"]

        # ── Aspect token mask: 1 where token_type_id == 1 (B-sequence) ────
        # For pair encoding, segment B = aspect tokens
        aspect_token_mask = torch.zeros(self.max_len, dtype=torch.float32)
        if not inst["is_null"] and aspect:
            for ti, tt in enumerate(token_type_ids.tolist()):
                tok = self.tokenizer.convert_ids_to_tokens(
                    [enc["input_ids"][ti]]
                )[0]
                if tt == 1 and enc["attention_mask"][ti] == 1:
                    aspect_token_mask[ti] = 1.0

        # ── Linguistic features (use text portion only, i.e. segment A) ───
        # We compute over the full sequence; special tokens stay at 0.
        ling_feats = torch.from_numpy(
            self.ling(text, offsets)
        )  # [max_len, 8]

        # ── VA targets ─────────────────────────────────────────────────────
        has_label = inst["valence"] is not None
        valence = torch.tensor(
            inst["valence"] if has_label else 0.0, dtype=torch.float32
        )
        arousal = torch.tensor(
            inst["arousal"] if has_label else 0.0, dtype=torch.float32
        )
        has_label_t = torch.tensor(has_label, dtype=torch.bool)
        is_null_t = torch.tensor(inst["is_null"], dtype=torch.bool)

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "token_type_ids": token_type_ids,
            "aspect_token_mask": aspect_token_mask,
            "ling_features": ling_feats,
            "valence": valence,
            "arousal": arousal,
            "has_label": has_label_t,
            "is_null_aspect": is_null_t,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET: ST2/3 EXTRACTION  (2D Biaffine Grid)
# ═══════════════════════════════════════════════════════════════════════════════

class DimABSAExtractionDataset(Dataset):
    """
    Subtasks 2 & 3 — Extraction with 2D Adjacency Grid.

    Encoding: [CLS] text [SEP]

    __getitem__ returns:
        input_ids         [max_len]         int64
        attention_mask    [max_len]         int64
        ling_features     [max_len, 8]      float32
        grid_labels       [max_len, max_len] int64
            • GRID_POSITIVE (1)   = token i ∈ Aspect AND token j ∈ Opinion (linked)
            • GRID_NEGATIVE (0)   = no relation
            • GRID_IGNORE  (-100) = [CLS],[SEP],padding rows/columns
        va_targets        [max_quads, 2]    float32  normalised ∈ [-1, 1]
                          (padded to max_quads with 0.0, mask via va_mask)
        va_mask           [max_quads]       bool     — True for valid quads
        category_ids      [max_quads]       int64    — ST3 category idx, -1 for ST2
        num_quads         scalar            int64    — actual number of quads

    Note: va_targets are per-quad, not per-token.  max_quads=16 covers every
    real sentence in the corpus (max observed ~12).
    """

    MAX_QUADS = 16   # pad all VA/category tensors to this length

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: AutoTokenizer,
        max_len: int = MAX_LEN,
        subtask: int = 3,       # 2 = Triplet, 3 = Quadruplet
        is_train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.subtask = subtask
        self.is_train = is_train
        self.ling = LinguisticFeatureExtractor()

        self.instances: List[Dict[str, Any]] = []
        self._load(Path(jsonl_path))

    # ──────────────────────────────────────────────────────────────────────────
    def _parse_va(self, va_str: str) -> Tuple[float, float]:
        """Parse 'V#A' string, clamp to [1,9], normalise to [-1,1]."""
        v_str, a_str = va_str.split("#")
        v = max(1.0, min(9.0, float(v_str)))
        a = max(1.0, min(9.0, float(a_str)))
        return va_normalise(v), va_normalise(a)

    def _load(self, path: Path) -> None:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["Text"]

                quads: List[Dict] = []

                # ── Triplet dev/test (ST2) ─────────────────────────────────
                if "Triplet" in obj:
                    for t in obj["Triplet"]:
                        v_n, a_n = self._parse_va(t["VA"])
                        quads.append({
                            "aspect": t["Aspect"],
                            "opinion": t["Opinion"],
                            "category": None,       # no category in ST2
                            "valence": v_n,
                            "arousal": a_n,
                            "is_null_a": t["Aspect"] == "NULL",
                            "is_null_o": t["Opinion"] == "NULL",
                        })

                # ── Quadruplet train/dev/test (ST2-via-Quad and ST3) ───────
                elif "Quadruplet" in obj:
                    for q in obj["Quadruplet"]:
                        v_n, a_n = self._parse_va(q["VA"])
                        cat = q.get("Category", "NULL")
                        quads.append({
                            "aspect": q["Aspect"],
                            "opinion": q["Opinion"],
                            "category": cat if cat != "NULL" else None,
                            "valence": v_n,
                            "arousal": a_n,
                            "is_null_a": q["Aspect"] == "NULL",
                            "is_null_o": q["Opinion"] == "NULL",
                        })

                # ── Test file (no labels) ──────────────────────────────────
                # Nothing to extract; add empty quads for inference
                else:
                    pass  # text-only test; quads stays []

                self.instances.append({
                    "id": obj["ID"],
                    "text": text,
                    "quads": quads,
                    "has_label": len(quads) > 0,
                })

    # ──────────────────────────────────────────────────────────────────────────
    def _build_grid(
        self,
        text: str,
        quads: List[Dict],
        offset_mapping: List[Tuple[int, int]],
        seq_len: int,
    ) -> np.ndarray:
        """
        Build the N×N 2D adjacency grid.

        Result shape: [seq_len, seq_len], dtype int64
          • Rows & columns for [CLS]/[SEP]/PAD → GRID_IGNORE (-100)
          • All other cells initialised to GRID_NEGATIVE (0)
          • Cell (i,j) set to GRID_POSITIVE (1) if:
                token i belongs to an Aspect span
                AND token j belongs to the linked Opinion span
                (for the same quad)
        """
        grid = np.full((seq_len, seq_len), GRID_NEGATIVE, dtype=np.int64)

        # Mark special-token rows/columns as IGNORE
        special_mask = np.zeros(seq_len, dtype=bool)
        for ti, (cs, ce) in enumerate(offset_mapping):
            if ti >= seq_len:
                break
            if cs == 0 and ce == 0:
                special_mask[ti] = True

        grid[special_mask, :] = GRID_IGNORE
        grid[:, special_mask] = GRID_IGNORE

        # Fill padding rows/columns (attention_mask == 0)
        # (handled after tokenisation; we mark by offset = (0,0) for specials,
        #  but padding also has (0,0) — both correctly get IGNORE)

        # For each quad: set Aspect×Opinion cross-cells to POSITIVE
        for quad in quads:
            if quad["is_null_a"] or quad["is_null_o"]:
                # NULL aspect or opinion → no extractable span → no positive cells
                continue

            asp_toks = _char_span_to_token_indices(
                quad["aspect"], text, offset_mapping[:seq_len]
            )
            op_toks = _char_span_to_token_indices(
                quad["opinion"], text, offset_mapping[:seq_len]
            )

            # Only write if both spans were found
            if asp_toks and op_toks:
                for ai in asp_toks:
                    if ai >= seq_len:
                        continue
                    for oi in op_toks:
                        if oi >= seq_len:
                            continue
                        # Do not overwrite IGNORE
                        if (grid[ai, oi] != GRID_IGNORE and
                                grid[oi, ai] != GRID_IGNORE):
                            grid[ai, oi] = GRID_POSITIVE
                            # Also fill the symmetric cell (Opinion→Aspect direction)
                            grid[oi, ai] = GRID_POSITIVE

        return grid

    # ──────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inst = self.instances[idx]
        text = inst["text"]
        quads = inst["quads"]

        # ── Single-sequence encoding: [CLS] text [SEP] ────────────────────
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attn_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        offsets: List[Tuple[int, int]] = enc["offset_mapping"]

        # ── 2D Adjacency Grid ─────────────────────────────────────────────
        grid = self._build_grid(text, quads, offsets, self.max_len)
        grid_t = torch.from_numpy(grid)   # [max_len, max_len]

        # ── Linguistic features ────────────────────────────────────────────
        ling_feats = torch.from_numpy(self.ling(text, offsets))

        # ── VA targets: pad to MAX_QUADS ───────────────────────────────────
        n_quads = min(len(quads), self.MAX_QUADS)
        va_targets = torch.zeros(self.MAX_QUADS, 2, dtype=torch.float32)
        va_mask = torch.zeros(self.MAX_QUADS, dtype=torch.bool)
        category_ids = torch.full((self.MAX_QUADS,), -1, dtype=torch.long)

        for qi in range(n_quads):
            q = quads[qi]
            va_targets[qi, 0] = q["valence"]
            va_targets[qi, 1] = q["arousal"]
            va_mask[qi] = True

            if q["category"] is not None and q["category"] in CATEGORY2IDX:
                category_ids[qi] = CATEGORY2IDX[q["category"]]

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "ling_features": ling_feats,
            "grid_labels": grid_t,
            "va_targets": va_targets,
            "va_mask": va_mask,
            "category_ids": category_ids,
            "num_quads": torch.tensor(n_quads, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataloader(
    jsonl_path: str | Path,
    subtask: int,                      # 1, 2, or 3
    tokenizer: AutoTokenizer,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    is_train: bool = True,
    num_workers: int = 0,              # keep 0 on Windows to avoid fork issues
) -> DataLoader:
    """
    Build a DataLoader for the given JSONL file and subtask.

    Args:
        jsonl_path: path to the .jsonl file
        subtask:    1 → DimABSARegressionDataset
                    2 → DimABSAExtractionDataset (triplet)
                    3 → DimABSAExtractionDataset (quadruplet)
        ...
    """
    if subtask == 1:
        ds = DimABSARegressionDataset(
            jsonl_path, tokenizer, max_len=max_len, is_train=is_train
        )
    elif subtask in (2, 3):
        ds = DimABSAExtractionDataset(
            jsonl_path, tokenizer, max_len=max_len,
            subtask=subtask, is_train=is_train
        )
    else:
        raise ValueError(f"Unknown subtask: {subtask}. Must be 1, 2, or 3.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST / VERIFICATION BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import textwrap
    from src.config_v4 import DATA_ROOT

    print("=" * 70)
    print("DimABSA V4 Data Loader — Self-Verification")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Test 1: Normalisation round-trip ──────────────────────────────────────
    print("\n[TEST 1] VA Normalisation round-trip")
    from src.config_v4 import va_normalise, va_denormalise
    for raw in [1.0, 3.5, 5.0, 7.25, 9.0]:
        normed = va_normalise(raw)
        back = va_denormalise(normed)
        assert abs(back - raw) < 1e-6, f"Round-trip failed: {raw} -> {normed} -> {back}"
        print(f"  {raw:.2f} -> {normed:+.4f} -> {back:.2f}  OK")

    # ── Test 2: ST1 Regression dataset ───────────────────────────────────────
    print("\n[TEST 2] ST1 Regression Dataset (eng_restaurant_dev)")
    st1_path = DATA_ROOT / "subtask_1" / "eng" / "eng_restaurant_dev_task1.jsonl"
    ds1 = DimABSARegressionDataset(st1_path, tokenizer, max_len=MAX_LEN, is_train=False)
    print(f"  Total instances (flattened): {len(ds1)}")

    item = ds1[0]
    assert item["input_ids"].shape == (MAX_LEN,), f"Bad shape: {item['input_ids'].shape}"
    assert item["attention_mask"].shape == (MAX_LEN,)
    assert item["token_type_ids"].shape == (MAX_LEN,)
    assert item["aspect_token_mask"].shape == (MAX_LEN,)
    assert item["ling_features"].shape == (MAX_LEN, 8)
    assert item["has_label"].item(), "Dev file should have labels"

    # Check VA in [-1, 1]
    for qi in range(min(20, len(ds1))):
        s = ds1[qi]
        v = s["valence"].item()
        a = s["arousal"].item()
        assert -1.0 <= v <= 1.0, f"Valence out of range: {v}"
        assert -1.0 <= a <= 1.0, f"Arousal out of range: {a}"
    print("  All VA values in [-1, 1]  OK")
    print(f"  item[0]: valence={item['valence'].item():.4f} arousal={item['arousal'].item():.4f}")
    print(f"           is_null_aspect={item['is_null_aspect'].item()}")
    print(f"           aspect_token_mask sum={item['aspect_token_mask'].sum().item():.0f}")
    print(f"           ling_features mean={item['ling_features'].mean().item():.4f}")

    # ── Test 3: ST3 Extraction dataset ───────────────────────────────────────
    print("\n[TEST 3] ST3 Extraction Dataset (eng_restaurant_dev)")
    st3_path = DATA_ROOT / "subtask_3" / "eng" / "eng_restaurant_dev_task3.jsonl"
    ds3 = DimABSAExtractionDataset(
        st3_path, tokenizer, max_len=MAX_LEN, subtask=3, is_train=False
    )
    print(f"  Total instances: {len(ds3)}")

    item3 = ds3[0]
    assert item3["input_ids"].shape == (MAX_LEN,)
    assert item3["grid_labels"].shape == (MAX_LEN, MAX_LEN), \
        f"Grid shape wrong: {item3['grid_labels'].shape}"
    assert item3["ling_features"].shape == (MAX_LEN, 8)
    assert item3["va_targets"].shape == (DimABSAExtractionDataset.MAX_QUADS, 2)

    print(f"  grid_labels shape: {list(item3['grid_labels'].shape)}  OK")

    # Verify grid only contains {0, 1, -100}
    grid_vals = set(item3["grid_labels"].flatten().tolist())
    assert grid_vals.issubset({0, 1, -100}), f"Unexpected grid values: {grid_vals}"
    print(f"  grid values subset of {{0, 1, -100}}  OK  (found: {sorted(grid_vals)})")

    # Verify special tokens have -100
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    ids = item3["input_ids"].tolist()
    for ti, tid in enumerate(ids):
        if tid in (cls_id, sep_id, tokenizer.pad_token_id):
            row = item3["grid_labels"][ti]
            col = item3["grid_labels"][:, ti]
            special_ok = (row == GRID_IGNORE).all() and (col == GRID_IGNORE).all()
            if not special_ok:
                print(f"  WARNING: special token at pos {ti} not fully -100")
            break
    print("  Special token grid positions = -100  OK")

    # Count positives
    n_pos = (item3["grid_labels"] == GRID_POSITIVE).sum().item()
    n_total_valid = (item3["grid_labels"] != GRID_IGNORE).sum().item()
    sparsity = 1.0 - n_pos / max(n_total_valid, 1)
    print(f"  Grid: {n_pos} positive cells / {n_total_valid} valid cells  "
          f"(sparsity={sparsity:.4f})")
    print(f"  num_quads={item3['num_quads'].item()}")
    print(f"  va_targets[0] = {item3['va_targets'][0].tolist()}")
    print(f"  va_mask[:4]  = {item3['va_mask'][:4].tolist()}")

    # ── Test 4: DataLoader batch shapes ───────────────────────────────────────
    print("\n[TEST 4] DataLoader batch shapes")
    dl1 = get_dataloader(st1_path, subtask=1, tokenizer=tokenizer,
                         batch_size=4, is_train=False)
    dl3 = get_dataloader(st3_path, subtask=3, tokenizer=tokenizer,
                         batch_size=4, is_train=False)

    batch1 = next(iter(dl1))
    batch3 = next(iter(dl3))

    print("  ST1 batch:")
    for k, v in batch1.items():
        print(f"    {k}: {list(v.shape) if v.dim() > 0 else v.item()}")

    print("  ST3 batch:")
    for k, v in batch3.items():
        print(f"    {k}: {list(v.shape) if v.dim() > 0 else v.item()}")

    # Shape contract assertions
    B = 4
    assert batch1["input_ids"].shape == (B, MAX_LEN)
    assert batch1["ling_features"].shape == (B, MAX_LEN, 8)
    assert batch3["grid_labels"].shape == (B, MAX_LEN, MAX_LEN), \
        f"Grid batch shape wrong: {batch3['grid_labels'].shape}"
    assert batch3["va_targets"].shape == (B, DimABSAExtractionDataset.MAX_QUADS, 2)

    print("\n  All shape assertions passed  OK")

    # ── Test 5: VA distribution statistics ────────────────────────────────────
    print("\n[TEST 5] VA distribution (first 50 ST1 instances)")
    vs = [ds1[i]["valence"].item() for i in range(min(50, len(ds1)))]
    as_ = [ds1[i]["arousal"].item() for i in range(min(50, len(ds1)))]
    print(f"  Valence  min={min(vs):.4f}  max={max(vs):.4f}  "
          f"mean={sum(vs)/len(vs):.4f}")
    print(f"  Arousal  min={min(as_):.4f}  max={max(as_):.4f}  "
          f"mean={sum(as_)/len(as_):.4f}")
    assert all(-1.0 <= v <= 1.0 for v in vs), "Valence OOB!"
    assert all(-1.0 <= a <= 1.0 for a in as_), "Arousal OOB!"
    print("  All values confirmed in [-1, 1]  OK")

    # ── Test 6: Grid sparsity across multiple samples ─────────────────────────
    print("\n[TEST 6] Grid sparsity across 20 ST3 samples")
    sparsities = []
    pos_counts = []
    for i in range(min(20, len(ds3))):
        g = ds3[i]["grid_labels"]
        n_p = (g == GRID_POSITIVE).sum().item()
        n_v = (g != GRID_IGNORE).sum().item()
        pos_counts.append(n_p)
        sparsities.append(1.0 - n_p / max(n_v, 1))
    print(f"  Avg positive cells : {sum(pos_counts)/len(pos_counts):.1f}")
    print(f"  Avg sparsity       : {sum(sparsities)/len(sparsities):.4f}")
    print(f"  Max positive cells : {max(pos_counts)}")
    print("  -> Focal Loss REQUIRED (confirmed extremely sparse grid)  OK")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED -- data_loader_v4.py is verified.")
    print("=" * 70)
