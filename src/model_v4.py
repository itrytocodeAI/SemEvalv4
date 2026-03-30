"""
model_v4.py — DimABSA V4 Neural Architecture.

Single DeBERTa-v3-base backbone with two task heads:

  ST1 — Regression Head (Tanh):
        [CLS text SEP aspect SEP]
        → DeBERTa hidden states [B, L, 768]
        → inject 8-dim linguistic features via LayerNorm + proj
        → token-index slicing cross-attention  (Q=CLS, K=V=aspect tokens)
        → LayerNorm(cat(cls, cross)) → FFN(768*2, 256) → Linear(256,2) → Tanh
        → pred_va  [B, 2]  in (-1, 1)

  ST2/3 — Biaffine Extraction Head:
        [CLS text SEP]
        → DeBERTa hidden states [B, L, 768]
        → inject linguistic features
        → project [B,L,768] → [B,L,256]  (VRAM safety before N×N multiply)
        → Biaffine: h_asp @ W @ h_op^T + b  →  grid_logits [B, L, L]
        → category head: [B, L, num_cats]

  Shared:
        log_vars [3]   — homoscedastic uncertainty params (LogSigma)
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config_v4 import (
    MODEL_NAME, HIDDEN_SIZE, BIAFFINE_DIM, LING_DIM,
    NUM_CATEGORIES, MAX_LEN, BATCH_SIZE,
)
from src.losses_v4 import (
    hybrid_regression_loss, focal_loss, uncertainty_weighted_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class DimABSAV4Model(nn.Module):
    """
    V4 DimABSA model.  Single-pass DeBERTa-v3-base encoder shared across
    all subtasks for VRAM efficiency.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        hidden_size: int = HIDDEN_SIZE,
        biaffine_dim: int = BIAFFINE_DIM,
        ling_dim: int = LING_DIM,
        num_categories: int = NUM_CATEGORIES,
        dropout: float = 0.2,
        use_ling_features: bool = True,
        use_hybrid_loss: bool = True,
        use_focal_loss: bool = True,
        use_uncertainty_weight: bool = True,
    ):
        super().__init__()
        H = hidden_size
        D = biaffine_dim
        
        self.use_ling_features = use_ling_features
        self.use_hybrid_loss = use_hybrid_loss
        self.use_focal_loss = use_focal_loss
        self.use_uncertainty_weight = use_uncertainty_weight

        # ── 1. Backbone ──────────────────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(model_name)

        # ── 2. Linguistic feature injection (shared) ─────────────────────────
        # 8-dim raw features → LayerNorm → project to H → add to hidden states
        self.ling_norm = nn.LayerNorm(ling_dim)
        self.ling_proj = nn.Linear(ling_dim, H, bias=False)

        # ── 3a. Cross-attention  [V3 foundation] ─────────────────────────────
        # Q = CLS token, K = V = aspect token hidden states (token-index sliced)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=H, num_heads=8, dropout=dropout, batch_first=True
        )

        # ── 3b. Regression head (ST1) ─────────────────────────────────────────
        self.reg_norm = nn.LayerNorm(H * 2)
        self.reg_ffn  = nn.Sequential(
            nn.Linear(H * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Tanh final activation → output naturally in (-1, 1)
        self.reg_head = nn.Sequential(
            nn.Linear(256, 2),
            nn.Tanh(),
        )

        # ── 4a. Biaffine projection (VRAM safety: 768 → 256) ─────────────────
        self.biaffine_proj_asp = nn.Linear(H, D, bias=False)
        self.biaffine_proj_op  = nn.Linear(H, D, bias=False)

        # ── 4b. Biaffine weight matrix + bias ─────────────────────────────────
        # score[b,i,j] = h_asp[b,i] @ W @ h_op[b,j]^T + b
        self.biaffine_W = nn.Parameter(torch.empty(D, D))
        self.biaffine_b = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.biaffine_W.unsqueeze(0))

        # ── 5. Category head (ST3) ────────────────────────────────────────────
        # Applied to the hidden state at each token position;
        # at training time, supervision at gold aspect-start tokens.
        self.category_head = nn.Sequential(
            nn.Linear(H, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_categories),
        )

        # ── 6. Uncertainty weights ─────────────────────────────────────────────
        # [0]=reg_valence, [1]=reg_arousal, [2]=extraction  (LogSigma §4)
        self.log_vars = nn.Parameter(torch.zeros(3))

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _inject_ling(
        self,
        hidden: torch.Tensor,        # [B, L, H]
        ling_features: torch.Tensor, # [B, L, 8]
    ) -> torch.Tensor:
        """Normalise ling features, project to H, residual-add to hidden."""
        if not self.use_ling_features:
            return hidden
        ling = self.ling_norm(ling_features.float())  # [B, L, 8]
        ling = self.ling_proj(ling)                   # [B, L, H]
        return hidden + ling

    def _cross_attend(
        self,
        hidden: torch.Tensor,           # [B, L, H]
        aspect_token_mask: torch.Tensor,# [B, L]  float; 1=aspect token
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Token-index slicing cross-attention (V3 foundation).

        Slices aspect token hidden states from each sample, builds a padded
        Key/Value tensor, then runs:
            Q = CLS token  [B, 1, H]
            K = V = aspect token hidden states  [B, max_asp, H]

        Returns:
            cls_repr   [B, H]  — raw CLS for fusion
            cross_repr [B, H]  — CLS after attending to aspect (or CLS for NULLs)
        """
        B, L, H = hidden.shape
        cls_repr  = hidden[:, 0, :]    # [B, H]

        asp_count = aspect_token_mask.sum(dim=1).long()          # [B]
        max_asp   = max(int(asp_count.max().item()), 1)

        # Build padded KV tensor via token-index slicing
        asp_hidden = hidden.new_zeros(B, max_asp, H)
        asp_kpm    = hidden.new_ones(B, max_asp).bool()  # True = ignore

        for b in range(B):
            idx = aspect_token_mask[b].bool().nonzero(as_tuple=True)[0]
            n   = idx.shape[0]
            if n > 0:
                asp_hidden[b, :n] = hidden[b, idx]  # token-index slice
                asp_kpm[b, :n]    = False            # these are valid

        # Cross-attention: ensure every sample has at least one unmasked token.
        # Otherwise MultiheadAttention can return NaNs if any row in key_padding_mask is all True.
        safe_kpm = asp_kpm.clone()
        safe_kpm[:, 0] = False # Unmask the first token for every sample to avoid MHA NaNs
        
        cls_q = cls_repr.unsqueeze(1)          # [B, 1, H]
        cross_out, _ = self.cross_attn(
            query=cls_q,
            key=asp_hidden,
            value=asp_hidden,
            key_padding_mask=safe_kpm,
        )                                      # [B, 1, H]
        cross_repr = cross_out.squeeze(1)      # [B, H]

        # NULL-aspect fallback: if a specific sample has no aspect tokens, use its CLS directly
        is_null = (asp_count == 0)
        if is_null.any():
            cross_repr = cross_repr.clone()
            cross_repr[is_null] = cls_repr[is_null].to(cross_repr.dtype)

        return cls_repr, cross_repr

    # ─────────────────────────────────────────────────────────────────────────
    # Task-specific forward passes
    # ─────────────────────────────────────────────────────────────────────────

    def forward_regression(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        ling_features: torch.Tensor | None = None,
        aspect_token_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        ST1 regression forward.
        Supports inputs_embeds for Integrated Gradients.
        """
        # 1. Encode
        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        hidden = enc.last_hidden_state      # [B, L, H]

        # 2. Inject linguistic features
        if ling_features is not None:
            hidden = self._inject_ling(hidden, ling_features)

        # 3. Token-index slicing cross-attention
        cls_repr, cross_repr = self._cross_attend(hidden, aspect_token_mask)

        # 4. Fuse CLS + cross-attended representation
        fused = torch.cat([cls_repr, cross_repr], dim=-1)  # [B, 2H]
        fused = self.reg_norm(fused)
        fused = self.reg_ffn(fused)                        # [B, 256]

        # 5. Tanh regression head → (-1, 1) output
        pred_va = self.reg_head(fused)                     # [B, 2]
        return pred_va

    def forward_extraction(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        ling_features: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ST2/3 extraction forward.
            grid_logits  [B, L, L]          — raw biaffine grid scores
            cat_logits   [B, L, num_cats]   — per-token category logits
        """
        # 1. Encode text-only: [CLS text SEP]
        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = enc.last_hidden_state     # [B, L, H]

        # 2. Inject linguistic features
        hidden = self._inject_ling(hidden, ling_features)

        # 3. VRAM-safe projection: [B, L, 768] → [B, L, 256]
        h_asp = self.biaffine_proj_asp(hidden)   # [B, L, D]
        h_op  = self.biaffine_proj_op(hidden)    # [B, L, D]

        # 4. Biaffine grid: score[b,i,j] = h_asp[b,i] @ W @ h_op[b,j] + b
        #    (B,L,D) @ (D,D) = (B,L,D)  →  bmm with (B,D,L)  →  (B,L,L)
        aw = h_asp @ self.biaffine_W              # [B, L, D]
        grid_logits = torch.bmm(aw, h_op.transpose(1, 2)) + self.biaffine_b  # [B, L, L]

        # 5. Mask padding rows/columns with -inf so they don't affect loss
        pad = (attention_mask == 0)               # [B, L]
        grid_logits = grid_logits.masked_fill(pad.unsqueeze(2), float("-inf"))
        grid_logits = grid_logits.masked_fill(pad.unsqueeze(1), float("-inf"))

        # 6. Category head — per-token logits
        cat_logits = self.category_head(hidden)   # [B, L, num_cats]

        return grid_logits, cat_logits

    # ─────────────────────────────────────────────────────────────────────────
    # Unified forward + loss computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_regression_loss(
        self,
        pred_va: torch.Tensor,   # [B, 2]
        valence: torch.Tensor,   # [B]
        arousal: torch.Tensor,   # [B]
        has_label: torch.Tensor, # [B] bool
        use_uw: bool = False,
        alpha_huber: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Compute regression loss on labelled samples only.

        Returns dict with keys: loss, loss_huber_ccc, pred_va_valid
        """
        valid = has_label.bool()
        if valid.sum() == 0:
            zero = pred_va.sum() * 0.0
            return {"loss": zero, "loss_reg": zero}

        pred_v  = pred_va[valid]                          # [N, 2]
        gold_va = torch.stack([valence[valid],
                               arousal[valid]], dim=-1)   # [N, 2]

        active_alpha = alpha_huber if self.use_hybrid_loss else 1.0  # 1.0 Huber is purely MSE-like without CCC
        loss_reg = hybrid_regression_loss(pred_v, gold_va, alpha=active_alpha)

        # Uncertainty weighting
        if use_uw and self.use_uncertainty_weight:
            log_v = self.log_vars[0].clamp(-3, 3)
            log_a = self.log_vars[1].clamp(-3, 3)
            lv    = hybrid_regression_loss(
                pred_v[:, :1], gold_va[:, :1], alpha=active_alpha
            )
            la    = hybrid_regression_loss(
                pred_v[:, 1:], gold_va[:, 1:], alpha=active_alpha
            )
            loss_reg = (torch.exp(-log_v) * lv + log_v +
                        torch.exp(-log_a) * la + log_a)

        return {"loss": loss_reg, "loss_reg": loss_reg, "v_loss": lv if (use_uw and self.use_uncertainty_weight) else F.mse_loss(pred_v[:, :1], gold_va[:, :1]), "a_loss": la if (use_uw and self.use_uncertainty_weight) else F.mse_loss(pred_v[:, 1:], gold_va[:, 1:])}

    def compute_extraction_loss(
        self,
        grid_logits: torch.Tensor,   # [B, L, L]
        cat_logits: torch.Tensor,    # [B, L, num_cats]
        grid_labels: torch.Tensor,   # [B, L, L]  {0,1,-100}
        category_ids: torch.Tensor,  # [B, MAX_QUADS]  {-1 or cat_idx}
        va_mask: torch.Tensor,       # [B, MAX_QUADS]
        use_uw: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> dict[str, torch.Tensor]:
        """
        Focal loss on biaffine grid + CrossEntropy on category labels.

        Category supervision: for each valid quad, the category label is
        attached to the *first token* of the sentence as a proxy
        (refined in Phase 3 when we have aspect-token-level alignment).
        """
        # Grid loss
        if self.use_focal_loss:
            loss_focal = focal_loss(
                grid_logits, grid_labels,
                alpha=focal_alpha, gamma=focal_gamma,
            )
        else:
            valid = grid_labels != -100
            if valid.sum() > 0:
                loss_focal = F.binary_cross_entropy_with_logits(
                    grid_logits[valid], grid_labels[valid].float()
                )
            else:
                loss_focal = grid_logits.sum() * 0.0

        # Category loss — only on valid quads that have a real category
        # Use [CLS] representation's category prediction as a sentence-level proxy
        # cat_logits[:, 0, :] = category logits at CLS position [B, num_cats]
        cat_valid_mask = (category_ids >= 0) & va_mask  # [B, MAX_QUADS]
        loss_cat = torch.tensor(0.0, device=grid_logits.device)
        n_cat = cat_valid_mask.sum().item()
        if n_cat > 0:
            # Gather CLS logits per batch sample
            cls_cat = cat_logits[:, 0, :]   # [B, num_cats]
            # For simplicity in Phase 2: supervise with the first valid category
            # Phase 3 will align to gold aspect token positions
            first_cat = torch.zeros(
                cat_logits.shape[0], dtype=torch.long, device=cat_logits.device
            )
            for b in range(cat_logits.shape[0]):
                valid_q = cat_valid_mask[b].nonzero(as_tuple=True)[0]
                if len(valid_q) > 0:
                    first_cat[b] = category_ids[b, valid_q[0]]
            has_any_cat = cat_valid_mask.any(dim=1)  # [B]
            if has_any_cat.sum() > 0:
                loss_cat = F.cross_entropy(
                    cls_cat[has_any_cat], first_cat[has_any_cat]
                )

        loss_ext = loss_focal + 0.5 * loss_cat

        if use_uw and self.use_uncertainty_weight:
            log_e = self.log_vars[2].clamp(-3, 3)
            loss_ext = torch.exp(-log_e) * loss_ext + log_e

        return {
            "loss": loss_ext,
            "loss_focal": loss_focal,
            "loss_cat": loss_cat,
        }

    @torch.no_grad()
    def get_integrated_gradients(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        ling_features: torch.Tensor,
        aspect_token_mask: torch.Tensor,
        target_dim: int = 0, # 0=Valence, 1=Arousal
        n_steps: int = 20,
    ) -> torch.Tensor:
        """
        XAI: Compute Integrated Gradients for a single regression sample.
        Returns: [L] attribution scores per token.
        """
        self.eval()
        # 1. Get word embeddings baseline (zero)
        embeddings_layer = self.encoder.get_input_embeddings()
        original_embeddings = embeddings_layer(input_ids) # [1, L, H]
        baseline = torch.zeros_like(original_embeddings)
        
        # 2. Linear interpolate 
        alphas = torch.linspace(0, 1, n_steps).to(input_ids.device)
        total_grads = torch.zeros_like(original_embeddings)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (original_embeddings - baseline)
            interpolated.requires_grad_(True)
            
            with torch.enable_grad():
                pred = self.forward_regression(
                    inputs_embeds=interpolated,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    ling_features=ling_features,
                    aspect_token_mask=aspect_token_mask
                )
                target = pred[0, target_dim]
                grad = torch.autograd.grad(target, interpolated)[0]
                total_grads += grad
                
        # 3. Final attribution: (orig - base) * avg_grad
        avg_grads = total_grads / n_steps
        attributions = (original_embeddings - baseline) * avg_grads
        attributions = attributions.sum(dim=-1).squeeze(0) # [L]
        
        return attributions # [L]


# ─────────────────────────────────────────────────────────────────────────────
# Self-test / Verification
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import gc
    from src.config_v4 import (
        DATA_ROOT, BATCH_SIZE, MAX_LEN,
        GRID_POSITIVE, GRID_IGNORE,
    )
    from src.data_loader_v4 import get_dataloader, DimABSAExtractionDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n[TEST 1] Model instantiation")
    model = DimABSAV4Model().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total params    : {n_params:.1f}M")
    print(f"  Trainable params: {n_train:.1f}M")
    assert n_params > 100, "Sanity: DeBERTa-v3-base has ~184M params"
    print("  Instantiation OK")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── ST1 regression forward ────────────────────────────────────────────────
    print("\n[TEST 2] ST1 Regression forward pass")
    st1_path = DATA_ROOT / "subtask_1" / "eng" / "eng_restaurant_dev_task1.jsonl"
    dl1 = get_dataloader(st1_path, subtask=1, tokenizer=tokenizer,
                         batch_size=4, is_train=False)
    batch1 = next(iter(dl1))

    model.eval()
    amp_dtype = torch.bfloat16 if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ) else torch.float16 if torch.cuda.is_available() else torch.float32

    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=torch.cuda.is_available()
    ):
        pred_va = model.forward_regression(
            input_ids=batch1["input_ids"].to(device),
            attention_mask=batch1["attention_mask"].to(device),
            token_type_ids=batch1["token_type_ids"].to(device),
            ling_features=batch1["ling_features"].to(device),
            aspect_token_mask=batch1["aspect_token_mask"].to(device),
        )

    assert pred_va.shape == (4, 2), f"Bad shape: {pred_va.shape}"
    assert (pred_va.abs() <= 1.0).all(), f"Tanh OOB: {pred_va}"
    print(f"  pred_va shape : {list(pred_va.shape)}  (expect [4, 2])  OK")
    print(f"  pred_va range : [{pred_va.min().item():.4f}, {pred_va.max().item():.4f}]  (expect in (-1,1))")

    # Test regression loss
    loss_dict = model.compute_regression_loss(
        pred_va=pred_va.float(),
        valence=batch1["valence"].to(device),
        arousal=batch1["arousal"].to(device),
        has_label=batch1["has_label"].to(device),
    )
    print(f"  Regression loss: {loss_dict['loss'].item():.4f}  OK")

    # ── ST3 extraction forward ────────────────────────────────────────────────
    print("\n[TEST 3] ST3 Extraction forward pass (Biaffine grid)")
    st3_path = DATA_ROOT / "subtask_3" / "eng" / "eng_restaurant_dev_task3.jsonl"
    dl3 = get_dataloader(st3_path, subtask=3, tokenizer=tokenizer,
                         batch_size=4, is_train=False)
    batch3 = next(iter(dl3))

    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=torch.cuda.is_available()
    ):
        grid_logits, cat_logits = model.forward_extraction(
            input_ids=batch3["input_ids"].to(device),
            attention_mask=batch3["attention_mask"].to(device),
            ling_features=batch3["ling_features"].to(device),
        )

    assert grid_logits.shape == (4, MAX_LEN, MAX_LEN), \
        f"Grid shape wrong: {grid_logits.shape}"
    assert cat_logits.shape == (4, MAX_LEN, NUM_CATEGORIES), \
        f"Cat shape wrong: {cat_logits.shape}"

    print(f"  grid_logits shape : {list(grid_logits.shape)}  OK")
    print(f"  cat_logits  shape : {list(cat_logits.shape)}  OK")

    # Test extraction loss
    loss_ext = model.compute_extraction_loss(
        grid_logits=grid_logits.float(),
        cat_logits=cat_logits.float(),
        grid_labels=batch3["grid_labels"].to(device),
        category_ids=batch3["category_ids"].to(device),
        va_mask=batch3["va_mask"].to(device),
    )
    print(f"  Focal loss : {loss_ext['loss_focal'].item():.4f}  OK")
    print(f"  Cat loss   : {loss_ext['loss_cat'].item():.4f}  OK")
    print(f"  Total ext  : {loss_ext['loss'].item():.4f}  OK")

    # ── VRAM check ────────────────────────────────────────────────────────────
    print("\n[TEST 4] VRAM usage check")
    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated() / 1e9
        reserv = torch.cuda.memory_reserved() / 1e9
        total  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Allocated : {alloc:.2f} GB")
        print(f"  Reserved  : {reserv:.2f} GB")
        print(f"  Total     : {total:.2f} GB")
        assert reserv < total * 0.9, "WARNING: >90% VRAM in use"
        print(f"  VRAM safe ({reserv/total*100:.0f}% used)  OK")
    else:
        print("  CPU mode (no VRAM to check)")

    # ── Biaffine weight sanity ────────────────────────────────────────────────
    print("\n[TEST 5] Biaffine weight and tensor shape contract")
    W = model.biaffine_W
    assert W.shape == (BIAFFINE_DIM, BIAFFINE_DIM), f"W shape wrong: {W.shape}"
    print(f"  biaffine_W shape: {list(W.shape)}  (expect [{BIAFFINE_DIM},{BIAFFINE_DIM}])  OK")
    print(f"  biaffine_b     : {model.biaffine_b.item():.4f}")
    print(f"  log_vars       : {model.log_vars.tolist()}  (expect [0,0,0] at init)")

    # ── Tensor shape contract summary ─────────────────────────────────────────
    print("\n[TEST 6] Complete tensor shape contract")
    print(f"  ST1 pred_va   : [B=4, 2]  in (-1,1)  -- Tanh bounded  OK")
    print(f"  ST3 grid      : [B=4, L={MAX_LEN}, L={MAX_LEN}]  -- biaffine NxN  OK")
    print(f"  ST3 cat       : [B=4, L={MAX_LEN}, C={NUM_CATEGORIES}]  -- per-token  OK")
    print(f"  Denormalise   : pred*4.0+5.0 maps (-1,1) -> (1,9)")

    print("\n" + "=" * 60)
    print("ALL MODEL TESTS PASSED -- model_v4.py is verified.")
    print("=" * 60)

    # cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
