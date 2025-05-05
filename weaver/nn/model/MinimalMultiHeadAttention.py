import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MinimalMultiheadAttention(nn.Module):
    r"""
    A distilled re-implementation of ``torch.nn.MultiheadAttention`` that
    delegates the heavy lifting to :pyfunc:`torch.nn.functional.scaled_dot_product_attention`.

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model (must be divisible by ``num_heads``).
    num_heads : int
        Number of attention heads.
    add_bias_kv : bool, default = False
        If ``True``, learns the extra bias-vectors ``bias_k`` and ``bias_v``
        exactly as the reference module does (often used for transformer
        decoders).
    dropout : float, default = 0.0
        Drop-out probability applied to the attention weights.
    batch_first : bool, default = False
        If ``True``, expects inputs as *(N, L, E)* instead of *(L, N, E)*.

    Notes
    -----
    Throughout the comments below we use the following dimension symbols::

        L  - target-sequence length (a.k.a. *tgt_len*)
        S  - source-sequence length (a.k.a. *src_len*)
        N  - batch size
        E  - embed_dim (model width)
        H  - num_heads
        Dh - head_dim  (E // H)

    Shapes such as ``(L, N, E)`` therefore read as:
    "sequence length L, batch size N, embedding width E".
    """

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            add_bias_kv: bool = False,
            dropout: float = 0.0,
            batch_first: bool = False,
    ) -> None:
        super().__init__()

        # ---- sanity checks ------------------------------------------------
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        # ---- hyper-parameters -------------------------------------------
        self.embed_dim = embed_dim                      # E
        self.num_heads = num_heads                      # H
        self.head_dim = embed_dim // num_heads          # Dh
        self.dropout = float(dropout)
        self.batch_first = batch_first                  # layout convention flag

        # ---- learnable projections --------------------------------------
        # Each projects (E) → (E).  Separating Q/K/V keeps the code easy to
        # understand at the slight cost of one extra GEMM vs the fused version.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # ---- optional decoder bias token ---------------------------------
        if add_bias_kv:
            # Parameter shapes: (1, 1, E) so they broadcast across both time and batch.
            self.bias_k = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:  # keep attributes for type-checking convenience
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        # Pre-compute the scaling factor 1/√Dh.  Multiplying Q by this factor
        # before the dot-product is mathematically equivalent to dividing the
        # logits afterwards and saves an extra kernel call.
        self.scaling = self.head_dim ** -0.5

    # ------------------------------------------------------------------
    # helper utilities
    # ------------------------------------------------------------------
    def _shape(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshape a *flat* projection ``x`` of shape ``(seq_len, batch_size, E)``
        into the 4-D tensor expected by SDPA::

            (N, H, seq_len, Dh)
        """
        return (
            x.contiguous()                                   # ensure we can view()
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)                                # move H before L
        )

    @staticmethod
    def _merge_bool_masks(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """Merge two boolean masks by logical-OR while handling broadcast rules."""
        if mask1 is None:
            return mask2
        if mask2 is None:
            return mask1
        return mask1 | mask2

    @staticmethod
    def _bool_to_additive(mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        """Convert a boolean mask (True = masked) to an *additive* float mask whose
        unmasked entries are 0 and masked entries are ``-inf``.
        """
        return mask.to(dtype).masked_fill(mask, float("-inf"))


    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------
    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute multi-head attention.

        All three inputs **must** share the same embed_dim = E.

        Accepted layouts::

            batch_first = False (default):    (L, N, E)
            batch_first = True:               (N, L, E)
        """

        # ---- 1. normalise layout to (L, N, E) ---------------------------
        if self.batch_first:  # user passed (N, L, E)
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        tgt_len, batch_size, _ = query.size()  # L, N
        src_len = key.size(0)                  # S (may grow by +1 if bias token)

        # ---- 2. bias_k / bias_v trick (optional) ------------------------
        # Adds a learned *global* token to K and V that every Q can attend to.
        if self.bias_k is not None and self.bias_v is not None:
            # Concatenate along the sequence dimension → shapes become (S+1, N, E)
            key   = torch.cat([key,   self.bias_k.repeat(1, batch_size, 1)], dim=0)
            value = torch.cat([value, self.bias_v.repeat(1, batch_size, 1)], dim=0)

            # Expand padding mask so the new token is *never* masked out.
            if key_padding_mask is not None:
                pad = torch.zeros((batch_size, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat([key_padding_mask, pad], dim=1)

            src_len += 1  # keep the bookkeeping consistent

        # ---- 3. linear projections  (still flat → (L/S, N, E)) ---------
        q = self.q_proj(query) * self.scaling  # scale only Q → (L, N, E)
        k = self.k_proj(key)                   # (S, N, E)
        v = self.v_proj(value)                 # (S, N, E)

        # ---- 4. reshape to multi-head view  (N, H, L/S, Dh) ------------
        q = self._shape(q, batch_size, tgt_len)   # (N, H, L, Dh)
        k = self._shape(k, batch_size, src_len)   # (N, H, S, Dh)
        v = self._shape(v, batch_size, src_len)   # (N, H, S, Dh)

        # ------------------------------------------------------------------
        # 5. Mask handling – *preserve* additive float mask & apply bool mask
        # ------------------------------------------------------------------
        bool_attn_mask: Optional[torch.Tensor] = None  # True → masked
        float_attn_mask: Optional[torch.Tensor] = None # additive

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                bool_attn_mask = attn_mask
            else:
                float_attn_mask = attn_mask

        # --------------------------------------------------------------
        # 5a. Build *additive* mask if a float mask is supplied
        # --------------------------------------------------------------
        if float_attn_mask is not None:
            merged_mask = float_attn_mask  # keep additive information

            # Convert any *extra* bool mask to additive and add it in
            if bool_attn_mask is not None:
                merged_mask = merged_mask + self._bool_to_additive(
                    bool_attn_mask, dtype=merged_mask.dtype
                )

            # Convert key_padding_mask (bool) to additive and add in
            if key_padding_mask is not None:
                # Broadcast to (N, 1, 1, S) so it lines up with (N, H, L, S)
                kp_add = self._bool_to_additive(
                    key_padding_mask[:, None, None, :], dtype=merged_mask.dtype
                )
                merged_mask = merged_mask + kp_add

        # --------------------------------------------------------------
        # 5b. *No* float mask provided → fall back to pure boolean logic
        # --------------------------------------------------------------
        else:
            merged_bool = self._merge_bool_masks(bool_attn_mask, key_padding_mask)
            merged_mask = merged_bool  # may still be None

        # merged_mask is now either:
        #   • additive float   (preferred path – keeps logits bias plus padding)
        #   • boolean          (if user supplied no float mask)
        #   • None             (no masking requested)

        # ---- 6. core scaled-dot-product attention -----------------------
        # Calls into the highly-optimised fused kernel (FlashAttention where available).
        # Output shape: (N, H, L, Dh)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=merged_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=1
        )

        # ---- 7. combine heads & final projection ------------------------
        attn_output = attn_output.transpose(1, 2).contiguous()         # (N, L, H, Dh)
        attn_output = attn_output.view(batch_size, tgt_len, self.embed_dim)  # (N, L, E)
        attn_output = self.out_proj(attn_output)                       # (N, L, E)

        # ---- 8. restore original layout if needed ----------------------
        if not self.batch_first:            # user expects (L, N, E)
            attn_output = attn_output.transpose(0, 1)

        # Return the output and a placeholder ``None`` to mirror the stock API
        # (the original layer can optionally return attention weights here).
        return attn_output, None
