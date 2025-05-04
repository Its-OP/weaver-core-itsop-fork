from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalMultiheadAttention(nn.Module):
    r"""
    A distilled re‑implementation of ``torch.nn.MultiheadAttention`` that
    delegates the heavy lifting to
    ``torch.nn.functional.scaled_dot_product_attention``.

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model (must be divisible by ``num_heads``).
    num_heads : int
        Number of attention heads.
    add_bias_kv : bool, default = False
        If ``True``, learns the extra bias‐vectors ``bias_k`` and ``bias_v``
        exactly as the reference module does (often used for transformer
        decoders).
    dropout : float, default = 0.0
        Drop‑out probability applied to the attention weights.
    batch_first : bool, default = False
        If ``True``, expects inputs as *(N, L, E)* instead of *(L, N, E)*.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        add_bias_kv: bool = False,
        dropout: float = 0.0,
        batch_first: bool = False,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = float(dropout)
        self.batch_first = batch_first

        # Independent linear projections match the fused pattern used
        # internally by the official module.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        if add_bias_kv:
            # Shape: (1, 1, E) so it broadcasts over both sequence and batch
            self.bias_k = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:  # keep attributes for type‑checking convenience
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.scaling = self.head_dim**-0.5  # identical to PyTorch’s

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _shape(
        self, x: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """
        Returns a view of ``x`` with dimensions
        (batch_size, num_heads, seq_len, head_dim).
        """
        return (
            x.contiguous()
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

    @staticmethod
    def _merge_bool_masks(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """
        Combines two boolean masks by logical‑OR while broadcasting shapes.
        """
        if mask1 is None:
            return mask2
        if mask2 is None:
            return mask1
        # Broadcast to a common shape
        return mask1 | mask2

    # --------------------------------------------------------------------- #
    # forward
    # --------------------------------------------------------------------- #
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        avg_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Mirrors the signature of the reference implementation.  See
        ``torch.nn.MultiheadAttention`` for shape conventions.
        """
        if self.batch_first:
            # Convert to (L, N, E)
            query, key, value = (
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
            )

        tgt_len, batch_size, _ = query.size()
        src_len = key.size(0)

        # Optional bias_k / bias_v (sequence length grows by 1)
        if self.bias_k is not None and self.bias_v is not None:
            key = torch.cat([key, self.bias_k.repeat(1, batch_size, 1)], dim=0)
            value = torch.cat([value, self.bias_v.repeat(1, batch_size, 1)], dim=0)

            if key_padding_mask is not None:
                pad = torch.zeros(
                    (batch_size, 1),
                    dtype=key_padding_mask.dtype,
                    device=key_padding_mask.device,
                )
                key_padding_mask = torch.cat([key_padding_mask, pad], dim=1)

            src_len += 1  # one extra token

        # Linear projections
        q = self.q_proj(query) * self.scaling  # (L, N, E)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi‑head attention
        q = self._shape(q, batch_size, tgt_len)        # (N, H, L, Dh)
        k = self._shape(k, batch_size, src_len)        # (N, H, S, Dh)
        v = self._shape(v, batch_size, src_len)        # (N, H, S, Dh)

        # ------------------------------------------------------------------ #
        # attention masks
        # ------------------------------------------------------------------ #
        bool_attn_mask: Optional[torch.Tensor] = None
        float_attn_mask: Optional[torch.Tensor] = None

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                bool_attn_mask = attn_mask
            else:
                float_attn_mask = attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (N, S)  -> (N, 1, 1, S)
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            bool_attn_mask = self._merge_bool_masks(bool_attn_mask, padding_mask)

        # ``scaled_dot_product_attention`` accepts ONE mask argument that can be:
        #    • bool  (True = masked)
        #    • float (additive: −inf for masked, 0 otherwise)
        # Choose whichever makes sense.
        merged_mask: Optional[torch.Tensor]
        if bool_attn_mask is not None and float_attn_mask is not None:
            # Convert float mask to bool so we can OR them together
            float_to_bool = float_attn_mask == float("-inf")
            merged_mask = self._merge_bool_masks(bool_attn_mask, float_to_bool)
        else:
            merged_mask = bool_attn_mask if bool_attn_mask is not None else float_attn_mask

        # ------------------------------------------------------------------ #
        # core SDPA call
        # ------------------------------------------------------------------ #
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=merged_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )  # (N, H, L, Dh)

        # Collapse heads and run final projection
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        )  # (N, L, E)
        attn_output = self.out_proj(attn_output)  # (N, L, E)

        # Return to (L, N, E) if user passed sequence‑first
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, None