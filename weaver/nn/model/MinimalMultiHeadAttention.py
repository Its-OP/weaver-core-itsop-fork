from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalMultiheadAttention(nn.Module):
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

        # Use a single QKV projection instead of separate ones
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

    def _merge_masks(self, attn_mask, key_padding_mask, batch_size, src_len):
        """Merge attention mask and key padding mask efficiently."""
        if key_padding_mask is None:
            return attn_mask

        # Convert key_padding_mask to the right shape for broadcasting
        key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len)

        if attn_mask is None:
            return key_padding_mask

        # Merge masks based on dtype
        if attn_mask.dtype == torch.bool and key_padding_mask.dtype == torch.bool:
            return attn_mask | key_padding_mask
        else:
            # Convert bool masks to float if needed
            if attn_mask.dtype == torch.bool:
                float_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
                float_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = float_mask

            if key_padding_mask.dtype == torch.bool:
                float_padding = torch.zeros_like(attn_mask, dtype=torch.float)
                float_padding.masked_fill_(key_padding_mask, float("-inf"))
                key_padding_mask = float_padding

            return attn_mask + key_padding_mask

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        tgt_len, batch_size, _ = query.size()
        src_len = key.size(0)

        # Handle bias_k and bias_v
        if self.bias_k is not None and self.bias_v is not None:
            key = torch.cat([key, self.bias_k.expand(1, batch_size, -1)], dim=0)
            value = torch.cat([value, self.bias_v.expand(1, batch_size, -1)], dim=0)

            if key_padding_mask is not None:
                pad = torch.zeros(
                    (batch_size, 1),
                    dtype=key_padding_mask.dtype,
                    device=key_padding_mask.device,
                )
                key_padding_mask = torch.cat([key_padding_mask, pad], dim=1)

            src_len += 1

        # Efficient projections using a single QKV matrix
        if query is key and key is value:
            # Self-attention case - use a single projection
            qkv = self.qkv_proj(query)
            q, k, v = torch.chunk(qkv, 3, dim=2)
        else:
            # Cross-attention case
            q = F.linear(query, self.qkv_proj.weight[:self.embed_dim],
                         self.qkv_proj.bias[:self.embed_dim])
            k = F.linear(key, self.qkv_proj.weight[self.embed_dim:2*self.embed_dim],
                         self.qkv_proj.bias[self.embed_dim:2*self.embed_dim])
            v = F.linear(value, self.qkv_proj.weight[2*self.embed_dim:],
                         self.qkv_proj.bias[2*self.embed_dim:])

        # Reshape for multi-head attention without unnecessary contiguous calls
        q = q.view(tgt_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Merge masks efficiently
        merged_mask = self._merge_masks(attn_mask, key_padding_mask, batch_size, src_len)

        # Core attention computation
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=merged_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape and project output efficiently
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(tgt_len, batch_size, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # Return in the correct format
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, None