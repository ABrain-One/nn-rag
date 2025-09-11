# Auto-generated single-file for AttentionCl
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional, Callable
import os

# ---- timm.layers.config._EXPORTABLE ----
_EXPORTABLE = False

# ---- timm.layers.config._HAS_FUSED_ATTN ----
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

# ---- timm.layers.config._USE_FUSED_ATTN ----
_USE_FUSED_ATTN = int(os.environ['TIMM_FUSED_ATTN'])

# ---- timm.layers.config.use_fused_attn ----
def use_fused_attn(experimental: bool = False) -> bool:
    # NOTE: ONNX export cannot handle F.scaled_dot_product_attention as of pytorch 2.0
    if not _HAS_FUSED_ATTN or _EXPORTABLE:
        return False
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0

# ---- AttentionCl (target) ----
class AttentionCl(nn.Module):
    """Channels-last multi-head attention (B, ..., C)."""
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            dim_head: int = 32,
            bias: bool = True,
            expand_first: bool = True,
            head_first: bool = True,
            rel_pos_cls: Optional[Callable] = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        """
        Args:
            dim: Input dimension.
            dim_out: Output dimension (defaults to input dimension).
            dim_head: Dimension per attention head.
            bias: Whether to use bias in qkv and projection.
            expand_first: Whether to expand channels before or after qkv.
            head_first: Whether heads are first in tensor layout.
            rel_pos_cls: Relative position class to use.
            attn_drop: Attention dropout rate.
            proj_drop: Projection dropout rate.
        """
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first and dim_out > dim else dim
        assert dim_attn % dim_head == 0, 'attn dim should be divisible by head_dim'
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = dim_head ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim_attn * 3, bias=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_attn, dim_out, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, shared_rel_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        if self.head_first:
            q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        else:
            q, k, v = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.dim_head).transpose(1, 3).unbind(2)

        if self.fused_attn:
            attn_bias = None
            if self.rel_pos is not None:
                attn_bias = self.rel_pos.get_bias()
            elif shared_rel_pos is not None:
                attn_bias = shared_rel_pos

            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if self.rel_pos is not None:
                attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
            elif shared_rel_pos is not None:
                attn = attn + shared_rel_pos
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
