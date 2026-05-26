import torch
from torch import nn
from torch.nn import functional as F

from wt._core.components import flash_attention


def wrap_dinov2_attention_with_fa3(module: nn.Module):
    class _AttentionWrapper(module.__class__):
        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            assert attn_bias is None, "attn_bias is not supported"
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            x = flash_attention.flash_attention(q, k, v, use_bshc=True)
            x = x.reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    module.__class__ = _AttentionWrapper
    return module


def wrap_dinov2_attention_with_sdpa(module: nn.Module):
    assert torch.__version__ >= "2.0", "SDPA requires PyTorch 2.0 or later"

    class _AttentionWrapper(module.__class__):
        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )  # (3, B, H, N, C // H)

            q, k, v = torch.unbind(qkv, 0)  # (B, H, N, C // H)

            x = F.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    module.__class__ = _AttentionWrapper
    return module
