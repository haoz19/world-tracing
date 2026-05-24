# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
DINOv2 Vision Transformer components module.

This module implements core components used in the DINOv2 Vision Transformer
architecture, including patch embedding, attention, feed-forward networks,
and transformer blocks. These components are used to build the full
transformer model for processing image inputs.

References:
    - DINOv2: https://github.com/facebookresearch/dinov2
    - https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
"""

import logging
import os
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

logger = logging.getLogger("dinov2")


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """
    Apply stochastic depth (drop path) to a tensor.

    Randomly drops entire samples from the batch during training by zeroing
    them out and scaling the remaining samples to preserve expected values.

    Args:
        x: Input tensor of shape `(batch_size, ...)`.
        drop_prob: Probability of dropping a sample. Default: 0.0.
        training: Whether the model is in training mode.

    Returns:
        Tensor with same shape as input, with some samples randomly zeroed.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """
    Stochastic depth (drop path) per sample.

    Randomly drops entire samples from the batch during training when applied
    in the main path of residual blocks. This regularization technique helps
    prevent overfitting in deep networks.
    """

    def __init__(self, drop_prob: float | None = None):
        """
        Args:
            drop_prob: Probability of dropping a sample. Default: None (no drop).
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass applying stochastic depth.

        Args:
            x: Input tensor of shape `(batch_size, ...)`.

        Returns:
            Tensor with same shape as input.
        """
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    """
    Layer scale module that applies learnable per-channel scaling.

    Each feature dimension is scaled by a learnable parameter gamma, helping
    stabilize training of deep networks by controlling the magnitude of
    residual contributions.
    """

    def __init__(
        self,
        dim: int,
        init_values: float | Tensor = 1e-5,
        inplace: bool = False,
    ):
        """
        Args:
            dim: Number of input feature dimensions to scale.
            init_values: Initial value for the scaling parameters gamma.
                Can be a float or tensor. Default: 1e-5.
            inplace: Whether to apply scaling in-place. Default: False.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass applying layer scaling.

        Args:
            x: Input tensor to scale.

        Returns:
            Scaled tensor where each feature dimension is multiplied by gamma.
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    """
    Multi-layer perceptron module.

    Standard two-layer MLP that processes input through linear projection,
    activation, and dropout layers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. If None, equals in_features.
            out_features: Number of output features. If None, equals in_features.
            act_layer: Activation layer class to use. Default: `nn.GELU`.
            drop: Dropout rate. Default: 0.0.
            bias: Whether to use bias in linear layers. Default: True.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Input tensor of shape `(..., in_features)`.

        Returns:
            Output tensor of shape `(..., out_features)`.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def make_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    """
    Convert an int or 2-tuple to a 2-tuple.

    Args:
        x: An integer (duplicated) or a 2-tuple (returned as-is).

    Returns:
        A 2-tuple of ints.
    """
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding layer.

    Converts input images into sequences of patch embeddings by splitting
    the image into fixed-size patches and projecting each patch into an
    embedding space via a convolutional layer: `(B, C, H, W) -> (B, N, D)`.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable[..., nn.Module] | None = None,
        flatten_embedding: bool = True,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Size of each image patch.
            in_chans: Number of input image channels.
            embed_dim: Number of linear projection output channels.
            norm_layer: Optional normalization layer applied after projection.
            flatten_embedding: Whether to flatten spatial dims into a sequence.
                If False, returns shape `(B, H', W', D)` instead of `(B, N, D)`.
        """
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass converting image to patch embeddings.

        Args:
            x: Input tensor of shape `(B, C, H, W)`.

        Returns:
            Patch embeddings of shape `(B, N, D)` when `flatten_embedding`
            is True, or `(B, H', W', D)` otherwise.
        """
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        """Estimate FLOPs for this layer at the default image size."""
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import (
            SwiGLU,
            fmha,
            index_select_cat,
            memory_efficient_attention,
            scaled_index_add,
            unbind,
        )

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    """
    Multi-head self-attention module.

    Computes scaled dot-product attention over input sequences using
    learned query, key, and value projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Args:
            dim: Total dimension of the model (embedding size).
            num_heads: Number of attention heads.
            qkv_bias: If True, add bias to query, key, value projections.
            proj_bias: If True, add bias to output projection.
            attn_drop: Dropout rate for attention weights.
            proj_drop: Dropout rate for output projection.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        """
        Forward pass computing multi-head self-attention.

        Args:
            x: Input tensor of shape `(batch_size, seq_len, dim)`.
            attn_bias: Optional attention bias tensor.

        Returns:
            Output tensor of shape `(batch_size, seq_len, dim)`.
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (3, B, H, N, C // H)

        q, k, v = qkv.unbind(0)  # (B, H, N, C // H)

        x = F.scaled_dot_product_attention(q, k, v, attn_bias)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    """
    Memory-efficient attention using xFormers.

    Falls back to standard `Attention` when xFormers is not available.
    When xFormers is available, uses `memory_efficient_attention` for
    reduced memory usage and supports nested tensor attention bias.
    """

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        """
        Forward pass using memory-efficient attention when available.

        Args:
            x: Input tensor of shape `(batch_size, seq_len, dim)`.
            attn_bias: Optional attention bias (requires xFormers for nested tensors).

        Returns:
            Output tensor of shape `(batch_size, seq_len, dim)`.
        """
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network module.

    Implements the SwiGLU variant of feed-forward networks which uses a gated
    linear unit with SiLU activation. A single linear layer projects the input
    to twice the hidden dimension, which is then split into gate and value paths.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] | None = None,
        drop: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. If None, equals in_features.
            out_features: Number of output features. If None, equals in_features.
            act_layer: Unused, kept for API compatibility with `Mlp`.
            drop: Unused, kept for API compatibility with `Mlp`.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through SwiGLU FFN.

        Args:
            x: Input tensor of shape `(..., in_features)`.

        Returns:
            Output tensor of shape `(..., out_features)`.
        """
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


if not XFORMERS_AVAILABLE:
    SwiGLU = SwiGLUFFN


class SwiGLUFFNFused(SwiGLU):
    """
    SwiGLU FFN with fused xFormers kernel and adjusted hidden dimension.

    Wraps the xFormers `SwiGLU` (or falls back to `SwiGLUFFN`) and
    adjusts the hidden dimension to `round_up(hidden * 2/3, 8)` for
    hardware-friendly alignment.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] | None = None,
        drop: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features before alignment.
                If None, equals in_features.
            out_features: Number of output features. If None, equals in_features.
            act_layer: Unused, kept for API compatibility with `Mlp`.
            drop: Unused, kept for API compatibility with `Mlp`.
            bias: Whether to use bias in linear layers.
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


class Block(nn.Module):
    """
    Transformer block combining self-attention and feed-forward network.

    Implements the standard pre-norm transformer architecture with optional
    layer scaling and stochastic depth for regularization during training.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ):
        """
        Args:
            dim: Feature dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for FFN hidden dimension.
            qkv_bias: Whether to use bias in QKV projections.
            proj_bias: Whether to use bias in attention output projection.
            ffn_bias: Whether to use bias in FFN layers.
            drop: Dropout rate for attention output and FFN.
            attn_drop: Dropout rate for attention weights.
            init_values: Initial values for layer scale. None disables layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer class for the FFN.
            norm_layer: Normalization layer class.
            attn_class: Attention module class.
            ffn_layer: Feed-forward network module class.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the transformer block.

        Applies self-attention and FFN with residual connections. During
        training with `drop_path > 0.1`, uses sample-level stochastic
        depth for efficient regularization.

        Args:
            x: Input tensor of shape `(batch_size, seq_len, dim)`.

        Returns:
            Output tensor of shape `(batch_size, seq_len, dim)`.
        """

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    """
    Apply a residual function to a random subset of batch samples.

    Selects a random subset of the batch, applies the residual function only
    to that subset, and adds the scaled result back to the full batch. This
    implements sample-level stochastic depth more efficiently than per-element
    masking for high drop rates.

    Args:
        x: Input tensor of shape `(batch_size, seq_len, dim)`.
        residual_func: Function to apply to the selected subset.
        sample_drop_ratio: Fraction of samples to drop.

    Returns:
        Tensor of shape `(batch_size, seq_len, dim)` with residual added.
    """
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


def get_branges_scales(
    x: Tensor, sample_drop_ratio: float = 0.0
) -> tuple[Tensor, float]:
    """
    Compute random batch indices and the corresponding scale factor for stochastic depth.

    Args:
        x: Input tensor of shape `(batch_size, seq_len, dim)`.
        sample_drop_ratio: Fraction of samples to drop.

    Returns:
        Tuple of `(batch_indices, scale_factor)`.
    """
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(
    x: Tensor,
    brange: Tensor,
    residual: Tensor,
    residual_scale_factor: float,
    scaling_vector: Tensor | None = None,
) -> Tensor:
    """
    Add a scaled residual to selected batch samples.

    When `scaling_vector` is None, uses `torch.index_add` on flattened
    tensors. Otherwise, uses the xFormers `scaled_index_add` kernel.

    Args:
        x: Input tensor of shape `(batch_size, seq_len, dim)`.
        brange: Batch indices for the residual subset.
        residual: Residual tensor for the selected subset.
        residual_scale_factor: Multiplicative scale for the residual.
        scaling_vector: Optional per-channel scaling (e.g. from `LayerScale`).

    Returns:
        Tensor with the residual added at the selected indices.
    """
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(
            x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
        )
    else:
        x_plus_residual = scaled_index_add(
            x,
            brange,
            residual.to(dtype=x.dtype),
            scaling=scaling_vector,
            alpha=residual_scale_factor,
        )
    return x_plus_residual


attn_bias_cache: dict[tuple, Any] = {}


def get_attn_bias_and_cat(
    x_list: list[Tensor], branges: list[Tensor] | None = None
) -> tuple[Any, Tensor]:
    """
    Build a block-diagonal attention mask and concatenate tensors for nested attention.

    Uses xFormers `BlockDiagonalMask` to construct an attention bias that
    prevents cross-attention between different tensors in the list. Results
    are cached by shape for efficiency.

    Args:
        x_list: List of tensors to concatenate.
        branges: Optional list of batch index tensors for subset selection.

    Returns:
        Tuple of `(attn_bias, concatenated_tensor)`.
    """
    batch_sizes = (
        [b.shape[0] for b in branges]
        if branges is not None
        else [x.shape[0] for x in x_list]
    )
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache:
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(
            1, -1, x_list[0].shape[-1]
        )
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: list[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector: Tensor | None = None,
) -> list[Tensor]:
    """
    Apply stochastic depth with residual addition over a list of tensors.

    This is the nested-tensor variant of :func:`drop_add_residual_stochastic_depth`,
    operating on a list of tensors that are concatenated with a block-diagonal
    attention mask for efficient batched processing.

    Args:
        x_list: List of input tensors.
        residual_func: Function taking `(x, attn_bias)` and returning a residual.
        sample_drop_ratio: Fraction of samples to drop.
        scaling_vector: Optional per-channel scaling (e.g. from `LayerScale`).

    Returns:
        List of tensors with residuals added.
    """
    branges_scales = [
        get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list
    ]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(
        x_list, branges, residual_list, residual_scale_factors
    ):
        outputs.append(
            add_residual(
                x, brange, residual, residual_scale_factor, scaling_vector
            ).view_as(x)
        )
    return outputs


class NestedTensorBlock(Block):
    """
    Transformer block with nested tensor support via xFormers.

    Extends `Block` to accept either a single tensor or a list of tensors.
    When given a list, uses xFormers block-diagonal attention masks for
    efficient batched processing of variable-length sequences.
    """

    def forward_nested(self, x_list: list[Tensor]) -> list[Tensor]:
        """
        Forward pass for a list of tensors using nested attention.

        Args:
            x_list: List of tensors to process jointly with block-diagonal
                attention masking.

        Returns:
            List of output tensors, one per input tensor.
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias: Any = None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias: Any = None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma
                if isinstance(self.ls1, LayerScale)
                else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma
                if isinstance(self.ls1, LayerScale)
                else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias: Any = None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias: Any = None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
        """
        Forward pass accepting either a single tensor or a list of tensors.

        Args:
            x_or_x_list: Either a single tensor of shape
                `(batch_size, seq_len, dim)`, or a list of such tensors
                for nested processing.

        Returns:
            Single output tensor or list of output tensors matching the input type.
        """
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
