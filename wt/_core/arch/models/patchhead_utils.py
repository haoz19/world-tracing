import functools
from copy import deepcopy
from typing import Callable, Literal

import einops
import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.utils
import torch.utils.checkpoint
import torch.version
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F

from wt._core.components import nn_layers
from wt._core.splat.utils import embedding
from wt._core.arch.models import blocks

logger = structlog.get_logger(__name__)

MAX_TOKEN_SIZE = 64  # max number of tokens to be merged along spatial dimension


@functools.lru_cache
def _make_xy_grid(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> Float[Tensor, "hw 2"]:
    """
    Returns grid of normalized coords in [-1, 1], shape [H*W, 2],
    arranged in h w 2 order storing x,y coords
    """
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1).reshape(height * width, 2)
    return xy


@torch.no_grad()
def _make_posenc_feats(
    posenc: embedding.PosEmbedding, height: int, width: int, device, dtype
) -> Float[Tensor, "hw d"]:
    """
    Make positional encoding features.
    """
    xy = _make_xy_grid(height, width, device, dtype)  # [hw, 2]
    feats = posenc(xy)  # [hw, d]
    return feats


class PredictionHead(nn.Module):
    """
    Linear head with transformer blocks.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: list[int],
        activations: list[Callable],
        pred_names: list[str],
        patch_size: int,
        num_head_blks: int = 4,
        norm_layer=nn.LayerNorm,
        head_mode: Literal["linear", "perceiver", "patchnerf"] = "linear",
        head_merge_ratio: float = 0.0,
        head_subsample_ratio: float = 0.2,
    ):
        super().__init__()
        assert len(dim_out) == len(activations), "dim_out and activations must match"
        self.patch_size = patch_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_head_blks = num_head_blks
        self.activations = activations
        self.pred_names = pred_names
        self.head_mode = head_mode
        self.head_merge_ratio = head_merge_ratio
        self.head_subsample_ratio = head_subsample_ratio

        if self.head_mode == "linear":
            head_cls = LinearHead
        elif self.head_mode == "perceiver":
            head_cls = PerceiverHead
        elif self.head_mode == "patchnerf":
            head_cls = PatchnerfHead
        else:
            raise ValueError(
                f"must be 'linear' 'perceiver' or 'patchnerf', got {self.head_mode}"
            )
        self.model = head_cls(
            dim_in=dim_in,
            dim_out=sum(dim_out),
            patch_size=self.patch_size,
            num_blocks=num_head_blks,
            norm_layer=norm_layer,
            merge_ratio=self.head_merge_ratio,
            subsample_ratio=self.head_subsample_ratio,
        )

    def forward(
        self,
        x_in: Float[Tensor, "b vp d"],
        xpos: Float[Tensor, "b vp 2"],
        img: Float[Tensor, "b v c h w"],
    ) -> dict[str, Float[Tensor, "b v h w c"]]:
        """
        Args:
            img: image with range [0, 1]
        """
        bs, num_view, _, height, width = img.shape
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Height {height} and width {width} not divisible by {self.patch_size}"
        t_h = height // self.patch_size
        t_w = width // self.patch_size

        x_out = x_in.reshape(bs, num_view, t_h * t_w, -1)
        xpos = xpos.reshape(bs, num_view, t_h * t_w, 2)
        out_dict = {}
        # use tf32 for last linear layer
        with torch.autocast(device_type="cuda", enabled=False):
            if x_out.dtype != torch.float32:
                x_out = x_out.float()
            x_out = self.model(x_out, height, width, xpos=xpos)
            # output: b v h w c
            # deal with subsampled queries in perceiver head
            invalid_mask = torch.isnan(x_out[..., 0])
            x_out = x_out.masked_fill(invalid_mask[..., None], 0)

            # split into semantic predictions
            pred_list = list(torch.split(x_out, self.dim_out, dim=-1))
            for pred, activation, pred_name in zip(
                pred_list, self.activations, self.pred_names
            ):
                out_dict[pred_name] = activation(pred)
            out_dict["invalid_mask"] = invalid_mask
        return out_dict


class NerfBlock(nn.Module):
    """
    Nerf block.
    """

    def __init__(
        self, hidden_size_s, hidden_size_x, mlp_ratio=4, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.param_generator1 = nn.Sequential(
            nn.Linear(hidden_size_s, 2 * hidden_size_x**2 * mlp_ratio, bias=True),
        )
        self.norm = norm_layer(hidden_size_x)
        self.mlp_ratio = mlp_ratio

    def forward(
        self, x: Float[Tensor, "b n d"], s: Float[Tensor, "b d"]
    ) -> Float[Tensor, "b n d"]:
        batch_size, _, hidden_size_x = x.shape
        mlp_params1 = self.param_generator1(s)
        fc1_param1, fc2_param1 = mlp_params1.chunk(2, dim=-1)
        fc1_param1 = fc1_param1.view(
            batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio
        )
        fc2_param1 = fc2_param1.view(
            batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x
        )

        # normalize fc1
        normalized_fc1_param1 = torch.nn.functional.normalize(fc1_param1, dim=-2)
        # normalize fc2
        normalized_fc2_param1 = torch.nn.functional.normalize(fc2_param1, dim=-2)
        # mlp 1
        res_x = x
        x = self.norm(x)
        x = torch.bmm(x, normalized_fc1_param1)
        x = torch.nn.functional.silu(x)
        x = torch.bmm(x, normalized_fc2_param1)
        x = x + res_x
        return x


class RMSNorm(nn.Module):
    """
    RMSNorm module.
    """

    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class PerceiverHead(nn.Module):
    """
    Perceiver head.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        patch_size: int,
        num_blocks: int,
        norm_layer=nn.LayerNorm,
        dim_hidden: int = 512,
        num_heads: int = 16,
        qkv_bias: bool = False,
        n_freqs: int = 7,
        scale_factor: float = 0.0,
        subsample_ratio: float = 0.2,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.n_freqs = n_freqs
        self.subsample_ratio = subsample_ratio

        self.posenc = embedding.PosEmbedding(
            in_channels=2, n_freqs=n_freqs, logscale=True
        )
        self.query_proj = nn_layers.Linear(self.posenc.out_channels, dim_hidden)
        self.kv_proj1 = nn_layers.Linear(dim_in, dim_hidden)
        self.kv_proj2 = nn_layers.Linear(dim_hidden + self.posenc.out_channels, dim_hidden)
        self.blocks = nn.ModuleList(
            [
                blocks.DecoderBlockCA(
                    dim=dim_hidden,
                    num_heads=num_heads,
                    use_qk_norm=True,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    scale_factor=scale_factor,
                    init_values=0.1,  # use 0.1 to avoid gradient explosion
                    add_query_residual=False,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_proj = nn_layers.Linear(dim_hidden, dim_out)
        self.q_posec_cache = {}
        self.kv_posec_cache = {}

    def build_queries(
        self,
        bv: int,
        height: int,
        width: int,
        device,
        dtype,
    ) -> Float[Tensor, "bv hw d"]:
        # [hw, d]
        if (height, width) in self.q_posec_cache:
            feats = self.q_posec_cache[(height, width)]
        else:
            feats = _make_posenc_feats(self.posenc, height, width, device, dtype)
            self.q_posec_cache[(height, width)] = feats
        q = self.query_proj(feats)  # [hw, d]
        q = einops.repeat(q, "... -> bv ...", bv=bv)  # [bv, hw, d]
        return q

    def build_kv(
        self,
        kv: Float[Tensor, "bv p d"],
        height: int,
        width: int,
        device,
        dtype,
    ) -> Float[Tensor, "bv hw d"]:
        bv, _, _ = kv.shape
        if (height, width) in self.kv_posec_cache:
            kv_pe = self.kv_posec_cache[(height, width)]
        else:
            kv_pe = _make_posenc_feats(self.posenc, height, width, device, dtype)
            self.kv_posec_cache[(height, width)] = kv_pe
        # p, d -> bv, p, d
        # TODO: add v embedding
        kv_pe = einops.repeat(kv_pe, "... -> bv ...", bv=bv)
        # TODO: balance channel dim
        kv = self.kv_proj1(kv)
        kv = torch.cat([kv, kv_pe], dim=-1)
        kv = self.kv_proj2(kv)  # [bv, p, d]
        return kv

    def _cross_attend_chunked(
        self, q: Float[Tensor, "bv hw d"], kv: Float[Tensor, "bv p d"]
    ) -> Float[Tensor, "bv hw d"]:
        if self.training:
            for blk in self.blocks:
                q = blk(q, kv)
            return q

        # memory-friendly: process queries in chunks
        _, nq, _ = q.shape
        out = torch.empty_like(q)
        chunk_size = int(nq * self.subsample_ratio)
        for start in range(0, nq, chunk_size):
            end = min(start + chunk_size, nq)
            q_chunk = q[:, start:end, :]
            for blk in self.blocks:
                q_chunk = blk(q_chunk, kv)
            out[:, start:end, :] = q_chunk
        return out

    def selective_decode(
        self,
        q: Float[Tensor, "bv hw d"],
        kv: Float[Tensor, "bv p d"],
    ) -> Float[Tensor, "bv hw d"]:
        # for training, subsample the query and key
        if self.training:
            out = torch.empty(
                q.shape[:-1] + (self.dim_out,), device=q.device, dtype=q.dtype
            )
            out.fill_(float("nan"))
            # sample different locations per view
            num_view, num_query, _ = q.shape
            num_query_to_sample = int(num_query * self.subsample_ratio)
            for idx in range(num_view):
                # sample query
                query_idx = torch.randperm(num_query)[:num_query_to_sample]
                q_idx = q[idx : idx + 1, query_idx, :]
                kv_idx = kv[idx : idx + 1]
                out_idx = self.attend_and_project(q_idx, kv_idx)
                out[idx : idx + 1, query_idx, :] = out_idx
        else:
            out = self.attend_and_project(q, kv)
        return out

    def attend_and_project(
        self, q: Float[Tensor, "bv hw d"], kv: Float[Tensor, "bv p d"]
    ) -> Float[Tensor, "bv hw d"]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self._cross_attend_chunked(q, kv)  # [bv, hw, d]

        with torch.autocast(device_type="cuda", enabled=False):
            out = out.to(torch.float32)
            out = self.final_proj(out)  # [bv, hw, d]
        return out

    def forward(
        self,
        tokens: Float[Tensor, "b v p d"],
        height: int,
        width: int,
        xpos: Float[Tensor, "b v p 2"] | None = None,  # pylint: disable=unused-argument
    ) -> Float[Tensor, "b v h w c"]:
        batch_size, num_views, num_patches, dim = tokens.shape
        assert dim == self.dim_in, f"tokens dim {dim} must match dim_in={self.dim_in}"
        kv = tokens.reshape(batch_size * num_views, num_patches, dim)  # [bv, p, d]
        kv = self.build_kv(
            kv,
            height // self.patch_size,
            width // self.patch_size,
            tokens.device,
            tokens.dtype,
        )  # [bv, p, d]

        q = self.build_queries(
            bv=batch_size * num_views,
            height=height,
            width=width,
            device=tokens.device,
            dtype=tokens.dtype,
        )  # [bv, hw, d]

        out = self.selective_decode(q, kv)

        out = einops.rearrange(out, "(b v) (h w) c -> b v h w c", v=num_views, h=height)
        return out


class PatchnerfHead(nn.Module):
    """
    Perceiver head.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        patch_size: int,
        num_blocks: int,
        norm_layer=nn.LayerNorm,
        dim_hidden: int = 128,  # 128 is better but takes too much memory
        n_freqs: int = 4,
        subsample_ratio: float = 0.2,
        merge_ratio: float = 0.0,  # 1.0 means full merge, 0.0 means no merge
    ):
        super().__init__()
        merge_size = max(1, int(merge_ratio * MAX_TOKEN_SIZE))
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.patch_size = patch_size
        self.num_blocks = num_blocks + int(np.log2(merge_size).round())
        self.n_freqs = n_freqs + int(np.log2(merge_size).round())
        self.subsample_ratio = subsample_ratio
        self.merge_ratio = merge_ratio
        logger.info(
            num_blocks=self.num_blocks,
            n_freqs=self.n_freqs,
            merge_ratio=merge_ratio,
            merge_size=merge_size,
        )

        self.posenc = embedding.PosEmbedding(
            in_channels=2, n_freqs=self.n_freqs, logscale=True
        )
        self.query_proj = nn_layers.Linear(self.posenc.out_channels, self.dim_hidden)
        self.kv_proj = nn_layers.Linear(dim_in, self.dim_hidden)
        blk = NerfBlock(
            hidden_size_s=self.dim_hidden,
            hidden_size_x=self.dim_hidden,
            norm_layer=norm_layer,
        )
        self.blocks = nn.ModuleList([deepcopy(blk) for _ in range(self.num_blocks)])
        self.final_proj = nn_layers.Linear(self.dim_hidden, self.dim_out)
        self.q_posec_cache = {}

    def build_queries(
        self,
        bv: int,
        height: int,
        width: int,
        device,
        dtype,
    ) -> Float[Tensor, "bv hw d"]:
        # [hw, d]
        if (height, width) in self.q_posec_cache:
            feats = self.q_posec_cache[(height, width)]
        else:
            feats = _make_posenc_feats(self.posenc, height, width, device, dtype)
            self.q_posec_cache[(height, width)] = feats
        q = self.query_proj(feats)  # [hw, d]
        q = einops.repeat(q, "... -> bv ...", bv=bv)  # [bv, hw, d]
        return q

    def _cross_attend_chunked(
        self, q: Float[Tensor, "bv hw d"], kv: Float[Tensor, "bv p d"]
    ) -> Float[Tensor, "bv hw d"]:
        if self.training:
            for blk in self.blocks:
                q = blk(q, kv)
            return q

        # memory-friendly: process queries in chunks
        _, nq, _ = q.shape
        out = torch.empty_like(q)
        chunk_size = int(nq * self.subsample_ratio)
        for start in range(0, nq, chunk_size):
            end = min(start + chunk_size, nq)
            q_chunk = q[:, start:end, :]
            for blk in self.blocks:
                q_chunk = blk(q_chunk, kv)
            out[:, start:end, :] = q_chunk
        return out

    def selective_decode(
        self,
        q: Float[Tensor, "bv hw d"],
        kv: Float[Tensor, "bv d"],
    ) -> Float[Tensor, "bv hw d"]:
        # for training, subsample the query and key
        if self.training:
            out = torch.empty(
                q.shape[:-1] + (self.dim_out,), device=q.device, dtype=q.dtype
            )
            out.fill_(float("nan"))
            # sample different patches to evaluate
            num_view, num_query, _ = q.shape
            if num_view > num_query:
                num_views_to_sample = max(1, int(num_view * self.subsample_ratio))
                view_idx = torch.randperm(num_view)[:num_views_to_sample]
                out[view_idx] = self.attend_and_project(q[view_idx], kv[view_idx])
            else:
                num_query_to_sample = int(num_query * self.subsample_ratio)
                query_idx = torch.randperm(num_query)[:num_query_to_sample]
                out[:, query_idx] = self.attend_and_project(q[:, query_idx], kv)
        else:
            out = self.attend_and_project(q, kv)
        return out

    def attend_and_project(
        self, q: Float[Tensor, "bv hw d"], kv: Float[Tensor, "bv d"]
    ) -> Float[Tensor, "bv hw d"]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self._cross_attend_chunked(q, kv)  # [bv, hw, d]

        with torch.autocast(device_type="cuda", enabled=False):
            out = out.to(torch.float32)
            out = self.final_proj(out)  # [bv, hw, d]
        return out

    def forward(
        self,
        tokens: Float[Tensor, "b v p d"],
        height: int,
        width: int,
        xpos: Float[Tensor, "b v p 2"] | None = None,  # pylint: disable=unused-argument
    ) -> Float[Tensor, "b v h w c"]:
        # spatial merging
        _, num_views, _, dim = tokens.shape
        assert dim == self.dim_in, f"tokens dim {dim} must match dim_in={self.dim_in}"
        ph, pw = height // self.patch_size, width // self.patch_size
        # get merge size from merge ratio
        merge_h = max(1, int(ph * self.merge_ratio))
        merge_w = max(1, int(pw * self.merge_ratio))
        tokens = einops.rearrange(
            tokens,
            "b v (sp_h s1 sp_w s2) d -> b v sp_h s1 sp_w s2 d",
            sp_h=ph // merge_h,
            sp_w=pw // merge_w,
            s1=merge_h,
            s2=merge_w,
        )
        tokens = tokens.mean([3, 5])
        tokens = einops.rearrange(
            tokens, "b v p_h p_w d -> (b v p_h p_w) d"
        )  # [bvpp,d]

        bvpp = tokens.shape[0]
        q = self.build_queries(
            bv=bvpp,
            height=self.patch_size * merge_h,
            width=self.patch_size * merge_w,
            device=tokens.device,
            dtype=tokens.dtype,
        )  # [bvpp, ss, d]
        kv = self.kv_proj(tokens)
        out = self.selective_decode(q, kv)

        out = einops.rearrange(
            out,
            "(b v p_h p_w) (s1 s2) d -> b v (p_h s1) (p_w s2) d",
            v=num_views,
            p_h=ph // merge_h,
            p_w=pw // merge_w,
            s1=self.patch_size * merge_h,
            s2=self.patch_size * merge_w,
        )
        return out


class LinearHead(nn.Module):
    """
    Linear decoder head.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        patch_size,
        num_blocks,
        norm_layer=nn.LayerNorm,
        dim_hidden: int = 1024,
        init_values: float | None = None,
        qkv_bias: bool = False,
        attn_mode: Literal["local", "global", "aa"] = "local",
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.attn_mode = attn_mode

        self.token_proj = nn_layers.Linear(dim_in, dim_hidden)
        blk = blocks.DecoderBlockSA(
            dim_hidden,
            num_heads=dim_hidden // 64,
            norm_layer=norm_layer,
            use_qk_norm=True,
            init_values=init_values,
            rope=blocks.RoPE2D(freq=100.0),
            qkv_bias=qkv_bias,
        )
        self.blocks = nn.ModuleList([deepcopy(blk) for _ in range(self.num_blocks)])
        self.project = nn_layers.Linear(
            dim_hidden, self.patch_size * self.patch_size * dim_out
        )

    def forward(
        self,
        tokens: Float[Tensor, "b v p d"],
        height: int,
        width: int,
        xpos: Float[Tensor, "b v p 2"] | None = None,
    ) -> Float[Tensor, "b v h w c"]:
        bs, num_view, num_patches, _ = tokens.shape
        t_h = height // self.patch_size
        t_w = width // self.patch_size

        x_out = self.token_proj(tokens)
        for idx, blk in enumerate(self.blocks):
            if self.attn_mode == "local" or (self.attn_mode == "aa" and idx % 2 == 0):
                # frame-wise attention
                x_out = x_out.reshape(bs * num_view, num_patches, -1)
                xpos = xpos.reshape(bs * num_view, num_patches, 2)
            elif self.attn_mode == "global" or (
                self.attn_mode == "aa" and idx % 2 == 1
            ):
                # global attention
                x_out = x_out.reshape(bs, num_view * num_patches, -1)
                xpos = xpos.reshape(bs, num_view * num_patches, 2)
            else:
                raise ValueError(f"Unknown attention mode: {self.attn_mode}")
            x_out = blk(x_out, xpos, use_fa_interface=True)
        x_out = x_out.reshape(bs * num_view, num_patches, -1)

        x_out = self.project(x_out)
        x_out = einops.rearrange(
            x_out,
            "(b v) (t_h t_w) (p_h p_w c) -> b v (t_h p_h) (t_w p_w) c",
            v=num_view,
            p_h=self.patch_size,
            p_w=self.patch_size,
            t_h=t_h,
            t_w=t_w,
        )
        return x_out


class ResidualConvBlock(nn.Module):
    """
    Residual convolution block for conv head.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int | None = None,
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
        norm: Literal["group_norm", "layer_norm"] = "group_norm",
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation == "relu":
            activation_cls = lambda: nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            activation_cls = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == "silu":
            activation_cls = lambda: nn.SiLU(inplace=True)
        elif activation == "elu":
            activation_cls = lambda: nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activation_cls(),
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GroupNorm(
                hidden_channels // 32 if norm == "group_norm" else 1, hidden_channels
            ),
            activation_cls(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
        )

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x


class ConvHead(nn.Module):
    """
    Modified Head from MoGe with upsampling and convs, specifically designed for
    3RV2. Ops are autocasted to fp16, besides the last output block which is fp/tf32.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: list[int],
        activations: list[Callable],
        pred_names: list[str],
        patch_size: int,
        dim_upsample: list[int] | None = None,
        num_res_blocks: int = 2,
        res_block_norm: str = "group_norm",
        last_conv_channels: int = 32,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activations = activations
        self.pred_names = pred_names
        if dim_upsample is None:
            dim_upsample = [256, 128, 64]

        self.upsample_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    self._make_upsampler(in_ch, out_ch),
                    *(
                        ResidualConvBlock(
                            out_ch,
                            out_ch,
                            2 * out_ch,
                            activation="relu",
                            norm=res_block_norm,
                        )
                        for _ in range(num_res_blocks)
                    ),
                )
                for in_ch, out_ch in zip(
                    [dim_in] + dim_upsample[:-1], dim_upsample, strict=False
                )
            ]
        )

        self.output_block = nn.ModuleList(
            [
                self._make_output_block(
                    dim_upsample[-1],
                    dim_out_,
                    last_conv_channels,
                )
                for dim_out_ in dim_out
            ]
        )

    def _make_upsampler(self, in_channels: int, out_channels: int):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(
        self,
        dim_in: int,
        dim_out: int,
        last_conv_channels: int,
    ):
        return nn.Sequential(
            nn.Conv2d(
                dim_in,
                last_conv_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                last_conv_channels,
                dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(
        self, tokens: Float[Tensor, "b v p d"], num_view: int, img_h: int, img_w: int
    ) -> Float[Tensor, "b v h w c"]:
        patch_h, patch_w = img_h // self.patch_size, img_w // self.patch_size
        tokens = einops.rearrange(
            tokens,
            "b (v p_h p_w) d -> (b v) d p_h p_w",
            v=num_view,
            p_h=patch_h,
            p_w=patch_w,
        )
        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) ->
        # (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for _, block in enumerate(self.upsample_blocks):
            for layer in block:
                tokens = layer(tokens)

        # (patch_h * 8, patch_w * 8) -> (img_h, img_w)
        tokens = F.interpolate(
            tokens, (img_h, img_w), mode="bilinear", align_corners=False
        )

        # use tf32 for last output block
        with torch.autocast(device_type="cuda", enabled=False):
            if tokens.dtype != torch.float32:
                tokens = tokens.float()
            pred_list = [block(tokens) for block in self.output_block]

        for idx in range(len(pred_list)):
            pred_list[idx] = einops.rearrange(
                pred_list[idx], "(b v) c h w -> b v h w c", v=num_view
            )

        out_dict = {}
        for pred, activation, pred_name in zip(
            pred_list, self.activations, self.pred_names
        ):
            out_dict[pred_name] = activation(pred)
        out_dict["invalid_mask"] = torch.isnan(pred_list[0][..., 0])
        return out_dict


class RaymapHead(nn.Module):
    """ConvHead-isomorphic head that produces a *single* per-view raymap.

    Designed for r81 multi-view raymap diffusion.  Unlike the regular ConvHead
    which produces a per-layer per-pixel output, RaymapHead consumes the same
    decoder tokens but COLLAPSES the layer dimension into the channel axis
    before any spatial conv:

        input  tokens  [B, L*P, D]   (flattened over L)
                 |
                 v  rearrange + reshape
        per-view feature map  [B, L*D, p_h, p_w]  (input ch = L*D)
                 |
                 v  ConvTranspose2d upsample (×2 ×2 ×2) + ResBlocks
                 v  bilinear F.interpolate to (img_h, img_w)
        output spatial features [B, dim_upsample[-1], H, W]
                 |
                 v  Conv 3x3 → ReLU → Conv 1x1 (zero-init)
        output ray-map  [B, 6, H, W]  (1 ray-map per view)

    The 1x1 final conv is zero-initialised (weight + bias = 0) so warm-starting
    from r69e/r80 leaves predicted raymap == 0 at iter 0.

    The forward returns shape ``[B, num_view, H, W, 6]`` to mimic ConvHead's
    return contract; broadcasting over L is the caller's responsibility.

    Args:
        dim_in:           input channel per token (typically decoder_embed_dim)
        num_layers:       L — collapsed into channel dim
        patch_size:       patch_h = patch_w = noise_patchify_size
        dim_upsample:     same default as ConvHead — [256, 128, 64]
        num_res_blocks:   per-stage residual blocks, default 2
        last_conv_channels: final 3x3 → 1x1 hidden width, default 32
    """

    def __init__(
        self,
        dim_in: int,
        num_layers: int,
        patch_size: int,
        dim_upsample: list[int] | None = None,
        num_res_blocks: int = 2,
        res_block_norm: str = "group_norm",
        last_conv_channels: int = 32,
        out_channels: int = 6,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.dim_in = dim_in
        self.out_channels = out_channels
        if dim_upsample is None:
            dim_upsample = [256, 128, 64]

        # Collapsed input channel per spatial location: L * D
        ch_in = num_layers * dim_in

        self.upsample_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    self._make_upsampler(in_ch, out_ch),
                    *(
                        ResidualConvBlock(
                            out_ch,
                            out_ch,
                            2 * out_ch,
                            activation="relu",
                            norm=res_block_norm,
                        )
                        for _ in range(num_res_blocks)
                    ),
                )
                for in_ch, out_ch in zip(
                    [ch_in] + dim_upsample[:-1], dim_upsample, strict=False
                )
            ]
        )

        self.output_block = self._make_output_block(
            dim_upsample[-1],
            out_channels,
            last_conv_channels,
        )

        # CRITICAL: zero-initialise the final 1x1 Conv (both weight and bias).
        # This makes the raymap prediction be exactly zero at init, matching
        # the warm-start convention (r69e ckpt has no RaymapHead → it falls
        # into ckpt missing_keys and we want it to be a no-op until it learns).
        last_conv = self.output_block[-1]
        assert isinstance(last_conv, nn.Conv2d)
        assert last_conv.kernel_size == (1, 1)
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def _make_upsampler(self, in_channels: int, out_channels: int):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(
        self,
        dim_in: int,
        dim_out: int,
        last_conv_channels: int,
    ):
        return nn.Sequential(
            nn.Conv2d(
                dim_in,
                last_conv_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                last_conv_channels,
                dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(
        self, tokens: Float[Tensor, "b vp d"], num_view: int, img_h: int, img_w: int
    ) -> Float[Tensor, "b v h w c"]:
        """Produce one (H, W, 6) raymap per view.

        Args:
            tokens:    [B, V*P, D]  flattened decoder output (V == L here)
            num_view:  V  (== num_layers; passed for shape arithmetic)
            img_h, img_w: target spatial size

        Returns:
            [B, V, H, W, 6]   — one raymap per view (V is the layer dim,
            since RaymapHead is invariant to depth-layer; the caller can
            either average over V or just take the first).  When the caller
            wants a single raymap broadcast to all L layers, it consumes
            the V dim by indexing [:, 0:1, :, :, :].

        NOTE on V == L: in r81 we reuse the same decoder geo tokens (which
        have a layer dim L) for the RaymapHead.  Conceptually the raymap is
        a per-VIEW (camera) quantity but at this point we are inside one
        camera view (B*V_camera was already folded into B by the outer
        train loop).  So L here is the depth-layer count — we collapse it
        into the channel dim and produce a single (H, W, 6) raymap that
        characterises THIS camera view.
        """
        patch_h, patch_w = img_h // self.patch_size, img_w // self.patch_size
        L = num_view

        # tokens: [B, L*P, D] -> [B, L*D, p_h, p_w]
        # Step 1: split L*P -> L, P=p_h*p_w; rearrange so L groups with D into channels
        x = einops.rearrange(
            tokens,
            "b (v p_h p_w) d -> b (v d) p_h p_w",
            v=L,
            p_h=patch_h,
            p_w=patch_w,
        )
        # x.shape == [B, L*D, p_h, p_w]

        for block in self.upsample_blocks:
            for layer in block:
                x = layer(x)

        x = F.interpolate(
            x, (img_h, img_w), mode="bilinear", align_corners=False
        )

        with torch.autocast(device_type="cuda", enabled=False):
            if x.dtype != torch.float32:
                x = x.float()
            raymap = self.output_block(x)  # [B, 6, H, W]

        # Reshape to ConvHead-compatible shape: [B, V, H, W, C].  We produce
        # ONE raymap per camera (=batch element), so V dim is degenerate
        # (=1).  We replicate it to L for broadcastable concat, but the
        # caller can also take [:, 0:1] to keep it explicit.
        raymap = einops.rearrange(raymap, "b c h w -> b 1 h w c")    # [B, 1, H, W, 6]
        raymap = raymap.expand(-1, L, -1, -1, -1)                    # [B, L, H, W, 6]
        return raymap
