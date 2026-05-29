"""Slim helpers used by ``MultilayerXYZModel`` and its ``MultilayerBackbone``.

Training-only utilities (gaussian merging, camera-solve, splat conversion,
distributed memory stats, ...) are intentionally dropped from the release
build.
"""

from __future__ import annotations

import collections.abc
from itertools import repeat

import einops
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional as F

from wt._core.engine import activation_checkpoint


def _ntuple(n):
    """Convert input to tuple of length n. Legacy of dust3r."""

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def reset_parameters(modules):
    if hasattr(modules, "reset_parameters"):
        modules.reset_parameters()


def freeze_all_params(modules):
    for module in modules:
        try:
            for _, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            module.requires_grad = False


def unfreeze_all_params(modules):
    for module in modules:
        try:
            for _, param in module.named_parameters():
                param.requires_grad = True
        except AttributeError:
            module.requires_grad = True


class PositionGetter:
    """Return positions of patches."""

    def __init__(self):
        self.cache_positions: dict[tuple[int, int], Tensor] = {}

    def __call__(self, b: int, h: int, w: int, device) -> Tensor:
        if (h, w) not in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h, w] = torch.cartesian_prod(y, x)
        pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()
        return pos


class PositionGetter3D:
    """Return positions of 3D patches."""

    def __init__(self):
        self.cache_positions: dict[tuple[int, int, int], Tensor] = {}

    def __call__(self, b, h, w, d, device) -> Tensor:
        if (h, w, d) not in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            z = torch.arange(d, device=device)
            self.cache_positions[h, w, d] = torch.cartesian_prod(y, x, z)
        pos = self.cache_positions[h, w, d]
        pos = einops.repeat(pos, "... -> b ...", b=b).clone()
        return pos


def one_dropout(x: Tensor, p: float = 0.0, training: bool = True) -> Tensor:
    """Custom dropout replacing dropped values with 1 (used in raymap paths)."""
    if (not training or p == 0) and p != 1:
        return x
    mask = torch.bernoulli(torch.full_like(x, 1 - p))
    return mask * x + (1 - mask) * 1


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Raymap helpers (kept as no-op-safe placeholders).  The release configs
# (r75b, r69e, r76) all run with ``encoder_model='moge'``, ``use_raymap=False``
# at inference time, so ``compute_patch_raymap`` is not invoked.  We still
# expose stubs that raise so misconfiguration is caught early.
# ---------------------------------------------------------------------------


class CameraScaleNormalizerV2:
    """Stub: never invoked for r75b/r69e/r76 release configs."""

    def __init__(self):
        self.static_threshold = 0.05
        self.scale: Tensor | None = None

    def apply(self, trans: Tensor) -> Tensor:
        _, nview, _ = trans.shape
        max_trans = (trans - trans[:, :1]).norm(2, -1).max(1)[0]
        static_mask = max_trans < self.static_threshold
        scale_static = torch.ones_like(max_trans)
        self.scale = torch.where(static_mask, scale_static, max_trans)
        scale_broadcast = einops.repeat(self.scale, "b -> b v c", v=nview, c=3)
        return trans / scale_broadcast

    def get_scale(self) -> Tensor:
        return self.scale  # type: ignore[return-value]

    def reset(self):
        self.scale = None


class RaymapProjector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.raymap_multiplier = dim // 6
        self.project = nn.Linear(dim + self.raymap_multiplier * 6, dim)
        zero_module(self.project)

    def forward(
        self, tokens: Float[Tensor, "b v p d"], raymap: Float[Tensor, "b v p 6"]
    ) -> Float[Tensor, "b v p d"]:
        raymap = einops.repeat(raymap, "... d -> ... (d k)", k=self.raymap_multiplier)
        tokens_out = torch.cat([tokens, raymap], dim=-1)
        tokens = tokens + self.project(tokens_out)
        return tokens


class RaymapProjectorV2(nn.Module):
    def __init__(self, dim: int, hidden: int = 4, in_dim: int = 6):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(in_dim, hidden * dim),
            nn.GELU(),
            nn.Linear(hidden * dim, 2 * dim),
        )
        zero_module(self.cond[-1])

    def forward(
        self, tokens: Float[Tensor, "... d1"], raymap: Float[Tensor, "... d2"]
    ) -> Float[Tensor, "... d1"]:
        with torch.autocast(device_type="cuda", enabled=False):
            raymap = raymap.float()
            gamma, beta = self.cond(raymap).chunk(2, dim=-1)
            tokens = tokens * (1 + gamma) + beta
        return tokens


def compute_patch_raymap(camera, patch_size: int) -> Tensor:
    raise NotImplementedError(
        "compute_patch_raymap is a training-only helper.  "
        "Release configs (r75b/r69e/r76) do not exercise it."
    )


def create_raymap_dropout_mask(bs: int, nview: int, dev: torch.device) -> Tensor:
    uvar = torch.rand(bs, 1, 1, 1, device=dev)
    keep_all = (uvar < 0.5).float()
    drop_some = (uvar >= 0.75).float()
    keep_view = (torch.rand(bs, nview, 1, 1, device=dev) > 0.5).float()
    return keep_all + drop_some * keep_view


# ---------------------------------------------------------------------------
# Image patch / norm helpers — used by the encoder front-end.
# ---------------------------------------------------------------------------


def masked_var_mean(
    x: Float[Tensor, "b c t h w"],
    mask: Float[Tensor, "b t h w"] | None,
    dim: int | list[int],
    keepdim: bool = True,
) -> tuple[Tensor, Tensor]:
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    else:
        mask = mask[:, None].expand_as(x)
    mask = mask.to(x.dtype)
    count = mask.sum(dim=dim, keepdim=keepdim)
    count_safe = count.clamp_min(1)
    mean = (x * mask).sum(dim=dim, keepdim=keepdim) / count_safe
    diff2 = ((x - mean) * mask) ** 2
    var_num = diff2.sum(dim=dim, keepdim=keepdim)
    var = var_num / count_safe
    return var, mean


def layer_norm_2d(
    x: Float[Tensor, "b c t h w"],
    dim: int | list[int],
    eps: float = 1e-6,
) -> Float[Tensor, "b c t h w"]:
    var, mean = torch.var_mean(x, dim=dim, keepdim=True, correction=0)
    return (x - mean) * torch.rsqrt(var + eps)


def masked_layer_norm_2d(
    x: Float[Tensor, "b c t h w"],
    dim: int | list[int],
    eps: float = 1e-6,
    valid_mask: Float[Tensor, "b t h w"] | None = None,
    scale_only: bool = False,
) -> Float[Tensor, "b c t h w"]:
    var, mean = masked_var_mean(x, valid_mask, dim, keepdim=True)
    x_out = x if scale_only else (x - mean)
    return x_out * torch.rsqrt(var + eps)


def patchify_image(
    img: Float[Tensor, "b s d1"], height: int, width: int, patch_size: int
) -> Float[Tensor, "b v p d2"]:
    return einops.rearrange(
        img,
        "b (v h ph w pw) d -> b v (h w) (ph pw d)",
        h=height // patch_size,
        w=width // patch_size,
        ph=patch_size,
        pw=patch_size,
    )


def unpatchify_image(
    img: Float[Tensor, "b v p d1"], height: int, width: int, patch_size: int
) -> Float[Tensor, "b s d2"]:
    return einops.rearrange(
        img,
        "b v (h w) (ph pw d) -> b (v h ph w pw) d",
        h=height // patch_size,
        w=width // patch_size,
        ph=patch_size,
        pw=patch_size,
    )


def image_plane_uv(
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """UV grid on the image plane (matches MoGe convention)."""
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5
    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def query_patches(
    signal: Float[Tensor, "b t c h w"], query_indices: Int[Tensor, "b s"], ksize: int
) -> Float[Tensor, "b s d"]:
    """Query k×k patches at given pixel indices via im2col (training utility)."""
    _, _, nc, ht, wt = signal.shape
    pad = ksize // 2
    hw = ht * wt
    signal_padded = F.pad(signal, (pad, pad, pad, pad))
    h_pad, w_pad = ht + 2 * pad, wt + 2 * pad
    query_t = query_indices // hw
    query_y = (query_indices % hw) // wt + pad
    query_x = (query_indices % hw) % wt + pad
    grid_1d = torch.arange(-pad, pad + 1, device=signal.device)
    dy, dx = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
    dy, dx = dy.reshape(1, 1, -1), dx.reshape(1, 1, -1)
    query_indices_img = (
        query_t[..., None] * (h_pad * w_pad)
        + (query_y[..., None] + dy) * w_pad
        + (query_x[..., None] + dx)
    ).long()
    query_indices_img = einops.repeat(
        query_indices_img, "bs s kk -> bs (s kk) nc", nc=nc
    )
    flat = einops.rearrange(signal_padded, "b t c h w -> b (t h w) c")
    gathered = flat.gather(dim=1, index=query_indices_img)
    return einops.rearrange(
        gathered, "b (s kk) c -> b s (c kk)", s=query_indices.shape[1]
    )


def apply_ac_module(module: nn.Module, ac_mode: str = "full") -> nn.Module:
    return activation_checkpoint.apply_activation_checkpointing(module, mode=ac_mode)


def apply_ac(blocks: nn.ModuleList, ac_mode: str = "full") -> nn.ModuleList:
    for layer_id, block in enumerate(blocks):
        blocks[layer_id] = activation_checkpoint.apply_activation_checkpointing(
            block, mode=ac_mode
        )
    if hasattr(activation_checkpoint, "monkey_patch_named_parameters"):
        blocks = activation_checkpoint.monkey_patch_named_parameters(blocks)
    return blocks


def check_model_diff(model1, model2):
    """No-op debug helper (kept for API parity)."""
    for name, param in model1.named_parameters():
        if not torch.allclose(param.data, model2.state_dict()[name].data):
            print(f"[check_model_diff] {name} differs")
