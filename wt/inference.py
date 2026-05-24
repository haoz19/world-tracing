"""Inference entry points for ``MultilayerXYZModel``.

This is the release counterpart of :mod:`wt._legacy_train_utils`.  It only
contains the helpers the inference path actually needs:

* Normalisation constants and de-normalisers (XYZ / depth)
* :func:`inference_diffusion`              -- single image / single view
* :func:`inference_diffusion_multiview`    -- r80-style multi-view scene
* :func:`inference_video_diffusion`        -- r76-style video clip
* :func:`_depth_to_xyz`                    -- pinhole unprojection helper
* :func:`_bypass_activation_checkpointing` -- unwrap AC wrappers under no_grad

The original implementation used :class:`FMLossWrapper.denoise`.  We replace
it with a small Euler ODE sampler in :mod:`wt.sampling`.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from wt.sampling import denoise_geometry

# ---------------------------------------------------------------------------
# Global normalisation constants (computed from 500K samples; see
# ``compute_depth_stats.py`` in the all-source release).
# ---------------------------------------------------------------------------

DEPTH_MEAN = 1.904278
DEPTH_STD = 0.706423
LOG_DEPTH_MEAN = 0.625089
LOG_DEPTH_STD = 0.189539
DISP_MEAN = 0.546635
DISP_STD = 0.167720
XYZ_MEAN = np.array([-0.002366, -0.004008, 1.904278], dtype=np.float32)
XYZ_STD = np.array([0.280865, 0.315927, 0.706423], dtype=np.float32)

#: Depth contraction reference distance (meters).  Overridable at runtime
#: (e.g. by scene-style configs that want ``exp(scene_log_depth_mean)``).
DEPTH_CONTRACTION_REF: float = 1.87

VALID_XYZ_NORM_MODES = (
    "contraction",
    "zscore",
    "log_zscore",
    "disp_zscore",
    "median_log_global",
    "depth_contraction",
    "median_log",
)


# ---------------------------------------------------------------------------
# Denormalisation
# ---------------------------------------------------------------------------


def denormalize_depth(depth_norm: torch.Tensor, use_log_depth: bool) -> torch.Tensor:
    """Inverse of ``normalize_depth``.

    Linear mode: ``depth = depth_norm * DEPTH_STD + DEPTH_MEAN``.
    Log mode:    ``depth = exp(depth_norm * LOG_DEPTH_STD + LOG_DEPTH_MEAN)``.
    """
    if use_log_depth:
        return torch.exp(depth_norm * LOG_DEPTH_STD + LOG_DEPTH_MEAN)
    return depth_norm * DEPTH_STD + DEPTH_MEAN


def denormalize_xyz_torch(xyz_norm: torch.Tensor, mode: str) -> torch.Tensor:
    """Inverse of ``normalize_xyz``.

    ``mode`` must match the value used during training (which is recorded in
    the experiment registry, e.g. ``r75b -> zscore``, ``r69e -> median_log``,
    ``r76 -> zscore``).  Supported modes: see :data:`VALID_XYZ_NORM_MODES`.
    """
    if mode == "contraction":
        # Per-sample contraction is not invertible without the scale; the
        # caller is expected to keep the contracted output.
        return xyz_norm

    if mode == "zscore":
        mean = torch.tensor(XYZ_MEAN, device=xyz_norm.device, dtype=xyz_norm.dtype)
        std = torch.tensor(XYZ_STD, device=xyz_norm.device, dtype=xyz_norm.dtype)
        return xyz_norm * std + mean

    if mode == "log_zscore":
        xyz = torch.empty_like(xyz_norm)
        xyz[..., 0] = xyz_norm[..., 0] * XYZ_STD[0] + XYZ_MEAN[0]
        xyz[..., 1] = xyz_norm[..., 1] * XYZ_STD[1] + XYZ_MEAN[1]
        xyz[..., 2] = torch.exp(xyz_norm[..., 2] * LOG_DEPTH_STD + LOG_DEPTH_MEAN)
        return xyz

    if mode == "disp_zscore":
        xyz = torch.empty_like(xyz_norm)
        xyz[..., 0] = xyz_norm[..., 0] * XYZ_STD[0] + XYZ_MEAN[0]
        xyz[..., 1] = xyz_norm[..., 1] * XYZ_STD[1] + XYZ_MEAN[1]
        disp = xyz_norm[..., 2] * DISP_STD + DISP_MEAN
        xyz[..., 2] = 1.0 / disp.clamp(min=1e-6)
        return xyz

    if mode == "depth_contraction":
        D = DEPTH_CONTRACTION_REF
        z_norm = xyz_norm[..., 2].clamp(-1 + 1e-6, 1 - 1e-6)
        inv_1mz = 1.0 / (1.0 - z_norm)
        xyz = torch.empty_like(xyz_norm)
        xyz[..., 0] = xyz_norm[..., 0] * 2.0 * D * inv_1mz
        xyz[..., 1] = xyz_norm[..., 1] * 2.0 * D * inv_1mz
        xyz[..., 2] = D * (1.0 + z_norm) * inv_1mz
        return xyz

    if mode in ("median_log", "median_log_global"):
        # See ``normalize_xyz_median_log`` in the all-source release for the
        # forward transform.  At inference time we don't know the per-sample
        # median ``m``; we invert in the ``m=1`` (relative-scale) frame so
        # downstream tools (rerun, depth metrics) all operate in the same
        # normalised frame.
        xyz = torch.empty_like(xyz_norm)
        for ch in (0, 1):
            t = xyz_norm[..., ch]
            xyz[..., ch] = torch.sign(t) * (torch.exp(torch.abs(t)) - 1.0)
        xyz[..., 2] = torch.exp(xyz_norm[..., 2])
        return xyz

    raise ValueError(
        f"Unknown xyz_norm_mode: {mode}. Expected one of {VALID_XYZ_NORM_MODES}"
    )


# ---------------------------------------------------------------------------
# Depth → XYZ unprojection
# ---------------------------------------------------------------------------


def _depth_to_xyz(
    depth: torch.Tensor, intrinsics: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Pinhole unprojection of depth to camera-space XYZ.

    Args:
        depth: ``[B, L, H, W]`` metric depth.
        intrinsics: ``[B, 3, 3]`` or ``[3, 3]``.
        mask: ``[B, L, H, W]`` bool valid mask (invalid positions zeroed out).

    Returns:
        ``[B, L, H, W, 3]`` camera-space XYZ.
    """
    batch_size, num_layers, height, width = depth.shape
    device = depth.device
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1, -1)
    fx = intrinsics[:, 0, 0][:, None, None, None]
    fy = intrinsics[:, 1, 1][:, None, None, None]
    cx = intrinsics[:, 0, 2][:, None, None, None]
    cy = intrinsics[:, 1, 2][:, None, None, None]
    u = torch.arange(width, device=device, dtype=depth.dtype)
    v = torch.arange(height, device=device, dtype=depth.dtype)
    v, u = torch.meshgrid(v, u, indexing="ij")
    u = u[None, None].expand(batch_size, num_layers, -1, -1)
    v = v[None, None].expand(batch_size, num_layers, -1, -1)
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    xyz = torch.stack([x, y, z], dim=-1)
    xyz[~mask] = 0.0
    return xyz


# ---------------------------------------------------------------------------
# Activation-checkpoint bypass (required around no_grad model calls when the
# checkpoint frame state would otherwise be polluted)
# ---------------------------------------------------------------------------

_AC_ATTR = "_checkpoint_wrapped_module"


@contextlib.contextmanager
def _bypass_activation_checkpointing(model):
    """Temporarily unwrap activation-checkpoint wrappers on all decoder blocks.

    Running the model under ``torch.no_grad()`` through checkpoint-wrapped
    blocks can corrupt the global checkpoint frame state used by
    ``torch.utils.checkpoint`` (non-reentrant mode), which then triggers a
    tensor-metadata mismatch on the next training backward pass.  Since
    checkpointing is pointless under ``no_grad`` anyway, we temporarily
    replace each wrapped block with its inner module.
    """
    net = getattr(model, "net", model)
    saved: list[tuple[torch.nn.ModuleList, int, torch.nn.Module]] = []
    for attr in ("decoder_blocks", "global_encoder_blocks"):
        blocks = getattr(net, attr, None)
        if blocks is None or isinstance(blocks, torch.nn.Identity):
            continue
        for i, blk in enumerate(blocks):
            inner = getattr(blk, _AC_ATTR, None)
            if inner is not None:
                saved.append((blocks, i, blk))
                blocks[i] = inner
    try:
        yield
    finally:
        for blocks, i, original in saved:
            blocks[i] = original


# ---------------------------------------------------------------------------
# inference_diffusion (single image)
# ---------------------------------------------------------------------------


@torch.no_grad()
def inference_diffusion(
    model,
    rgb: torch.Tensor,
    num_steps: int = 50,
    total_elements: int | None = None,
    gt_mask: torch.Tensor | None = None,
    use_gt_mask: bool = False,
    output_mode: str = "xyz",
    intrinsics: torch.Tensor | None = None,
    use_log_depth: bool = False,
    cfm_mask: bool = False,
    cfm_uniform_noise: bool = False,
    cfm_noise_type: str = "fixed_0.5",
    xyz_norm_mode: str = "contraction",
    predict_color: bool = False,
    model_task: str = "joint",
    depth_only: bool = False,
    invalid_fill_mode: str | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """Run multilayer-geometry diffusion sampling on a single RGB image.

    Args:
        model: a ``MultilayerXYZModel``.
        rgb: ``[B, 3, H, W]`` in ``[0, 1]``.
        num_steps: number of Euler ODE steps.
        total_elements: kept for parity (the release sampler does not depend
            on it because ``hunyuan_shift_factor=1.0`` is the only mode used).
        gt_mask: ``[B, L, H, W]`` bool GT mask.  Required for the ``depth_only``
            path (the model does not produce a mask in that mode).
        use_gt_mask: override the predicted mask with ``gt_mask`` (joint mode).
        output_mode: ``"xyz"`` (default for the release configs) or ``"depth"``.
        intrinsics: ``[3,3]`` or ``[B,3,3]`` — required for depth mode.
        use_log_depth, xyz_norm_mode, cfm_*, model_task, depth_only,
            predict_color: must match the values used during training.

    Returns:
        ``(xyz_pred, mask_pred, rgb_pred)``:
            * ``xyz_pred``: ``[B, L, H, W, 3]`` camera-space XYZ (or ``None``
              for the mask-only path).
            * ``mask_pred``: ``[B, L, H, W]`` bool, AND-accumulated across L.
            * ``rgb_pred``: ``[B, L, H, W, 3]`` in ``[0, 1]`` when
              ``predict_color=True``, otherwise ``None``.
    """
    del total_elements
    model.eval()
    batch_size, _, height, width = rgb.shape
    num_layers = model.num_layers
    device = rgb.device

    if model_task == "xyz_only":
        # Same canonicalisation as in the model __init__.
        model_task = "split_token"
        depth_only = True

    if model_task == "mask":
        nc = 1
    elif model_task == "geo":
        nc = 3 if output_mode == "xyz" else 1
    elif model_task == "split_token":
        nc_geo = 3 if output_mode == "xyz" else 1
        nc = nc_geo if depth_only else nc_geo + 1
    else:
        nc = 7 if predict_color else 4

    img = rgb * 2 - 1
    img = img[:, None].repeat(1, num_layers, 1, 1, 1)
    conditioning = {
        "rgb": img * 0.5 + 0.5,
        "noise_height": height,
        "noise_width": width,
        "noise_nview": num_layers,
        "batch_size": batch_size,
    }
    if invalid_fill_mode is not None and gt_mask is not None:
        valid_mask_flat = gt_mask.bool().reshape(batch_size, -1, 1).float()
        conditioning["valid_mask"] = valid_mask_flat
        conditioning["invalid_fill_mode"] = invalid_fill_mode

    x_t = torch.randn(batch_size, nc, num_layers, height, width, device=device)
    if cfm_mask and model_task in ("mask", "joint", "split_token"):
        if cfm_noise_type == "uniform_0_1" or cfm_uniform_noise:
            x_t[:, nc - 1] = torch.rand_like(x_t[:, nc - 1])
        elif cfm_noise_type == "normal_0.5_1":
            x_t[:, nc - 1] = torch.randn_like(x_t[:, nc - 1]) + 0.5
        elif cfm_noise_type == "normal_0_1_sym":
            x_t[:, nc - 1] = torch.randn_like(x_t[:, nc - 1])
        else:
            x_t[:, nc - 1] = 0.5

    pred = denoise_geometry(model, x_t, conditioning, iterations=num_steps)

    # ---- Mask extraction ----
    if model_task == "mask":
        if cfm_noise_type == "normal_0_1_sym":
            mask_pred = pred[..., 0] > 0.0
        else:
            mask_pred = pred[..., 0] > 0.5
        for k in range(1, num_layers):
            mask_pred[:, k] = mask_pred[:, k] & mask_pred[:, k - 1]
        return None, mask_pred, None

    if model_task == "geo" or (model_task == "split_token" and depth_only):
        if gt_mask is None:
            raise ValueError(
                "model_task='geo' / depth_only requires gt_mask for inference."
            )
        mask_pred = gt_mask.bool()
    else:
        mask_ch = nc - 1
        if use_gt_mask and gt_mask is not None:
            mask_pred = gt_mask.bool()
        elif cfm_mask:
            if cfm_noise_type == "normal_0_1_sym":
                mask_pred = pred[..., mask_ch] > 0.0
            else:
                mask_pred = pred[..., mask_ch] > 0.5
        else:
            mask_pred = torch.sigmoid(pred[..., mask_ch]) > 0.5

    for k in range(1, num_layers):
        mask_pred[:, k] = mask_pred[:, k] & mask_pred[:, k - 1]

    # ---- Geometry extraction ----
    if output_mode == "xyz":
        xyz_pred = pred[..., :3]
        if xyz_norm_mode != "contraction":
            xyz_pred = denormalize_xyz_torch(xyz_pred, mode=xyz_norm_mode)
        xyz_pred[~mask_pred] = 0.0
    elif output_mode == "depth":
        if intrinsics is None:
            raise ValueError("Depth mode requires the 'intrinsics' parameter.")
        depth_normalized = pred[..., 0]
        depth = denormalize_depth(depth_normalized, use_log_depth)
        depth = depth.clamp(min=0.1)
        for k in range(1, num_layers):
            both_valid = mask_pred[:, k] & mask_pred[:, k - 1]
            depth[:, k][both_valid] = torch.maximum(
                depth[:, k][both_valid], depth[:, k - 1][both_valid]
            )
        xyz_pred = _depth_to_xyz(depth, intrinsics, mask_pred)
    else:
        raise ValueError(f"Unknown output_mode: {output_mode}")

    rgb_pred = None
    if predict_color and model_task == "joint":
        rgb_pred = (pred[..., 3:6] + 1) / 2
        rgb_pred = rgb_pred.clamp(0, 1)
        rgb_pred[~mask_pred] = 0.0

    return xyz_pred, mask_pred, rgb_pred


# ---------------------------------------------------------------------------
# inference_diffusion_multiview (r80 / r69e-style)
# ---------------------------------------------------------------------------


@torch.no_grad()
def inference_diffusion_multiview(
    model,
    rgb_mv: torch.Tensor,
    num_steps: int = 50,
    total_elements: int | None = None,
    gt_mask_mv: torch.Tensor | None = None,
    use_gt_mask: bool = False,
    output_mode: str = "xyz",
    cfm_mask: bool = False,
    cfm_uniform_noise: bool = False,
    cfm_noise_type: str = "fixed_0.5",
    xyz_norm_mode: str = "median_log_global",
    predict_color: bool = False,
    model_task: str = "split_token",
    depth_only: bool = True,
    invalid_fill_mode: str | None = "noise",
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """Multi-view counterpart of :func:`inference_diffusion`.

    The release scene model ``r69e_v2_evermotion_ithappy_504`` is multi-view
    and uses this function.  ``rgb_mv`` has shape ``[B, V, 3, H, W]``; ``V``
    views are folded into the leading batch dim so the model sees an
    effective batch ``B*V``, with ``conditioning["num_view"] = V`` threading
    view-block attention through the decoder.

    Returns ``(xyz_pred_mv, mask_pred_mv, rgb_pred_mv)`` shaped
    ``[B, V, L, H, W, ...]``.  ``output_mode='xyz'`` is the only supported
    mode (depth-mode unprojection would require per-view intrinsics).
    """
    del total_elements
    model.eval()
    assert rgb_mv.ndim == 5, f"rgb_mv must be [B,V,3,H,W], got {tuple(rgb_mv.shape)}"
    batch_size, num_view, _, height, width = rgb_mv.shape
    num_layers = model.num_layers
    device = rgb_mv.device

    if model_task == "xyz_only":
        model_task = "split_token"
        depth_only = True

    if model_task == "mask":
        nc = 1
    elif model_task == "geo":
        nc = 3 if output_mode == "xyz" else 1
    elif model_task == "split_token":
        nc_geo = 3 if output_mode == "xyz" else 1
        nc = nc_geo if depth_only else nc_geo + 1
    else:
        nc = 7 if predict_color else 4

    bv = batch_size * num_view
    rgb_flat = rgb_mv.reshape(bv, 3, height, width)
    img = rgb_flat * 2 - 1
    img = img[:, None].repeat(1, num_layers, 1, 1, 1)
    conditioning = {
        "rgb": img * 0.5 + 0.5,
        "noise_height": height,
        "noise_width": width,
        "noise_nview": num_layers,
        "batch_size": bv,
        "num_view": num_view,
    }
    if invalid_fill_mode is not None and gt_mask_mv is not None:
        gt_mask_flat = gt_mask_mv.reshape(bv, num_layers, height, width).bool()
        valid_mask_flat = gt_mask_flat.reshape(bv, -1, 1).float()
        conditioning["valid_mask"] = valid_mask_flat
        conditioning["invalid_fill_mode"] = invalid_fill_mode
    else:
        gt_mask_flat = None

    x_t = torch.randn(bv, nc, num_layers, height, width, device=device)
    if cfm_mask and model_task in ("mask", "joint", "split_token"):
        if cfm_noise_type == "uniform_0_1" or cfm_uniform_noise:
            x_t[:, nc - 1] = torch.rand_like(x_t[:, nc - 1])
        elif cfm_noise_type == "normal_0.5_1":
            x_t[:, nc - 1] = torch.randn_like(x_t[:, nc - 1]) + 0.5
        elif cfm_noise_type == "normal_0_1_sym":
            x_t[:, nc - 1] = torch.randn_like(x_t[:, nc - 1])
        else:
            x_t[:, nc - 1] = 0.5

    pred = denoise_geometry(model, x_t, conditioning, iterations=num_steps)

    if model_task == "mask":
        if cfm_noise_type == "normal_0_1_sym":
            mask_pred_flat = pred[..., 0] > 0.0
        else:
            mask_pred_flat = pred[..., 0] > 0.5
        for k in range(1, num_layers):
            mask_pred_flat[:, k] = mask_pred_flat[:, k] & mask_pred_flat[:, k - 1]
        return None, mask_pred_flat.reshape(
            batch_size, num_view, num_layers, height, width
        ), None

    if model_task == "geo" or (model_task == "split_token" and depth_only):
        if gt_mask_flat is None:
            raise ValueError(
                "inference_diffusion_multiview: depth_only path requires "
                "gt_mask_mv ([B,V,L,H,W])."
            )
        mask_pred_flat = gt_mask_flat
    else:
        mask_ch = nc - 1
        if use_gt_mask and gt_mask_flat is not None:
            mask_pred_flat = gt_mask_flat
        elif cfm_mask:
            if cfm_noise_type == "normal_0_1_sym":
                mask_pred_flat = pred[..., mask_ch] > 0.0
            else:
                mask_pred_flat = pred[..., mask_ch] > 0.5
        else:
            mask_pred_flat = torch.sigmoid(pred[..., mask_ch]) > 0.5
    for k in range(1, num_layers):
        mask_pred_flat[:, k] = mask_pred_flat[:, k] & mask_pred_flat[:, k - 1]

    if output_mode != "xyz":
        raise NotImplementedError(
            "inference_diffusion_multiview currently supports output_mode='xyz' "
            "only (per-view depth-mode unprojection is not wired in the release)."
        )
    xyz_pred_flat = pred[..., :3]
    if xyz_norm_mode != "contraction":
        xyz_pred_flat = denormalize_xyz_torch(xyz_pred_flat, mode=xyz_norm_mode)
    xyz_pred_flat[~mask_pred_flat] = 0.0

    rgb_pred_flat = None
    if predict_color and model_task == "joint":
        rgb_pred_flat = (pred[..., 3:6] + 1) / 2
        rgb_pred_flat = rgb_pred_flat.clamp(0, 1)
        rgb_pred_flat[~mask_pred_flat] = 0.0

    xyz_pred_mv = xyz_pred_flat.reshape(
        batch_size, num_view, num_layers, height, width, 3
    )
    mask_pred_mv = mask_pred_flat.reshape(
        batch_size, num_view, num_layers, height, width
    )
    rgb_pred_mv = (
        rgb_pred_flat.reshape(batch_size, num_view, num_layers, height, width, 3)
        if rgb_pred_flat is not None
        else None
    )
    return xyz_pred_mv, mask_pred_mv, rgb_pred_mv


# ---------------------------------------------------------------------------
# inference_video_diffusion (r76 dynamic clip)
# ---------------------------------------------------------------------------


@torch.no_grad()
def inference_video_diffusion(
    model,
    rgb_clip: torch.Tensor,
    num_steps: int = 50,
    total_elements: int | None = None,
    gt_mask_clip: torch.Tensor | None = None,
    use_gt_mask: bool = False,
    output_mode: str = "xyz",
    intrinsics: torch.Tensor | None = None,
    use_log_depth: bool = False,
    cfm_mask: bool = False,
    cfm_uniform_noise: bool = False,
    cfm_noise_type: str = "fixed_0.5",
    xyz_norm_mode: str = "contraction",
    predict_color: bool = False,
    model_task: str = "split_token",
    depth_only: bool = False,
    invalid_fill_mode: str | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """r76-style video diffusion (T frames jointly).

    Identical semantics to :func:`inference_diffusion` per-frame, but all T
    frames are denoised jointly so the model's temporal attention blocks can
    couple them.  ``rgb_clip`` has shape ``[B, T, 3, H, W]``.
    """
    del total_elements
    model.eval()
    assert rgb_clip.ndim == 5, f"rgb_clip must be [B,T,3,H,W], got {tuple(rgb_clip.shape)}"
    batch_size, num_time, _, height, width = rgb_clip.shape
    num_layers = model.num_layers
    device = rgb_clip.device
    TL = num_time * num_layers

    if model_task == "mask":
        nc = 1
    elif model_task == "geo":
        nc = 3 if output_mode == "xyz" else 1
    elif model_task == "split_token":
        nc_geo = 3 if output_mode == "xyz" else 1
        nc = nc_geo if depth_only else nc_geo + 1
    else:
        nc = 7 if predict_color else 4

    img = rgb_clip * 2 - 1
    img_tl = (
        img.unsqueeze(2)
        .expand(-1, -1, num_layers, -1, -1, -1)
        .reshape(batch_size, TL, 3, height, width)
        .contiguous()
    )
    conditioning = {
        "rgb": img_tl * 0.5 + 0.5,
        "noise_height": height,
        "noise_width": width,
        "noise_nview": TL,
        "batch_size": batch_size,
        "num_time": int(num_time),
    }
    if invalid_fill_mode is not None and gt_mask_clip is not None:
        valid_mask_flat = gt_mask_clip.bool().reshape(batch_size, -1, 1).float()
        conditioning["valid_mask"] = valid_mask_flat
        conditioning["invalid_fill_mode"] = invalid_fill_mode
    elif depth_only and gt_mask_clip is not None:
        valid_mask_flat = gt_mask_clip.bool().reshape(batch_size, -1, 1).float()
        conditioning["valid_mask"] = valid_mask_flat

    x_t = torch.randn(batch_size, nc, TL, height, width, device=device)
    if cfm_mask and model_task in ("mask", "joint", "split_token"):
        if cfm_noise_type == "uniform_0_1" or cfm_uniform_noise:
            x_t[:, nc - 1] = torch.rand_like(x_t[:, nc - 1])
        elif cfm_noise_type == "normal_0.5_1":
            x_t[:, nc - 1] = torch.randn_like(x_t[:, nc - 1]) + 0.5
        elif cfm_noise_type == "normal_0_1_sym":
            x_t[:, nc - 1] = torch.randn_like(x_t[:, nc - 1])
        else:
            x_t[:, nc - 1] = 0.5

    pred = denoise_geometry(model, x_t, conditioning, iterations=num_steps)
    pred_tl = pred.reshape(batch_size, num_time, num_layers, height, width, nc)

    if model_task == "mask":
        if cfm_noise_type == "normal_0_1_sym":
            mask_pred = pred_tl[..., 0] > 0.0
        else:
            mask_pred = pred_tl[..., 0] > 0.5
        for k in range(1, num_layers):
            mask_pred[:, :, k] = mask_pred[:, :, k] & mask_pred[:, :, k - 1]
        return None, mask_pred, None

    if model_task == "geo" or (model_task == "split_token" and depth_only):
        if gt_mask_clip is None:
            raise ValueError(
                "inference_video_diffusion: depth_only path requires gt_mask_clip "
                "([B,T,L,H,W])."
            )
        mask_pred = gt_mask_clip.bool()
    else:
        mask_ch = nc - 1
        if use_gt_mask and gt_mask_clip is not None:
            mask_pred = gt_mask_clip.bool()
        elif cfm_mask:
            if cfm_noise_type == "normal_0_1_sym":
                mask_pred = pred_tl[..., mask_ch] > 0.0
            else:
                mask_pred = pred_tl[..., mask_ch] > 0.5
        else:
            mask_pred = torch.sigmoid(pred_tl[..., mask_ch]) > 0.5
    for k in range(1, num_layers):
        mask_pred[:, :, k] = mask_pred[:, :, k] & mask_pred[:, :, k - 1]

    if output_mode == "xyz":
        xyz_pred = pred_tl[..., :3]
        if xyz_norm_mode != "contraction":
            xyz_pred = denormalize_xyz_torch(xyz_pred, mode=xyz_norm_mode)
        xyz_pred = xyz_pred.contiguous()
        xyz_pred[~mask_pred] = 0.0
    elif output_mode == "depth":
        if intrinsics is None:
            raise ValueError("Depth mode requires the 'intrinsics' parameter.")
        depth_normalized = pred_tl[..., 0]
        depth = denormalize_depth(depth_normalized, use_log_depth)
        depth = depth.clamp(min=0.1)
        for k in range(1, num_layers):
            both_valid = mask_pred[:, :, k] & mask_pred[:, :, k - 1]
            depth[:, :, k][both_valid] = torch.maximum(
                depth[:, :, k][both_valid], depth[:, :, k - 1][both_valid]
            )
        xyz_frames = []
        for t in range(num_time):
            xyz_frames.append(_depth_to_xyz(depth[:, t], intrinsics, mask_pred[:, t]))
        xyz_pred = torch.stack(xyz_frames, dim=1)
    else:
        raise ValueError(f"Unknown output_mode: {output_mode}")

    rgb_pred = None
    if predict_color and model_task == "joint":
        rgb_pred = (pred_tl[..., 3:6] + 1) / 2
        rgb_pred = rgb_pred.clamp(0, 1).contiguous()
        rgb_pred[~mask_pred] = 0.0

    return xyz_pred, mask_pred, rgb_pred
