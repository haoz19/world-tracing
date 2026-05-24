"""Lightweight data loading helpers for the released inference path.

Functions are grouped by entry point:

* ``load_rgba_image`` + ``preprocess_rgba_for_model``: single image
  (used by ``examples/infer_rgba.py`` and ``examples/infer_scene.py``).
* ``load_video_clip`` + ``preprocess_clip_for_model``: T-frame clip with
  a single shared crop (used by ``examples/infer_video.py``).

All helpers produce ``(rgb_tensor, mask_tensor, intrinsics_tensor)`` (with
an extra leading T dim for clips) ready for the corresponding
``inference_*`` entry point.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

#: Default pinhole intrinsics for the training data (Objaverse renders,
#: 512×512 with horizontal FoV ≈ 54.7°).  Used when the user does not supply
#: explicit intrinsics; downstream code can also recover them from the
#: predicted XYZ via :func:`wt.intrinsics.solve_intrinsics_from_xyz`.
FIXED_FX = 500.0
FIXED_FY = 500.0
FIXED_CX = 256.0
FIXED_CY = 256.0
FIXED_RES = 512


def make_default_intrinsics(h: int, w: int) -> np.ndarray:
    """Build a pinhole K matrix matching the training data at resolution H×W.

    The training renders use ``fx=fy=500``, ``cx=cy=256`` at ``512×512``.  We
    scale proportionally so the field of view is preserved across image
    sizes.
    """
    sx, sy = w / FIXED_RES, h / FIXED_RES
    return np.array(
        [
            [FIXED_FX * sx, 0.0, FIXED_CX * sx],
            [0.0, FIXED_FY * sy, FIXED_CY * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


_SKY_SEGMENTER: dict | None = None  # cached {"processor", "model", "sky_ids"}


def segment_sky_mask(
    rgb_uint8: np.ndarray,
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    device: str | torch.device | None = None,
    min_ratio: float = 0.0,
) -> np.ndarray:
    """Predict an ``H×W`` boolean sky mask using a lightweight ADE20K SegFormer.

    The model and processor are cached after first use.  The default
    ``segformer-b0-finetuned-ade-512-512`` is ~14M params and runs in well
    under a second per image on a modern GPU; you can swap in the larger
    ``...-b5-finetuned-ade-640-640`` for higher quality on tricky outdoor
    scenes.

    Args:
        rgb_uint8: ``H×W×3`` uint8 RGB image (NOT RGBA).
        model_name: Hugging Face checkpoint to use.
        device: torch device for the segmenter.  Defaults to ``cuda`` if
            available, else ``cpu``.
        min_ratio: if the predicted sky covers less than this fraction of the
            image, return an all-zero mask (lets you no-op on scenes that
            don't actually contain sky).

    Returns:
        ``H×W`` bool array — True where the pixel is classified as sky.
    """
    try:
        from transformers import (  # type: ignore[import]
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Sky segmentation requires `transformers`. "
            "Install with `pip install 'transformers>=4.40'`."
        ) from exc

    if rgb_uint8.dtype != np.uint8 or rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(
            f"segment_sky_mask expects uint8 H×W×3 RGB; got "
            f"{rgb_uint8.dtype}, shape={rgb_uint8.shape}"
        )

    global _SKY_SEGMENTER
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if _SKY_SEGMENTER is None or _SKY_SEGMENTER.get("name") != model_name:
        print(f"[wt] loading sky segmenter ({model_name}) on {device} ...")
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        model = model.to(device).eval()
        sky_ids = [
            int(k) for k, v in model.config.id2label.items() if v.lower() == "sky"
        ]
        if not sky_ids:
            raise RuntimeError(
                f"Model {model_name} has no 'sky' class; got labels "
                f"{list(model.config.id2label.values())[:10]}..."
            )
        _SKY_SEGMENTER = {
            "name": model_name,
            "processor": processor,
            "model": model,
            "sky_ids": sky_ids,
            "device": device,
        }

    state = _SKY_SEGMENTER
    pil = Image.fromarray(rgb_uint8, mode="RGB")
    inputs = state["processor"](images=pil, return_tensors="pt").to(state["device"])
    with torch.no_grad():
        logits = state["model"](**inputs).logits
    seg = torch.nn.functional.interpolate(
        logits,
        size=(rgb_uint8.shape[0], rgb_uint8.shape[1]),
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)[0].cpu().numpy()
    sky_mask = np.isin(seg, state["sky_ids"])
    if sky_mask.mean() < min_ratio:
        return np.zeros_like(sky_mask, dtype=bool)
    return sky_mask.astype(bool)


def apply_sky_mask(
    rgba_uint8: np.ndarray,
    sky_mask: np.ndarray,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Zero-out the alpha and overwrite RGB for pixels marked as sky.

    Args:
        rgba_uint8: ``H×W×4`` uint8 RGBA image.
        sky_mask: ``H×W`` bool array of sky pixels (e.g. from
            :func:`segment_sky_mask`).
        bg_color: RGB triple written into sky pixels.  Defaults to black to
            match the training distribution (most scene renders sit on a
            black background outside the rendered viewport).

    Returns:
        A new RGBA array with sky pixels set to ``(bg_color, alpha=0)``.
    """
    if rgba_uint8.shape[:2] != sky_mask.shape:
        raise ValueError(
            f"sky_mask shape {sky_mask.shape} does not match RGBA "
            f"{rgba_uint8.shape[:2]}"
        )
    out = rgba_uint8.copy()
    out[sky_mask, :3] = np.asarray(bg_color, dtype=out.dtype)
    out[sky_mask, 3] = 0
    return out


def _auto_alpha_from_near_white(
    rgb: np.ndarray, threshold: int = 245, border_check: bool = True
) -> np.ndarray | None:
    """Heuristic: if the image was matted onto a near-uniform light background
    (very common for stock-photo / Stable-Diffusion / SAM matting outputs),
    detect that background and return a binary alpha that excludes it.

    Returns ``None`` if the heuristic cannot find a clean background.
    """
    if rgb.dtype != np.uint8:
        return None
    near_white = (rgb >= threshold).all(axis=-1)
    if border_check:
        h, w = rgb.shape[:2]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        if near_white[border].mean() < 0.95:
            return None
    if not (0.10 < near_white.mean() < 0.95):
        return None
    alpha = (~near_white).astype(np.uint8) * 255
    return alpha


def load_rgba_image(
    path: str | os.PathLike, auto_alpha: bool = True
) -> np.ndarray:
    """Load an image as ``uint8 H×W×4`` RGBA.

    For images that already carry an alpha channel, the alpha is used
    verbatim.

    For RGB images (no alpha channel), the behaviour depends on
    ``auto_alpha``:

    * ``auto_alpha=True`` (default): try a near-white-background heuristic.
      If the image looks like a matted object on a near-uniform light
      background (e.g. ``case_new.png``, SAM/SDXL outputs), the heuristic
      builds a binary alpha that excludes the background.  This avoids
      feeding background pixels into the model as if they were valid
      foreground geometry.
    * If the heuristic fails (e.g. genuine scene RGB), or ``auto_alpha`` is
      ``False``, a fully-opaque alpha is synthesised and a warning is
      printed.  In that mode the entire image is treated as foreground and
      the resulting "ghost" geometry over the background is the expected
      behaviour.
    """
    pil = Image.open(str(path))
    if pil.mode == "RGBA":
        return np.array(pil)
    rgb = np.array(pil.convert("RGB"))

    if auto_alpha:
        auto = _auto_alpha_from_near_white(rgb)
        if auto is not None:
            print(
                "[wt] auto-detected near-white background; using heuristic alpha. "
                "Pass --no-auto-alpha (or provide a proper RGBA) to disable."
            )
            return np.dstack([rgb, auto])

    print(
        "[wt] WARNING: input image has no alpha channel and no auto-mask "
        "could be derived.  Treating the entire image as foreground; the "
        "model may produce 'ghost' geometry over the background.  Pass a "
        "proper RGBA image with the object alpha-matted out."
    )
    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    return np.dstack([rgb, alpha])


def list_image_dir(image_dir: str | os.PathLike) -> list[tuple[str, np.ndarray]]:
    """Enumerate all images in ``image_dir`` and load them as RGBA arrays.

    Returns a list of ``(filename, rgba_uint8_hwc)`` tuples sorted by name.
    Skips files with extensions outside ``.png/.jpg/.jpeg/.webp``.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    entries: list[tuple[str, np.ndarray]] = []
    p = Path(image_dir)
    for child in sorted(p.iterdir()):
        if child.suffix.lower() in exts:
            entries.append((child.name, load_rgba_image(child)))
    return entries


def compute_object_crop(
    rgba: np.ndarray, max_object_ratio: float = 2.0 / 3.0
) -> tuple[int, int, int]:
    """Centre-crop on the alpha foreground.

    The object is centred in the crop, and its longest dimension occupies at
    most ``max_object_ratio`` of the crop side.  If the required crop exceeds
    the image, padding will be applied later.

    Default ``max_object_ratio=2/3`` (≈0.667) gives the object a roughly
    1/6-side empty margin on each side, matching the framing used during
    training (Objaverse renders) and during the video-selection inference
    run.  Use higher values (e.g. 0.8) for tighter framing.

    Returns ``(y1, x1, side)`` where ``y1``/``x1`` may be negative.
    """
    img_h, img_w = rgba.shape[:2]
    fg = rgba[:, :, 3] > 127
    if not fg.any():
        side = min(img_h, img_w)
        return (img_h - side) // 2, (img_w - side) // 2, side
    ys, xs = np.where(fg)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    obj_longest = max(bbox_h, bbox_w)
    side = int(np.ceil(obj_longest / max_object_ratio))
    cy = (y_min + y_max) / 2.0
    cx = (x_min + x_max) / 2.0
    y1 = int(round(cy - side / 2))
    x1 = int(round(cx - side / 2))
    return y1, x1, side


def crop_with_padding(image: np.ndarray, y1: int, x1: int, side: int) -> np.ndarray:
    """Crop ``image`` to ``[y1:y1+side, x1:x1+side]`` with zero-padding."""
    h, w = image.shape[:2]
    src_y1, src_x1 = max(0, y1), max(0, x1)
    src_y2, src_x2 = min(h, y1 + side), min(w, x1 + side)
    dst_y1, dst_x1 = src_y1 - y1, src_x1 - x1
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    out_shape = (side, side) + image.shape[2:]
    out = np.zeros(out_shape, dtype=image.dtype)
    out[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    return out


def preprocess_rgba_for_model(
    rgba_uint8: np.ndarray,
    image_size: int,
    num_layers: int,
    intrinsics_override: np.ndarray | None = None,
    alpha_erode_px: int = 0,
    center_crop: bool = True,
    max_object_ratio: float = 2.0 / 3.0,
    bg_color: tuple[int, int, int] | None = (0, 0, 0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a single RGBA image for the multilayer-depth model.

    Args:
        rgba_uint8: ``H×W×4`` uint8 RGBA image.  Alpha channel doubles as the
            object mask (foreground ⇔ alpha > 127).
        image_size: target square resolution (the three release configs use
            ``504`` for r75b/r69e and ``336`` for r76).
        num_layers: model ``num_layers`` (e.g. 6).
        intrinsics_override: optional ``[3, 3]`` K at the *original*
            resolution.  Defaults to :func:`make_default_intrinsics`.
        alpha_erode_px: if > 0, erode the foreground mask by this many
            pixels.  Helps suppress deep-layer "plume" artifacts caused by
            over-segmented matting / SAM masks.
        center_crop: if True, first centre-crop on the alpha foreground so
            the object roughly fills ``max_object_ratio`` of the canvas.
            Useful for unframed real-world images; pass ``False`` when the
            image is already framed.
        bg_color: RGB triple in 0-255 uint8.  When set, the RGB is
            **alpha-blended** against this colour before being fed to the
            model: ``rgb_blended = rgb * alpha + bg * (1 - alpha)``.  This
            preserves soft cutout edges (hair, feathers, thin straps)
            instead of binary-painting them, and matches the training-set
            RGBA rendering that uses the same alpha-compositing.  Common
            choices: ``(128, 128, 128)`` mid-gray (matches what we used
            during the video-selection inference run for ``r75b``), or
            ``(0, 0, 0)`` black.  Pass ``None`` to skip the blend entirely
            and feed the raw RGB to the encoder (use this when the input
            image's RGB is already pre-composited against the desired
            background, e.g. the generated_object PNGs we ship).

    Returns:
        rgb_tensor:        ``[1, 3, image_size, image_size]`` float32 in [0, 1]
        mask_tensor:       ``[1, L, image_size, image_size]`` bool
        intrinsics_tensor: ``[1, 3, 3]`` float32 in model-pixel units
    """
    if center_crop:
        y1, x1, side = compute_object_crop(rgba_uint8, max_object_ratio)
        rgba_crop = crop_with_padding(rgba_uint8, y1, x1, side)
    else:
        rgba_crop = rgba_uint8
        side = max(rgba_crop.shape[:2])

    rgb = rgba_crop[:, :, :3]
    alpha = rgba_crop[:, :, 3]

    # If bg_color is provided, alpha-blend the RGB against it *before* resize.
    # This preserves the soft-alpha rendering that the model was trained on
    # (objaverse RGBA renders are alpha-premultiplied + blended with the
    # training-time bg colour), and crucially avoids the "hard binary paint"
    # that destroys soft cutout edges (hair, feathers, leaves, thin straps).
    if bg_color is not None:
        bg_arr = np.asarray(bg_color, dtype=np.float32).reshape(1, 1, 3)
        alpha_f = alpha.astype(np.float32) / 255.0
        rgb_f = rgb.astype(np.float32) * alpha_f[..., None] + bg_arr * (
            1.0 - alpha_f[..., None]
        )
        rgb = np.clip(rgb_f, 0.0, 255.0).astype(np.uint8)

    rgb_resized = cv2.resize(
        rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    )
    alpha_resized = cv2.resize(
        alpha, (image_size, image_size), interpolation=cv2.INTER_NEAREST
    )

    fg_mask = alpha_resized > 127
    if alpha_erode_px > 0:
        k = 2 * alpha_erode_px + 1
        kernel = np.ones((k, k), np.uint8)
        fg_mask = cv2.erode(fg_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    rgb_01 = rgb_resized.astype(np.float32) / 255.0
    rgb_tensor = torch.from_numpy(rgb_01).permute(2, 0, 1).unsqueeze(0)
    mask_l0 = torch.from_numpy(fg_mask)[None, None]
    mask_tensor = mask_l0.expand(1, num_layers, -1, -1).contiguous()

    orig_h, orig_w = rgba_crop.shape[:2]
    intr = (
        intrinsics_override.copy()
        if intrinsics_override is not None
        else make_default_intrinsics(orig_h, orig_w)
    )
    sx, sy = image_size / orig_w, image_size / orig_h
    intr[0, 0] *= sx
    intr[1, 1] *= sy
    intr[0, 2] *= sx
    intr[1, 2] *= sy
    intr_tensor = torch.from_numpy(intr).unsqueeze(0)

    return rgb_tensor, mask_tensor, intr_tensor


# ---------------------------------------------------------------------------
# Video clip helpers (used by examples/infer_video.py with the r76 config).
# ---------------------------------------------------------------------------


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _list_frames(image_dir: str | os.PathLike) -> list[Path]:
    p = Path(image_dir)
    return sorted(c for c in p.iterdir() if c.suffix.lower() in _IMAGE_EXTS)


def load_video_clip(
    image_dir: str | os.PathLike,
    frame_indices: list[int] | None = None,
    auto_alpha: bool = True,
) -> tuple[list[str], list[np.ndarray]]:
    """Load T frames from ``image_dir`` as RGBA arrays.

    Args:
        image_dir: directory with one image per frame (sorted by filename).
        frame_indices: optional list of 0-based indices into the sorted file
            list.  If ``None``, every frame in the directory is loaded in
            order.  Indices outside the range are skipped with a warning.
        auto_alpha: forwarded to :func:`load_rgba_image` for RGB-only
            frames.

    Returns:
        ``(frame_names, rgba_list)`` where each entry is uint8 ``H×W×4``.
    """
    files = _list_frames(image_dir)
    if not files:
        raise FileNotFoundError(
            f"No images with extensions {sorted(_IMAGE_EXTS)} in {image_dir!r}"
        )

    if frame_indices is None:
        chosen = list(range(len(files)))
    else:
        chosen = []
        for i in frame_indices:
            if i < 0 or i >= len(files):
                print(
                    f"[wt] WARN: frame index {i} out of range (0..{len(files) - 1}), "
                    "skipping."
                )
                continue
            chosen.append(i)
        if not chosen:
            raise ValueError("All requested frame_indices are out of range.")

    names: list[str] = []
    rgba_list: list[np.ndarray] = []
    for i in chosen:
        f = files[i]
        rgba = load_rgba_image(f, auto_alpha=auto_alpha)
        names.append(f.name)
        rgba_list.append(rgba)
    return names, rgba_list


def compute_clip_shared_crop(
    rgba_list: list[np.ndarray], max_object_ratio: float = 2.0 / 3.0
) -> tuple[int, int, int]:
    """Compute one square crop that covers the alpha foreground across T frames.

    Identical to ``compute_object_crop`` but the bbox is taken over the
    union of per-frame foreground masks.  Sharing the crop across frames is
    critical for ``r76``-style temporal attention: pixel ``(u, v)`` must
    refer to comparable scene content in every frame, otherwise the
    temporal blocks see noise.

    Falls back to a centred square over the smallest frame when no frame
    has any foreground pixel.
    """
    img_h = min(rgba.shape[0] for rgba in rgba_list)
    img_w = min(rgba.shape[1] for rgba in rgba_list)
    y_min, y_max = np.inf, -np.inf
    x_min, x_max = np.inf, -np.inf
    any_fg = False
    for rgba in rgba_list:
        fg = rgba[:, :, 3] > 127
        if not fg.any():
            continue
        ys, xs = np.where(fg)
        y_min = min(y_min, float(ys.min()))
        y_max = max(y_max, float(ys.max()))
        x_min = min(x_min, float(xs.min()))
        x_max = max(x_max, float(xs.max()))
        any_fg = True
    if not any_fg:
        side = min(img_h, img_w)
        return (img_h - side) // 2, (img_w - side) // 2, side

    bbox_h = int(y_max - y_min + 1)
    bbox_w = int(x_max - x_min + 1)
    obj_longest = max(bbox_h, bbox_w)
    side = int(np.ceil(obj_longest / max_object_ratio))
    cy = (y_min + y_max) / 2.0
    cx = (x_min + x_max) / 2.0
    y1 = int(round(cy - side / 2))
    x1 = int(round(cx - side / 2))
    return y1, x1, side


def preprocess_clip_for_model(
    rgba_list: list[np.ndarray],
    image_size: int,
    num_layers: int,
    intrinsics_override: np.ndarray | None = None,
    alpha_erode_px: int = 0,
    center_crop: bool = True,
    max_object_ratio: float = 2.0 / 3.0,
    bg_color: tuple[int, int, int] | None = (0, 0, 0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray]]:
    """Prepare a T-frame RGBA clip with one shared crop.

    Args:
        rgba_list: T frames as ``uint8 H_t × W_t × 4`` arrays.  Frames may
            have different ``(H_t, W_t)`` but they will all be cropped to
            the same square side at the end.
        image_size: target square resolution (336 for r76).
        num_layers: model ``num_layers`` (6 for r76).
        intrinsics_override / alpha_erode_px / bg_color: same semantics as
            :func:`preprocess_rgba_for_model` (applied per frame).
        center_crop: if True (default), the shared crop is computed by
            :func:`compute_clip_shared_crop` from the union of per-frame
            foreground bboxes.  Pass ``False`` to take a centre crop of the
            smallest frame.

    Returns:
        rgb_clip:    ``[1, T, 3, image_size, image_size]`` float32 in [0, 1]
        mask_clip:   ``[1, T, num_layers, image_size, image_size]`` bool
        intr_tensor: ``[1, 3, 3]`` float32 (shared across frames)
        rgb_resized_list: ``T × uint8 [H, W, 3]`` for downstream logging
    """
    if center_crop:
        y1, x1, side = compute_clip_shared_crop(rgba_list, max_object_ratio)
        rgba_cropped = [crop_with_padding(r, y1, x1, side) for r in rgba_list]
    else:
        img_h = min(r.shape[0] for r in rgba_list)
        img_w = min(r.shape[1] for r in rgba_list)
        side = min(img_h, img_w)
        y1 = (img_h - side) // 2
        x1 = (img_w - side) // 2
        rgba_cropped = [crop_with_padding(r, y1, x1, side) for r in rgba_list]

    rgb_frames: list[np.ndarray] = []
    mask_frames: list[np.ndarray] = []
    for rgba in rgba_cropped:
        rgb_r = cv2.resize(
            rgba[:, :, :3], (image_size, image_size), interpolation=cv2.INTER_LINEAR
        )
        alpha_r = cv2.resize(
            rgba[:, :, 3], (image_size, image_size), interpolation=cv2.INTER_NEAREST
        )
        fg = alpha_r > 127
        if alpha_erode_px > 0:
            k = 2 * alpha_erode_px + 1
            kernel = np.ones((k, k), np.uint8)
            fg = cv2.erode(fg.astype(np.uint8), kernel, iterations=1).astype(bool)
        if bg_color is not None:
            rgb_r = rgb_r.copy()
            rgb_r[~fg] = np.asarray(bg_color, dtype=rgb_r.dtype)
        rgb_frames.append(rgb_r)
        mask_frames.append(fg)

    rgb_01 = np.stack(rgb_frames, axis=0).astype(np.float32) / 255.0  # [T,H,W,3]
    rgb_clip = (
        torch.from_numpy(rgb_01)
        .permute(0, 3, 1, 2)  # [T, 3, H, W]
        .unsqueeze(0)
        .contiguous()
    )
    mask_stack = np.stack(mask_frames, axis=0)  # [T, H, W]
    mask_clip = (
        torch.from_numpy(mask_stack)[:, None]  # [T, 1, H, W]
        .expand(-1, num_layers, -1, -1)
        .unsqueeze(0)
        .contiguous()
    )

    orig_h, orig_w = rgba_cropped[0].shape[:2]
    intr = (
        intrinsics_override.copy()
        if intrinsics_override is not None
        else make_default_intrinsics(orig_h, orig_w)
    )
    sx, sy = image_size / orig_w, image_size / orig_h
    intr[0, 0] *= sx
    intr[1, 1] *= sy
    intr[0, 2] *= sx
    intr[1, 2] *= sy
    intr_tensor = torch.from_numpy(intr).unsqueeze(0)

    return rgb_clip, mask_clip, intr_tensor, rgb_frames
