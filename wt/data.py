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


#: Default foreground-segmentation model.  ``BiRefNet_HR`` (MIT) outputs an
#: 8-bit grayscale alpha matte and is currently SOTA for high-resolution
#: dichotomous image segmentation.  Override with :func:`segment_foreground`'s
#: ``model_name`` argument if you have a different fine-tune locally.
DEFAULT_FG_SEGMENTER = "ZhengPeng7/BiRefNet_HR"

# Module-level cache: {"name": str, "model": nn.Module, "device": torch.device}.
# Re-used across calls so the BiRefNet weights only load once per process.
_FG_SEGMENTER: dict | None = None


def segment_foreground(
    rgb_uint8: np.ndarray,
    model_name: str = DEFAULT_FG_SEGMENTER,
    device: str | torch.device | None = None,
    threshold: int = 64,
    matte: bool = False,
) -> np.ndarray:
    """Predict a foreground alpha matte using BiRefNet (Hugging Face).

    Loads ``ZhengPeng7/BiRefNet_HR`` by default — MIT-licensed, BiRefNet
    architecture, currently SOTA on DIS / HRSOD / COD benchmarks.  Outputs a
    high-quality 8-bit alpha matte (hair, fur, semi-transparent edges all
    preserved).

    Args:
        rgb_uint8: ``H×W×3`` uint8 RGB image (NOT RGBA).
        model_name: Hugging Face repo id.  ``BiRefNet_HR`` (2048² input) is
            the default for highest quality on 1024² source images; pass
            ``ZhengPeng7/BiRefNet`` for a smaller / faster version.
        device: torch device for the segmenter.  Defaults to ``cuda`` if
            available, else ``cpu``.
        threshold: Cut-off on the predicted matte (0-255) used when
            ``matte=False``.  Pixels below this are treated as background.
            Defaults to 64 — generous on edges, harsh on dark interior
            shadows.
        matte: If True return the raw ``H×W`` uint8 alpha matte; if False
            (default) return a hard ``H×W`` bool mask after thresholding.

    Returns:
        ``H×W`` ``np.uint8`` (when ``matte=True``) or ``np.bool_``
        (default).  True / >0 = foreground.
    """
    if rgb_uint8.dtype != np.uint8 or rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(
            f"segment_foreground expects uint8 H×W×3 RGB; got "
            f"{rgb_uint8.dtype}, shape={rgb_uint8.shape}"
        )

    try:
        from transformers import AutoModelForImageSegmentation  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Foreground segmentation requires `transformers`. "
            "Install with `pip install 'transformers>=4.40'`."
        ) from exc
    try:
        from torchvision import transforms as _tv_transforms  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Foreground segmentation requires `torchvision` (for BiRefNet "
            "preprocessing).  Install with `pip install torchvision`."
        ) from exc

    global _FG_SEGMENTER
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if _FG_SEGMENTER is None or _FG_SEGMENTER.get("name") != model_name:
        print(f"[wt] loading foreground segmenter ({model_name}) on {device} ...")
        # trust_remote_code=True is required to pull BiRefNet's custom
        # model class from the HF repo.  All MIT-licensed by ZhengPeng7.
        model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = model.to(device).to(torch.float32).eval()
        # BiRefNet_HR is trained at 2048; BiRefNet (base) at 1024.
        input_size = 2048 if "BiRefNet_HR" in model_name else 1024
        _FG_SEGMENTER = {
            "name": model_name,
            "model": model,
            "device": device,
            "input_size": input_size,
        }

    state = _FG_SEGMENTER
    input_size = state["input_size"]
    pil = Image.fromarray(rgb_uint8, mode="RGB")
    transform = _tv_transforms.Compose([
        _tv_transforms.Resize((input_size, input_size)),
        _tv_transforms.ToTensor(),
        _tv_transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),
    ])
    x = transform(pil).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        preds = state["model"](x)[-1].sigmoid().cpu()
    # BiRefNet returns ``[B, 1, S, S]`` after sigmoid; resize back to source.
    matte_t = preds[0, 0]
    matte_pil = _tv_transforms.ToPILImage()(matte_t)
    matte_pil = matte_pil.resize((rgb_uint8.shape[1], rgb_uint8.shape[0]))
    alpha = np.array(matte_pil, dtype=np.uint8)
    if matte:
        return alpha
    return alpha > threshold


def apply_fg_mask(
    rgba_uint8: np.ndarray,
    fg: np.ndarray,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Replace the RGB+alpha of background pixels with ``(bg_color, 0)``.

    Args:
        rgba_uint8: ``H×W×4`` uint8 RGBA image.
        fg: ``H×W`` bool array (or ``H×W`` uint8 alpha matte) of foreground
            pixels.  An alpha-matte input is binarised at 0.
        bg_color: RGB triple written into background pixels.

    Returns:
        A new RGBA array with background pixels set to ``(bg_color, 0)``.
    """
    if rgba_uint8.shape[:2] != fg.shape[:2]:
        raise ValueError(
            f"foreground mask shape {fg.shape} does not match RGBA "
            f"{rgba_uint8.shape[:2]}"
        )
    fg_bool = fg.astype(bool) if fg.dtype != bool else fg
    out = rgba_uint8.copy()
    bg = ~fg_bool
    out[bg, :3] = np.asarray(bg_color, dtype=out.dtype)
    out[bg, 3] = 0
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
    path: str | os.PathLike,
    auto_alpha: bool = True,
    fg_segmenter: str = DEFAULT_FG_SEGMENTER,
) -> np.ndarray:
    """Load an image as ``uint8 H×W×4`` RGBA.

    For images that already carry an alpha channel, the alpha is used
    verbatim — no model is loaded.

    For RGB images (no alpha channel), the behaviour depends on
    ``auto_alpha``:

    * ``auto_alpha=True`` (default): try a fast near-white-background
      heuristic first.  If the image is a clean matte (white-on-object,
      e.g. e-commerce / SDXL / SAM outputs) this returns immediately with
      ``< 10ms``.  Otherwise we fall back to BiRefNet
      (``ZhengPeng7/BiRefNet_HR``, MIT) to predict a proper foreground
      matte.  BiRefNet weights are auto-downloaded from Hugging Face the
      first time you call this function.
    * ``auto_alpha=False``: synthesise a fully-opaque alpha and print a
      warning.  The model will treat the whole image as foreground; expect
      "ghost" geometry over the background.
    """
    pil = Image.open(str(path))
    if pil.mode == "RGBA":
        return np.array(pil)
    rgb = np.array(pil.convert("RGB"))

    if auto_alpha:
        # Fast path: near-white background heuristic.  Skips loading
        # BiRefNet (~880MB) on clean stock-photo / SDXL inputs.
        auto = _auto_alpha_from_near_white(rgb)
        if auto is not None:
            print(
                "[wt] auto-detected near-white background; using heuristic "
                "alpha (no segmentation model loaded)."
            )
            return np.dstack([rgb, auto])

        # Quality path: BiRefNet foreground matting.
        try:
            print(
                f"[wt] running foreground segmentation ({fg_segmenter}) ..."
            )
            fg = segment_foreground(rgb, model_name=fg_segmenter)
            alpha = (fg.astype(np.uint8)) * 255
            return np.dstack([rgb, alpha])
        except (ImportError, OSError, RuntimeError) as exc:
            print(
                f"[wt] WARNING: foreground segmentation failed ({exc}); "
                "falling back to fully-opaque alpha.  Pass a proper RGBA "
                "image or `pip install transformers torchvision` to enable "
                "BiRefNet-based auto-matting."
            )

    print(
        "[wt] WARNING: input image has no alpha channel.  Treating the "
        "entire image as foreground; the model may produce 'ghost' "
        "geometry over the background.  Pass a proper RGBA image to "
        "disable this warning."
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
    """Prepare a single RGBA image for the multilayer-geometry model.

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
