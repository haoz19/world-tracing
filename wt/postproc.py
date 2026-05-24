"""Post-processing for predicted multilayer geometry.

Per-layer presets
-----------------
Each numeric param (``z_thresh_frac``, ``z_thresh_abs``, ``sor_k``,
``sor_n_std``) accepts either a scalar or a per-layer mapping
(``dict`` keyed by layer index, ``list`` indexed by layer, or a plain
scalar applied to every layer).  This lets callers be **gentle on
layer 0** (the visible RGB surface where over-filtering carves a
visible seam between L0 and the hidden gray layers) and **strict on
the hidden layers** (where stray gray ghost dots are still the most
visually obvious flyers in 360° orbits).

The shipped preset for dynamic clips is :data:`DYNAMIC_PRESET_SOFT_L0`.


Provides :func:`filter_edge_flyers` and the video variant
:func:`filter_edge_flyers_clip`, used to suppress the "flying point"
artifacts that the dynamic (``r76``) model occasionally emits.

After staring at top-down (X-Z) renders of the layer-0 cloud for the
``davis__tennis / breakdance / dance-twirl / hike`` clips, the flyers
turn out to NOT be thin halos hugging the silhouette boundary, as the
initial implementation assumed. They're **arc-shaped sprays of points
fanning out in front of and behind the main body** — pixels whose
predicted depth is several local-medians off from their image-space
neighbours. Many of those pixels live well inside the eroded interior
of the foreground mask, so an edge-band-only filter does almost
nothing.

The current implementation therefore combines:

1. **Image-space depth-discontinuity test** (catches depth-spike arcs).
   For each layer-0 pixel inside the foreground mask, compare its
   predicted z to the median z of a small image-space window (default
   5×5). Drop the pixel if ``|z - z_median| > z_thresh_frac * median_z``.
   Catches *both* edge flyers (large jumps at the silhouette) and
   interior depth-spike flyers (the X-Z arcs).
2. **3D statistical outlier removal (SOR)** (catches the 3D-isolated
   "ghost dots" that survive the depth test because each ghost has
   same-z neighbours next to it). Computes the mean distance from each
   point to its ``sor_k`` nearest neighbours; drops points whose mean
   k-NN distance is more than ``sor_n_std × MAD`` above the cloud
   median. MAD (median-absolute-deviation, scaled to std-equivalent
   1.4826) is used instead of std for robustness — a small handful of
   far-flung flyers can wildly inflate std but barely touches MAD.
   Connected-component / cluster filters fail here because the
   breakdance subject is a thin, sparse stick-figure where many
   legitimate limb tips have few neighbours; SOR uses *distance*
   rather than connectivity and so distinguishes "thin geometry" from
   "isolated flyer" reliably.
3. **Optional boundary-band 3D outlier removal** (off by default), kept
   for the cases where neither test catches a particular flyer class.
   Preserves the previous ``erode_px / radius_frac / min_neighbours`` API.

The depth-discontinuity radius is **per-pixel adaptive**: pixels with a
larger local-median depth get a proportionally larger threshold, so a
foreground pixel at z=2.0 m and a background pixel at z=8.0 m are
evaluated against the same fractional cutoff.

Layers to filter: by default ``layers_to_filter=None`` (= ALL layers).
Earlier versions defaulted to ``(0,)`` (visible-surface only), but
the hidden layers ``L ≥ 1`` turned out to ALSO contain noticeable
3D-isolated ghost dots: in the Rerun viewer the hidden layers render
as small **gray** disks, and a few stray gray disks floating beside
the colored layer-0 subject are the most visually obvious flyers in
rotated views. Filtering only layer 0 left those completely untouched
(filter dropped only ~1.5 % of total points instead of ~10 %), which
is why the user saw "no improvement" between raw and filtered. Pass
``layers_to_filter=(0,)`` explicitly if you need the old behaviour.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.spatial import cKDTree as _KDTree

    _HAS_KDTREE = True
except ImportError:  # pragma: no cover
    _KDTree = None  # type: ignore[assignment]
    _HAS_KDTREE = False


# ---------------------------------------------------------------------------
# Image-space depth-discontinuity test (the heavy lifter)
# ---------------------------------------------------------------------------


def _local_median_z(z_map: np.ndarray, mask: np.ndarray, ksize: int) -> np.ndarray:
    """Compute the local-median depth at every valid pixel.

    Pixels outside ``mask`` are treated as "no contribution" (replaced
    with the mean of valid neighbours via a `cv2.inpaint`-style trick:
    we fill them with the mean depth first so they don't pull the
    median, then take a median blur).  We use ``cv2.medianBlur`` for
    speed; it requires uint8/uint16/float32 with ksize ∈ {3, 5} for
    float, so we re-scale to uint16 for ksize > 5.
    """
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(f"ksize must be odd and >=3, got {ksize}")

    if not mask.any():
        return np.zeros_like(z_map)

    z_valid = z_map[mask]
    z_mean = float(np.mean(z_valid))
    z_lo, z_hi = float(np.min(z_valid)), float(np.max(z_valid))
    span = max(z_hi - z_lo, 1e-6)

    # Fill invalid pixels with the mean depth so the median blur isn't
    # biased toward 0.  This is a conservative choice — invalid pixels
    # always sit near the centre of the valid distribution.
    z_filled = np.where(mask, z_map, z_mean).astype(np.float32)

    if ksize <= 5:
        # cv2.medianBlur supports float32 directly only for ksize <= 5.
        z_med = cv2.medianBlur(z_filled, ksize)
    else:
        # ksize > 5 must use uint8 in OpenCV's medianBlur.  We quantise
        # to 256 levels over the local span — plenty for a flyer test
        # since we compare against the SAME quantised median.
        z_norm = (z_filled - z_lo) / span
        z_u8 = np.clip(z_norm * 255.0, 0, 255).astype(np.uint8)
        z_med_u8 = cv2.medianBlur(z_u8, ksize)
        z_med = z_med_u8.astype(np.float32) / 255.0 * span + z_lo

    return z_med


def _depth_discontinuity_mask(
    z_map: np.ndarray,
    fg_mask: np.ndarray,
    *,
    ksize: int = 5,
    z_thresh_frac: float = 0.02,
    z_thresh_abs: float = 0.0,
) -> np.ndarray:
    """Return a bool image marking pixels to DROP (True = flyer).

    A pixel is marked as a flyer if
        ``|z(p) - localmedian_z(p)| > max(z_thresh_frac * localmedian_z(p), z_thresh_abs)``.
    Only ``fg_mask`` pixels are evaluated; invalid pixels are False.
    """
    if not fg_mask.any():
        return np.zeros_like(fg_mask, dtype=bool)
    z_med = _local_median_z(z_map, fg_mask, ksize)
    devi = np.abs(z_map - z_med)
    thresh = np.maximum(z_thresh_frac * np.maximum(z_med, 1e-3), z_thresh_abs)
    return fg_mask & (devi > thresh)


# ---------------------------------------------------------------------------
# 3D statistical outlier removal (SOR)
# ---------------------------------------------------------------------------


def _sor_outlier_mask(
    xyz: np.ndarray,
    fg_mask: np.ndarray,
    *,
    sor_k: int = 8,
    sor_n_std: float = 2.5,
) -> np.ndarray:
    """Return a bool image marking 3D statistical outlier pixels.

    True means "drop this pixel".

    Algorithm (Open3D-style ``remove_statistical_outlier`` but with a
    robust MAD threshold instead of std):
        1. Build a KDTree over the foreground 3D points.
        2. For each point, compute the mean distance to its
           ``sor_k`` nearest neighbours (excluding self).
        3. Compute the median of those mean distances and the
           MAD (median absolute deviation), scaled to std-equivalent
           by ``1.4826``.
        4. Drop points whose mean k-NN distance is more than
           ``median + sor_n_std × MAD``.

    MAD is used instead of std because a handful of badly placed
    flyers can wildly inflate std but barely touches MAD, making
    the threshold much more stable across frames.
    """
    if not _HAS_KDTREE:
        return np.zeros_like(fg_mask, dtype=bool)
    if not fg_mask.any():
        return np.zeros_like(fg_mask, dtype=bool)

    ys, xs = np.where(fg_mask)
    pts = xyz[ys, xs]
    finite = np.all(np.isfinite(pts), axis=1)
    ys, xs, pts = ys[finite], xs[finite], pts[finite]
    if len(pts) < sor_k + 1:
        return np.zeros_like(fg_mask, dtype=bool)

    tree = _KDTree(pts)
    # query returns distances + indices; +1 because the closest match
    # is always the point itself (distance 0).
    dists, _ = tree.query(pts, k=sor_k + 1, workers=-1)
    mean_d = dists[:, 1:].mean(axis=1)

    med = float(np.median(mean_d))
    mad = float(np.median(np.abs(mean_d - med)) * 1.4826)
    if mad <= 0:
        return np.zeros_like(fg_mask, dtype=bool)
    bad = mean_d > med + sor_n_std * mad
    if not bad.any():
        return np.zeros_like(fg_mask, dtype=bool)

    out = np.zeros_like(fg_mask, dtype=bool)
    out[ys[bad], xs[bad]] = True
    return out


# ---------------------------------------------------------------------------
# Boundary band (used by the optional 3D outlier removal step)
# ---------------------------------------------------------------------------


def _boundary_band(mask: np.ndarray, erode_px: int) -> np.ndarray:
    """Return a bool image marking the thin boundary band of ``mask``."""
    if erode_px <= 0:
        return np.zeros_like(mask, dtype=bool)
    k = 2 * erode_px + 1
    kernel = np.ones((k, k), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask & (~eroded)


def _boundary_3d_outlier_mask(
    xyz: np.ndarray,
    fg_mask: np.ndarray,
    *,
    erode_px: int = 3,
    radius_frac: float = 0.012,
    min_neighbours: int = 6,
) -> np.ndarray:
    """Return a bool image marking 3D-outlier pixels in the boundary band."""
    if not _HAS_KDTREE:
        return np.zeros_like(fg_mask, dtype=bool)
    band = _boundary_band(fg_mask, erode_px=erode_px)
    if not band.any():
        return np.zeros_like(fg_mask, dtype=bool)
    pts_all = xyz[fg_mask]
    finite = np.all(np.isfinite(pts_all), axis=1)
    if finite.sum() < 8:
        return np.zeros_like(fg_mask, dtype=bool)
    pts_all = pts_all[finite]
    lo, hi = pts_all.min(axis=0), pts_all.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    if diag <= 0:
        return np.zeros_like(fg_mask, dtype=bool)
    radius = radius_frac * diag

    tree = _KDTree(pts_all)
    cand_y, cand_x = np.where(band)
    cand_pts = xyz[cand_y, cand_x]
    cand_finite = np.all(np.isfinite(cand_pts), axis=1)
    cand_y = cand_y[cand_finite]
    cand_x = cand_x[cand_finite]
    cand_pts = cand_pts[cand_finite]
    if len(cand_pts) == 0:
        return np.zeros_like(fg_mask, dtype=bool)
    counts = tree.query_ball_point(
        cand_pts, r=radius, return_length=True, workers=-1
    )
    bad = np.asarray(counts) < min_neighbours
    out = np.zeros_like(fg_mask, dtype=bool)
    out[cand_y[bad], cand_x[bad]] = True
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Per-layer preset used by dynamic clip inference.  Layer 0 (the
# visible RGB surface) is filtered far more gently than the hidden
# layers L≥1, because over-filtering L0 carves a visible seam where
# the silhouette boundary points get dropped and the gray hidden
# layer behind them shows through.  The user noticed this as "接缝
# 非常大" (huge seam) after the first dynamic refine pass.
#
#  * L0: z_thresh_frac=0.04 (was 0.02), sor_n_std=3.5 (was 2.5)
#        → drops only the most extreme depth spikes and the rarest
#          3D-isolated points, keeps the silhouette band intact.
#  * L≥1: z_thresh_frac=0.025 (slightly looser than 0.02),
#         sor_n_std=2.7 (slightly looser than 2.5)
#        → still catches gray ghost dots in 360° orbits without
#          eating into the hidden geometry.
DYNAMIC_PRESET_SOFT_L0 = dict(
    depth_ksize=5,
    z_thresh_frac={0: 0.04, "default": 0.025},
    z_thresh_abs=0.0,
    sor=True,
    sor_k=8,
    sor_n_std={0: 3.5, "default": 2.7},
    layers_to_filter=None,
)


def _resolve_per_layer(
    val,
    layer: int,
    n_layers: int,
    default,
):
    """Look up a scalar param that may be per-layer.

    Accepts a plain scalar (used for every layer) OR a dict mapping
    ``layer_idx -> value`` (with optional key ``"default"`` for fallback)
    OR a list/tuple indexed by layer (uses last value if longer).
    """
    if val is None:
        return default
    if isinstance(val, dict):
        if layer in val:
            return val[layer]
        return val.get("default", default)
    if isinstance(val, (list, tuple)):
        if layer < len(val):
            return val[layer]
        return val[-1] if val else default
    return val


def filter_edge_flyers(
    xyz: np.ndarray,
    mask: np.ndarray,
    *,
    # Depth-discontinuity test (catches X-Z spike arcs).
    # Each parameter accepts a scalar, a per-layer dict ({0: x, 1: y, "default": z}),
    # or a list/tuple indexed by layer.
    depth_ksize: int = 5,
    z_thresh_frac=0.02,
    z_thresh_abs=0.0,
    # 3D statistical outlier removal (catches 3D-isolated ghost dots)
    sor: bool = True,
    sor_k=8,
    sor_n_std=2.5,
    # Optional 3D boundary outlier test (off by default; legacy fallback)
    boundary_3d: bool = False,
    erode_px: int = 3,
    radius_frac: float = 0.012,
    min_neighbours: int = 6,
    # Which layers to clean
    layers_to_filter: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop flying-point pixels from a single ``[L, H, W, 3]`` xyz.

    Three independent tests can fire:

    * **Depth-discontinuity** (always on) — drop pixels where the
      predicted z deviates by more than ``z_thresh_frac × local_median_z``
      from a small image-space median filter of depth. Catches edge
      halos and interior depth-spike flyers (X-Z arcs).
    * **Statistical outlier removal (SOR)** (on by default) — drop
      points whose mean distance to their ``sor_k`` nearest neighbours
      exceeds ``median + sor_n_std × MAD`` over the whole cloud. This
      is what catches the 3D-isolated "ghost dot" clusters that
      survive the depth test because each ghost has same-z neighbours
      around it; SOR uses *distance*, not connectivity, so it
      distinguishes legitimate thin geometry (limb tips with few but
      *close* neighbours) from isolated flyers (few neighbours that
      are *far*).
    * **Boundary 3D outlier** (off by default) — for pixels in the
      thin band ``erode_px`` wide just inside the silhouette boundary,
      drop those whose 3D neighbour count within ``radius_frac × diag``
      is below ``min_neighbours``.

    Args:
        xyz: ``[L, H, W, 3]`` camera-space XYZ (float).
        mask: ``[L, H, W]`` bool validity mask.
        depth_ksize: image-space median-filter window for the depth
            discontinuity test. Must be odd and >=3.
        z_thresh_frac: drop a pixel if ``|z - localmedian| > frac × median``.
            0.02 (2 %) is a sensible default; 0.015 more aggressive,
            0.03 more conservative.
        z_thresh_abs: optional floor (meters) for the depth threshold.
        sor: enable statistical outlier removal.
        sor_k: number of nearest neighbours per point. 8 is a good
            default; 6 makes it more sensitive to local density, 12
            smooths out small holes in the neighbourhood graph.
        sor_n_std: drop points whose mean k-NN distance exceeds
            ``median + sor_n_std × MAD``. 2.5 is conservative (drops
            ~5-15 %), 2.0 catches more flyers at risk of trimming
            limb tips, 3.0 only catches the most extreme outliers.
        boundary_3d: enable the optional 3D boundary outlier pass.
        erode_px / radius_frac / min_neighbours: parameters for the 3D
            boundary outlier pass (legacy).
        layers_to_filter: tuple of layer indices to clean. ``None``
            (default) means **all** layers. Pass ``(0,)`` to restore
            the layer-0-only behaviour (kept for back-compat — but
            note hidden layers carry the majority of visible flyers
            in the Rerun viewer).

    Returns:
        New ``(xyz, mask)`` arrays with the same shape; dropped pixels
        have ``mask = False`` and ``xyz = 0``.
    """
    xyz = xyz.copy()
    mask = mask.copy()
    L = xyz.shape[0]
    if layers_to_filter is None:
        layers_to_filter = tuple(range(L))

    for layer in layers_to_filter:
        if layer >= L:
            continue
        m = mask[layer]
        if not m.any():
            continue
        z_map = xyz[layer, ..., 2]

        ztf = _resolve_per_layer(z_thresh_frac, layer, L, 0.02)
        zta = _resolve_per_layer(z_thresh_abs, layer, L, 0.0)
        sk = _resolve_per_layer(sor_k, layer, L, 8)
        sn = _resolve_per_layer(sor_n_std, layer, L, 2.5)

        flyer = _depth_discontinuity_mask(
            z_map, m, ksize=depth_ksize,
            z_thresh_frac=ztf, z_thresh_abs=zta,
        )

        if sor:
            m_after_depth = m & (~flyer)
            extra = _sor_outlier_mask(
                xyz[layer], m_after_depth,
                sor_k=sk, sor_n_std=sn,
            )
            flyer = flyer | extra

        if boundary_3d:
            extra = _boundary_3d_outlier_mask(
                xyz[layer], m, erode_px=erode_px,
                radius_frac=radius_frac, min_neighbours=min_neighbours,
            )
            flyer = flyer | extra

        if flyer.any():
            mask[layer][flyer] = False
            xyz[layer][flyer] = 0.0

    return xyz, mask


def filter_edge_flyers_clip(
    xyz_clip: np.ndarray,
    mask_clip: np.ndarray,
    *,
    depth_ksize: int = 5,
    z_thresh_frac=0.02,
    z_thresh_abs=0.0,
    sor: bool = True,
    sor_k=8,
    sor_n_std=2.5,
    boundary_3d: bool = False,
    erode_px: int = 3,
    radius_frac: float = 0.012,
    min_neighbours: int = 6,
    layers_to_filter: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """T-frame wrapper around :func:`filter_edge_flyers`.

    Applies the same filter independently per frame; temporal
    consistency is not enforced because the dynamic model already
    aligns frames jointly during denoising.
    """
    out_xyz = xyz_clip.copy()
    out_mask = mask_clip.copy()
    T = xyz_clip.shape[0]
    for t in range(T):
        out_xyz[t], out_mask[t] = filter_edge_flyers(
            out_xyz[t],
            out_mask[t],
            depth_ksize=depth_ksize,
            z_thresh_frac=z_thresh_frac,
            z_thresh_abs=z_thresh_abs,
            sor=sor,
            sor_k=sor_k,
            sor_n_std=sor_n_std,
            boundary_3d=boundary_3d,
            erode_px=erode_px,
            radius_frac=radius_frac,
            min_neighbours=min_neighbours,
            layers_to_filter=layers_to_filter,
        )
    return out_xyz, out_mask
