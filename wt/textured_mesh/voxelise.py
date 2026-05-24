"""Cloud → voxel-coords voxelisation for TRELLIS.2 stage-2 injection.

The ``v4_ray_fill`` strategy is the production default (``ours_v4``):

1. Densify between adjacent depth layers (linear interp in 3D — preserves
   the model's x/y ratio exactly, unlike pinhole back-projection).
2. Canonicalise into ``[-0.5, 0.5]^3``.
3. Quantise onto a ``res^3`` voxel grid.
4. Morphologically close (heal 1-cell gaps) + flood-fill the interior
   so the voxel structure is solid, matching TRELLIS.2's training
   distribution.

All operations are pure ``numpy`` + ``scipy.ndimage`` — no CUDA kernel
dependencies until you feed the coords back into TRELLIS.2.
"""

from __future__ import annotations

import numpy as np


def _xyz_to_grid(cloud: np.ndarray, res: int) -> np.ndarray:
    """``(N, 3)`` float in ``[-0.5, 0.5]^3`` → bool grid ``(res, res, res)``."""
    grid = np.zeros((res, res, res), dtype=bool)
    if cloud.size == 0:
        return grid
    idx = np.floor((cloud + 0.5) * res).astype(np.int64)
    idx = np.clip(idx, 0, res - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def _grid_to_coords(grid: np.ndarray) -> np.ndarray:
    """bool grid → ``(M, 3) int32`` coords of the True cells."""
    return np.argwhere(grid).astype(np.int32)


def expand_cloud_ray_xyz(
    xyz: np.ndarray, mask: np.ndarray, steps: int = 4
) -> np.ndarray:
    """Densify a multilayer XYZ tensor by interpolating between adjacent
    layers directly in 3D.

    Unlike pinhole back-projection (``X = (u - cx) / fx * Z``), this
    routine preserves the model's predicted ``(x, y, z)`` ratio exactly,
    which matters for ``output_mode="xyz"`` models where the predicted
    X / Y do not satisfy the input camera's pinhole equation.

    Args:
        xyz: ``(L, H, W, 3) float32`` per-layer camera-frame XYZ.
        mask: ``(L, H, W) bool`` validity mask.
        steps: interior samples per adjacent-layer pair, excluding the
            two endpoints.  Defaults to 4.

    Returns:
        ``(N, 3) float32`` cam-frame cloud (surface + 3D-interpolated
        interior).
    """
    L = xyz.shape[0]
    pts_list: list[np.ndarray] = []

    for li in range(L):
        m = mask[li]
        if m.any():
            pts_list.append(xyz[li][m])

    if steps > 0 and L > 1:
        ts = np.linspace(0.0, 1.0, steps + 2)[1:-1]
        for li in range(L - 1):
            m = mask[li] & mask[li + 1]
            if not m.any():
                continue
            p0 = xyz[li][m]
            p1 = xyz[li + 1][m]
            for t in ts:
                pts_list.append(p0 * (1.0 - float(t)) + p1 * float(t))

    if not pts_list:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(pts_list, axis=0).astype(np.float32)


def v4_ray_fill(
    xyz: np.ndarray,
    mask: np.ndarray,
    canon_apply,
    res: int = 64,
    ray_steps: int = 4,
    close_iters: int = 1,
    max_voxels: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """``ours_v4`` voxelisation: ray-densify → canonicalise → close → fill.

    Args:
        xyz: ``(L, H, W, 3) float32`` per-layer camera XYZ.
        mask: ``(L, H, W) bool`` validity mask.
        canon_apply: callable mapping camera XYZ to canonical XYZ (use
            :meth:`CanonicalTransform.apply`).
        res: voxel-grid resolution (must match the TRELLIS.2 stage you
            feed into; 32 for ``ss_res=32``, 64 for the 1024 pipeline).
        ray_steps: interior samples per layer-pair (see
            :func:`expand_cloud_ray_xyz`).
        close_iters: ``scipy.ndimage.binary_closing`` iterations.  Default
            1 (heals 1-cell pinholes).
        max_voxels: optional random subsample cap.  ``None`` (default)
            disables it.
        seed: RNG seed for the optional subsample.

    Returns:
        ``(coords, n_voxels)``: ``(M, 3) int32`` coords in ``[0, res-1]``
        and the raw voxel count BEFORE any cap (for diagnostics).
    """
    try:
        import scipy.ndimage as ndi
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "v4_ray_fill requires scipy.  Install with `pip install scipy`."
        ) from exc

    cloud_cam = expand_cloud_ray_xyz(xyz, mask, steps=int(ray_steps))
    if cloud_cam.size == 0:
        return np.empty((0, 3), dtype=np.int32), 0
    cloud_canon = canon_apply(cloud_cam)
    in_bb = np.all(np.abs(cloud_canon) <= 0.5 - 1e-6, axis=1)
    cloud_canon = cloud_canon[in_bb]
    if cloud_canon.size == 0:
        return np.empty((0, 3), dtype=np.int32), 0
    grid = _xyz_to_grid(cloud_canon, res)
    if close_iters > 0:
        grid = ndi.binary_closing(grid, iterations=int(close_iters))
    grid = ndi.binary_fill_holes(grid)
    coords = _grid_to_coords(grid)
    n_raw = int(coords.shape[0])
    if max_voxels is not None and n_raw > max_voxels:
        sel = np.random.RandomState(int(seed)).choice(
            n_raw, int(max_voxels), replace=False
        )
        coords = coords[sel]
    return coords, n_raw


__all__ = [
    "expand_cloud_ray_xyz",
    "v4_ray_fill",
]
