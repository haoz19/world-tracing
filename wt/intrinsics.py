"""Recover camera intrinsics from the model's predicted XYZ.

When running on real-world images we usually do not know the ground-truth
camera intrinsics.  Our diffusion model is trained to output XYZ in camera
space, so we can fit a pinhole projection on the layer-0 prediction:

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

This avoids the need for an external MoGe / VGGT pose estimator at
inference time.  The recovered ``K`` can then be used to:

* render the predicted point cloud through a known camera, or
* run the depth-mode unprojection path (:func:`wt.inference._depth_to_xyz`).
"""

from __future__ import annotations

import numpy as np


def solve_intrinsics_from_xyz(
    xyz_layer0: np.ndarray,
    mask_layer0: np.ndarray,
    image_size: int | None = None,
) -> tuple[np.ndarray | None, float]:
    """Least-squares fit of a pinhole ``K`` to the predicted layer-0 XYZ.

    Args:
        xyz_layer0: ``[H, W, 3]`` XYZ prediction for the front-most layer.
        mask_layer0: ``[H, W]`` bool foreground mask for that layer.
        image_size: optional square image side (assumes H==W).  Only used to
            return the horizontal field of view; ignored otherwise.

    Returns:
        ``(K, fov_x_deg)`` where ``K`` is ``[3, 3]`` float32 (None if fewer
        than 10 valid pixels) and ``fov_x_deg`` is the horizontal FoV in
        degrees.
    """
    if image_size is None:
        image_size = xyz_layer0.shape[1]

    valid = mask_layer0.copy().astype(bool)
    Z = xyz_layer0[..., 2]
    valid &= Z > 1e-6
    n_valid = int(valid.sum())
    if n_valid < 10:
        return None, 0.0

    vs, us = np.where(valid)
    u = us.astype(np.float64) + 0.5
    v = vs.astype(np.float64) + 0.5
    X = xyz_layer0[valid, 0].astype(np.float64)
    Y = xyz_layer0[valid, 1].astype(np.float64)
    Zv = xyz_layer0[valid, 2].astype(np.float64)

    A_u = np.stack([X / Zv, np.ones(n_valid)], axis=1)
    fx_cx, _, _, _ = np.linalg.lstsq(A_u, u, rcond=None)
    fx, cx = fx_cx

    A_v = np.stack([Y / Zv, np.ones(n_valid)], axis=1)
    fy_cy, _, _, _ = np.linalg.lstsq(A_v, v, rcond=None)
    fy, cy = fy_cy

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    fov_x_deg = (
        float(2.0 * np.rad2deg(np.arctan(image_size / (2.0 * fx)))) if fx > 0 else 0.0
    )
    return K, fov_x_deg
