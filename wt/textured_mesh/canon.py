"""Camera-frame ↔ TRELLIS.2 canonical-frame transform.

TRELLIS.2 expects 3D content inside the cube ``[-0.5, 0.5]^3`` with **Z-up**
(``utils3d.torch.extrinsics_look_at(origin, (0,0,0), (0,0,1))`` inside
``trellis2/utils/render_utils.py``).  Our multilayer-geometry model emits XYZ
in OpenCV camera space: ``+x`` right, ``+y`` down, ``+z`` forward.

This module provides the bidirectional mapping (forward for voxelisation,
inverse for cam-frame mesh export).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CanonicalTransform:
    """Camera XYZ → TRELLIS canonical XYZ.

    The forward map is::

        (x, y, z)_cam  →  (x, z, -y)_canonical_zup
        v_canon = (R @ v_cam - centroid) * scale

    so that the cloud lands inside ``[-half_target, half_target]^3`` ⊂
    ``[-0.5, 0.5]^3`` with its "front" (smallest ``z_cam``) facing the
    TRELLIS yaw=0 camera.

    Attributes:
        centroid: ``(3,)`` translation in canonical space (applied AFTER
            the axis remap).
        scale: isotropic scale factor.
        axis_map: ``"cam2canonical_zup"`` (default), ``"identity"`` (skip
            the remap entirely), or ``"y_flip"`` (negate y only -- legacy).
    """

    centroid: np.ndarray  # (3,) -- applied AFTER axis remap
    scale: float
    axis_map: str = "cam2canonical_zup"

    def _apply_axis_map(self, xyz: np.ndarray) -> np.ndarray:
        if self.axis_map == "cam2canonical_zup":
            return np.stack([xyz[..., 0], xyz[..., 2], -xyz[..., 1]], axis=-1)
        if self.axis_map in ("identity", None):
            return xyz.copy()
        if self.axis_map == "y_flip":
            out = xyz.copy()
            out[..., 1] = -out[..., 1]
            return out
        raise ValueError(f"Unknown axis_map: {self.axis_map}")

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        """Camera XYZ → canonical XYZ."""
        out = self._apply_axis_map(xyz)
        out = out - self.centroid[None, :]
        out = out * self.scale
        return out

    def to_dict(self) -> dict:
        return {
            "centroid": self.centroid.astype(np.float32).tolist(),
            "scale": float(self.scale),
            "axis_map": self.axis_map,
        }


def compute_canonical_transform(
    cloud_cam: np.ndarray,
    half_target: float = 0.45,
    axis_map: str = "cam2canonical_zup",
) -> CanonicalTransform:
    """Fit a ``CanonicalTransform`` that centres + isotropic-scales the
    cloud so it fits in ``[-half_target, half_target]^3``.

    Args:
        cloud_cam: ``(N, 3) float`` cloud in camera space.
        half_target: target half-extent in canonical space (default 0.45,
            keeping a small margin inside the [-0.5, 0.5] cube).
        axis_map: orientation convention (see :class:`CanonicalTransform`).
    """
    if cloud_cam.size == 0:
        raise ValueError("compute_canonical_transform: empty cloud")
    tf = CanonicalTransform(
        centroid=np.zeros(3, dtype=np.float32),
        scale=1.0,
        axis_map=axis_map,
    )
    pts = tf._apply_axis_map(cloud_cam.copy())
    centroid = pts.mean(axis=0)
    centered = pts - centroid[None, :]
    half_extent = float(np.abs(centered).max())
    scale = half_target / max(half_extent, 1e-6)
    return CanonicalTransform(
        centroid=centroid.astype(np.float32),
        scale=scale,
        axis_map=axis_map,
    )


def canon_inverse(
    v_canon: np.ndarray, tf: CanonicalTransform
) -> np.ndarray:
    """Canonical XYZ → camera XYZ (inverse of :meth:`CanonicalTransform.apply`).

    Useful for rendering the textured mesh in the same world as the input
    camera (e.g. side-by-side comparison with the raw point cloud).
    """
    v = np.asarray(v_canon, dtype=np.float64)
    v = v / float(tf.scale) + tf.centroid[None, :].astype(np.float64)
    if tf.axis_map == "cam2canonical_zup":
        v_cam = np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)
    elif tf.axis_map in ("identity", None):
        v_cam = v
    elif tf.axis_map == "y_flip":
        v_cam = v.copy()
        v_cam[..., 1] = -v_cam[..., 1]
    else:
        raise ValueError(f"Unknown axis_map: {tf.axis_map}")
    return v_cam.astype(np.float32)


def aggregate_camera_cloud(
    xyz_cam: np.ndarray,
    mask: np.ndarray,
    max_points: int = 400_000,
    seed: int = 0,
) -> np.ndarray:
    """Flatten ``[L, H, W, 3]`` + ``[L, H, W]`` mask → ``(N, 3)`` cloud.

    Drops non-finite points and randomly subsamples to ``max_points`` if
    the cloud is larger.  The subsampling RNG is seeded for determinism.
    """
    pts = xyz_cam[mask.astype(bool)]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return pts
    if pts.shape[0] > max_points:
        sel = np.random.RandomState(seed).choice(
            pts.shape[0], max_points, replace=False
        )
        pts = pts[sel]
    return pts.astype(np.float32)


__all__ = [
    "CanonicalTransform",
    "aggregate_camera_cloud",
    "canon_inverse",
    "compute_canonical_transform",
]
