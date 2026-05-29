"""Local Rerun visualisation for predicted multilayer point clouds.

Outputs an ``.rrd`` file you can open with ``rerun /path/to/foo.rrd`` or
stream live with ``rr.serve_grpc()``.  No remote upload, no HTML packaging
-- just a self-contained record file you can ship with the release.
"""

from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb

    _HAS_RERUN = True
except ModuleNotFoundError:  # pragma: no cover - graceful fallback
    _HAS_RERUN = False
    rr = None  # type: ignore[assignment]
    rrb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def depth_to_colormap(
    depth: np.ndarray, mask: np.ndarray | None = None, cmap: int = cv2.COLORMAP_TURBO
) -> np.ndarray:
    """Map a single-layer depth array to an RGB visualisation.

    ``depth`` is normalised by its valid-pixel min/max.  Invalid pixels are
    painted white.
    """
    valid = mask if mask is not None else np.isfinite(depth)
    out = np.full(depth.shape + (3,), 255, dtype=np.uint8)
    if not valid.any():
        return out
    d_valid = depth[valid]
    d_lo, d_hi = float(d_valid.min()), float(d_valid.max())
    span = max(d_hi - d_lo, 1e-6)
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = (depth[valid] - d_lo) / span
    norm = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cmap)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    out[valid] = coloured[valid]
    return out


def _later_layer_grayscale_colors(
    z: np.ndarray,
    z_min: float,
    z_max: float,
    gray_near: int = 220,
    gray_far: int = 60,
) -> np.ndarray:
    """Map later-layer depths to grayscale (near = light, far = dark)."""
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        gray = (gray_near + gray_far) // 2
        col = np.full((z.shape[0], 3), gray, dtype=np.uint8)
        return col
    norm = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    gray = gray_near + (gray_far - gray_near) * norm
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _subsample_points(
    pts: np.ndarray, colors: np.ndarray, max_pts: int
) -> tuple[np.ndarray, np.ndarray]:
    if max_pts <= 0 or len(pts) <= max_pts:
        return pts, colors
    idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
    return pts[idx], colors[idx]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_prediction(
    rgb_uint8: np.ndarray,
    xyz: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray | None = None,
    name: str = "sample",
    max_pts_per_layer: int = 50_000,
    later_gray_near: int = 220,
    later_gray_far: int = 60,
    recording=None,
) -> None:
    """Log a single sample to the current (or supplied) Rerun recording.

    Layer 0 is coloured with the input RGB.  Later layers use a grayscale
    ramp based on their depth (lighter = closer to camera).  Useful for
    skimming many samples in the same recording.

    Args:
        rgb_uint8: ``[H, W, 3]`` uint8 input image (after preprocess).
        xyz: ``[L, H, W, 3]`` predicted camera-space XYZ.
        mask: ``[L, H, W]`` bool predicted validity mask.
        intrinsics: optional ``[3, 3]`` K -- if given, also logs a
            ``world/camera`` Pinhole + image plane.
        name: text label written to ``world/info``.
        max_pts_per_layer: random subsample cap per layer.
        recording: optional ``rr.RecordingStream``.
    """
    if not _HAS_RERUN:
        raise ImportError(
            "rerun-sdk is required for visualisation; install with "
            "`pip install rerun-sdk`."
        )

    rr.log("world/info", rr.TextLog(f"image={name}"), recording=recording)
    rr.log("world/rgb", rr.Image(rgb_uint8), recording=recording)

    if intrinsics is not None:
        rr.log(
            "world/camera",
            rr.Pinhole(
                image_from_camera=intrinsics,
                resolution=[rgb_uint8.shape[1], rgb_uint8.shape[0]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.3,
            ),
            recording=recording,
        )
        rr.log("world/camera/image", rr.Image(rgb_uint8), recording=recording)

    num_layers = xyz.shape[0]
    later_z = xyz[1:, ..., 2][mask[1:]] if num_layers > 1 else np.empty(0)
    later_z = later_z[np.isfinite(later_z)]
    z_lo = float(later_z.min()) if later_z.size else np.inf
    z_hi = float(later_z.max()) if later_z.size else -np.inf

    all_pts: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for layer in range(num_layers):
        m = mask[layer]
        if not m.any():
            continue
        pts = xyz[layer][m]
        if layer == 0:
            colors = rgb_uint8[m]
        else:
            colors = _later_layer_grayscale_colors(
                pts[:, 2], z_lo, z_hi, later_gray_near, later_gray_far
            )
        pts, colors = _subsample_points(pts, colors, max_pts_per_layer)
        rr.log(
            f"world/pointcloud/layer_{layer:02d}",
            rr.Points3D(
                positions=pts, colors=colors, radii=rr.Radius.ui_points(2.0)
            ),
            recording=recording,
        )
        all_pts.append(pts)
        all_colors.append(colors)

    if all_pts:
        rr.log(
            "world/pointcloud/all",
            rr.Points3D(
                positions=np.concatenate(all_pts),
                colors=np.concatenate(all_colors),
                radii=rr.Radius.ui_points(2.0),
            ),
            recording=recording,
        )


def log_prediction_layer_timeline(
    rgb_uint8: np.ndarray,
    xyz: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray | None = None,
    name: str = "sample",
    max_pts_per_layer: int = 50_000,
    later_gray_near: int = 220,
    later_gray_far: int = 60,
    timeline_name: str = "layer",
    recording=None,
) -> None:
    """Log a single sample with a ``layer`` timeline that adds one layer per step.

    At time ``k`` the recording exposes the **cumulative** point cloud of
    layers ``[0, k]``; scrubbing the timeline forward reveals occluded
    layers one at a time, making it easy to inspect what the model
    "carves out" behind the front surface.  The ``world/pointcloud/all``
    entity holds the cumulative cloud and is what the default 3D view
    will display.

    Args mirror :func:`log_prediction`; the only addition is
    ``timeline_name`` (defaults to ``"layer"``).
    """
    if not _HAS_RERUN:
        raise ImportError(
            "rerun-sdk is required for visualisation; install with "
            "`pip install rerun-sdk`."
        )

    rr.log("world/info", rr.TextLog(f"image={name}"), recording=recording)
    rr.log("world/rgb", rr.Image(rgb_uint8), recording=recording)

    if intrinsics is not None:
        rr.log(
            "world/camera",
            rr.Pinhole(
                image_from_camera=intrinsics,
                resolution=[rgb_uint8.shape[1], rgb_uint8.shape[0]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.3,
            ),
            recording=recording,
        )
        rr.log("world/camera/image", rr.Image(rgb_uint8), recording=recording)

    num_layers = xyz.shape[0]
    later_z = xyz[1:, ..., 2][mask[1:]] if num_layers > 1 else np.empty(0)
    later_z = later_z[np.isfinite(later_z)]
    z_lo = float(later_z.min()) if later_z.size else np.inf
    z_hi = float(later_z.max()) if later_z.size else -np.inf

    cum_pts: list[np.ndarray] = []
    cum_colors: list[np.ndarray] = []
    for layer in range(num_layers):
        m = mask[layer]
        if m.any():
            pts = xyz[layer][m]
            if layer == 0:
                colors = rgb_uint8[m]
            else:
                colors = _later_layer_grayscale_colors(
                    pts[:, 2], z_lo, z_hi, later_gray_near, later_gray_far
                )
            pts, colors = _subsample_points(pts, colors, max_pts_per_layer)
            cum_pts.append(pts)
            cum_colors.append(colors)

        rr.set_time_sequence(timeline_name, layer, recording=recording)
        if cum_pts:
            rr.log(
                "world/pointcloud/all",
                rr.Points3D(
                    positions=np.concatenate(cum_pts),
                    colors=np.concatenate(cum_colors),
                    radii=rr.Radius.ui_points(2.0),
                ),
                recording=recording,
            )
            rr.log(
                "world/info",
                rr.TextLog(f"image={name}  layers 0..{layer} ({len(cum_pts)} layers)"),
                recording=recording,
            )

    rr.reset_time(recording=recording)


def init_recording_layer_timeline(application_id: str):
    """Recording + blueprint for :func:`log_prediction_layer_timeline`.

    Adds a ``TimePanel`` so the user can immediately scrub the ``layer``
    timeline; the 3D view shows the cumulative cloud at the current
    layer.
    """
    if not _HAS_RERUN:
        raise ImportError("rerun-sdk is required for visualisation.")
    rec = rr.new_recording(application_id, spawn=False)
    views = [
        rrb.Spatial3DView(
            name="Prediction (layer-by-layer)",
            origin="world",
            contents=["world/**", "- world/rgb"],
            background=[255, 255, 255],
        ),
        rrb.Spatial2DView(name="Input", origin="world/rgb", contents="/**"),
    ]
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(*views),
            rrb.TimePanel(state="expanded"),
            collapse_panels=True,
        ),
        recording=rec,
    )
    return rec


def init_recording(application_id: str, blueprint: bool = True):
    """Convenience wrapper around :func:`rerun.new_recording`.

    Sets up a single Spatial3D + Image blueprint that includes every entity
    under ``world/`` so both single-sample (``world/pointcloud/**``) and
    multi-seed (``world/seed_*/pointcloud/**``) recordings render
    correctly.
    """
    if not _HAS_RERUN:
        raise ImportError("rerun-sdk is required for visualisation.")
    rec = rr.new_recording(application_id, spawn=False)
    if blueprint:
        views = [
            rrb.Spatial3DView(
                name="Prediction",
                origin="world",
                contents=["world/**", "- world/rgb"],
                background=[255, 255, 255],
            ),
            rrb.Spatial2DView(name="Input", origin="world/rgb", contents="/**"),
        ]
        rr.send_blueprint(
            rrb.Blueprint(rrb.Horizontal(*views), collapse_panels=True),
            recording=rec,
        )
    return rec


def save_rrd(recording, path: str | os.PathLike) -> str:
    """Persist a recording to disk.  Returns the absolute path."""
    if not _HAS_RERUN:
        raise ImportError("rerun-sdk is required for visualisation.")
    rr.save(str(path), recording=recording)
    return str(os.path.abspath(path))


# ---------------------------------------------------------------------------
# Multi-seed logging (used by examples/infer_rgba.py / infer_scene.py /
# infer_video.py when running the default 4-seed sweep).
# ---------------------------------------------------------------------------


def _seeds_x_extent(
    seeds_xyz: list[np.ndarray], seeds_mask: list[np.ndarray], axis: int = 0
) -> float:
    """Per-axis extent of the layer-0 valid points across all seeds."""
    lo, hi = np.inf, -np.inf
    for xyz, mask in zip(seeds_xyz, seeds_mask):
        m0 = mask[0]
        if not m0.any():
            continue
        v = xyz[0][m0][:, axis]
        lo = min(lo, float(v.min()))
        hi = max(hi, float(v.max()))
    if lo == np.inf:
        return 1.0
    return max(hi - lo, 0.1) * 1.3


def log_multiseed_prediction(
    rgb_uint8: np.ndarray,
    seeds_xyz: list[np.ndarray],
    seeds_mask: list[np.ndarray],
    seed_values: list[int],
    intrinsics: np.ndarray | None = None,
    name: str = "multiseed",
    max_pts_per_layer: int = 50_000,
    later_gray_near: int = 220,
    later_gray_far: int = 60,
    recording=None,
) -> None:
    """Log N seeds of the same input image to one Rerun recording.

    Each seed's point cloud is offset along ``+X`` by an axis-aligned step
    of ``1.3 × layer-0 extent``, so the seeds can be compared at a glance
    in the 3D viewport.

    Args:
        rgb_uint8: ``[H, W, 3]`` uint8 input image (after preprocess).
        seeds_xyz: list of ``[L, H, W, 3]`` predicted XYZ, one per seed.
        seeds_mask: list of ``[L, H, W]`` bool masks, one per seed.
        seed_values: seed integer for each entry (logged in ``world/info``).
        intrinsics: optional ``[3, 3]`` camera matrix at model resolution.
        name: text label.
    """
    if not _HAS_RERUN:
        raise ImportError("rerun-sdk is required for visualisation.")

    n_seeds = len(seeds_xyz)
    if not (len(seeds_mask) == n_seeds == len(seed_values)):
        raise ValueError("seeds_xyz, seeds_mask, seed_values must agree in length")

    rr.log("world/info", rr.TextLog(f"image={name} ({n_seeds} seeds)"), recording=recording)
    rr.log("world/rgb", rr.Image(rgb_uint8), recording=recording)
    if intrinsics is not None:
        rr.log(
            "world/camera",
            rr.Pinhole(
                image_from_camera=intrinsics,
                resolution=[rgb_uint8.shape[1], rgb_uint8.shape[0]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.3,
            ),
            recording=recording,
        )
        rr.log("world/camera/image", rr.Image(rgb_uint8), recording=recording)

    x_step = _seeds_x_extent(seeds_xyz, seeds_mask, axis=0)
    for si, (xyz, mask, seed) in enumerate(zip(seeds_xyz, seeds_mask, seed_values)):
        x_offset = (si - (n_seeds - 1) / 2.0) * x_step
        seed_prefix = f"world/seed_{si:02d}_s{seed}"
        rr.log(f"{seed_prefix}/info", rr.TextLog(f"seed={seed}"), recording=recording)

        num_layers = xyz.shape[0]
        later_z = xyz[1:, ..., 2][mask[1:]] if num_layers > 1 else np.empty(0)
        later_z = later_z[np.isfinite(later_z)]
        z_lo = float(later_z.min()) if later_z.size else np.inf
        z_hi = float(later_z.max()) if later_z.size else -np.inf

        all_pts: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        for layer in range(num_layers):
            m = mask[layer]
            if not m.any():
                continue
            pts = xyz[layer][m].copy()
            pts[:, 0] += x_offset
            if layer == 0:
                colors = rgb_uint8[m]
            else:
                colors = _later_layer_grayscale_colors(
                    pts[:, 2], z_lo, z_hi, later_gray_near, later_gray_far
                )
            pts, colors = _subsample_points(pts, colors, max_pts_per_layer)
            rr.log(
                f"{seed_prefix}/pointcloud/layer_{layer:02d}",
                rr.Points3D(positions=pts, colors=colors, radii=rr.Radius.ui_points(2.0)),
                recording=recording,
            )
            all_pts.append(pts)
            all_colors.append(colors)

        if all_pts:
            rr.log(
                f"{seed_prefix}/pointcloud/all",
                rr.Points3D(
                    positions=np.concatenate(all_pts),
                    colors=np.concatenate(all_colors),
                    radii=rr.Radius.ui_points(2.0),
                ),
                recording=recording,
            )


# ---------------------------------------------------------------------------
# Video clip logging (used by examples/infer_video.py)
# ---------------------------------------------------------------------------


def init_recording_video(application_id: str):
    """Like :func:`init_recording` but tuned for the video timeline scrubbing.

    The viewport gets two views: a 3D Spatial view on the left showing
    layered point clouds, and a 2D image view on the right showing the
    current frame's RGB.  Use ``rr.set_time_sequence("frame", t)`` to
    scrub.
    """
    if not _HAS_RERUN:
        raise ImportError("rerun-sdk is required for visualisation.")
    rec = rr.new_recording(application_id, spawn=False)
    views = [
        rrb.Spatial3DView(
            name="Prediction",
            origin="world",
            contents=["world/**", "- world/rgb"],
            background=[255, 255, 255],
        ),
        rrb.Spatial2DView(name="Frame", origin="world/rgb", contents="/**"),
    ]
    rr.send_blueprint(
        rrb.Blueprint(rrb.Horizontal(*views), collapse_panels=True),
        recording=rec,
    )
    return rec


def log_video_clip_prediction(
    rgb_clip: list[np.ndarray],
    xyz_clip: np.ndarray,
    mask_clip: np.ndarray,
    intrinsics: np.ndarray | None = None,
    frame_names: list[str] | None = None,
    name: str = "clip",
    max_pts_per_layer: int = 50_000,
    later_gray_near: int = 220,
    later_gray_far: int = 60,
    recording=None,
) -> None:
    """Log a T-frame clip prediction across the ``frame`` Rerun timeline.

    Args:
        rgb_clip: list of T ``[H, W, 3]`` uint8 input frames (after preprocess).
        xyz_clip: ``[T, L, H, W, 3]`` predicted XYZ.
        mask_clip: ``[T, L, H, W]`` bool predicted mask.
        intrinsics: optional shared ``[3, 3]`` K.
        frame_names: optional list of T filenames for the ``world/info`` log.
    """
    if not _HAS_RERUN:
        raise ImportError("rerun-sdk is required for visualisation.")

    T = len(rgb_clip)
    if not (xyz_clip.shape[0] == mask_clip.shape[0] == T):
        raise ValueError(
            f"Inconsistent T: rgb={T}, xyz={xyz_clip.shape[0]}, mask={mask_clip.shape[0]}"
        )

    later_z = xyz_clip[:, 1:, ..., 2][mask_clip[:, 1:]] if xyz_clip.shape[1] > 1 else np.empty(0)
    later_z = later_z[np.isfinite(later_z)]
    z_lo = float(later_z.min()) if later_z.size else np.inf
    z_hi = float(later_z.max()) if later_z.size else -np.inf

    for t in range(T):
        rr.set_time_sequence("frame", t, recording=recording)
        label = frame_names[t] if frame_names is not None else f"frame_{t:03d}"
        rr.log("world/info", rr.TextLog(f"clip={name}, frame={label}"), recording=recording)
        rr.log("world/rgb", rr.Image(rgb_clip[t]), recording=recording)

        if intrinsics is not None:
            rr.log(
                "world/camera",
                rr.Pinhole(
                    image_from_camera=intrinsics,
                    resolution=[rgb_clip[t].shape[1], rgb_clip[t].shape[0]],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.3,
                ),
                recording=recording,
            )
            rr.log("world/camera/image", rr.Image(rgb_clip[t]), recording=recording)

        num_layers = xyz_clip.shape[1]
        all_pts: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        for layer in range(num_layers):
            m = mask_clip[t, layer]
            if not m.any():
                continue
            pts = xyz_clip[t, layer][m]
            if layer == 0:
                colors = rgb_clip[t][m]
            else:
                colors = _later_layer_grayscale_colors(
                    pts[:, 2], z_lo, z_hi, later_gray_near, later_gray_far
                )
            pts, colors = _subsample_points(pts, colors, max_pts_per_layer)
            rr.log(
                f"world/pointcloud/layer_{layer:02d}",
                rr.Points3D(positions=pts, colors=colors, radii=rr.Radius.ui_points(2.0)),
                recording=recording,
            )
            all_pts.append(pts)
            all_colors.append(colors)

        if all_pts:
            rr.log(
                "world/pointcloud/all",
                rr.Points3D(
                    positions=np.concatenate(all_pts),
                    colors=np.concatenate(all_colors),
                    radii=rr.Radius.ui_points(2.0),
                ),
                recording=recording,
            )


# ---------------------------------------------------------------------------
# Multi-seed video clip logging (used by examples/_infer_4seeds.py)
# ---------------------------------------------------------------------------


def log_video_multiseed_prediction(
    rgb_clip: list[np.ndarray],
    seeds_xyz: list[np.ndarray],
    seeds_mask: list[np.ndarray],
    seed_values: list[int],
    intrinsics: np.ndarray | None = None,
    frame_names: list[str] | None = None,
    name: str = "clip_multiseed",
    max_pts_per_layer: int = 30_000,
    later_gray_near: int = 220,
    later_gray_far: int = 60,
    recording=None,
) -> None:
    """Log N seeds of the same T-frame clip along a shared ``frame`` timeline.

    Each seed's point cloud is offset along ``+X`` by an axis-aligned step
    of ``1.3 × layer-0 extent`` (computed from seed-0/frame-0/layer-0 mask),
    so seeds can be compared at a glance while scrubbing through frames.

    Args:
        rgb_clip:    list of T ``[H, W, 3]`` uint8 frames (after preprocess).
        seeds_xyz:   list of ``[T, L, H, W, 3]`` arrays, one per seed.
        seeds_mask:  list of ``[T, L, H, W]`` bool masks, one per seed.
        seed_values: integer seed for each entry (logged in ``world/info``).
        intrinsics:  optional shared ``[3, 3]`` camera matrix.
        frame_names: optional list of T frame filenames.
    """
    if not _HAS_RERUN:
        raise ImportError("rerun-sdk is required for visualisation.")

    n_seeds = len(seeds_xyz)
    if not (len(seeds_mask) == n_seeds == len(seed_values)):
        raise ValueError("seeds_xyz, seeds_mask, seed_values must agree in length")

    T = len(rgb_clip)
    for s in range(n_seeds):
        if seeds_xyz[s].shape[0] != T or seeds_mask[s].shape[0] != T:
            raise ValueError(
                f"seed {s}: T mismatch (rgb T={T}, "
                f"xyz T={seeds_xyz[s].shape[0]}, mask T={seeds_mask[s].shape[0]})"
            )

    # x_step uses the same logic as log_multiseed_prediction but computed
    # from each seed's frame-0/layer-0 (the "anchor" frame's foreground).
    seeds_xyz_anchor = [s[0] for s in seeds_xyz]
    seeds_mask_anchor = [s[0] for s in seeds_mask]
    x_step = _seeds_x_extent(seeds_xyz_anchor, seeds_mask_anchor, axis=0)

    # Pre-compute later-z range per seed for grayscale coloring.
    z_lo_hi: list[tuple[float, float]] = []
    for xyz, mask in zip(seeds_xyz, seeds_mask):
        later_z = xyz[:, 1:, ..., 2][mask[:, 1:]] if xyz.shape[1] > 1 else np.empty(0)
        later_z = later_z[np.isfinite(later_z)]
        z_lo = float(later_z.min()) if later_z.size else np.inf
        z_hi = float(later_z.max()) if later_z.size else -np.inf
        z_lo_hi.append((z_lo, z_hi))

    for t in range(T):
        rr.set_time_sequence("frame", t, recording=recording)
        label = frame_names[t] if frame_names is not None else f"frame_{t:03d}"
        rr.log(
            "world/info",
            rr.TextLog(f"clip={name}, frame={label}, n_seeds={n_seeds}"),
            recording=recording,
        )
        rr.log("world/rgb", rr.Image(rgb_clip[t]), recording=recording)
        if intrinsics is not None:
            rr.log(
                "world/camera",
                rr.Pinhole(
                    image_from_camera=intrinsics,
                    resolution=[rgb_clip[t].shape[1], rgb_clip[t].shape[0]],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.3,
                ),
                recording=recording,
            )
            rr.log("world/camera/image", rr.Image(rgb_clip[t]), recording=recording)

        for si, (xyz, mask, seed) in enumerate(zip(seeds_xyz, seeds_mask, seed_values)):
            x_offset = (si - (n_seeds - 1) / 2.0) * x_step
            seed_prefix = f"world/seed_{si:02d}_s{seed}"
            z_lo, z_hi = z_lo_hi[si]

            num_layers = xyz.shape[1]
            all_pts: list[np.ndarray] = []
            all_colors: list[np.ndarray] = []
            for layer in range(num_layers):
                m = mask[t, layer]
                if not m.any():
                    continue
                pts = xyz[t, layer][m].copy()
                pts[:, 0] += x_offset
                if layer == 0:
                    colors = rgb_clip[t][m]
                else:
                    colors = _later_layer_grayscale_colors(
                        pts[:, 2], z_lo, z_hi, later_gray_near, later_gray_far
                    )
                pts, colors = _subsample_points(pts, colors, max_pts_per_layer)
                rr.log(
                    f"{seed_prefix}/pointcloud/layer_{layer:02d}",
                    rr.Points3D(positions=pts, colors=colors, radii=rr.Radius.ui_points(2.0)),
                    recording=recording,
                )
                all_pts.append(pts)
                all_colors.append(colors)
            if all_pts:
                rr.log(
                    f"{seed_prefix}/pointcloud/all",
                    rr.Points3D(
                        positions=np.concatenate(all_pts),
                        colors=np.concatenate(all_colors),
                        radii=rr.Radius.ui_points(2.0),
                    ),
                    recording=recording,
                )
