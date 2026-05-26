import math
from functools import partial

import numpy as np
from wt._core.vendor.moge import utils3d


def weighted_mean_numpy(
    x: np.ndarray,
    w: np.ndarray = None,
    axis: int | tuple[int, ...] = None,
    keepdims: bool = False,
    eps: float = 1e-7,
) -> np.ndarray:
    if w is None:
        return np.mean(x, axis=axis)
    else:
        w = w.astype(x.dtype)
        return (x * w).mean(axis=axis) / np.clip(w.mean(axis=axis), eps, None)


def harmonic_mean_numpy(
    x: np.ndarray,
    w: np.ndarray = None,
    axis: int | tuple[int, ...] = None,
    keepdims: bool = False,
    eps: float = 1e-7,
) -> np.ndarray:
    if w is None:
        return 1 / (1 / np.clip(x, eps, None)).mean(axis=axis)
    else:
        w = w.astype(x.dtype)
        return 1 / (
            weighted_mean_numpy(1 / (x + eps), w, axis=axis, keepdims=keepdims, eps=eps)
            + eps
        )


def image_plane_uv_numpy(
    width: int, height: int, aspect_ratio: float = None, dtype: np.dtype = np.float32
) -> np.ndarray:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5

    u = np.linspace(
        -span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype
    )
    v = np.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
    )
    u, v = np.meshgrid(u, v, indexing="xy")
    uv = np.stack([u, v], axis=-1)
    return uv


def focal_to_fov_numpy(focal: np.ndarray):
    return 2 * np.arctan(0.5 / focal)


def fov_to_focal_numpy(fov: np.ndarray):
    return 0.5 / np.tan(fov / 2)


def intrinsics_to_fov_numpy(intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fov_x = focal_to_fov_numpy(intrinsics[..., 0, 0])
    fov_y = focal_to_fov_numpy(intrinsics[..., 1, 1])
    return fov_x, fov_y


def point_map_to_depth_legacy_numpy(points: np.ndarray):
    height, width = points.shape[-3:-1]
    diagonal = (height**2 + width**2) ** 0.5
    uv = image_plane_uv_numpy(width, height, dtype=points.dtype)  # (H, W, 2)
    _, uv = np.broadcast_arrays(points[..., :2], uv)

    # Solve least squares problem
    b = (uv * points[..., 2:]).reshape(*points.shape[:-3], -1)  # (..., H * W * 2)
    A = np.stack([points[..., :2], -uv], axis=-1).reshape(
        *points.shape[:-3], -1, 2
    )  # (..., H * W * 2, 2)

    M = A.swapaxes(-2, -1) @ A
    solution = (
        np.linalg.inv(M + 1e-6 * np.eye(2)) @ (A.swapaxes(-2, -1) @ b[..., None])
    ).squeeze(-1)
    focal, shift = solution

    depth = points[..., 2] + shift[..., None, None]
    fov_x = np.arctan(width / diagonal / focal) * 2
    fov_y = np.arctan(height / diagonal / focal) * 2
    return depth, fov_x, fov_y, shift


def solve_optimal_shift_focal(
    uv: np.ndarray,
    xyz: np.ndarray,
    ransac_iters: int = None,
    ransac_hypothetical_size: float = 0.1,
    ransac_threshold: float = 0.1,
):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    initial_shift = 0  # -z.min(keepdims=True) + 1.0

    if ransac_iters is None:
        solution = least_squares(
            partial(fn, uv, xy, z), x0=initial_shift, ftol=1e-3, method="lm"
        )
        optim_shift = solution["x"].squeeze().astype(np.float32)
    else:
        best_err, best_shift = np.inf, None
        for _ in range(ransac_iters):
            maybe_inliers = np.random.choice(
                len(z), size=int(ransac_hypothetical_size * len(z)), replace=False
            )
            solution = least_squares(
                partial(fn, uv[maybe_inliers], xy[maybe_inliers], z[maybe_inliers]),
                x0=initial_shift,
                ftol=1e-3,
                method="lm",
            )
            maybe_shift = solution["x"].squeeze().astype(np.float32)
            confirmed_inliers = (
                np.linalg.norm(fn(uv, xy, z, maybe_shift).reshape(-1, 2), axis=-1)
                < ransac_threshold
            )
            if confirmed_inliers.sum() > 10:
                solution = least_squares(
                    partial(
                        fn,
                        uv[confirmed_inliers],
                        xy[confirmed_inliers],
                        z[confirmed_inliers],
                    ),
                    x0=maybe_shift,
                    ftol=1e-3,
                    method="lm",
                )
                better_shift = solution["x"].squeeze().astype(np.float32)
            else:
                better_shift = maybe_shift
            err = (
                np.linalg.norm(fn(uv, xy, z, better_shift).reshape(-1, 2), axis=-1)
                .clip(max=ransac_threshold)
                .mean()
            )
            if err < best_err:
                best_err, best_shift = err, better_shift
                initial_shift = best_shift

        optim_shift = best_shift

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / (xy_proj * xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)

    return optim_shift


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal
