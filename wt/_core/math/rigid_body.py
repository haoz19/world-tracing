# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Disable the invalid-name linter due to the use of math notation in this module.
# pylint: disable=invalid-name

"""
Functions and transforms for rigid body dynamics.

Many equations are from the Modern Robotics textbook available online at:
  http://hades.mech.northwestern.edu/index.php/Modern_Robotics

Note that many operations here use a `torch.where` to avoid evaluating at
numerically unstable or undefined regions of the domain. In addition, to avoid
NaNs accumulating through `torch.where` expressions of unsafe math operations,
we also wrap the argument of those operations in another `torch.where` call

Adapted from:
  https://github.com/google/nerfies/blob/main/nerfies/rigid_body.py
"""

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F

from wt._core.math import quaternion as quat_lib
from wt._core.math import safe_math

EPS = torch.finfo(torch.float32).eps


def skew(vector: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3 3"]:
    """
    Builds a skew matrix ("cross product matrix") for a vector.

    References:
        https://en.wikipedia.org/wiki/Skew-symmetric_matrix.

    Args:
        vector: A 3-vector.

    Returns:
        A (3, 3) skew-symmetric matrix such that W @ v == w x v where w is the input
        vector, W is the corresponding skew-symmatrix matrix, and v is another
        vector.
    """
    *batch_shape, _ = vector.shape
    matrix = vector.new_zeros(*batch_shape, 3, 3)
    matrix[..., 0, 1] = -vector[..., 2]
    matrix[..., 0, 2] = vector[..., 1]
    matrix[..., 1, 0] = vector[..., 2]
    matrix[..., 1, 2] = -vector[..., 0]
    matrix[..., 2, 0] = -vector[..., 1]
    matrix[..., 2, 1] = vector[..., 0]
    return matrix


def unskew(matrix: Float[Tensor, "*b 3 3"]) -> Float[Tensor, "*b 3"]:
    """
    Convert a skew matrix to a vector w.

    See `skew()` for documentation.

    Args:
      matrix: (3, 3) A skew matrix.

    Returns:
      w: (3,) A 3-vector corresponding to the skew matrix.
    """
    return torch.stack(
        [
            matrix[..., 2, 1],
            matrix[..., 0, 2],
            matrix[..., 1, 0],
        ],
        dim=-1,
    )


def eye(
    n: int, m: int | None = None, batch_shape: tuple[int, ...] = (), **kwargs
) -> torch.Tensor:
    """Convience function for creating batched identity matrices."""
    m = n if m is None else m
    matrix = torch.eye(n=n, m=m, **kwargs)
    n, m = matrix.shape[-2:]
    matrix = matrix.view(*[1 for _ in batch_shape], n, m).expand(*batch_shape, n, m)
    return matrix


def rt_to_se3(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """
    Rotation and translation to homogeneous transform.

    Args:
      rotation: (..., 3, 3) An orthonormal rotation matrix.
      translation: (..., 3,) A 3-vector representing an offset.

    Returns:
      (4, 4) The homogeneous transformation matrix described by rotating by R and
        translating by p.
    """
    device = rotation.device
    dtype = rotation.dtype
    batch_shape = rotation.shape[:-2]
    return torch.concatenate(
        [
            torch.concatenate([rotation, translation.view(*batch_shape, 3, 1)], dim=-1),
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).expand(
                (*batch_shape, 1, 4)
            ),
        ],
        dim=-2,
    )


def se3_to_rt(transform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a homogeneous transform to a rotation and translation.

    Args:
      transform: (..., 4, 4) A homogeneous transformation matrix.

    Returns:
      rotation: (..., 3, 3) An orthonormal rotation matrix.
      translation: (..., 3,) A 3-vector representing an offset.
    """
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return rotation, translation


def exp_so3(axis_angle: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

    Args:
      axis_angle: A 3-vector where the direction is the axis of rotation and the
        magnitude is the angle of rotation.
      eps: an epsilon value for numerical stability.

    Returns:
      (3, 3) An orthonormal rotation matrix representing the same rotation.
    """
    device = axis_angle.device
    theta_squared = torch.sum(axis_angle**2, dim=-1)
    theta = safe_math.safe_sqrt(theta_squared)

    # Near zero, we switch to using the first order Taylor expansion.
    R_taylor = torch.eye(3, device=device) + skew(axis_angle)

    # Prevent bad gradients from propagating back when theta is small.
    axis_angle_safe = torch.where(theta > eps, axis_angle, 0.0)
    theta_safe = torch.where(theta > eps, theta, 1.0)
    axis = axis_angle_safe / theta_safe
    W = skew(axis)
    R = (
        torch.eye(3, device=device)
        + torch.sin(theta_safe) * W
        + (1.0 - torch.cos(theta_safe)) * (W @ W)
    )

    return torch.where(theta > eps, R, R_taylor)


def log_so3(rotation: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Matrix logarithm from the Lie group SO3 to the Lie algebra so3.

    Modern Robotics Eqn 3.53.

    Args:
      rotation: (3, 3) An orthonormal rotation matrix.
      eps: an epsilon value for numerical stability.

    Returns:
      w: (3,) The unit vector representing the axis of rotation.
      theta: The angle of rotation.
    """
    q = quat_lib.from_rotation_matrix(rotation, eps)
    axis_angle = quat_lib.to_axis_angle(q, eps)
    return axis_angle


def so3_align_vectors(
    v_from: Float[Tensor, "*batch 3"],
    v_to: Float[Tensor, "*batch 3"],
    eps: float = EPS,
) -> Float[Tensor, "*batch 3 3"]:
    r"""Compute rotation matrix aligning v_from to v_to.

    Computes the rotation :math:`R \in SO(3)` such that
    :math:`R \mathbf{v}_1 = \mathbf{v}_2`.

    Uses Rodrigues' rotation formula:

    .. math::

        R = I + [\mathbf{w}]_\times + [\mathbf{w}]_\times^2 \frac{1 - c}{s^2}

    where :math:`\mathbf{w} = \mathbf{v}_1 \times \mathbf{v}_2`,
    :math:`c = \mathbf{v}_1 \cdot \mathbf{v}_2`, and
    :math:`s = \|\mathbf{w}\|`.

    Args:
        v_from: Source vector with shape ``[..., 3]``.
        v_to: Target vector with shape ``[..., 3]``.
        eps: Small epsilon for numerical stability.

    Returns:
        Rotation matrix with shape ``[..., 3, 3]``.

    Note:
        This is equivalent to computing the exponential map of the cross
        product scaled by the angle: :math:`\exp([v_1 \times v_2]_\times)`.
    """
    # Compute dot product (cosine of angle)
    cos_theta = torch.sum(v_from * v_to, dim=-1, keepdim=True)

    # Compute cross product (axis of rotation, scaled by sine)
    w = torch.cross(v_from, v_to, dim=-1)
    sin_theta = torch.norm(w, dim=-1, keepdim=True)

    # Skew-symmetric matrix [w]_×
    batch_shape = v_from.shape[:-1]
    skew_mat = skew(w)

    # Identity matrix
    eye_mat = torch.eye(3, device=v_from.device, dtype=v_from.dtype)
    eye_mat = eye_mat.expand(*batch_shape, 3, 3)

    # Rodrigues formula: R = I + [w]_× + [w]_×² * (1-c)/s²
    W_sq = torch.einsum("...ik,...kj->...ij", skew_mat, skew_mat)
    factor = safe_math.safe_div(1.0 - cos_theta, sin_theta**2 + eps, eps=eps)
    R = eye_mat + skew_mat + W_sq * factor.unsqueeze(-1)

    return R


def exp_se3(screw_axis: Float[Tensor, "6"], eps: float = EPS) -> Float[Tensor, "4 4"]:
    """
    Exponential map from Lie algebra se3 to Lie group SE3.

    Modern Robotics Eqn 3.88.

    Args:
      screw_axis: A 6-vector encoding a screw axis of motion. This can be broken
        down into [w, v] where w is an angle-axis rotation and v represents a
        translation. ||w|| corresponds to the magnitude of motion.
      eps: an epsilon value for numerical stability.

    Returns:
      (4, 4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.
    """
    w, v = torch.split(screw_axis, [3, 3])
    R = exp_so3(w)
    theta_squared = torch.sum(w**2, dim=-1)
    theta = safe_math.safe_sqrt(theta_squared)
    W = skew(w / theta)
    # Note that p = 0 when theta = 0.
    p = torch.matmul(
        (
            theta * torch.eye(3, device=screw_axis.device)
            + (1.0 - torch.cos(theta)) * W
            + (theta - torch.sin(theta)) * (W @ W)
        ),
        v / theta,
    )
    # If theta^2 is close to 0 it means this is a pure translation so p = v.
    p = torch.where(theta_squared > eps**2, p, v)
    return rt_to_se3(R, p)


def log_se3(transform: Float[Tensor, "4 4"], eps: float = EPS) -> Float[Tensor, "6"]:
    """
    Matrix logarithm from the Lie group SE3 to the Lie algebra se3.

    Modern Robotics Eqn 3.91-3.92.

    Args:
      transform: (4,4) A homogeneous transformation matrix.
      eps: an epsilon value for numerical stability.

    Returns:
      screw_axis: A 6-vector encoding a screw axis of motion. This can be broken
        down into [w, v] where w is an angle-axis rotation and v represents a
        translation. The ||w|| and ||v|| both correspond to the magnitude of
        motion.
    """
    R, p = se3_to_rt(transform)
    w = log_so3(R, eps)
    theta_squared = torch.sum(w**2, dim=-1)
    theta = safe_math.safe_sqrt(theta_squared)
    W = skew(w / theta)

    G_inv1 = torch.eye(3, device=transform.device)
    G_inv2 = theta * -W / 2.0
    G_inv3 = (1.0 - 0.5 * theta / torch.tan(theta / 2.0)) * (W @ W)
    G_inv = G_inv1 + G_inv2 + G_inv3

    v = (G_inv @ p[..., None]).squeeze(-1)
    # If theta = 0 then the transformation is a pure translation and v = p.
    # This avoids using the numerically unstable G matrix when theta is near zero.
    v = torch.where(theta_squared > eps, v, p)
    S = torch.concatenate([w, v], dim=-1)
    return S


def rts_to_sim3(
    rotation: torch.Tensor | None = None,
    translation: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convert a rotation, translation and scale to a homogeneous transform.

    Args:
      rotation: (3, 3) An orthonormal rotation matrix. If None does not rotate.
      translation: (3,) A vector representing a translation. If None does not translate.
      scale: A (3,) tensor containing the scale for each axis, or a scalar factor.
        If None does not scale.

    Returns:
      (4, 4) A homogeneous transformation matrix.
    """
    if rotation is not None:
        dtype = rotation.dtype
        device = rotation.device
        *batch_shape, _, _ = rotation.shape
    elif translation is not None:
        dtype = translation.dtype
        device = translation.device
        *batch_shape, _ = translation.shape
    elif scale is not None:
        dtype = scale.dtype
        device = scale.device
        batch_shape = scale.shape
    else:
        raise ValueError("At least one input must not be None.")

    if rotation is None:
        rotation = eye(3, dtype=dtype, device=device, batch_shape=batch_shape)

    if translation is None:
        translation = torch.zeros((*batch_shape, 3), dtype=dtype, device=device)

    if scale is None:
        scale = torch.full(batch_shape, 1.0, dtype=dtype, device=device)

    # Expand scalar scales into per-axis scales.
    if scale.ndim == len(batch_shape):
        scale = scale[..., None]

    transform = torch.concatenate(
        [rotation * scale[..., None], translation[..., :, None]], dim=-1
    )
    return pad_matrix_to_4x4(transform)


def sim3_to_rts(transform: torch.Tensor) -> torch.Tensor:
    """
    Convert a homogeneous transform to rotation, translation and scale.

    Args:
      transform: (4, 4) A homogeneous transformation matrix.

    Returns:
      rotation: (3, 3) An orthonormal rotation matrix.
      translation: (3,) A 3-vector representing a translation.
      scale: A scalar factor.
    """
    rotation_scale = transform[..., :3, :3]
    # Assumes rotation is an orthonormal transform, thus taking norm of first row.
    scale = torch.linalg.norm(rotation_scale, dim=1)[0]
    rotation = rotation_scale / scale
    translation = transform[..., :3, 3]
    return rotation, translation, scale


def sim3_remove_scale(transform: torch.Tensor) -> torch.Tensor:
    """Remove scale component from a transform (useful for camera matrices)."""
    rotation, translation, _ = sim3_to_rts(transform)
    return rt_to_se3(rotation, translation)


def ortho6d_from_rotation_matrix(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert a matrix to an ortho6d by taking the first two columns."""
    return rotation_matrix[..., :2, :].view(*rotation_matrix.shape[:-2], 6)


def rotation_matrix_from_ortho6d(ortho6d: torch.Tensor) -> torch.Tensor:
    """
    Compute the 3D rotation matrix from the 6D representation.

    Zhou et al. have proposed a novel 6D representation for the rotation in
    SO(3) which is completely continuous. This is highly benificial and produces
    better results than most standard rotation representations for many tasks,
    especially when the predicted value is close to the discontinuity of the
    utilized rotation represantation. This function converts from the proposed 6
    dimensional representation to the classic 3x3 rotation matrix.

    See https://arxiv.org/pdf/1812.07035.pdf for more information.

    Args:
      ortho6d: 6D represantion for the rotation according Zhou et al. of shape
        [6].

    Returns:
      (3, 3) The associated 3x3 rotation matrices.
    """
    if ortho6d.shape[-1] != 6:
        raise ValueError("The shape of the input ortho 6D vector needs to be (6).")

    a1, a2 = ortho6d[..., :3], ortho6d[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdims=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def pad_matrix_to_4x4(matrix: Float[Tensor, "*b i j"]) -> Float[Tensor, "*b 4 4"]:
    """Convert a matrix to a 4x4 matrix."""
    *_, num_rows, num_cols = matrix.shape
    # Nothing to do.
    if num_rows == 4 and num_cols == 4:
        return matrix

    if num_rows > 4 or num_cols > 4:
        raise ValueError("Matrix dimensions must be a maximum of 4x4.")

    padding = (0, 4 - num_cols, 0, 4 - num_rows)
    matrix_4x4 = F.pad(matrix, padding, value=0.0)

    # Make sure padded diagonals correspond to the identity.
    for i in [1, 2, 3]:
        if num_rows <= i or num_cols <= i:
            matrix_4x4[..., i, i] = 1.0

    return matrix_4x4


def invert_se3(transform: Float[Tensor, "... 4 4"]) -> Float[Tensor, "... 4 4"]:
    """
    Invert an SE3 transformation.

    Given a homogeneous transformation matrix T representing a rigid body transform,
    computes the inverse transform T^{-1} such that T @ T^{-1} = I.
    """
    rotation, translation = se3_to_rt(transform)
    inverse_rotation = rotation.transpose(-1, -2)  # For rotation matrices
    inverse_translation = -torch.einsum(
        "...ij,...j->...i", inverse_rotation, translation
    )
    inverse_transform = rt_to_se3(inverse_rotation, inverse_translation[..., None])
    return inverse_transform


def interpolate_se3(
    transform_0: Float[Tensor, "4 4"], transform_1: Float[Tensor, "4 4"], t: float
) -> Float[Tensor, "4 4"]:
    """
    Interpolate between two SE(3) transforms using the parameter t.

    Args:
        transform_0: The starting transforms (4, 4).
        transform_1: The ending transforms (4, 4).
        t: The interpolation coefficients, where each element 0 <= t[i] <= 1.

    Returns:
        The interpolated transform (4, 4).
    """
    if transform_0.shape != (4, 4) or transform_1.shape != (4, 4):
        raise ValueError("Input transforms must have shape (4, 4)")

    if t < 0 or t > 1:
        raise ValueError("Interpolation coefficient t must be between 0 and 1")

    # Compute the inverse of start_transforms
    transform_0_inv = torch.linalg.inv(transform_0)

    # Compute the relative transforms
    relative_transform = transform_1 @ transform_0_inv

    # Compute the logarithm of the relative transforms
    log_relative = log_se3(relative_transform)

    # Scale the logarithm by the interpolation parameter t
    scaled_log = log_relative * t

    # Compute the exponential map to get the interpolated transforms
    interpolated_transform = exp_se3(scaled_log) @ transform_0
    return interpolated_transform
