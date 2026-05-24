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
"""
Quaternion math.

This module assumes the wxyz quaternion format where w is the real part
and xyz is the imaginary part.

Functions in this module support both batched and unbatched quaternions.

References:
    - https://github.com/google/nerfies/blob/main/nerfies/quaternion.py
    - https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/rotation.h
    - https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
"""

import torch
from beartype import beartype
from jaxtyping import Float
from torch import Tensor

from wt._internal.math import safe_math

EPS = torch.finfo(torch.float32).eps


@beartype
def im(q: Float[Tensor, "*n 4"]) -> Float[Tensor, "*n 3"]:
    """Fetch the imaginary part of the quaternion."""
    return q[..., 1:]


@beartype
def re(q: Float[Tensor, "*n 4"]) -> Float[Tensor, "*n 1"]:
    """Fetch the real part of the quaternion."""
    return q[..., :1]


@beartype
def identity(device: torch.device | str | None = None) -> Float[Tensor, "4"]:
    """Return the identity quaternion."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)


@beartype
def conjugate(q: Float[Tensor, "*n 4"]) -> Float[Tensor, "*n 4"]:
    """Compute the conjugate of a quaternion."""
    return torch.concatenate([re(q), -im(q)], dim=-1)


@beartype
def inverse(q: Float[Tensor, "*n 4"]) -> Float[Tensor, "*n 4"]:
    """Compute the inverse of a quaternion."""
    return normalize(conjugate(q))


@beartype
def normalize(q: Float[Tensor, "*n 4"]) -> Float[Tensor, "*n 4"]:
    """Normalize a quaternion."""
    return q / norm(q)


@beartype
def norm(q: Float[Tensor, "*n 4"]) -> Float[Tensor, "*n 1"]:
    """Return the norm of a quaternion."""
    return torch.linalg.norm(q, dim=-1, keepdim=True)


@beartype
def multiply(
    q1: Float[Tensor, "*n 4"], q2: Float[Tensor, "*n 4"]
) -> Float[Tensor, "*n 4"]:
    """Multiply two quaternions."""
    w = re(q1) * re(q2) - (im(q1) * im(q2)).sum(dim=-1, keepdim=True)
    c = re(q1) * im(q2) + re(q2) * im(q1) + torch.cross(im(q1), im(q2), dim=-1)
    return torch.concatenate([w, c], dim=-1)


@beartype
def rotate(q: Float[Tensor, "*n 4"], v: Float[Tensor, "*n 3"]) -> Float[Tensor, "*n 3"]:
    """Rotate a vector using a quaternion."""
    # Create the quaternion representation of the vector.
    q_v = torch.concatenate([torch.zeros_like(v[..., :1]), v], dim=-1)
    return im(multiply(multiply(q, q_v), conjugate(q)))


@beartype
def log(q: Float[Tensor, "*n 4"], eps: float = EPS) -> Float[Tensor, "*n 4"]:
    """
    Compute the quaternion logarithm.

    References:
      https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions

    Args:
      q: the quaternion in (w,x,y,z) format.
      eps: an epsilon value for numerical stability.

    Returns:
      The logarithm of q.
    """
    mag = torch.linalg.norm(q, dim=-1, keepdim=True)
    v = im(q)
    s = re(q)
    w = torch.log(mag)
    denom = torch.maximum(
        torch.linalg.norm(v, dim=-1, keepdim=True), eps * torch.ones_like(v)
    )
    xyz = v / denom * safe_math.safe_arccos(s)
    return torch.concatenate((w, xyz), dim=-1)


@beartype
def exp(
    q: Float[Tensor, "*n 4"] | Float[Tensor, "*n 3"], eps: float = EPS
) -> Float[Tensor, "*n 4"]:
    """
    Compute the quaternion exponential.

    References:
      https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions

    Args:
      q: a (*, 4) or (*, 3) array containing a quaternion. If the quaternion has 4
        channels it is interpreted as a (w,x,y,z) quaternion while if it has 3 channels
        it is assumed to be a pure quaternion in (x,y,z) format.
      eps: an epsilon value for numerical stability.

    Returns:
      The exponential of q.
    """
    is_pure = q.shape[-1] == 3
    if is_pure:
        s = torch.zeros_like(q[..., :1])
        v = q
    else:
        v = im(q)
        s = re(q)

    norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
    exp_s = torch.exp(s)
    w = torch.cos(norm_v)
    xyz = torch.sin(norm_v) * v / torch.maximum(norm_v, eps * torch.ones_like(norm_v))
    return exp_s * torch.concatenate((w, xyz), dim=-1)


@beartype
def to_rotation_matrix(q: Float[Tensor, "*n 4"]) -> Float[Tensor, "*n 3 3"]:
    """
    Construct a rotation matrix from a quaternion.

    Args:
        q: Tensor of shape `(*, 4)` containing quaternions in WXYZ format.

    Returns:
        Tensor of shape `(*, 3, 3)` containing rotation matrices.
    """
    w, x, y, z = torch.unbind(q, dim=-1)
    s = 1.0 / torch.sum(q**2, dim=-1)
    return torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * s * (y**2 + z**2),
                    2 * s * (x * y - z * w),
                    2 * s * (x * z + y * w),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * s * (x * y + z * w),
                    1 - s * 2 * (x**2 + z**2),
                    2 * s * (y * z - x * w),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * s * (x * z - y * w),
                    2 * s * (y * z + x * w),
                    1 - 2 * s * (x**2 + y**2),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )


@beartype
def from_axis_angle(
    axis_angle: Float[Tensor, "*n 3"], eps: float = EPS
) -> Float[Tensor, "*n 4"]:
    """
    Construct a quaternion for the given axis/angle rotation.

    Args:
        axis_angle: A 3-vector where the direction is the axis of rotation and
            the magnitude is the angle of rotation.
        eps: A small number used for numerical stability around zero rotations.

    Returns:
        A quaternion encoding the same rotation in WXYZ format.
    """
    theta_squared = torch.sum(axis_angle**2, dim=-1)
    theta = safe_math.safe_sqrt(theta_squared)
    half_theta = theta / 2.0
    k = torch.sin(half_theta) / theta
    # Avoid evaluating sqrt when theta is close to zero.
    k = torch.where(theta_squared > eps**2, k, 0.5)
    qw = torch.where(theta_squared > eps**2, torch.cos(half_theta), 1.0)
    qx = axis_angle[..., 0] * k
    qy = axis_angle[..., 1] * k
    qz = axis_angle[..., 2] * k

    return torch.stack([qw, qx, qy, qz], dim=-1)


@beartype
def to_axis_angle(q: Float[Tensor, "*n 4"], eps: float = EPS) -> Float[Tensor, "*n 3"]:
    """
    Convert a quaternion to an axis-angle representation.

    Args:
        q: a 4-vector representing a unit quaternion in WXYZ format.
        eps: A small number used for numerical stability around zero rotations.

    Returns:
        A 3-vector where the direction is the axis of rotation and the magnitude
            is the angle of rotation.
    """
    sin_sq_theta = torch.sum(im(q) ** 2, dim=-1, keepdim=True)

    sin_theta = safe_math.safe_sqrt(sin_sq_theta)
    cos_theta = re(q)

    # If cos_theta is negative, theta is greater than pi/2, which
    # means that angle for the angle_axis vector which is 2 * theta
    # would be greater than pi.
    #
    # While this will result in the correct rotation, it does not
    # result in a normalized angle-axis vector.
    #
    # In that case we observe that 2 * theta ~ 2 * theta - 2 * pi,
    # which is equivalent saying
    #
    #   theta - pi = atan(sin(theta - pi), cos(theta - pi))
    #              = atan(-sin(theta), -cos(theta))
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.arctan2(-sin_theta, -cos_theta),
        torch.arctan2(sin_theta, cos_theta),
    )

    # For zero rotation, sqrt() will produce NaN in the derivative since
    # the argument is zero. We avoid this by directly returning the value in
    # such cases.
    k = torch.where(sin_sq_theta > eps**2, two_theta / sin_theta, 2.0)

    return im(q) * k


# BSD License
#
# For PyTorch3D software
#
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name Meta nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


@beartype
def from_rotation_matrix(
    matrix: Float[Tensor, "*b 3 3"], eps: float = EPS
) -> Float[Tensor, "*b 4"]:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape `(..., 3, 3)`.

    Returns:
        Quaternions with real part first, as tensor of shape `(..., 4)`.
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = safe_math.safe_sqrt(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        ).clamp(min=0),
        eps=eps,
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    q_abs_max_inds = q_abs.argmax(dim=-1)
    out = torch.gather(
        quat_candidates,
        dim=-2,
        index=q_abs_max_inds[..., None, None].expand(*q_abs_max_inds.shape, 1, 4),
    )
    out = out.reshape((*batch_dim, 4))

    return standardize(out)


def standardize(quaternions: Float[Tensor, "*b 4"]) -> Float[Tensor, "*b 4"]:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first, as tensor of shape `(..., 4)`.

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)
