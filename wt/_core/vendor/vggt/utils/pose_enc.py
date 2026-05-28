# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from wt._core.arch.utils import geometry_utils


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
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
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,  # e.g., (256, 512)
    pose_encoding_type="absT_quaR_FoV",
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    This function transforms camera parameters into a unified pose encoding format,
    which can be used for various downstream tasks like pose prediction or representation.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
            where B is batch size and S is sequence length.
            In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world transformation.
            The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
            Defined in pixels, with format:
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            where fx, fy are focal lengths and (cx, cy) is the principal point
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for computing field of view values. For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding to use. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).

    Returns:
        torch.Tensor: Encoded camera pose parameters with shape BxSx9.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
    """

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3

    if pose_encoding_type == "absT_quaR_FoV":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3

        quat = mat_to_quat(R)
        # Note the order of h and w here
        H, W = image_size_hw
        fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        pose_encoding = torch.cat(
            [T, quat, fov_h[..., None], fov_w[..., None]], dim=-1
        ).float()
    else:
        raise NotImplementedError

    return pose_encoding


def pose_encoding_to_extri_intri(
    pose_encoding,
    image_size_hw=None,  # e.g., (256, 512)
    pose_encoding_type="absT_quaR_FoV",
    build_intrinsics=True,
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.

    This function performs the inverse operation of extri_intri_to_pose_encoding,
    reconstructing the full camera parameters from the compact encoding.

    Args:
        pose_encoding (torch.Tensor): Encoded camera pose parameters with shape BxSx9,
            where B is batch size and S is sequence length.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for reconstructing intrinsics from field of view values.
            For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding used. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).
        build_intrinsics (bool): Whether to reconstruct the intrinsics matrix.
            If False, only extrinsics are returned and intrinsics will be None.

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4.
              In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world
              transformation. The format is [R|t] where R is a 3x3 rotation matrix and t is
              a 3x1 translation vector.
            - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape BxSx3x3,
              or None if build_intrinsics is False. Defined in pixels, with format:
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              where fx, fy are focal lengths and (cx, cy) is the principal point,
              assumed to be at the center of the image (W/2, H/2).
    """

    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)
        extrinsics = torch.cat(
            [extrinsics, torch.zeros_like(extrinsics[..., :1, :])], dim=-2
        )
        extrinsics[..., 3, 3] = 1.0
        extrinsics = geometry_utils.invert_se3(extrinsics)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(
                pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device
            )
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
    else:
        raise NotImplementedError

    return extrinsics, intrinsics
