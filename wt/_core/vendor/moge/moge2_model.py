import functools
import itertools
import json
from collections.abc import Sequence
from numbers import Number
from pathlib import Path
from typing import Any, Literal

import safetensors
import structlog
import torch
from wt._core.vendor.moge import utils3d
from wt._core.vendor.moge.dinov2.vision_transformer import DinoVisionTransformer
from wt._core.vendor.moge.utils.geometry_torch import (
    normalized_view_plane_uv,
    recover_focal_shift,
)
from wt._core.vendor.moge.utils.wrap_module import wrap_dinov2_attention_with_sdpa
from torch import nn
from torch.nn import functional as F

from wt._core import file_io, path_utils

logger = structlog.get_logger(__name__)

MODEL_ZOO_DIR = None
"""Default download location for pre-trained MoGe-2 checkpoints.

See ``moge_model.MODEL_ZOO_DIR`` for context; this constant is only consulted
during *training* warm-start and is never read at inference time.  Set this
to a local directory if you intend to fine-tune from MoGe-2 weights.
"""


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = None,
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
        in_norm: Literal[
            "group_norm", "layer_norm", "instance_norm", "none"
        ] = "layer_norm",
        hidden_norm: Literal[
            "group_norm", "layer_norm", "instance_norm"
        ] = "group_norm",
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation == "relu":
            activation_cls = nn.ReLU
        elif activation == "leaky_relu":
            activation_cls = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        elif activation == "silu":
            activation_cls = nn.SiLU
        elif activation == "elu":
            activation_cls = nn.ELU
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.layers = nn.Sequential(
            nn.GroupNorm(in_channels // 32, in_channels)
            if in_norm == "group_norm"
            else nn.GroupNorm(1, in_channels)
            if in_norm == "layer_norm"
            else nn.InstanceNorm2d(in_channels)
            if in_norm == "instance_norm"
            else nn.Identity(),
            activation_cls(),
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
            nn.GroupNorm(hidden_channels // 32, hidden_channels)
            if hidden_norm == "group_norm"
            else nn.GroupNorm(1, hidden_channels)
            if hidden_norm == "layer_norm"
            else nn.InstanceNorm2d(hidden_channels)
            if hidden_norm == "instance_norm"
            else nn.Identity(),
            activation_cls(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
        )

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x


class DINOv2Encoder(nn.Module):
    """
    Wrapped DINOv2 encoder supporting gradient checkpointing.
    Input is RGB image in range [0, 1].
    """

    backbone: DinoVisionTransformer
    image_mean: torch.Tensor
    image_std: torch.Tensor
    dim_features: int

    def __init__(
        self,
        backbone: str,
        intermediate_layers: int | list[int],
        dim_out: int,
        **deprecated_kwargs,
    ):
        super().__init__()

        self.intermediate_layers = intermediate_layers

        # Load the backbone
        from wt._core.vendor.moge.dinov2 import hub

        self.hub_loader = getattr(hub, backbone)
        self.backbone_name = backbone
        self.backbone = self.hub_loader(pretrained=False)

        self.dim_features = self.backbone.blocks[0].attn.qkv.in_features
        self.num_features = (
            intermediate_layers
            if isinstance(intermediate_layers, int)
            else len(intermediate_layers)
        )

        self.output_projections = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.dim_features,
                    out_channels=dim_out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for _ in range(self.num_features)
            ]
        )

        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.backbone.onnx_compatible_mode = value

    def init_weights(self):
        pretrained_backbone_state_dict = self.hub_loader(pretrained=True).state_dict()
        self.backbone.load_state_dict(pretrained_backbone_state_dict)

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)

    def forward(
        self,
        image: torch.Tensor,
        token_rows: int | torch.LongTensor,
        token_cols: int | torch.LongTensor,
        return_class_token: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_14 = F.interpolate(
            image,
            (token_rows * 14, token_cols * 14),
            mode="bilinear",
            align_corners=False,
            antialias=not self.onnx_compatible_mode,
        )
        image_14 = (image_14 - self.image_mean) / self.image_std

        # Get intermediate layers from the backbone
        features = self.backbone.get_intermediate_layers(
            image_14, n=self.intermediate_layers, return_class_token=True
        )

        # Project features to the desired dimensionality
        x = torch.stack(
            [
                proj(
                    feat.permute(0, 2, 1)
                    .unflatten(2, (token_rows, token_cols))
                    .contiguous()
                )
                for proj, (feat, clstoken) in zip(self.output_projections, features)
            ],
            dim=1,
        ).sum(dim=1)

        if return_class_token:
            return x, features[-1][1]
        else:
            return x


class Resampler(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        type_: Literal[
            "pixel_shuffle",
            "nearest",
            "bilinear",
            "conv_transpose",
            "pixel_unshuffle",
            "avg_pool",
            "max_pool",
        ],
        scale_factor: int = 2,
    ):
        if type_ == "pixel_shuffle":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels * (scale_factor**2),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
            for i in range(1, scale_factor**2):
                self[0].weight.data[i :: scale_factor**2] = self[0].weight.data[
                    0 :: scale_factor**2
                ]
                self[0].bias.data[i :: scale_factor**2] = self[0].bias.data[
                    0 :: scale_factor**2
                ]
        elif type_ in ["nearest", "bilinear"]:
            nn.Sequential.__init__(
                self,
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode=type_,
                    align_corners=False if type_ == "bilinear" else None,
                ),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
        elif type_ == "conv_transpose":
            nn.Sequential.__init__(
                self,
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=scale_factor,
                    stride=scale_factor,
                ),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
            self[0].weight.data[:] = self[0].weight.data[:, :, :1, :1]
        elif type_ == "pixel_unshuffle":
            nn.Sequential.__init__(
                self,
                nn.PixelUnshuffle(scale_factor),
                nn.Conv2d(
                    in_channels * (scale_factor**2),
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
        elif type_ == "avg_pool":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        elif type_ == "max_pool":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        else:
            raise ValueError(f"Unsupported resampler type: {type_}")


class MLP(nn.Sequential):
    def __init__(self, dims: Sequence[int]):
        nn.Sequential.__init__(
            self,
            *itertools.chain(
                *[
                    (nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True))
                    for dim_in, dim_out in zip(dims[:-2], dims[1:-1])
                ]
            ),
            nn.Linear(dims[-2], dims[-1]),
        )


class ConvStack(nn.Module):
    def __init__(
        self,
        dim_in: list[int | None],
        dim_res_blocks: list[int],
        dim_out: list[int | None],
        resamplers: (
            Literal[
                "pixel_shuffle",
                "nearest",
                "bilinear",
                "conv_transpose",
                "pixel_unshuffle",
                "avg_pool",
                "max_pool",
            ]
            | list
        ),
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_in_norm: Literal[
            "layer_norm", "group_norm", "instance_norm", "none"
        ] = "layer_norm",
        res_block_hidden_norm: Literal[
            "layer_norm", "group_norm", "instance_norm", "none"
        ] = "group_norm",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
    ):
        super().__init__()
        self.input_blocks = nn.ModuleList(
            [
                nn.Conv2d(dim_in_, dim_res_block_, kernel_size=1, stride=1, padding=0)
                if dim_in_ is not None
                else nn.Identity()
                for dim_in_, dim_res_block_ in zip(
                    dim_in
                    if isinstance(dim_in, Sequence)
                    else itertools.repeat(dim_in),
                    dim_res_blocks,
                )
            ]
        )
        self.resamplers = nn.ModuleList(
            [
                Resampler(dim_prev, dim_succ, scale_factor=2, type_=resampler)
                for i, (dim_prev, dim_succ, resampler) in enumerate(
                    zip(
                        dim_res_blocks[:-1],
                        dim_res_blocks[1:],
                        resamplers
                        if isinstance(resamplers, Sequence)
                        else itertools.repeat(resamplers),
                    )
                )
            ]
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *(
                        ResidualConvBlock(
                            dim_res_block_,
                            dim_res_block_,
                            dim_times_res_block_hidden * dim_res_block_,
                            activation=activation,
                            in_norm=res_block_in_norm,
                            hidden_norm=res_block_hidden_norm,
                        )
                        for _ in range(
                            num_res_blocks[i]
                            if isinstance(num_res_blocks, list)
                            else num_res_blocks
                        )
                    )
                )
                for i, dim_res_block_ in enumerate(dim_res_blocks)
            ]
        )
        self.output_blocks = nn.ModuleList(
            [
                nn.Conv2d(dim_res_block_, dim_out_, kernel_size=1, stride=1, padding=0)
                if dim_out_ is not None
                else nn.Identity()
                for dim_out_, dim_res_block_ in zip(
                    dim_out
                    if isinstance(dim_out, Sequence)
                    else itertools.repeat(dim_out),
                    dim_res_blocks,
                )
            ]
        )

    def forward(self, in_features: list[torch.Tensor]):
        out_features = []
        for i in range(len(self.res_blocks)):
            feature = self.input_blocks[i](in_features[i])
            if i == 0:
                x = feature
            elif feature is not None:
                x = x + feature
            x = self.res_blocks[i](x)
            out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out_features


class MoGeModel(nn.Module):
    encoder: DINOv2Encoder
    neck: ConvStack
    points_head: ConvStack
    mask_head: ConvStack
    scale_head: MLP
    onnx_compatible_mode: bool

    def __init__(
        self,
        encoder: dict[str, Any],
        neck: dict[str, Any],
        points_head: dict[str, Any] = None,
        mask_head: dict[str, Any] = None,
        normal_head: dict[str, Any] = None,
        scale_head: dict[str, Any] = None,
        remap_output: Literal["linear", "sinh", "exp", "sinh_exp"] = "linear",
        num_tokens_range: list[int] | None = None,
        weights_dtype: torch.dtype = torch.float16,
        torch_compile: bool = False,
        use_torch_focal_shift_solver: bool = False,
        **deprecated_kwargs,
    ):
        """
        Args:
            weights_dtype: Target model weights dtype. Defaults to `torch.float16`
                on CUDA and falls back to `torch.float32` when CUDA is unavailable.
            torch_compile: If True, apply `torch.compile` on the forward pass. MoGe
                works best when images are kept at their native aspect ratio, so
                compilation is only recommended for fixed-size images (static shapes).
        """
        super().__init__()
        if deprecated_kwargs:
            logger.warn(
                "The following deprecated arguments are ignored", **deprecated_kwargs
            )

        self.remap_output = remap_output
        self.num_tokens_range = (
            num_tokens_range if num_tokens_range is not None else [1200, 3600]
        )
        self.use_torch_focal_shift_solver = use_torch_focal_shift_solver
        # Fall back to FP32 if CUDA is not available.
        self.weights_dtype = (
            weights_dtype if torch.cuda.is_available() else torch.float32
        )

        self.encoder = DINOv2Encoder(**encoder)
        self.neck = ConvStack(**neck)
        if points_head is not None:
            self.points_head = ConvStack(**points_head)
        if mask_head is not None:
            self.mask_head = ConvStack(**mask_head)
        if normal_head is not None:
            self.normal_head = ConvStack(**normal_head)
        if scale_head is not None:
            self.scale_head = MLP(**scale_head)

        self.to(dtype=self.weights_dtype)
        if torch_compile:
            self.compile()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def onnx_compatible_mode(self) -> bool:
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.encoder.onnx_compatible_mode = value

    @classmethod
    def from_pretrained(
        cls,
        model_name: str | None = None,
        model_config: str | Path | None = None,
        model_weights: str | Path | None = None,
        model_kwargs: dict | None = None,
        weights_dtype: torch.dtype = torch.float16,
        torch_compile: bool = False,
        use_torch_focal_shift_solver: bool = False,
    ) -> "MoGeModel":
        """
        Creates a MoGeModel instance from pretrained config and weights files.

        This method loads a MoGeModel from separate config and weights files. The
        config file contains model architecture parameters as JSON, while the weights
        file contains the pretrained parameters in safetensors format.

        Either model_name must be specified to use a predefined model architecture,
        or both `model_config` and `model_weights` must be provided to load a
        custom model.

        Args:
            model_name: Predefined model name, this could be either `moge-2-vitl`
                or `moge-2-vitl-normal`.
            model_config: Path to the model config JSON file.
            model_weights: Path to safetensors file of model weights.
            model_kwargs: Additional keyword arguments to override the parameters
                in the `model_config`.
            weights_dtype: Target weights dtype. Defaults to `torch.float16`.
                Falls back to `torch.float32` when CUDA is unavailable.

        Returns:
            MoGeModel instance initialized with the pretrained config and weights.

        Raises:
            ValueError: If model_name is not valid, or if neither model_name nor both
                config/weights paths are specified.
        """
        if model_name is None:
            if model_config is None or model_weights is None:
                raise ValueError(
                    "Either model_name must be specified or both model_config "
                    "and model_weights must be provided."
                )
        elif model_config is not None or model_weights is not None:
            raise ValueError(
                "Cannot specify both model_name and model_config/weights. "
                "Use one or the other."
            )

        if model_name is not None:
            if model_name not in ["moge-2-vitl", "moge-2-vitl-normal"]:
                raise ValueError(
                    f"Unsupported model: {model_name}. "
                    f"Supported models: 'moge-2-vitl', 'moge-2-vitl-normal'"
                )
            else:
                zoo_dir = path_utils.ensure_pathlike(MODEL_ZOO_DIR)
                model_config = zoo_dir / f"{model_name}-config.json"
                model_weights = zoo_dir / f"{model_name}.safetensors"

        model_config = path_utils.ensure_pathlike(model_config)
        model_weights = path_utils.ensure_pathlike(model_weights)

        with file_io.cache_remote_path(model_config).open("r") as f:
            model_config = json.load(f)
            if model_kwargs is not None:
                model_config.update(model_kwargs)

        local_weights_path = file_io.cache_remote_path(model_weights)
        with safetensors.safe_open(local_weights_path, framework="pt") as weights:
            state_dict = {k: weights.get_tensor(k) for k in weights.keys()}  # noqa: SIM118

        model = cls(
            **model_config,
            weights_dtype=weights_dtype,
            torch_compile=torch_compile,
            use_torch_focal_shift_solver=use_torch_focal_shift_solver,
        )
        model.load_state_dict(state_dict)
        return model

    def init_weights(self):
        self.encoder.init_weights()

    def enable_pytorch_native_sdpa(self):
        self.encoder.enable_pytorch_native_sdpa()

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.remap_output == "linear":
            pass
        elif self.remap_output == "sinh":
            points = torch.sinh(points)
        elif self.remap_output == "exp":
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == "sinh_exp":
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def forward(self, image: torch.Tensor, num_tokens: int) -> dict[str, torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        base_h, base_w = (
            int((num_tokens / aspect_ratio) ** 0.5),
            int((num_tokens * aspect_ratio) ** 0.5),
        )
        num_tokens = base_h * base_w

        # Backbones encoding
        features, cls_token = self.encoder(
            image, base_h, base_w, return_class_token=True
        )
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2**level,
                height=base_h * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck
        features = self.neck(features)

        # Heads decoding
        points, normal, mask = (
            getattr(self, head)(features)[-1] if hasattr(self, head) else None
            for head in ["points_head", "normal_head", "mask_head"]
        )
        metric_scale = (
            self.scale_head(cls_token) if hasattr(self, "scale_head") else None
        )

        # Resize
        points, normal, mask = (
            F.interpolate(
                v, (img_h, img_w), mode="bilinear", align_corners=False, antialias=False
            )
            if v is not None
            else None
            for v in [points, normal, mask]
        )

        # Remap output
        if points is not None:
            points = points.permute(0, 2, 3, 1)
            points = self._remap_points(
                points
            )  # slightly improves the performance in case of very large output values
        if normal is not None:
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        return_dict = {
            "points": points,
            "normal": normal,
            "mask": mask,
            "metric_scale": metric_scale,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: Literal[False, True, "blend"] = True,
        fov_x: Number | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        User-friendly inference function for monocular depth and geometry
        estimation.

        Args:
            image: Input image tensor of shape `(batch_size, 3, height, width)`
                or `(3, height, width)`. If a 3D tensor is provided, a batch
                dimension will be added automatically.
            num_tokens: Number of base ViT tokens to use for inference. More
                tokens result in higher accuracy and finer details but slower
                inference. Suggested range: 1200-2500. If None, automatically
                determined based on `resolution_level`.
            resolution_level: Resolution level from 0 to 9 used to determine
                `num_tokens` when it is None. Higher values use more tokens.
            force_projection: If True, the output point map will be computed
                using the actual depth map.
            apply_mask: If True, the output point map will be masked using the
                predicted mask. Can also be "blend" for soft masking.
            fov_x: Horizontal camera field of view in degrees. If None, it will
                be inferred from the predicted point map.

        Returns:
            Dictionary containing the following keys:
            - `points`: Output tensor of shape `(batch_size, height, width, 3)`
                or `(height, width, 3)` containing 3D point coordinates.
            - `depth`: Tensor of shape `(batch_size, height, width)` or
                `(height, width)` containing the depth map.
            - `intrinsics`: Tensor of shape `(batch_size, 3, 3)` or `(3, 3)`
                containing the camera intrinsics matrix.
        """
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.weights_dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(
                min_tokens + (resolution_level / 9) * (max_tokens - min_tokens)
            )

        output = self.forward(image, num_tokens=num_tokens)
        points, normal, mask, metric_scale = (
            output.get(k, None) for k in ["points", "normal", "mask", "metric_scale"]
        )

        # Always process the output in fp32 precision
        points, normal, mask, metric_scale, fov_x = map(
            lambda x: x.float() if isinstance(x, torch.Tensor) else x,
            [points, normal, mask, metric_scale, fov_x],
        )
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            if mask is not None:
                mask_binary = mask > 0.5
            else:
                mask_binary = None

            if points is not None:
                focal_shift_downsample_size = (
                    None if self.use_torch_focal_shift_solver else (64, 64)
                )
                # Convert affine point map to camera-space. Recover depth and
                # intrinsics from point map. NOTE: Focal here is the focal length
                # relative to half the image diagonal.
                if fov_x is None:
                    # Recover focal and shift from predicted point map.
                    focal, shift = recover_focal_shift(
                        points,
                        mask_binary,
                        downsample_size=focal_shift_downsample_size,
                        use_torch_solver=self.use_torch_focal_shift_solver,
                    )
                else:
                    # Focal is known, recover shift only.
                    fov_x_tensor = torch.as_tensor(
                        fov_x, device=points.device, dtype=points.dtype
                    )
                    half_fov_x = fov_x_tensor / 2
                    half_fov_x_rad = torch.deg2rad(half_fov_x)
                    tan_half_fov = torch.tan(half_fov_x_rad)
                    diagonal_factor = (1 + aspect_ratio**2) ** 0.5
                    focal = aspect_ratio / diagonal_factor / tan_half_fov
                    if focal.ndim == 0:
                        focal = focal[None].expand(points.shape[0])
                    _, shift = recover_focal_shift(
                        points,
                        mask_binary,
                        focal=focal,
                        downsample_size=focal_shift_downsample_size,
                        use_torch_solver=self.use_torch_focal_shift_solver,
                    )
                fx, fy = (
                    focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio,
                    focal / 2 * (1 + aspect_ratio**2) ** 0.5,
                )
                intrinsics = utils3d.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
                points[..., 2] += shift[..., None, None]
                if mask_binary is not None:
                    # Filter out points with negative depth values (this should not
                    # occur in practice).
                    mask_binary &= points[..., 2] > 0
                depth = points[..., 2].clone()
            else:
                depth, intrinsics = None, None

            # If projection constraint is forced, recompute the point map using the
            # actual depth map & intrinsics
            if force_projection and depth is not None:
                points = utils3d.unproject_cv(
                    utils3d.image_uv(
                        width=depth.shape[-1],
                        height=depth.shape[-2],
                        dtype=depth.dtype,
                        device=depth.device,
                    ),
                    depth,
                    extrinsics=None,
                    intrinsics=intrinsics[..., None, :, :],
                )

            # Apply metric scale
            if metric_scale is not None:
                if points is not None:
                    points *= metric_scale[:, None, None, None]
                if depth is not None:
                    depth *= metric_scale[:, None, None]

            if apply_mask and mask_binary is not None:
                if points is not None:
                    points = torch.where(mask_binary[..., None], points, torch.inf)
                if depth is not None:
                    depth = torch.where(mask_binary, depth, torch.inf)
                if normal is not None:
                    normal = torch.where(
                        mask_binary[..., None], normal, torch.zeros_like(normal)
                    )

        return_dict = {
            "points": points,
            "intrinsics": intrinsics,
            "depth": depth,
            "mask": mask_binary,
            "mask_confidence": mask,  # Soft confidence mask in [0, 1]
            "normal": normal,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

        return return_dict
