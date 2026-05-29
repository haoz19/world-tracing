from contextlib import nullcontext
from copy import deepcopy
from typing import Literal, Mapping

import einops
import structlog
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from wt._core.vendor.moge.moge_model import MoGeModel
from torch import Tensor
from torch.nn import functional as F

from wt._core import camera as _camera
from wt._core.components import nn_layers
from wt._core.diffusion import constants
from wt._core.models.wan_video import layers as wan_video_layers
from wt._core.splat.utils import embedding
from wt._core.utils import torch_utils
from wt._core.arch.models import blocks, head_utils, model_utils, patchhead_utils
from wt._core.arch.models import config as model_config
from wt._core.arch.utils import geometry_utils

logger = structlog.get_logger(__name__)


class MultilayerBackbone(nn.Module):
    """Transformer backbone for multilayer geometry diffusion.

    A dust3r / vggt style encoder-decoder used as the denoising network
    inside :class:`wt.model.MultilayerXYZModel`.
    """

    def __init__(
        self,
        # general args
        model_type: Literal[
            "regression", "diffusion", "query_diffusion"
        ] = "regression",
        img_fusion_mode: Literal["concat", "concat_nostd", "group_concat"] = "concat",
        fuse_raw_rgb: bool = False,
        encoder_model: Literal["moge", "null"] = "moge",
        patch_size: int = 14,
        num_decoder_blocks: int = 12,
        num_decoder_heads: int = 16,
        decoder_embed_dim: int = 1024,
        img_token_scale: float = 0.3,
        num_global_encoder_blocks: int = 0,
        alternate_attention: bool = True,
        num_head_blks: int = 4,
        use_gated_attn: bool = False,
        head_mode: Literal["linear", "perceiver", "patchnerf", "conv"] = "linear",
        output_mode: Literal["pointmaps", "depth"] = "pointmaps",
        use_camera_head: bool = False,
        use_scale_head: bool = False,
        use_camera_scale_normalizer: bool = True,
        use_raymap_dropout: bool = False,
        head_merge_ratio: float = 0.0,
        head_subsample_ratio: float = 0.2,
        output_scale_modulation: float = 0.2,
        norm_layer=nn.LayerNorm,
        ls_init_values: float | None = None,
        positional_encoding: Literal["rope", "pope", "none"] = "rope",
        use_raymap: bool = False,
        raymap_version: Literal["v1", "v2"] = "v1",
        use_dense_raycond: bool = False,
        inference_mode: bool = False,
        freeze_encoder: bool = True,
        use_activation_checkpoint: bool = False,
        activation_checkpoint_ratio: float = 1.0,
        # diffusion args
        time_fusion_mode: Literal["adaln"] = "adaln",
        t_embed_channels: int = 256,
        noise_channel: int = 1024,
        noise_patchify_size: int = 1,
        use_x0_prediction: bool = False,
        # query diffusion args
        num_query_blocks: int = 4,
        num_denoising_global_tokens: int = 1024,
        query_chunk_size: int = 65536,
        use_rgb_query: bool = False,
        use_noise_patch_query: bool = False,
        rgb_query_patch_size: int = 9,
        use_noise_free_decoding: bool = False,
        use_diffusion_query: bool = True,
        use_noise_embed: bool = False,
        use_feat_interpolation: bool = False,
        use_noise_interpolation: bool = False,
    ):
        super().__init__()
        # general args
        self.model_type = model_type
        self.img_fusion_mode = img_fusion_mode
        self.fuse_raw_rgb = fuse_raw_rgb
        self.encoder_model = encoder_model
        self.patch_size = patch_size
        self.num_decoder_blocks = num_decoder_blocks
        self.num_decoder_heads = num_decoder_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.img_token_scale = img_token_scale  # only used in diffusion model
        self.num_global_encoder_blocks = num_global_encoder_blocks
        self.alternate_attention = alternate_attention
        self.num_head_blks = num_head_blks
        self.use_gated_attn = use_gated_attn
        self.head_mode = head_mode
        self.output_mode = output_mode
        self.use_camera_head = use_camera_head
        self.use_scale_head = use_scale_head
        self.use_camera_scale_normalizer = use_camera_scale_normalizer
        self.use_raymap_dropout = use_raymap_dropout
        self.head_merge_ratio = head_merge_ratio
        self.head_subsample_ratio = head_subsample_ratio
        self.output_scale_modulation = output_scale_modulation
        self.norm_layer = norm_layer
        self.ls_init_values = ls_init_values
        self.positional_encoding = positional_encoding
        self.use_raymap = use_raymap
        self.raymap_version = raymap_version
        self.use_dense_raycond = use_dense_raycond
        self.inference_mode = inference_mode
        self.freeze_encoder = freeze_encoder
        self.use_activation_checkpoint = use_activation_checkpoint
        self.activation_checkpoint_ratio = activation_checkpoint_ratio
        # diffusion args
        self.time_fusion_mode = time_fusion_mode
        self.t_embed_channels = t_embed_channels
        self.noise_channel = noise_channel
        self.noise_patchify_size = noise_patchify_size
        self.use_x0_prediction = use_x0_prediction
        # query diffusion args
        self.num_query_blocks = num_query_blocks
        self.num_denoising_global_tokens = num_denoising_global_tokens
        self.query_chunk_size = query_chunk_size
        self.use_rgb_query = use_rgb_query
        self.use_noise_patch_query = use_noise_patch_query
        self.rgb_query_patch_size = rgb_query_patch_size
        self.use_noise_free_decoding = use_noise_free_decoding
        self.use_diffusion_query = use_diffusion_query
        self.use_noise_embed = use_noise_embed
        self.use_feat_interpolation = use_feat_interpolation
        self.use_noise_interpolation = use_noise_interpolation
        assert (
            self.decoder_embed_dim % self.num_decoder_heads == 0
        ), "decoder_embed_dim must be divisible by num_decoder_heads"

        # positional encoding
        self.build_positional_encoding()

        # encoder
        self.build_encoder()

        # decoder
        self.build_decoder()

        # output heads
        self.build_head()

        # freeze some part of the model
        self.selective_freeze()

    def build_positional_encoding(self):
        self.position_getter = model_utils.PositionGetter()
        if self.positional_encoding == "rope":
            self.rope = blocks.RoPE2D(freq=100.0)
        elif self.positional_encoding == "pope":
            self.rope = blocks.PoPEND(freq=100.0)
        elif self.positional_encoding == "none":
            self.rope = None
        else:
            raise ValueError(
                f"Invalid positional encoding model: {self.positional_encoding}"
            )

        if self.model_type == "regression":
            self.time_embedding = None
            self.time_projection = None
        else:
            self.build_diffusion_projections()

    def build_diffusion_projections(self):
        """
        Build the projections for diffusion model.
        """
        self.time_embedding = nn.Sequential(
            nn_layers.Linear(self.t_embed_channels, self.decoder_embed_dim),
            nn.SiLU(),
            nn_layers.Linear(self.decoder_embed_dim, self.decoder_embed_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn_layers.Linear(self.decoder_embed_dim, self.decoder_embed_dim * 6),
        )
        if self.model_type == "diffusion":
            raw_input_dim = self.noise_channel * self.noise_patchify_size**2
        elif self.model_type == "query_diffusion":
            if self.use_noise_interpolation:
                raw_input_dim = self.noise_channel * self.patch_size**2
            else:
                raw_input_dim = self.noise_channel
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        if self.fuse_raw_rgb:
            raw_input_dim += 3 * self.patch_size**2
        fused_dim = self.decoder_embed_dim + raw_input_dim

        if self.img_fusion_mode == "concat" or self.img_fusion_mode == "concat_nostd":
            self.pixel_projection = nn.Sequential(
                nn_layers.Linear(fused_dim, self.decoder_embed_dim),
                nn.SiLU(),
                nn_layers.Linear(self.decoder_embed_dim, self.decoder_embed_dim),
            )
        elif self.img_fusion_mode == "group_concat":
            assert (
                self.decoder_embed_dim % 2 == 0
            ), "decoder_embed_dim must be even for group_concat"
            self.noise_projection = nn.Sequential(
                nn_layers.Linear(raw_input_dim, self.decoder_embed_dim // 2),
            )
            self.feature_projection = nn.Sequential(
                nn_layers.Linear(self.decoder_embed_dim, self.decoder_embed_dim // 2),
            )
        else:
            raise ValueError(f"Invalid image fusion mode: {self.img_fusion_mode}")

        # Following WAN to initialize the time embedding.
        for m in self.time_embedding.modules():
            if isinstance(m, nn_layers.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Following WAN to initialize the time projection.
        for m in self.time_projection.modules():
            if isinstance(m, nn_layers.Linear):
                wan_video_layers.wan_init_linear(m)

    def build_encoder(self):
        if self.encoder_model == "moge":
            if self.inference_mode:
                moge_model = MoGeModel(**model_config.MOGE_CONFIG_FA3)
            else:
                moge_model = MoGeModel.from_pretrained(
                    model_name="vitl", model_kwargs={"use_fa3": True}
                )
            self.encoder = deepcopy(moge_model.backbone)
            self.encoder.intermediate_layers = 4
            self.encoder.register_buffer("image_mean", moge_model.image_mean)
            self.encoder.register_buffer("image_std", moge_model.image_std)
            del moge_model
            self.encoder.final_project = nn_layers.Linear(4096, self.decoder_embed_dim)
        elif self.encoder_model == "null":
            self.encoder = nn.Identity()
        else:
            raise NotImplementedError(
                f"Encoder model {self.encoder_model} not implemented"
            )
        if self.num_global_encoder_blocks > 0:
            blk = blocks.DecoderBlockSA(
                self.decoder_embed_dim,
                self.num_decoder_heads,
                norm_layer=self.norm_layer,
                use_qk_norm=True,
                init_values=self.ls_init_values,
                rope=self.rope,
                use_gated_attn=self.use_gated_attn,
            )
            blks = [deepcopy(blk) for _ in range(self.num_global_encoder_blocks)]
            if self.use_activation_checkpoint:
                checkpoint_end_idx = int(
                    self.activation_checkpoint_ratio * self.num_global_encoder_blocks
                )
                for idx in range(checkpoint_end_idx):
                    blks[idx] = model_utils.apply_ac_module(blks[idx])
                logger.info(f"AC enabled in {checkpoint_end_idx} global encoder blks")
            self.global_encoder_blocks = nn.ModuleList(blks)
        else:
            self.global_encoder_blocks = nn.Identity()

    def build_decoder(self):
        if self.use_raymap:
            if self.raymap_version == "v1":
                raymap_project = model_utils.RaymapProjector(self.decoder_embed_dim)
            elif self.raymap_version == "v2":
                raymap_project = model_utils.RaymapProjectorV2(self.decoder_embed_dim)
            else:
                raise ValueError(
                    f"raymap_version must be 'v1' or 'v2', got {self.raymap_version}"
                )
            if self.use_dense_raycond:
                self.raymap_project = nn.ModuleList(
                    [deepcopy(raymap_project) for _ in range(self.num_decoder_blocks)]
                )
            else:
                self.raymap_project = raymap_project
            if self.num_global_encoder_blocks > 0:
                self.raymap_project_encoder_blks = nn.ModuleList(
                    [
                        deepcopy(raymap_project)
                        for _ in range(self.num_global_encoder_blocks)
                    ]
                )
        else:
            self.raymap_project = None
        self.camera_scale_normalizer = model_utils.CameraScaleNormalizerV2()
        if self.model_type == "regression":
            blk_cls = blocks.DecoderBlockSA
        else:
            blk_cls = blocks.DecoderBlockDiT
        blk = blk_cls(
            self.decoder_embed_dim,
            self.num_decoder_heads,
            norm_layer=self.norm_layer,
            use_qk_norm=True,
            init_values=self.ls_init_values,
            rope=self.rope,
            use_gated_attn=self.use_gated_attn,
        )
        self.decoder_blocks = [deepcopy(blk) for _ in range(self.num_decoder_blocks)]

        if self.use_activation_checkpoint:
            checkpoint_end_idx = int(
                self.activation_checkpoint_ratio * self.num_decoder_blocks
            )
            for idx in range(checkpoint_end_idx):
                self.decoder_blocks[idx] = model_utils.apply_ac_module(
                    self.decoder_blocks[idx]
                )
            logger.info(
                f"Activation checkpointing enabled in {checkpoint_end_idx} decoder blks"
            )
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

    def build_head(self):
        # pixel-aligned heads
        if self.model_type == "regression":
            self.build_regression_head()
        elif self.model_type == "diffusion":
            self.build_diffusion_head()
        elif self.model_type == "query_diffusion":
            self.build_diffusion_head()
            self.build_query_diffusion_head()
        else:
            raise ValueError(
                f"model_type must be 'regression' or 'diffusion', got {self.model_type}"
            )
        # global or per view heads
        self.build_head_transformer()
        self.build_camera_head()
        self.build_scale_head()

    def build_regression_head(self):
        manual_scale1 = 0.2
        manual_scale2 = self.output_scale_modulation
        if self.output_mode == "pointmaps":
            head_dict = {
                "pointmap": (
                    [3, 3, 1, 1],
                    [
                        lambda x: manual_scale1 * x,
                        lambda x: manual_scale1 * x,
                        lambda x: F.softplus(manual_scale2 * x),
                        lambda x: F.softplus(manual_scale2 * x),
                    ],
                    ["pointmap_g", "pointmap_l", "conf_g", "conf_l"],
                )
            }
        elif self.output_mode == "depth":
            head_dict = {
                "depth": (
                    [1],
                    [lambda x: (manual_scale1 * x).exp()],
                    ["depth"],
                )
            }
        else:
            raise ValueError(
                f"output_mode must be 'pointmaps' or 'depth', got {self.output_mode}"
            )

        self.head_dict = head_dict

        self.heads = nn.ModuleDict()
        for key, (num_channels, activations, pred_names) in self.head_dict.items():
            self.heads[key] = self.get_head_module(
                num_channels, activations, pred_names
            )

    def build_diffusion_head(self):
        if self.model_type == "diffusion":
            if self.head_mode == "conv":
                output_dim = self.noise_channel
            elif self.head_mode == "linear":
                output_dim = self.noise_channel * self.noise_patchify_size**2
            else:
                raise ValueError(f"Invalid head mode: {self.head_mode}")
        elif self.model_type == "query_diffusion":
            output_dim = self.noise_channel
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        if self.head_mode == "linear":
            self.latents_projection = nn.Sequential(
                nn.SiLU(), nn_layers.Linear(self.decoder_embed_dim, output_dim)
            )
        else:
            assert (
                self.model_type != "query_diffusion"
            ), "Query diffusion is only supported for linear head mode"
            self.latents_projection = self.get_head_module(
                [output_dim], [nn.Identity()], ["v_t"]
            )

    def get_head_module(self, num_channels, activations, pred_names):
        head_kwargs = {
            "dim_in": self.decoder_embed_dim,
            "dim_out": num_channels,
            "activations": activations,
            "pred_names": pred_names,
            "patch_size": self.patch_size,
        }
        if self.head_mode == "conv":
            head = patchhead_utils.ConvHead(**head_kwargs)
        else:
            head_kwargs.update(
                {
                    "num_head_blks": self.num_head_blks,
                    "norm_layer": self.norm_layer,
                    "head_mode": self.head_mode,
                    "head_merge_ratio": self.head_merge_ratio,
                    "head_subsample_ratio": self.head_subsample_ratio,
                }
            )
            head = patchhead_utils.PredictionHead(**head_kwargs)
        return head

    def build_query_diffusion_head(
        self,
        n_freqs: int = 8,
        init_values: float = 0.1,
    ):
        # projection and fusion layers
        if self.use_raymap:
            query_dim = 6
        else:
            query_dim = 3
        self.query_embed = embedding.PosEmbedding(
            in_channels=query_dim, n_freqs=n_freqs, logscale=True
        )
        self.query_token_projection = nn_layers.Linear(
            self.query_embed.out_channels, self.decoder_embed_dim
        )
        if self.use_rgb_query:
            self.query_rgb_token_projection = nn_layers.Linear(
                3 * self.rgb_query_patch_size**2, self.decoder_embed_dim
            )
            self.query_rgb_fusion = nn.Sequential(
                nn_layers.Linear(self.decoder_embed_dim * 2, self.decoder_embed_dim),
                nn.GELU(),
                nn_layers.Linear(self.decoder_embed_dim, self.decoder_embed_dim),
            )
        if self.use_noise_embed:
            self.noise_embed = embedding.PosEmbedding(
                in_channels=self.noise_channel, n_freqs=n_freqs, logscale=True
            )
            noise_embed_channel = self.noise_embed.out_channels
        else:
            self.noise_embed = nn.Identity()
            noise_embed_channel = self.noise_channel

        self.noise_query_fusion = nn.Sequential(
            nn_layers.Linear(self.decoder_embed_dim * 2, self.decoder_embed_dim),
            nn.GELU(),
            nn_layers.Linear(self.decoder_embed_dim, self.decoder_embed_dim),
        )
        post_proj_noise_dim = self.decoder_embed_dim
        if self.use_noise_patch_query:
            pre_proj_noise_dim = noise_embed_channel * self.rgb_query_patch_size**2
        else:
            pre_proj_noise_dim = noise_embed_channel
        self.query_noise_projection = nn_layers.Linear(
            pre_proj_noise_dim, post_proj_noise_dim
        )
        if self.use_feat_interpolation:
            # use MLPs
            if self.use_diffusion_query:
                blk_cls = blocks.MLPBlockDiT
                self.query_time_projection = blocks.TimestepProjection(
                    self.t_embed_channels, self.decoder_embed_dim, 3
                )
            else:
                blk_cls = blocks.MLPBlock
                self.query_time_projection = None
            mlp_block = blk_cls(
                self.decoder_embed_dim,
                norm_layer=self.norm_layer,
                zero_last=True,
            )
            self.mlp_blocks = nn.ModuleList(
                [deepcopy(mlp_block) for _ in range(self.num_query_blocks)]
            )
        else:
            # cross attention between query and global memory
            if self.use_diffusion_query:
                blk_cls = blocks.DecoderBlockCADiT
                self.query_time_projection = blocks.TimestepProjection(
                    self.t_embed_channels, self.decoder_embed_dim, 6
                )
            else:
                blk_cls = blocks.DecoderBlockCA
                self.query_time_projection = None
            # use cross attention
            ca_blk_kwargs = {
                "dim": self.decoder_embed_dim,
                "num_heads": self.num_decoder_heads,
                "use_qk_norm": True,
                "init_values": init_values,
                "use_gated_attn": True,
                "zero_bias_last": True,
                "norm_layer": self.norm_layer,
            }
            ca_block = blk_cls(**ca_blk_kwargs)
            self.ca_blocks = nn.ModuleList(
                [deepcopy(ca_block) for _ in range(self.num_query_blocks)]
            )

    def build_head_transformer(self):
        if self.use_scale_head or self.use_camera_head:
            self.head_transformer = blocks.HeadTransformer(
                self.decoder_embed_dim,
                self.num_decoder_heads,
                norm_layer=self.norm_layer,
                init_values=self.ls_init_values,
                use_gated_attn=self.use_gated_attn,
                rope=self.rope,
                num_blocks=self.num_head_blks,
            )
            self.head_pre_projection = nn.Linear(
                2 * self.decoder_embed_dim, self.decoder_embed_dim
            )
            if self.use_raymap:
                self.head_raymap_project = model_utils.RaymapProjector(
                    self.decoder_embed_dim
                )

    def build_camera_head(self):
        if self.use_camera_head:
            self.camera_head = head_utils.MLPCameraHead(dim=self.decoder_embed_dim)
        else:
            self.camera_head = None

    def build_scale_head(self):
        if self.use_scale_head:
            self.scale_projection = nn.Sequential(
                nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim),
                nn.SiLU(),
                nn.Linear(self.decoder_embed_dim, 2),
            )
            model_utils.zero_module(self.scale_projection[-1])
        else:
            self.scale_projection = None

    def selective_freeze(self):
        freeze_list = []
        if self.freeze_encoder:
            freeze_list.append(self.encoder)
        model_utils.freeze_all_params(freeze_list)

    def get_yx_positions(
        self, img: Float[Tensor, "b v c h w"]
    ) -> Float[Tensor, "b v p 2"]:
        """
        Get the yx positions of the image patches.
        """
        device = img.device
        batch_size, num_views, patch_h, patch_w = (
            img.shape[0],
            img.shape[1],
            img.shape[-2] // self.patch_size,
            img.shape[-1] // self.patch_size,
        )
        xpos = self.position_getter(batch_size * num_views, patch_h, patch_w, device)
        xpos = xpos.reshape(batch_size, num_views, -1, 2)  # (y,x)
        return xpos

    def encode_conditioning(
        self,
        img: Float[Tensor, "b v c h w"],
        camera: _camera.Camera | None = None,
    ) -> tuple[Float[Tensor, "b v p 2"], Float[Tensor, "b v p 6"] | None]:
        """
        Encode the conditioning information.
        """
        self.camera_scale_normalizer.reset()  # reset scale to None
        xpos = self.get_yx_positions(img)
        if self.use_raymap:
            dev = xpos.device
            bs, nview, npatch, _ = xpos.shape
            if camera is None:
                raymap = torch.zeros(bs, nview, npatch, 6, device=dev)
            else:
                # normalize the camera translation
                updated_position = camera.position
                if self.use_camera_scale_normalizer:
                    updated_position = self.camera_scale_normalizer.apply(
                        updated_position
                    )
                camera = _camera.update_extrinsic(camera, position=updated_position)
                raymap = model_utils.compute_patch_raymap(camera, self.patch_size)
                raymap = einops.rearrange(raymap, "b v h w ... -> b v (h w) ...")

            # raymap dropout
            if self.use_raymap_dropout and self.training:
                raymap_mask = model_utils.create_raymap_dropout_mask(bs, nview, dev)
                raymap = raymap * raymap_mask
        else:
            raymap = None
        return xpos, raymap

    @staticmethod
    def _get_norm_buffers(encoder: nn.Module, img: Tensor) -> tuple[Tensor, Tensor]:
        """
        mainly to make sure dtype of img and mean/std are consistent
        """
        mean = encoder.image_mean.to(dtype=img.dtype, device=img.device)
        std = encoder.image_std.to(dtype=img.dtype, device=img.device)
        return mean, std

    def encode_image_moge(
        self, img: Float[Tensor, "b v c h w"]
    ) -> Float[Tensor, "b v p d"]:
        """
        Encode the image into tokens.

        Args:
            img: images with range [0, 1]
        """
        batch_size, num_view, patch_h, patch_w = (
            img.shape[0],
            img.shape[1],
            img.shape[3] // self.patch_size,
            img.shape[4] // self.patch_size,
        )
        mean, std = self._get_norm_buffers(self.encoder, img)
        x_in = (img - mean) / std

        x_in = einops.rearrange(x_in, "b v c h w -> (b v) c h w")

        # resize input image to match moge weights
        resized_h = patch_h * model_config.MOGE_PATCH_SIZE
        resized_w = patch_w * model_config.MOGE_PATCH_SIZE
        x_in = F.interpolate(x_in, (resized_h, resized_w), mode="bilinear")

        x_out = self.encoder.get_intermediate_layers(
            x_in, self.encoder.intermediate_layers
        )
        x_out = torch.cat(x_out, -1)
        x_out = x_out.view(batch_size, num_view, patch_h * patch_w, x_out.shape[-1])
        x_out = self.encoder.final_project(x_out)
        return x_out

    def encode_image(self, img: Float[Tensor, "b v c h w"]) -> Float[Tensor, "b v p d"]:
        """
        Encode the image into tokens.

        Args:
            img: images with range [0, 1]
        """
        ctx = torch.no_grad() if self.freeze_encoder else nullcontext()

        if self.encoder_model == "moge":
            if self.freeze_encoder:
                autocast_dtype = torch.float16
            else:
                autocast_dtype = torch.bfloat16
            type_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype)
            with ctx, type_ctx:
                tokens = self.encode_image_moge(img)
            tokens = tokens.to(torch.float32)
        elif self.encoder_model == "null":
            batch_size, num_view, _, height, width = img.shape
            num_patches = height // self.patch_size * width // self.patch_size
            device = img.device
            tokens = torch.zeros(
                batch_size, num_view, num_patches, self.decoder_embed_dim, device=device
            )
        else:
            raise NotImplementedError(
                f"Encoder model {self.encoder_model} not implemented"
            )
        return tokens

    def insert_global_tokens(
        self,
        tokens: Float[Tensor, "b s1 d"],
        xpos: Float[Tensor, "b s1 2"],
        global_tokens: Float[Tensor, "b s2 d"] | None,
    ) -> tuple[Float[Tensor, "b s1+s2 d"], Float[Tensor, "b s1+s2 2"]]:
        """
        Insert global tokens to the begining of each frame token sequence.
        [global_tokens, patch_tokens]
        """
        tokens_g = torch.cat([global_tokens, tokens], dim=1)

        # set pos to 0 for special tokens
        xpos_g = xpos + 1
        pos_special = torch.zeros_like(xpos[:, 0])
        pos_special = einops.repeat(
            pos_special, "b ... -> b p ...", p=global_tokens.shape[1]
        )
        xpos_g = torch.cat([pos_special, xpos_g], dim=1)
        return tokens_g, xpos_g

    def remove_global_tokens(
        self,
        tokens: Float[Tensor, "b s1+s2 d"],
        xpos: Float[Tensor, "b s1+s2 2"],
        num_global_tokens: int,
    ) -> tuple[
        Float[Tensor, "b s1 d"],
        Float[Tensor, "b s1 2"],
        Float[Tensor, "b s2 d"],
    ]:
        """
        Remove global tokens from the tokens.
        [global_tokens, patch_tokens]
        """
        xpos = xpos - 1
        return (
            tokens[:, num_global_tokens:],
            xpos[:, num_global_tokens:],
            tokens[:, :num_global_tokens],
        )

    def fuse_timestep_cond(
        self,
        tokens: Float[Tensor, "b v p d"],
        xpos: Float[Tensor, "b v p 2"],
        timesteps: Float[Tensor, "b"] | None,
    ) -> tuple[
        Float[Tensor, "b v p d"], Float[Tensor, "b v p 2"], Float[Tensor, "b 1 6 d"]
    ]:
        time_tokens = self.project_timestep(timesteps)
        adaln_input = torch.zeros_like(time_tokens)
        # b v 6+p d
        time_tokens = einops.repeat(
            time_tokens, "b 1 ... -> b v ...", v=tokens.shape[1]
        )
        tokens = torch.cat([time_tokens, tokens], dim=2)
        # add xpos
        xpos_time = torch.zeros_like(xpos[:, :, :1])
        xpos_time = einops.repeat(xpos_time, "b v p d -> b v (p k) d", k=6)
        xpos = torch.cat([xpos_time, xpos + 1], dim=2)
        return tokens, xpos, adaln_input

    def decode_to_output_tokens(
        self,
        tokens: Float[Tensor, "b v p d"],
        xpos: Float[Tensor, "b v p 2"],
        raymap: Float[Tensor, "b v p 6"] | None = None,
        global_tokens: Float[Tensor, "b s d"] | None = None,
        timesteps: Float[Tensor, "b"] | None = None,
    ) -> tuple[Float[Tensor, "b v p d"], Float[Tensor, "b s d"] | None]:
        # add timestep conditioning
        if self.model_type == "regression":
            adaln_input = None
        else:
            if self.time_fusion_mode == "adaln":
                adaln_input = self.project_timestep(timesteps)  # b 1 6 d
            else:
                raise ValueError(f"Unknown time fusion mode: {self.time_fusion_mode}")

        # decode
        bs, nview, npatch, dim = tokens.shape
        num_global_tokens = global_tokens.shape[1] if global_tokens is not None else 0
        for blk_idx, blk in enumerate(self.decoder_blocks):
            # fuse raymap tokens
            if self.use_raymap and (blk_idx == 0 or self.use_dense_raycond):
                assert raymap is not None, "raymap is required when use_raymap is True"
                tokens = tokens.reshape(bs, nview, npatch, dim)
                tokens = self.fuse_raymap_tokens(tokens, raymap, blk_idx)
            # alternate attention
            is_framewise_attn = self.alternate_attention and blk_idx % 2 == 0
            if is_framewise_attn:
                # frame-wise attention
                tokens = tokens.reshape(bs * nview, npatch, dim)
                xpos = xpos.reshape(bs * nview, npatch, 2)
            else:
                # global attention
                tokens = tokens.reshape(bs, nview * npatch, dim)
                xpos = xpos.reshape(bs, nview * npatch, 2)

            # insert global tokens only in global layers
            use_global_tokens = global_tokens is not None and not is_framewise_attn
            if use_global_tokens:
                mul_n = tokens.shape[0] // bs
                global_tok_rep = einops.repeat(
                    global_tokens, "b ... -> (b m) ...", m=mul_n
                )
                tokens, xpos = self.insert_global_tokens(tokens, xpos, global_tok_rep)

            # b s d
            if self.model_type == "diffusion" or self.model_type == "query_diffusion":
                mul_n = tokens.shape[0] // bs
                adaln_rep = einops.repeat(adaln_input, "b ... -> (b m) ...", m=mul_n)
                tokens = blk(tokens, xpos, adaln_rep)
            else:
                tokens = blk(tokens, xpos)

            # split to tokens and global tokens
            if use_global_tokens:
                tokens, xpos, global_tokens = self.remove_global_tokens(
                    tokens, xpos, num_global_tokens
                )
                global_tokens = einops.rearrange(
                    global_tokens, "(b m) ... -> b m ...", b=bs
                )[:, 0]

        tokens = tokens.reshape(bs, nview, npatch, dim)
        return tokens, global_tokens

    def fuse_raymap_tokens(
        self,
        tokens: Float[Tensor, "b v p d"],
        raymap: Float[Tensor, "b v p 6"],
        blk_idx: int,
        is_enc_blocks: bool = False,
    ) -> Float[Tensor, "b v p d"]:
        if is_enc_blocks:
            raymap_project = self.raymap_project_encoder_blks[blk_idx]
        else:
            if self.use_dense_raycond:
                raymap_project = self.raymap_project[blk_idx]
            else:
                raymap_project = self.raymap_project
        if self.use_dense_raycond:
            raymap_in = raymap.reshape(tokens.shape[:-1] + (6,))  # ..., 6
            tokens = raymap_project(tokens, raymap_in)
        else:
            tokens = raymap_project(tokens, raymap)
        return tokens

    def decode_to_structures(
        self,
        tokens: Float[Tensor, "b v p d"],
        xpos: Float[Tensor, "b v p 2"],
        img: Float[Tensor, "b v c h w"],
    ) -> dict:
        """
        Decode the output tokens to structures.
        Returns
        -------
        {
          "pointmap_g":(B, V, H, W, 3),
          "pointmap_l":(B, V, H, W, 3),
          "conf_g":  (B, V, H, W, 1),
          "conf_l":  (B, V, H, W, 1),
        }
        """
        bs, nview, _, height, width = img.shape
        tokens = einops.rearrange(tokens, "b v p ... -> b (v p) ...")
        xpos = einops.rearrange(xpos, "b v p ... -> b (v p) ...")
        output_dict = {}
        # pixel aligned structures
        for _, head in self.heads.items():
            if self.head_mode == "conv":
                pred_dict = head(tokens, nview, height, width)
            else:
                pred_dict = head(tokens, xpos, img)
            output_dict.update(pred_dict)

        if self.use_camera_head:
            # per view structures, e.g., camera pose
            camera_tokens = tokens.reshape(bs * nview, -1, tokens.shape[-1])
            camera_xpos = xpos.reshape(bs * nview, -1, 2)
            camera_tokens = self.head_transformer(camera_tokens, xpos=camera_xpos)
            camera_poses = self.camera_head(camera_tokens).reshape(-1, nview, 4, 4)
            # canonicalize
            camera_poses = geometry_utils.invert_se3(camera_poses[:, :1]) @ camera_poses
            # match the input camera scale
            camera_scale = self.camera_scale_normalizer.get_scale()
            if camera_scale is not None:
                camera_poses[..., :3, 3] *= camera_scale[:, None, None]
            output_dict["camera_poses"] = camera_poses
        return output_dict

    def project_timestep(self, t: Float[Tensor, "b"]) -> Float[Tensor, "b 1 6 d"]:
        """
        Project diffusion timestep to get timestep embedding and adaln input.
        Adapted from the Wan2.1 video base-model time-embedding head.
        """
        with torch_utils.maybe_autocast(t.device, torch.float32):
            t = t * model_config.DIFFUSION_TIMESTEP_SCALE
            t_embed = self.time_embedding(
                wan_video_layers.sinusoidal_embedding_1d(
                    self.t_embed_channels, t
                ).float()
            )
            t_embed = einops.rearrange(t_embed, "b d -> b 1 d")
            adaln_input = self.time_projection(t_embed).unflatten(
                -1, (6, self.decoder_embed_dim)
            )
            if t.device.type == "cuda":
                assert (
                    t_embed.dtype == torch.float32
                    and adaln_input.dtype == torch.float32
                )
        return adaln_input

    def _forward_regression(
        self,
        img: Float[Tensor, "b v c h w"],
        camera: _camera.Camera | None = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            img: images with range [0, 1]
        """
        # encode conditioning info
        xpos, raymap = self.encode_conditioning(img, camera)

        # encode images
        tokens = self.encode_image(img)

        # decode to output tokens
        tokens_out, _ = self.decode_to_output_tokens(tokens, xpos, raymap=raymap)

        # structure output
        output_dict = self.decode_to_structures(tokens_out, xpos, img)
        return output_dict

    def fuse_context_tokens(
        self,
        img_tokens: Float[Tensor, "b v p d"],
        psi_t: Float[Tensor, "b v p d"],
    ) -> Float[Tensor, "b v p d"]:
        if self.img_fusion_mode == "concat":
            # scale down image tokens magnitude
            img_tokens = model_utils.layer_norm_2d(img_tokens, dim=[1, 2, 3])
            img_tokens = img_tokens * self.img_token_scale
            # since output is pixeled aligned to input, we concat tokens with noise.
            tokens = torch.cat([img_tokens, psi_t], dim=-1)
            tokens = self.pixel_projection(tokens)
        elif self.img_fusion_mode == "concat_nostd":
            tokens = torch.cat([img_tokens, psi_t], dim=-1)
            tokens = self.pixel_projection(tokens)
        elif self.img_fusion_mode == "group_concat":
            # scale down image tokens magnitude
            img_tokens = model_utils.layer_norm_2d(img_tokens, dim=[1, 2, 3])
            img_tokens = img_tokens * self.img_token_scale
            # grouped concat
            feature_tokens = self.feature_projection(img_tokens)
            noise_tokens = self.noise_projection(psi_t)
            tokens = torch.cat([feature_tokens, noise_tokens], dim=-1)
        else:
            raise ValueError(f"Invalid image fusion mode: {self.img_fusion_mode}")
        return tokens

    def _patchify_noise(
        self,
        psi_t: Float[Tensor, "b s d1"],
        conditioning: dict,
        noise_patchify_size: int,
    ) -> Float[Tensor, "b v p d2"]:
        noise_height = conditioning["noise_height"]
        noise_width = conditioning["noise_width"]
        # map invalid pixels to a random gaussian
        if self.training:
            assert "valid_mask" in conditioning, "valid_mask is required"
            # map invalid pixels to a random gaussian at the same timestep
            psi_t = torch.lerp(
                torch.randn_like(psi_t), psi_t, conditioning["valid_mask"].float()
            )
        psi_t_tokens = model_utils.patchify_image(
            psi_t, noise_height, noise_width, noise_patchify_size
        )
        if self.fuse_raw_rgb:
            img = conditioning[constants.RGB_KEY]
            rgb_height, rgb_width = img.shape[-2:]
            rgb_seq = einops.rearrange(img, "b v c h w -> b (v h w) c")
            rgb_patched = model_utils.patchify_image(
                rgb_seq, rgb_height, rgb_width, self.patch_size
            )
            psi_t_tokens = torch.cat([rgb_patched, psi_t_tokens], dim=-1)
        return psi_t_tokens

    def _unpatchify_prediction(
        self,
        v_t: Float[Tensor, "b v p d1"],
        conditioning: dict,
    ) -> Float[Tensor, "b s d2"]:
        noise_height = conditioning["noise_height"]
        noise_width = conditioning["noise_width"]
        v_t = model_utils.unpatchify_image(
            v_t, noise_height, noise_width, self.noise_patchify_size
        )
        return v_t

    def compute_position_query(
        self,
        conditioning: dict,
        query_indices: Int[Tensor, "b s"],
        query_mode: Literal["raymap", "xyt"],
        time_mode: Literal["absolute", "relative"] = "relative",
    ) -> Float[Tensor, "b s d"]:
        if query_mode == "raymap":
            query_position = self.compute_raymap_query(conditioning["camera"])
        elif query_mode == "xyt":
            # (x,y,t)
            bs = conditioning["batch_size"]
            noise_height = conditioning["noise_height"]
            noise_width = conditioning["noise_width"]
            noise_nview = conditioning["noise_nview"]
            device = conditioning["rgb"].device
            xpos = torch.linspace(-1, 1, noise_width, device=device)
            ypos = torch.linspace(-1, 1, noise_height, device=device)
            if time_mode == "relative":
                tpos = torch.linspace(-1, 1, noise_nview, device=device)
            elif time_mode == "absolute":
                tpos = torch.arange(noise_nview, device=device).float()
            else:
                raise ValueError(f"Invalid time mode: {time_mode}")
            # grid
            xyt = torch.stack(
                torch.meshgrid(tpos, ypos, xpos, indexing="ij"), dim=-1
            ).flip(-1)
            query_position = einops.repeat(xyt, "t h w ... -> b (t h w) ...", b=bs)
        else:
            raise ValueError(f"Invalid query mode: {query_mode}")
        query_position = self.index_query(query_position, query_indices)
        return query_position

    def compute_raymap_query(self, camera: _camera.Camera) -> Float[Tensor, "b s d"]:
        updated_position = camera.position
        if self.use_camera_scale_normalizer:
            updated_position = self.camera_scale_normalizer.apply(updated_position)
        camera = _camera.update_extrinsic(camera, position=updated_position)
        query_raymap = model_utils.compute_patch_raymap(camera, 1)
        query_raymap = einops.rearrange(query_raymap, "b v h w c -> b (v h w) c")
        return query_raymap

    def index_query(
        self,
        query: Float[Tensor, "b s1 d"],
        query_indices: Int[Tensor, "b s1"] | None = None,
    ) -> Float[Tensor, "b s2 d"]:
        # training time sampling
        if query_indices is not None:
            query = query.gather(
                dim=1,
                index=einops.repeat(query_indices, "... -> ... c", c=query.shape[-1]),
            )
        return query

    def compute_query_tokens(
        self,
        conditioning: dict,
        query_indices: Int[Tensor, "b s1"],
        feat_grid: Float[Tensor, "b s2 d"] | None = None,
    ) -> Float[Tensor, "b s1 d"]:
        """
        When use_feat_interpolation is True, return features at the query location.
        When use_feat_interpolation is False, return query tokens.
        """
        if self.use_feat_interpolation:
            # get xy coordinates in [-1,1], bs, thw, 2
            query_position = self.compute_position_query(
                conditioning, query_indices, query_mode="xyt", time_mode="absolute"
            )
            query_position, query_view = query_position[..., :2], query_position[..., 2]
            # reshape feat_grid to target dimension
            batch_size = query_indices.shape[0]
            nview, height, width = (
                conditioning["noise_nview"],
                conditioning["noise_height"],
                conditioning["noise_width"],
            )

            feat_grid = einops.rearrange(
                feat_grid,
                "b (t h w) c -> b t c h w",
                h=height // self.patch_size,
                w=width // self.patch_size,
            )
            query_tokens = torch.zeros(
                query_indices.shape + (self.decoder_embed_dim,),
                device=feat_grid.device,
                dtype=feat_grid.dtype,
            )
            for idx in range(batch_size):
                for jdx in range(nview):
                    feat_grid_sub = einops.rearrange(
                        feat_grid[idx, jdx], "c h w -> 1 c h w"
                    )
                    query_position_index = query_view[idx] == jdx
                    query_position_sub = query_position[idx][query_position_index]
                    query_position_sub = einops.rearrange(
                        query_position_sub, "n c -> 1 n 1 c"
                    )
                    query_tokens_sub = F.grid_sample(
                        feat_grid_sub.float(),
                        query_position_sub,
                        align_corners=True,
                    ).to(feat_grid.dtype)
                    query_tokens_sub = einops.rearrange(
                        query_tokens_sub, "1 c n 1 -> n c"
                    )
                    query_tokens[idx][query_position_index] = query_tokens_sub

        else:
            # compute query tokens
            if self.use_raymap:
                query_mode = "raymap"
            else:
                query_mode = "xyt"
            query_position = self.compute_position_query(
                conditioning, query_indices, query_mode
            )
            # project raymap query tokens
            query_embed = self.query_embed(query_position)
            query_tokens = self.query_token_projection(query_embed)
        # fuse raymap and color query tokens
        if self.use_rgb_query:
            query_patch_rgb = model_utils.query_patches(
                conditioning[constants.RGB_KEY],
                query_indices,
                self.rgb_query_patch_size,
            )
            query_rgb_tokens = self.query_rgb_token_projection(query_patch_rgb)
            query_tokens = self.query_rgb_fusion(
                torch.cat([query_tokens, query_rgb_tokens], dim=-1)
            )
        return query_tokens

    def fuse_query_noise(
        self,
        psi_t: Float[Tensor, "b s1 d"],
        conditioning: dict,
        decode_mode: bool,
        seq_idx: Int[Tensor, "s2"] | None = None,
        feat_grid: Float[Tensor, "b s1 d"] | None = None,
    ) -> Float[Tensor, "b s2 e"]:
        # bs, s2, d
        # training time query sampling
        if conditioning["query_indices"] is not None:
            assert self.training, "query_indices should only be used in training"
            query_indices = conditioning["query_indices"]
        else:
            if seq_idx is None:
                # inference time using all queries
                query_indices = torch.arange(psi_t.shape[1], device=psi_t.device)
            else:
                # inference time chunking
                assert not self.training, "chunking should only be used in inference"
                assert (
                    conditioning["query_indices"] is None
                ), "query_indices should be None in chunking"
                query_indices = seq_idx
            query_indices = einops.repeat(
                query_indices, "... -> b ...", b=psi_t.shape[0]
            )

        # subsample tokens as anchor context
        if decode_mode:
            pass
        else:
            num_provided_tokens = query_indices.shape[1]
            rand_indices = torch.randperm(num_provided_tokens)[
                : self.num_denoising_global_tokens
            ]
            query_indices = query_indices[:, rand_indices]

        query_tokens = self.compute_query_tokens(conditioning, query_indices, feat_grid)

        if decode_mode and self.use_noise_free_decoding:
            return query_tokens
        else:
            # compute noise tokens
            if self.use_noise_patch_query:
                # map invalid pixels to a randam gaussian at the same timestep
                if self.training:
                    assert (
                        "valid_mask" in conditioning
                    ), "valid_mask is not in conditioning"
                    psi_t = torch.lerp(
                        torch.randn_like(psi_t),
                        psi_t,
                        conditioning["valid_mask"].float(),
                    )
                noise_height = conditioning["noise_height"]
                noise_width = conditioning["noise_width"]
                psi_t_image = einops.rearrange(
                    psi_t, "b (t h w) c -> b t c h w", h=noise_height, w=noise_width
                )
                noise_tokens = model_utils.query_patches(
                    psi_t_image, query_indices, self.rgb_query_patch_size
                )
            else:
                noise_tokens = self.index_query(psi_t, query_indices)
            # fuse noise and query tokens
            noise_tokens = self.noise_embed(noise_tokens)
            noise_tokens = self.query_noise_projection(noise_tokens)
            fused_tokens = self.noise_query_fusion(
                torch.cat([noise_tokens, query_tokens], dim=-1)
            )
            return fused_tokens

    def query_memory_tokens_chunk(
        self,
        memory_tokens: Float[Tensor, "b s1 c"],
        conditioning: dict,
        psi_t: Float[Tensor, "b s2 e"],
        timesteps: Float[Tensor, "b"],
        seq_idx: Int[Tensor, "s2"] | None = None,
    ) -> Float[Tensor, "b s2 c"]:
        if self.use_diffusion_query:
            blk_kwargs = {"adaln_input": self.query_time_projection(timesteps)}
        else:
            blk_kwargs = {}
        query_tokens = self.fuse_query_noise(
            psi_t,
            conditioning,
            decode_mode=True,
            seq_idx=seq_idx,
            feat_grid=memory_tokens,
        )
        if self.use_feat_interpolation:
            # query with coordinate interpolation + MLP
            for blk in self.mlp_blocks:
                query_tokens = blk(query_tokens, **blk_kwargs)
        else:
            # query using perceiver IO style CA
            for blk in self.ca_blocks:
                query_tokens = blk(query_tokens, memory_tokens, **blk_kwargs)
        # project to latent dimension
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            v_t = self.latents_projection(query_tokens)
        return v_t

    def query_memory_tokens(
        self,
        memory_tokens: Float[Tensor, "b s1 c"],
        conditioning: dict,
        psi_t: Float[Tensor, "b s2 e"],
        timesteps: Float[Tensor, "b"],
    ) -> Float[Tensor, "b s2 c"]:
        # cross attention between query and global tokens
        if self.training:
            v_t = self.query_memory_tokens_chunk(
                memory_tokens, conditioning, psi_t, timesteps
            )
            return v_t

        # memory-friendly: process queries in chunks
        bs, nq, _ = psi_t.shape
        nc = self.noise_channel
        v_t = torch.empty((bs, nq, nc), dtype=psi_t.dtype, device=psi_t.device)
        chunk_size = self.query_chunk_size
        for start in range(0, nq, chunk_size):
            end = min(start + chunk_size, nq)
            seq_idx = torch.arange(start, end, device=psi_t.device)
            v_chunk = self.query_memory_tokens_chunk(
                memory_tokens,
                conditioning,
                psi_t,
                timesteps,
                seq_idx=seq_idx,
            )
            v_t[:, start:end, :] = v_chunk
        return v_t

    def get_scale_tokens(self, tokens: Float[Tensor, "b v p d"]):
        if self.use_scale_head:
            batch_size = tokens.shape[0]
            return einops.repeat(self.scale_tokens, "... -> b ...", b=batch_size)
        else:
            return None

    def predict_scale(self, tokens: Float[Tensor, "b s d"]) -> Float[Tensor, "b"]:
        # process frame separately
        scale_output = self.scale_projection(tokens)
        # fuse frame dimension
        logscale_norm_pred, attn_logits = scale_output.chunk(2, -1)  # b, s, 1
        attn_weights = attn_logits.softmax(dim=1)
        logscale_norm_pred = (logscale_norm_pred * attn_weights).sum(dim=1).squeeze(-1)
        # get metric scale
        logscale_pred = logscale_norm_pred
        # account for camera normalization before raymap construction
        camera_scale = self.camera_scale_normalizer.get_scale()
        if self.use_raymap and camera_scale is not None:
            logscale_pred += camera_scale.log()
        return logscale_norm_pred, logscale_pred

    def encode_all_frames_attention(
        self,
        tokens: Float[Tensor, "b v p d"],
        xpos: Float[Tensor, "b v p 2"],
        raymap: Float[Tensor, "b v p 6"],
    ) -> Float[Tensor, "b v p d"]:
        bs, nview, npatch, dim = tokens.shape
        if self.num_global_encoder_blocks > 0:
            for blk_idx, blk in enumerate(self.global_encoder_blocks):
                # fuse raymap tokens
                if self.use_raymap and (blk_idx == 0 or self.use_dense_raycond):
                    assert (
                        raymap is not None
                    ), "raymap is required when use_raymap is True"
                    tokens = tokens.reshape(bs, nview, npatch, dim)
                    tokens = self.fuse_raymap_tokens(
                        tokens, raymap, blk_idx, is_enc_blocks=True
                    )
                # alternate attention
                is_framewise_attn = self.alternate_attention and blk_idx % 2 == 0
                if is_framewise_attn:
                    # frame-wise attention
                    tokens = tokens.reshape(bs * nview, npatch, dim)
                    xpos = xpos.reshape(bs * nview, npatch, 2)
                else:
                    # global attention
                    tokens = tokens.reshape(bs, nview * npatch, dim)
                    xpos = xpos.reshape(bs, nview * npatch, 2)

                tokens = blk(tokens, xpos)
            tokens = tokens.reshape(bs, nview, npatch, dim)
        return tokens

    def _forward_denoising(
        self,
        psi_t: Float[Tensor, "b s d"],
        timesteps: Float[Tensor, "b"],
        conditioning: Mapping[str, Tensor],
    ) -> dict:
        """
        Single denoising step.
        """
        # encode conditioning info
        img = conditioning[constants.RGB_KEY]
        camera = conditioning.get(constants.CAMERA_KEY, None)
        xpos, raymap = self.encode_conditioning(img, camera)

        # encode images
        if "img_tokens" in conditioning:
            img_tokens = conditioning["img_tokens"]
            assert not self.training, "img_tokens should not be provided in training"
        else:
            img_tokens = self.encode_image(img)

        # add all frame attention encoding
        img_tokens = self.encode_all_frames_attention(img_tokens, xpos, raymap)

        # patchify noise
        psi_t_tokens = self._patchify_noise(
            psi_t, conditioning, self.noise_patchify_size
        )

        # fuse tokens, psi_t and t
        tokens = self.fuse_context_tokens(img_tokens, psi_t_tokens)

        # decode to output tokens
        tokens_out, _ = self.decode_to_output_tokens(
            tokens,
            xpos,
            raymap=raymap,
            timesteps=timesteps,
        )

        # project to latent dimension
        if self.head_mode == "linear":
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                v_t = self.latents_projection(tokens_out)
            # unpatchify prediction
            v_t = self._unpatchify_prediction(v_t, conditioning)
        elif self.head_mode == "conv":
            _, nview, _, height, width = img.shape
            v_t = self.latents_projection(
                einops.rearrange(tokens_out, "b v p ... -> b (v p) ..."),
                nview,
                height,
                width,
            )["v_t"]
            v_t = einops.rearrange(v_t, "b v h w c -> b (v h w) c")
        else:
            raise ValueError(f"Unsupported head mode: {self.head_mode}")

        if self.use_x0_prediction:
            v_t = (v_t + psi_t) / timesteps.clamp(min=model_config.T_MIN_CLAMP).reshape(
                -1, 1, 1
            )

        output_dict = {}
        output_dict["v_t"] = v_t
        output_dict_regr = self.run_regression_heads(
            img_tokens, tokens_out, raymap, xpos
        )
        output_dict.update(output_dict_regr)
        return output_dict

    def run_regression_heads(
        self,
        img_tokens: Float[Tensor, "b v p d"],
        tokens_out: Float[Tensor, "b v p d"],
        raymap: Float[Tensor, "b v p 6"] | None,
        xpos: Float[Tensor, "b v p 2"] | None = None,
    ) -> dict[str, Tensor]:
        output_dict_regr = {}
        if self.use_scale_head or self.use_camera_head:
            tokens = self.head_pre_projection(
                torch.cat([img_tokens, tokens_out.detach()], dim=-1)
            )
            if raymap is not None:
                tokens = self.head_raymap_project(tokens, raymap)
            bs, nview, _, _ = tokens.shape
            tokens = einops.rearrange(tokens, "b v p d -> b (v p) d")
            if xpos is not None:
                xpos = einops.rearrange(xpos, "b v p d -> b (v p) d")
            # fuse patch dimension
            tokens = self.head_transformer(tokens, xpos=xpos)
        if self.use_scale_head:
            # predict normalized scale
            logscale_norm_pred, logscale_pred = self.predict_scale(tokens)
            output_dict_regr["logscale_pred"] = logscale_pred
            output_dict_regr["logscale_norm_pred"] = logscale_norm_pred
        if self.use_camera_head:
            camera_tokens = einops.rearrange(tokens, "b (v p) d -> (b v) p d", v=nview)
            camera_poses = self.camera_head(camera_tokens).reshape(bs, nview, 4, 4)
            # canonicalize
            camera_poses = geometry_utils.invert_se3(camera_poses[:, :1]) @ camera_poses
            output_dict_regr["camera_poses"] = camera_poses
        return output_dict_regr

    def _forward_query_denoising(
        self,
        psi_t: Float[Tensor, "b s d"],
        timesteps: Float[Tensor, "b"],
        conditioning: Mapping[str, Tensor],
    ) -> dict:
        """
        Single denoising step for query denoising.
        """
        assert "query_indices" in conditioning, "query_indices should be provided"
        # encode conditioning info
        img = conditioning[constants.RGB_KEY]
        camera = conditioning.get(constants.CAMERA_KEY, None)
        xpos, raymap = self.encode_conditioning(img, camera)

        # obtain image tokens
        if "img_tokens" in conditioning:
            img_tokens = conditioning["img_tokens"]
            assert not self.training, "img_tokens should not be provided in training"
        else:
            # encode images
            img_tokens = self.encode_image(img)

        if self.use_noise_interpolation:
            # patchify noise
            psi_t_tokens = self._patchify_noise(psi_t, conditioning, self.patch_size)

            # fuse tokens, psi_t and t
            context_tokens = self.fuse_context_tokens(img_tokens, psi_t_tokens)

            # decode to output tokens
            memory_tokens, _ = self.decode_to_output_tokens(
                context_tokens,
                xpos,
                raymap=raymap,
                timesteps=timesteps,
            )
        else:
            # aggregate current noisy data into anchors
            anchor_tokens = self.fuse_query_noise(
                psi_t,
                conditioning,
                decode_mode=False,
                feat_grid=einops.rearrange(img_tokens, "b v p d -> b (v p) d"),
            )
            # decode global representation
            memory_tokens, anchor_tokens = self.decode_to_output_tokens(
                img_tokens,
                xpos,
                raymap=raymap,
                global_tokens=anchor_tokens,
                timesteps=timesteps,
            )
        memory_tokens = einops.rearrange(memory_tokens, "b v p d -> b (v p) d")
        if self.use_feat_interpolation:
            # retain the context image token structure of b (v h w) d
            pass
        else:
            memory_tokens = torch.cat([memory_tokens, anchor_tokens], dim=1)

        # decode query
        v_t = self.query_memory_tokens(memory_tokens, conditioning, psi_t, timesteps)

        # fill in v_t from sparse outputs
        if self.training:
            bs, ns, _ = conditioning["loss_mask"].shape
            nc = v_t.shape[-1]
            query_indices = conditioning["query_indices"]
            v_t = torch.scatter(
                torch.empty((bs, ns, nc), dtype=v_t.dtype, device=v_t.device),
                dim=1,
                index=einops.repeat(query_indices, "... -> ... c", c=nc),
                src=v_t,
            )

        if self.use_x0_prediction:
            v_t = (v_t + psi_t) / timesteps.clamp(min=model_config.T_MIN_CLAMP).reshape(
                -1, 1, 1
            )

        output_dict = {}
        output_dict["v_t"] = v_t
        return output_dict

    def forward(self, *args, **kwargs) -> dict:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.model_type == "regression":
                output_dict = self._forward_regression(*args, **kwargs)
            elif self.model_type == "diffusion":
                output_dict = self._forward_denoising(*args, **kwargs)
            elif self.model_type == "query_diffusion":
                output_dict = self._forward_query_denoising(*args, **kwargs)
            else:
                raise ValueError(f"Invalid model type: {self.model_type}")
        return output_dict

    def get_additional_kwargs(self) -> dict:
        return {
            "use_x0_prediction": self.use_x0_prediction,
            "t_min_clamp": model_config.T_MIN_CLAMP,
        }
