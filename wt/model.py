"""
Multilayer XYZ Diffusion Model.

This is a thin wrapper around :class:`MultilayerBackbone` to reuse the
original diffusion architecture and ``FMLossWrapper`` training logic
with minimal changes.  Mask is modeled as an extra diffusion channel
(xyz + mask_logit), and layer order is encoded with a learned layer
embedding.
"""

import os
from collections.abc import Mapping
from copy import deepcopy

import einops
import structlog
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from wt._core.vendor.vggt.layers import layer_scale
from wt._core.diffusion import constants
from wt._core.engine import activation_checkpoint
from wt._core.arch.models import blocks as model_blocks
from wt._core.arch.models import config as model_config
from wt._core.arch.models import backbone as _backbone
from wt._core.arch.models import model_utils

logger = structlog.get_logger(__name__)


def _unwrap_ac(module: nn.Module) -> nn.Module:
    """Unwrap activation-checkpoint wrapper to access the underlying module."""
    return getattr(module, "_checkpoint_wrapped_module", module)


_ATTN_PATTERN_PERIOD = {
    "frame_global": 2,
    "frame_ray_global": 3,
    "layer_ray_view_global": 4,  # r80 (multi-view scene model)
}


def _attn_pattern_period(pattern: str) -> int:
    """Number of decoder blocks in one attention pattern period.

    Used to compute ``num_global_blocks = num_decoder_blocks // period``
    for layer-embedding heads, AR cross-attention modules, and temporal
    block insertion bookkeeping.
    """
    if pattern not in _ATTN_PATTERN_PERIOD:
        raise ValueError(
            f"Unknown attention_pattern: {pattern!r}; "
            f"valid: {list(_ATTN_PATTERN_PERIOD.keys())}"
        )
    return _ATTN_PATTERN_PERIOD[pattern]


def _svd_orthogonalize(M: torch.Tensor) -> torch.Tensor:
    """Project a (..., 3, 3) matrix to the closest rotation via SVD.

    Returns U @ diag([1, 1, det(U @ V^T)]) @ V^T which lives in SO(3) and
    handles reflections by flipping the last singular value's sign so the
    determinant is +1.  Used by r80 PoseHead to convert the raw 9D output
    of the MLP into a valid rotation matrix.

    .. note::
        This function is **NOT used in r80 training** — see
        :func:`_gram_schmidt_6d_to_R` below for the production
        orthogonalisation path.  ``torch.linalg.svd``'s backward is
        numerically unstable near degenerate singular values
        (e.g. ``M = I``), producing NaN gradients when any pair of
        singular values is equal.  The 6D Gram-Schmidt
        representation has a *continuous* gradient everywhere on
        SO(3) (Zhou et al., CVPR 2019, "On the Continuity of Rotation
        Representations in Neural Networks") and is the canonical fix
        for this class of bugs.  We keep ``_svd_orthogonalize`` for
        ablation / inference utilities only.
    """
    with torch.autocast(device_type="cuda", enabled=False):
        M_fp32 = M.float()
        U, _S, Vh = torch.linalg.svd(M_fp32, full_matrices=False)
        det = torch.det(U @ Vh)
        D = torch.eye(3, device=M.device, dtype=torch.float32)
        D = D.unsqueeze(0).expand(*M.shape[:-2], 3, 3).clone()
        D[..., 2, 2] = det
        R = U @ D @ Vh
    return R


def _gram_schmidt_6d_to_R(rep_6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert a 6D continuous rotation representation to a 3×3 rotation.

    Following Zhou et al. (CVPR'19, "On the Continuity of Rotation
    Representations in Neural Networks"), the 6D representation maps to
    a valid rotation in SO(3) via Gram-Schmidt orthogonalisation:

        a1, a2 = rep_6d.split(3, dim=-1)
        b1 = a1 / |a1|
        b2 = (a2 - <b1, a2> b1)
        b2 = b2 / |b2|
        b3 = b1 × b2
        R  = stack([b1, b2, b3], dim=-1)   # column-major: R = [b1 | b2 | b3]

    Crucially, **the gradient of this map is finite and continuous
    everywhere away from the rank-deficient set** (a1 == 0 or a2 ∈
    span(a1)), unlike SVD whose backward is undefined for repeated
    singular values (e.g. R = identity).  This is the standard fix for
    the NaN-gradient pathology that affects 9D + SVD heads at
    initialization.

    Args:
        rep_6d: tensor of shape (..., 6).  ``rep_6d[..., :3]`` and
            ``rep_6d[..., 3:6]`` are the two raw vectors.  Recommended
            initialization: bias ``(1, 0, 0, 0, 1, 0)`` so the head
            outputs identity rotations at init.
        eps:    small constant added to the norms to avoid division by
            zero when both raw vectors are zero (early in training the
            biased init prevents this, but the eps is cheap insurance).

    Returns:
        tensor of shape (..., 3, 3) — a valid rotation matrix
        (det = +1, R^T R = I).  Output is column-major:
        ``R[..., :, 0] == b1``, ``R[..., :, 1] == b2``,
        ``R[..., :, 2] == b3``.
    """
    with torch.autocast(device_type="cuda", enabled=False):
        rep_fp32 = rep_6d.float()
        a1 = rep_fp32[..., :3]
        a2 = rep_fp32[..., 3:6]

        b1 = a1 / (a1.norm(dim=-1, keepdim=True) + eps)
        a2_proj = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = a2_proj / (a2_proj.norm(dim=-1, keepdim=True) + eps)
        b3 = torch.cross(b1, b2, dim=-1)

        # Column-major stack: R[..., :, 0]=b1, R[..., :, 1]=b2, R[..., :, 2]=b3.
        R = torch.stack([b1, b2, b3], dim=-1)
    return R


class DecoderBlockDiTCrossAttn(model_blocks.DecoderBlockDiT):
    """DecoderBlockDiT extended with an image cross-attention layer.

    Structure: Self-Attention (AdaLN) → Cross-Attention → FFN (AdaLN).

    The cross-attention output is gated by a LayerScale initialized to
    ``cross_attn_ls_init`` (default 0.1) so that training starts with a
    moderate amount of image information and doesn't destabilize the
    denoising signal.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cross_attn_ls_init: float = 0.1,
        **kwargs,
    ):
        super().__init__(dim, num_heads, **kwargs)
        self.cross_attn = model_blocks.CrossAttention(
            dim_q=dim,
            dim_kv=dim,
            rope=kwargs.get("rope"),
            num_heads=num_heads,
            use_qk_norm=kwargs.get("use_qk_norm", True),
        )
        self.norm_cross = nn.LayerNorm(dim)
        self.ls_cross = layer_scale.LayerScale(dim, init_values=cross_attn_ls_init)

    def forward(
        self,
        x,
        xpos,
        adaln_input,
        context=None,
        context_pos=None,
        kv_cache_in=None,
        return_kv_cache=False,
    ):
        x_dtype = x.dtype
        shift_a, scale_a, dscale_a, shift_m, scale_m, dscale_m = (
            self.modulation.view(self.modulation_shape).unsqueeze(1)
            + adaln_input.float()
        ).unbind(-2)

        attn_out = self.attn(
            (self.norm1(x).float() * (1 + scale_a) + shift_a).to(x_dtype),
            xpos,
            use_fa_interface=True,
            kv_cache_in=kv_cache_in,
            return_kv_cache=return_kv_cache,
        )
        if return_kv_cache:
            y, kv_cached = attn_out
        else:
            y = attn_out
            kv_cached = None

        y = self.ls1(y)
        x = (x + y * dscale_a).to(x_dtype)

        if context is not None:
            ca_out = self.cross_attn(
                self.norm_cross(x), context, qpos=xpos, kvpos=context_pos
            )
            x = x + self.ls_cross(ca_out).to(x_dtype)

        y = self.ls2(
            self.mlp((self.norm3(x).float() * (1 + scale_m) + shift_m).to(x_dtype))
        )
        x = (x + y * dscale_m).to(x_dtype)
        if return_kv_cache:
            return x, kv_cached
        return x


_MOGE_FILES = ("moge-vitl-config.json", "moge-vitl.safetensors")


def _ensure_moge_local_zoo() -> str:
    """Locate the MoGe encoder weights on disk for *training* warm-start.

    The published ``wt`` inference release does not need MoGe weights — the
    encoder parameters are baked into every released checkpoint and restored
    by ``checkpoint.build_model_and_load_ckpt``.  This helper exists only so
    that fine-tuning / retraining workflows can warm-start from an external
    MoGe checkpoint.  Set the environment variable ``MOGE_LOCAL_ZOO`` to a
    directory containing ``moge-vitl-config.json`` and
    ``moge-vitl.safetensors`` (download them from the official MoGe release,
    e.g. https://huggingface.co/Ruicheng/moge-vitl).  If the env var is not
    set we return an empty string and ``build_encoder`` falls back to a
    randomly-initialised backbone (its parameters will be overwritten by the
    checkpoint anyway when inference is the target).
    """
    local_zoo = os.environ.get("MOGE_LOCAL_ZOO", "")
    if not local_zoo:
        return ""
    if not os.path.isdir(local_zoo):
        return ""
    if all(os.path.isfile(os.path.join(local_zoo, n)) for n in _MOGE_FILES):
        return local_zoo
    return ""


class TemporalAttentionBlock(nn.Module):
    """Temporal self-attention block for video-clip (T-frame) mode.

    Operates on tokens shaped ``[B, T*L, P, D]`` by reshaping to
    ``[B*L*P, T, D]`` and performing self-attention along the T axis.  Uses
    1D RoPE (implemented by feeding ``[t, 0]`` positions to the shared
    ``RoPE2D`` module so the k/q dimensionality matches the rest of the
    decoder blocks).

    The block is LayerScale-initialized (default ``init=1e-5``) so the
    residual contribution is near zero at training start, letting us
    warm-start from a non-temporal (r75) checkpoint without perturbing
    the frozen single-frame behaviour.

    Only active when ``num_time > 1``; for single-frame inference it is
    never called (see ``MultilayerBackbonePatched.decode_to_output_tokens``).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float = 1e-5,
        use_qk_norm: bool = True,
        rope_freq: float = 100.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.rope = model_blocks.RoPE2D(freq=rope_freq)
        self.block = model_blocks.DecoderBlockDiT(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            rope=self.rope,
            use_qk_norm=use_qk_norm,
            init_values=init_values,
        )

    def forward(
        self,
        tokens: Float[Tensor, "b tl p d"],
        adaln_input: Float[Tensor, "b 1 6 d"],
        num_time: int,
        num_layers: int,
    ) -> Float[Tensor, "b tl p d"]:
        bs, tl, p, d = tokens.shape
        assert tl == num_time * num_layers, (
            f"tokens second dim ({tl}) must equal num_time*num_layers "
            f"({num_time}*{num_layers})"
        )

        x = tokens.reshape(bs, num_time, num_layers, p, d)
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        x = x.reshape(bs * num_layers * p, num_time, d)

        pos_t = torch.arange(num_time, device=tokens.device, dtype=torch.long)
        pos = torch.zeros(num_time, 2, device=tokens.device, dtype=torch.long)
        pos[:, 0] = pos_t
        pos = pos.unsqueeze(0).expand(bs * num_layers * p, -1, -1).contiguous()

        adaln_bcast = adaln_input.unsqueeze(1).expand(-1, num_layers * p, -1, -1, -1)
        adaln_bcast = adaln_bcast.reshape(bs * num_layers * p, *adaln_input.shape[1:])

        x = self.block(x, pos, adaln_bcast)

        x = x.reshape(bs, num_layers, p, num_time, d)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        x = x.reshape(bs, num_time * num_layers, p, d)
        return x


class MultilayerBackbonePatched(_backbone.MultilayerBackbone):
    """:class:`MultilayerBackbone` with _forward_denoising patched to allow img_tokens during
    training and optional dense layer-embedding injection in the decoder.

    Overrides:
      - build_encoder: optionally warm-starts MoGe weights from a local
        directory (env var ``MOGE_LOCAL_ZOO``).  Inference does not need
        this -- the encoder state is restored from the release checkpoint.
      - _forward_denoising: bypasses the upstream ``not self.training``
        assertion when ``img_tokens`` is supplied via conditioning, and
        passes ``layer_embed_info`` through to the decoder.
      - decode_to_output_tokens: supports dense layer-embedding injection
        modes (adaln / film / per_block / global_only) on top of the
        standard timestep AdaLN.  Also supports 3-way attention patterns
        (frame-wise / ray-wise / global) when ``attention_pattern`` is set.
        When ``num_time > 1`` is threaded through ``conditioning``, ray /
        global reshapes operate per-frame (spatial-temporal factoring)
        and optional ``TemporalAttentionBlock`` instances are invoked
        after designated global blocks (see ``temporal_insert_indices``).
    """

    def build_encoder(self):
        if self.encoder_model == "moge" and not self.inference_mode:
            local_zoo = _ensure_moge_local_zoo()
            if os.path.isdir(local_zoo):
                from wt._core.vendor.moge.moge_model import MoGeModel

                moge_model = MoGeModel.from_pretrained(
                    model_name="vitl",
                    model_kwargs={"use_fa3": True},
                    cache_path=local_zoo,
                )
                from copy import deepcopy

                self.encoder = deepcopy(moge_model.backbone)
                self.encoder.intermediate_layers = 4
                self.encoder.register_buffer("image_mean", moge_model.image_mean)
                self.encoder.register_buffer("image_std", moge_model.image_std)
                del moge_model
                from wt._core.components import nn_layers

                self.encoder.final_project = nn_layers.Linear(4096, self.decoder_embed_dim)
                return
        super().build_encoder()

    def _forward_denoising(
        self,
        psi_t: Float[Tensor, "b s d"],
        timesteps: Float[Tensor, "b"],
        conditioning: Mapping[str, Tensor],
    ) -> dict:
        img = conditioning[constants.RGB_KEY]
        camera = conditioning.get(constants.CAMERA_KEY, None)
        xpos, raymap = self.encode_conditioning(img, camera)

        use_cross_attn = getattr(self, "use_cross_attn", False)
        use_split_token = getattr(self, "use_split_token", False)

        # CFM: rescale mask channel for network input.
        if getattr(self, "cfm_mask", False):
            _noise_type = getattr(self, "cfm_noise_type", "fixed_0.5")
            if _noise_type == "normal_0_1_sym":
                psi_t_net = psi_t
            else:
                nc = psi_t.shape[-1]
                psi_t_net = torch.cat(
                    [
                        psi_t[..., : nc - 1],
                        psi_t[..., nc - 1 : nc] * 2 - 1,
                    ],
                    dim=-1,
                )
        else:
            psi_t_net = psi_t

        _invalid_fill = conditioning.get("invalid_fill_mode", None)

        if use_cross_attn:
            # --- Cross-attention path ---
            # Image context: [B, P, D] from conditioning (pre-computed, normalized)
            img_context = conditioning["img_context"]

            # Patchify noise WITHOUT raw RGB concatenation
            noise_height = conditioning["noise_height"]
            noise_width = conditioning["noise_width"]
            if self.training or _invalid_fill is not None:
                _vm = conditioning["valid_mask"].float()
                if _invalid_fill == "zeros":
                    psi_t_net = psi_t_net * _vm
                else:
                    psi_t_net = torch.lerp(
                        torch.randn_like(psi_t_net),
                        psi_t_net,
                        _vm,
                    )
            psi_t_tokens = model_utils.patchify_image(
                psi_t_net, noise_height, noise_width, self.noise_patchify_size
            )  # [B, L, P, noise_channel * patch²]

            # Project noise tokens to decoder dimension
            noise_proj = getattr(self, "noise_projection_ca")
            tokens = noise_proj(psi_t_tokens)  # [B, L, P, D]
        elif use_split_token:
            # --- Split-token fusion path ---
            nc_geo = getattr(self, "nc_geo", 1)
            depth_only = getattr(self, "depth_only", False)
            mask_only = getattr(self, "mask_only", False)
            noise_height = conditioning["noise_height"]
            noise_width = conditioning["noise_width"]

            if mask_only:
                psi_t_mask = psi_t_net  # [B, S, 1] — all channels are mask
                if self.training or _invalid_fill is not None:
                    mask_valid = conditioning["valid_mask"].float()
                    if _invalid_fill == "zeros":
                        psi_t_mask = psi_t_mask * mask_valid
                    else:
                        psi_t_mask = torch.lerp(
                            torch.randn_like(psi_t_mask), psi_t_mask, mask_valid
                        )
                mask_patches = model_utils.patchify_image(
                    psi_t_mask, noise_height, noise_width, self.noise_patchify_size
                )  # [B, L, P, p²]
            else:
                psi_t_geo = psi_t_net[..., :nc_geo]  # [B, S, nc_geo]
                if self.training or _invalid_fill is not None:
                    geo_valid = conditioning.get(
                        "depth_valid_mask", conditioning["valid_mask"]
                    ).float()
                    if _invalid_fill == "zeros":
                        psi_t_geo = psi_t_geo * geo_valid
                    else:
                        psi_t_geo = torch.lerp(
                            torch.randn_like(psi_t_geo), psi_t_geo, geo_valid
                        )
                geo_patches = model_utils.patchify_image(
                    psi_t_geo, noise_height, noise_width, self.noise_patchify_size
                )  # [B, L, P, nc_geo * p²]

                if not depth_only:
                    psi_t_mask = psi_t_net[..., nc_geo:]  # [B, S, 1]
                    if self.training or _invalid_fill is not None:
                        mask_valid = conditioning["valid_mask"].float()
                        if _invalid_fill == "zeros":
                            psi_t_mask = psi_t_mask * mask_valid
                        else:
                            psi_t_mask = torch.lerp(
                                torch.randn_like(psi_t_mask), psi_t_mask, mask_valid
                            )
                    mask_patches = model_utils.patchify_image(
                        psi_t_mask, noise_height, noise_width, self.noise_patchify_size
                    )  # [B, L, P, p²]

            if self.fuse_raw_rgb:
                rgb_height, rgb_width = img.shape[-2:]
                rgb_seq = einops.rearrange(img, "b v c h w -> b (v h w) c")
                rgb_patches = model_utils.patchify_image(
                    rgb_seq, rgb_height, rgb_width, self.patch_size
                )  # [B, L, P, 3*p²]
                if mask_only:
                    mask_patches = torch.cat([rgb_patches, mask_patches], dim=-1)
                else:
                    geo_patches = torch.cat([rgb_patches, geo_patches], dim=-1)
                    if not depth_only:
                        mask_patches = torch.cat([rgb_patches, mask_patches], dim=-1)

            if "img_tokens" in conditioning:
                img_tokens = conditioning["img_tokens"]
            else:
                img_tokens = self.encode_image(img)
            img_tokens = model_utils.layer_norm_2d(img_tokens, dim=[1, 2, 3])
            img_tokens = img_tokens * self.img_token_scale
            feat = self.feature_projection(img_tokens)  # [B, L, P, D/2]

            # AR: make image features context-aware via cross-attention
            # Skipped when _skip_input_ctx_attn is set (KV cache mode default).
            ar_context = conditioning.get("ar_context", None)
            ar_ctx_pos = conditioning.get("ar_context_pos", None)
            input_ctx_attn = getattr(self, "input_context_attn", None)
            _skip_input_ca = conditioning.get("_skip_input_ctx_attn", False)
            if (
                ar_context is not None
                and input_ctx_attn is not None
                and not _skip_input_ca
            ):
                B_f, L_f, P_f, D_half = feat.shape
                feat_2d = feat.reshape(B_f * L_f, P_f, D_half)
                ctx_2d = (
                    ar_context.unsqueeze(1)
                    .expand(-1, L_f, -1, -1)
                    .reshape(B_f * L_f, -1, ar_context.shape[-1])
                )
                feat_pos = xpos.reshape(B_f * L_f, P_f, 2)
                ctx_pos_2d = (
                    ar_ctx_pos.unsqueeze(1)
                    .expand(-1, L_f, -1, -1)
                    .reshape(B_f * L_f, -1, 2)
                )
                feat_2d = input_ctx_attn(
                    feat_2d, ctx_2d, qpos=feat_pos, kvpos=ctx_pos_2d
                )
                feat = feat_2d.reshape(B_f, L_f, P_f, D_half)

            if mask_only:
                mask_noise = self.mask_noise_projection(mask_patches)  # [B, L, P, D/2]
                tokens = torch.cat([feat, mask_noise], dim=-1)  # [B, L, P, D]
            elif depth_only:
                geo_noise = self.geo_noise_projection(geo_patches)  # [B, L, P, D/2]
                tokens = torch.cat([feat, geo_noise], dim=-1)  # [B, L, P, D]
            else:
                geo_noise = self.geo_noise_projection(geo_patches)  # [B, L, P, D/2]
                geo_tokens = torch.cat([feat, geo_noise], dim=-1)  # [B, L, P, D]
                mask_noise = self.mask_noise_projection(mask_patches)  # [B, L, P, D/2]
                mask_tokens = torch.cat([feat, mask_noise], dim=-1)  # [B, L, P, D]
                geo_tokens = geo_tokens + self.type_embed.weight[0]
                mask_tokens = mask_tokens + self.type_embed.weight[1]
                tokens = torch.cat([geo_tokens, mask_tokens], dim=2)  # [B, L, 2P, D]
                xpos = torch.cat([xpos, xpos], dim=2)  # [B, L, 2P, 2]
            img_context = None
        else:
            # --- Original concat-fusion path ---
            if "img_tokens" in conditioning:
                img_tokens = conditioning["img_tokens"]
            else:
                img_tokens = self.encode_image(img)
            psi_t_tokens = self._patchify_noise(
                psi_t_net, conditioning, self.noise_patchify_size
            )
            tokens = self.fuse_context_tokens(img_tokens, psi_t_tokens)
            img_context = None

        layer_embed_info = conditioning.get("layer_embed_info")
        _return_kv_cache = conditioning.get("_return_kv_cache", False)
        _return_per_block = conditioning.get("_return_per_block_tokens", False)

        decode_result = self.decode_to_output_tokens(
            tokens,
            xpos,
            raymap=raymap,
            timesteps=timesteps,
            layer_embed_info=layer_embed_info,
            img_context=img_context,
            ar_context=conditioning.get("ar_context", None),
            ar_context_pos=conditioning.get("ar_context_pos", None),
            ar_context_per_block=conditioning.get("ar_context_per_block", None),
            ar_context_pos_per_block=conditioning.get("ar_context_pos_per_block", None),
            ar_kv_cache=conditioning.get("ar_kv_cache", None),
            return_kv_cache=_return_kv_cache,
            return_per_block_tokens=_return_per_block,
            per_layer_t=conditioning.get("per_layer_t"),
            num_time=int(conditioning.get("num_time", 1)),
            num_view=int(conditioning.get("num_view", 1)),
            temporal_blocks=conditioning.get("_temporal_blocks", None),
            temporal_insert_indices=conditioning.get(
                "_temporal_insert_indices", None
            ),
        )

        per_block_tokens = None
        kv_cache_list = None
        if _return_per_block and _return_kv_cache:
            tokens_out, _, kv_cache_list, per_block_tokens = decode_result
        elif _return_per_block:
            tokens_out, _, per_block_tokens = decode_result
        elif _return_kv_cache:
            tokens_out, _, kv_cache_list = decode_result
        else:
            tokens_out, _ = decode_result

        # Fast path: return only decoder tokens, skip head + unpatchify.
        # Used by token-context generation (teacher forcing / self-forcing).
        if conditioning.get("_context_only", False):
            result = {"decoder_tokens": tokens_out}
            if _return_kv_cache:
                result["kv_cache"] = kv_cache_list
            if _return_per_block and per_block_tokens is not None:
                result["per_block_tokens"] = per_block_tokens
            return result

        num_time_int_head = int(conditioning.get("num_time", 1))

        def _fold_T_into_B(x: torch.Tensor) -> torch.Tensor:
            if num_time_int_head <= 1:
                return x
            B_head = x.shape[0]
            TL_head = x.shape[1]
            assert TL_head % num_time_int_head == 0, (
                f"tokens_out second dim ({TL_head}) not divisible by "
                f"num_time ({num_time_int_head})"
            )
            L_true = TL_head // num_time_int_head
            return x.reshape(
                B_head * num_time_int_head, L_true, *x.shape[2:]
            ).contiguous()

        def _unfold_B_to_T(x: torch.Tensor, B_orig: int) -> torch.Tensor:
            if num_time_int_head <= 1:
                return x
            return x.reshape(
                B_orig, num_time_int_head * x.shape[1], *x.shape[2:]
            ).contiguous()

        if use_split_token:
            depth_only = getattr(self, "depth_only", False)
            mask_only = getattr(self, "mask_only", False)
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                if getattr(self.latents_projection, "head_timestep", False):
                    if num_time_int_head > 1:
                        head_ts = timesteps.repeat_interleave(num_time_int_head)
                    else:
                        head_ts = timesteps
                    head_adaln = self.project_timestep(head_ts)
                else:
                    head_adaln = None
                target_layer_idx = conditioning.get("target_layer_idx", None)
                if num_time_int_head > 1 and target_layer_idx is not None:
                    raise NotImplementedError(
                        "AR mode (target_layer_idx) not supported with num_time>1."
                    )

                B_orig = tokens_out.shape[0]
                tokens_out_head = _fold_T_into_B(tokens_out)

                if mask_only:
                    mask_raw = self.latents_projection.forward_mask_only(
                        tokens_out_head,
                        adaln_input=head_adaln,
                        target_layer_idx=target_layer_idx,
                    )
                    mask_raw = _unfold_B_to_T(mask_raw, B_orig)
                    v_t = self._unpatchify_prediction(mask_raw, conditioning)
                elif depth_only:
                    geo_raw = self.latents_projection.forward_geo_only(
                        tokens_out_head,
                        adaln_input=head_adaln,
                        target_layer_idx=target_layer_idx,
                    )
                    geo_raw = _unfold_B_to_T(geo_raw, B_orig)
                    if geo_raw.ndim == 3:
                        # ConvHead path: already unpatchified to [B, V*H*W, C]
                        # which matches unpatchify_image's output contract,
                        # so we skip _unpatchify_prediction here.
                        v_t = geo_raw
                    else:
                        v_t = self._unpatchify_prediction(geo_raw, conditioning)
                else:
                    num_patches = tokens_out_head.shape[2] // 2
                    geo_out = tokens_out_head[:, :, :num_patches, :]
                    mask_out = tokens_out_head[:, :, num_patches:, :]
                    geo_raw, mask_raw = self.latents_projection.forward_split(
                        geo_out,
                        mask_out,
                        adaln_input=head_adaln,
                        target_layer_idx=target_layer_idx,
                    )
                    geo_raw = _unfold_B_to_T(geo_raw, B_orig)
                    mask_raw = _unfold_B_to_T(mask_raw, B_orig)
                    geo_v_t = self._unpatchify_prediction(geo_raw, conditioning)
                    mask_v_t = self._unpatchify_prediction(mask_raw, conditioning)
                    v_t = torch.cat([geo_v_t, mask_v_t], dim=-1)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                B_orig = tokens_out.shape[0]
                tokens_out_head = _fold_T_into_B(tokens_out)
                if getattr(self.latents_projection, "head_timestep", False):
                    if num_time_int_head > 1:
                        head_ts = timesteps.repeat_interleave(num_time_int_head)
                    else:
                        head_ts = timesteps
                    head_adaln = self.project_timestep(head_ts)
                    v_t_raw = self.latents_projection(
                        tokens_out_head, adaln_input=head_adaln
                    )
                else:
                    v_t_raw = self.latents_projection(tokens_out_head)
                v_t_raw = _unfold_B_to_T(v_t_raw, B_orig)

            v_t = self._unpatchify_prediction(v_t_raw, conditioning)

        if self.use_x0_prediction:
            plt_expand = conditioning.get("per_layer_t_expand")
            if plt_expand is not None:
                t_clamp = plt_expand.clamp(min=model_config.T_MIN_CLAMP)
            else:
                t_clamp = timesteps.clamp(min=model_config.T_MIN_CLAMP).reshape(
                    -1, 1, 1
                )

            if getattr(self, "cfm_mask", False):
                nc = v_t.shape[-1]  # 4 or 7
                z_mask = v_t[..., nc - 1 : nc].clone()  # [B, S, 1]

                v_t = (v_t + psi_t) / t_clamp

                pi_mask = torch.sigmoid(z_mask)
                _noise_type = getattr(self, "cfm_noise_type", "fixed_0.5")
                if _noise_type == "normal_0_1_sym":
                    mask_x0 = 2.0 * pi_mask - 1.0
                else:
                    mask_x0 = pi_mask
                v_t_mask_cfm = (psi_t[..., nc - 1 : nc] - mask_x0) / t_clamp
                v_t = torch.cat([v_t[..., : nc - 1], v_t_mask_cfm], dim=-1)

                return {"v_t": v_t, "z_mask": z_mask, "decoder_tokens": tokens_out}
            else:
                v_t = (v_t + psi_t) / t_clamp

        return {"v_t": v_t, "decoder_tokens": tokens_out}

    def decode_to_output_tokens(
        self,
        tokens: Float[Tensor, "b v p d"],
        xpos: Float[Tensor, "b v p 2"],
        raymap=None,
        global_tokens=None,
        timesteps=None,
        layer_embed_info=None,
        img_context=None,
        ar_context=None,
        ar_context_pos=None,
        ar_context_per_block=None,
        ar_context_pos_per_block=None,
        ar_kv_cache=None,
        return_kv_cache=False,
        return_per_block_tokens=False,
        per_layer_t=None,
        num_time: int = 1,
        num_view: int = 1,
        temporal_blocks: nn.ModuleList | None = None,
        temporal_insert_indices: list[int] | None = None,
    ):
        """Decoder loop with optional dense layer-embedding injection,
        cross-attention to image features, and AR context cross-attention.

        When all optional features are ``None`` the behaviour is identical
        to the upstream ``MultilayerBackbone.decode_to_output_tokens``.

        r80 multi-view path:
          When ``num_view > 1``, the caller folds V into the batch dim
          (input batch = B*V).  The 4-way ``layer_ray_view_global`` pattern
          unwraps V at view/global blocks to enable cross-view attention.
          Combinations of (num_view > 1, num_time > 1) are not yet
          jointly tested — scene model uses V only with T=1; dynamic model
          uses T only with V=1.

        Args:
            img_context: [B, P, D] normalized encoder features for cross-attn KV.
            ar_context: [B, K, D] AR context tokens from previous layer predictions.
                When provided, cross-attention is applied after each global block.
            ar_context_pos: [B, K, 2] spatial positions for context tokens (RoPE).
            ar_context_per_block: list of [B, K, D] tensors, one per global block.
                When provided, each AR CA layer uses its own block-specific context
                instead of the shared ``ar_context``.
            ar_context_pos_per_block: list of [B, K, 2] tensors matching above.
            ar_kv_cache: list of (K, V) tuples per decoder block from context
                layers. When provided, KV is prepended to self-attention and
                ARContextCrossAttention is skipped.
            return_kv_cache: if True, collect and return per-block (K, V) pairs.
            return_per_block_tokens: if True, collect decoder tokens after each
                global block (post AR CA) and return them as an extra list.
            per_layer_t: [B, L] per-layer diffusion timesteps for non-AR mode.
                When provided, frame-wise blocks use per-layer AdaLN conditioning
                while ray/global blocks use the mean timestep.
            num_time: video-clip frame count T.  ``num_time == 1`` (default)
                preserves the single-frame path exactly bit-for-bit.  When
                ``num_time > 1``, the input ``nview`` axis is interpreted as
                ``T * num_layers``; ray and global attention blocks are
                reshaped per-frame (spatial-temporal factoring) and
                ``temporal_blocks`` are invoked after the global blocks
                whose ``global_idx`` is listed in
                ``temporal_insert_indices``.
            temporal_blocks: ``nn.ModuleList`` of ``TemporalAttentionBlock``.
                ``None`` (default) disables the temporal path even if
                ``num_time > 1`` — useful for BC smoke tests.
            temporal_insert_indices: list of global-block indices (0-based
                over the sequence of global blocks only) where a temporal
                block is applied immediately after the global block's AR CA
                (if any).  Length must match ``len(temporal_blocks)``.
        """
        attn_pattern = getattr(self, "attention_pattern", "frame_global")

        if (
            layer_embed_info is None
            and img_context is None
            and ar_context is None
            and ar_context_per_block is None
            and ar_kv_cache is None
            and not return_kv_cache
            and not return_per_block_tokens
            and attn_pattern == "frame_global"
            and per_layer_t is None
            and num_time == 1
            and temporal_blocks is None
        ):
            return super().decode_to_output_tokens(
                tokens,
                xpos,
                raymap=raymap,
                global_tokens=global_tokens,
                timesteps=timesteps,
            )

        bs, nview, npatch, dim = tokens.shape
        num_time_int = int(num_time)
        num_view_int = int(num_view)
        if num_time_int > 1:
            assert nview % num_time_int == 0, (
                f"nview ({nview}) must be divisible by num_time ({num_time_int})"
            )
            num_layers_only = nview // num_time_int
        else:
            num_layers_only = nview

        # r80: when num_view > 1, the caller has folded V into the batch dim
        # (i.e. effective batch = B_user * V).  The view/global blocks unwrap
        # V from the batch dim to perform cross-view attention.
        if num_view_int > 1:
            assert bs % num_view_int == 0, (
                f"bs ({bs}) must be divisible by num_view ({num_view_int}); "
                "caller must fold V into the leading batch dim."
            )
            bs_orig = bs // num_view_int
        else:
            bs_orig = bs

        if temporal_blocks is not None and num_time_int <= 1:
            raise ValueError(
                "temporal_blocks provided but num_time=1; this is ambiguous. "
                "Either pass num_time>1 or drop temporal_blocks."
            )
        if temporal_blocks is not None:
            if temporal_insert_indices is None or len(temporal_insert_indices) != len(
                temporal_blocks
            ):
                raise ValueError(
                    "temporal_insert_indices must match temporal_blocks length."
                )

        # Per-layer timestep: compute per-layer adaln for frame-wise blocks
        # and a mean adaln for ray/global blocks.
        adaln_per_layer: torch.Tensor | None = None
        if per_layer_t is not None and per_layer_t.shape[1] == nview:
            adaln_parts = []
            for l_idx in range(nview):
                adaln_parts.append(self.project_timestep(per_layer_t[:, l_idx]))
            adaln_per_layer = torch.cat(adaln_parts, dim=1)  # [B, L, 6, D]
            adaln_input = adaln_per_layer.mean(dim=1, keepdim=True)  # [B, 1, 6, D]
        else:
            adaln_input = self.project_timestep(timesteps)  # [B, 1, 6, D]

        mode = layer_embed_info["mode"] if layer_embed_info is not None else None
        _ar_mode = (
            layer_embed_info.get("ar_mode", False)
            if layer_embed_info is not None
            else False
        )
        ar_ca_layers = getattr(self, "ar_cross_attn_layers", None)
        use_kv_cache = ar_kv_cache is not None

        if img_context is not None:
            ctx_for_ca = img_context.unsqueeze(1).expand(-1, nview, -1, -1)
            ctx_for_ca = ctx_for_ca.reshape(bs * nview, npatch, dim)
            ctx_pos = xpos.reshape(bs * nview, npatch, 2)
        else:
            ctx_for_ca = None
            ctx_pos = None

        # Temporal-block lookup: global_idx -> temporal_block_idx.
        # ``ar_ca_idx`` starts at 0 and is incremented after each global block,
        # so the global-block index we key on is ``ar_ca_idx`` at the moment
        # immediately after executing the global block (i.e. before increment).
        temporal_idx_map: dict[int, int] = {}
        if temporal_blocks is not None and temporal_insert_indices is not None:
            temporal_idx_map = {
                int(g_idx): t_idx
                for t_idx, g_idx in enumerate(temporal_insert_indices)
            }

        kv_cache_out: list[tuple[torch.Tensor, torch.Tensor]] = []
        per_block_tokens_out: list[torch.Tensor] | None = (
            [] if return_per_block_tokens else None
        )

        # Determine attention type per block based on pattern.
        def _get_attn_type(blk_idx: int) -> str:
            """Return 'frame', 'ray', 'view', or 'global' for this block index.

            r80 4-way pattern semantics (with V folded into batch as B*V):
              * 'frame' / 'layer'  : per-image, per-layer spatial attention along P
              * 'ray'              : per-pixel attention along L
              * 'view'             : per-(layer, pixel) attention along V (cross-view sync)
              * 'global'           : flatten V*L*P (or L*P when V==1) for full attention

            The legacy 2-way ('frame_global') and 3-way ('frame_ray_global')
            paths are unchanged.  The 4-way path treats 'frame' and 'layer'
            as synonyms (the existing reshape sends V*L into batch identically).
            """
            if attn_pattern == "layer_ray_view_global":
                return ("frame", "ray", "view", "global")[blk_idx % 4]
            if attn_pattern == "frame_ray_global":
                return ("frame", "ray", "global")[blk_idx % 3]
            # Default 2-way: frame_global
            return "frame" if blk_idx % 2 == 0 else "global"

        # Track AR CA index (incremented after each global block).
        ar_ca_idx = 0

        for blk_idx, blk in enumerate(self.decoder_blocks):
            attn_type = _get_attn_type(blk_idx)
            is_framewise = attn_type == "frame"
            is_raywise = attn_type == "ray"
            is_viewwise = attn_type == "view"
            is_global = attn_type == "global"

            if mode == "film" and is_framewise:
                tokens = tokens.reshape(bs, nview, npatch, dim)
                gamma = layer_embed_info["gamma"]
                beta = layer_embed_info["beta"]
                with torch.autocast(device_type="cuda", enabled=False):
                    if num_time_int > 1 and gamma.shape[1] == num_layers_only:
                        g5 = gamma.unsqueeze(1)
                        b5 = beta.unsqueeze(1)
                        tokens_v = tokens.reshape(
                            bs, num_time_int, num_layers_only, npatch, dim
                        )
                        tokens_v = tokens_v.float() * (1 + g5) + b5
                        tokens = tokens_v.reshape(bs, nview, npatch, dim)
                    else:
                        tokens = tokens.float() * (1 + gamma) + beta
            elif mode == "film_adaln" and is_framewise:
                tokens = tokens.reshape(bs, nview, npatch, dim)
                gamma = layer_embed_info["gamma"]
                beta = layer_embed_info["beta"]
                with torch.autocast(device_type="cuda", enabled=False):
                    if num_time_int > 1 and gamma.shape[1] == num_layers_only:
                        g5 = gamma.unsqueeze(1)
                        b5 = beta.unsqueeze(1)
                        tokens_v = tokens.reshape(
                            bs, num_time_int, num_layers_only, npatch, dim
                        )
                        tokens_v = tokens_v.float() * (1 + g5) + b5
                        tokens = tokens_v.reshape(bs, nview, npatch, dim)
                    else:
                        tokens = tokens.float() * (1 + gamma) + beta
            elif mode == "per_block":
                tokens = tokens.reshape(bs, nview, npatch, dim)
                emb = layer_embed_info["layer_embeds"][blk_idx]
                if _ar_mode:
                    tokens = tokens + emb[:, None, None, :]
                elif num_time_int > 1 and emb.shape[0] == num_layers_only:
                    emb_bcast = emb[None, None, :, None, :].expand(
                        bs, num_time_int, num_layers_only, 1, dim
                    )
                    emb_bcast = emb_bcast.reshape(1, nview, 1, dim)
                    tokens = tokens + emb_bcast
                else:
                    tokens = tokens + emb[None, :, None, :]
            elif mode == "global_only" and is_global:
                tokens = tokens.reshape(bs, nview, npatch, dim)
                emb = layer_embed_info["layer_embeds"][ar_ca_idx]
                if _ar_mode:
                    tokens = tokens + emb[:, None, None, :]
                elif num_time_int > 1 and emb.shape[0] == num_layers_only:
                    emb_bcast = emb[None, None, :, None, :].expand(
                        bs, num_time_int, num_layers_only, 1, dim
                    )
                    emb_bcast = emb_bcast.reshape(1, nview, 1, dim)
                    tokens = tokens + emb_bcast
                else:
                    tokens = tokens + emb[None, :, None, :]

            if is_framewise:
                tokens = tokens.reshape(bs * nview, npatch, dim)
                xpos = xpos.reshape(bs * nview, npatch, 2)
            elif is_raywise:
                tokens = tokens.reshape(bs, nview, npatch, dim)
                if num_time_int > 1:
                    tokens = tokens.reshape(
                        bs, num_time_int, num_layers_only, npatch, dim
                    )
                    tokens = tokens.permute(0, 1, 3, 2, 4).reshape(
                        bs * num_time_int * npatch, num_layers_only, dim
                    )
                    xpos = xpos.reshape(bs, nview, npatch, 2)
                else:
                    tokens = tokens.permute(0, 2, 1, 3).reshape(
                        bs * npatch, nview, dim
                    )
                    xpos = xpos.reshape(bs, nview, npatch, 2)
            elif is_viewwise:
                # r80: cross-view attention along V axis.
                # Input tokens shape: [B*V, L, P, D]  (V folded into batch).
                # Reshape to [B*L*P, V, D] so attn runs along V independently
                # for every (layer, patch) location.  RoPE is disabled for
                # this block (V is not a spatial axis).
                tokens = tokens.reshape(
                    bs_orig, num_view_int, num_layers_only, npatch, dim
                )
                tokens = tokens.permute(0, 2, 3, 1, 4).contiguous().reshape(
                    bs_orig * num_layers_only * npatch, num_view_int, dim
                )
                # xpos for view block is unused (no RoPE); keep prior shape
                # so it can be restored after the block runs.  We stash the
                # canonical [B, nview, npatch, 2] view here for the unwrap.
                xpos = xpos.reshape(bs, nview, npatch, 2)
            else:
                # global block
                if num_view_int > 1:
                    # r80: cross-view + cross-spatial global attention.
                    # Flatten V·L·P along the seq axis; keep B_orig as the
                    # outer batch dim so attention spans all V views.
                    tokens = tokens.reshape(
                        bs_orig, num_view_int * nview * npatch, dim
                    )
                    xpos = xpos.reshape(
                        bs_orig, num_view_int * nview * npatch, 2
                    )
                elif num_time_int > 1:
                    tokens = tokens.reshape(
                        bs * num_time_int, num_layers_only * npatch, dim
                    )
                    xpos = xpos.reshape(
                        bs * num_time_int, num_layers_only * npatch, 2
                    )
                else:
                    tokens = tokens.reshape(bs, nview * npatch, dim)
                    xpos = xpos.reshape(bs, nview * npatch, 2)

            # --- KV cache for this block ---
            blk_kv = ar_kv_cache[blk_idx] if use_kv_cache else None

            # --- Execute block ---
            # r80: when num_view > 1 and the block is global, tokens have
            # been reshaped to leading dim = bs_orig (not bs).  The adaln
            # repeat factor must be computed against the *current* leading
            # dim, which equals tokens.shape[0]; ratio is then //bs_eff
            # where bs_eff is bs_orig for global+multi-view, else bs.
            if is_global and num_view_int > 1:
                bs_eff = bs_orig
            else:
                bs_eff = bs
            mul_n = tokens.shape[0] // bs_eff
            is_ca_blk = isinstance(_unwrap_ac(blk), DecoderBlockDiTCrossAttn)

            if mode in ("adaln", "film_adaln") and is_framewise:
                layer_adaln = layer_embed_info["layer_adaln"]
                if (
                    num_time_int > 1
                    and not _ar_mode
                    and layer_adaln.shape[1] == num_layers_only
                ):
                    layer_adaln_exp = layer_adaln.repeat(
                        1, num_time_int, 1, 1, 1
                    )
                    adaln_combined = adaln_input.unsqueeze(1) + layer_adaln_exp
                elif adaln_per_layer is not None:
                    adaln_combined = adaln_per_layer.unsqueeze(2) + layer_adaln
                else:
                    adaln_combined = adaln_input.unsqueeze(1) + layer_adaln
                adaln_combined = adaln_combined.reshape(bs * nview, 1, 6, dim)
                if is_ca_blk:
                    blk_out = blk(
                        tokens,
                        xpos,
                        adaln_combined,
                        context=ctx_for_ca,
                        context_pos=ctx_pos,
                        kv_cache_in=blk_kv,
                        return_kv_cache=return_kv_cache,
                    )
                else:
                    kv_kwargs = (
                        dict(kv_cache_in=blk_kv, return_kv_cache=return_kv_cache)
                        if is_ca_blk
                        else {}
                    )
                    blk_out = blk(
                        tokens,
                        xpos,
                        adaln_combined,
                        **kv_kwargs,
                    )
            else:
                if is_framewise and adaln_per_layer is not None:
                    adaln_rep = adaln_per_layer.reshape(bs * nview, 1, 6, dim)
                elif is_global and num_view_int > 1:
                    # r80 cross-view global: collapse the V axis of adaln
                    # to match tokens batch dim (bs_orig).  Across views,
                    # timesteps are identical (set per batch element), so
                    # we take the first V slot (== mean for identical t's).
                    adaln_v = adaln_input.reshape(
                        bs_orig, num_view_int, *adaln_input.shape[1:]
                    )[:, 0]
                    adaln_rep = einops.repeat(
                        adaln_v, "b ... -> (b m) ...", m=mul_n
                    )
                else:
                    adaln_rep = einops.repeat(
                        adaln_input, "b ... -> (b m) ...", m=mul_n
                    )
                if is_framewise and is_ca_blk:
                    blk_out = blk(
                        tokens,
                        xpos,
                        adaln_rep,
                        context=ctx_for_ca,
                        context_pos=ctx_pos,
                        kv_cache_in=blk_kv,
                        return_kv_cache=return_kv_cache,
                    )
                elif is_raywise:
                    blk_out = blk(
                        tokens,
                        None,
                        adaln_rep,
                    )
                else:
                    kv_kwargs = (
                        dict(kv_cache_in=blk_kv, return_kv_cache=return_kv_cache)
                        if is_ca_blk
                        else {}
                    )
                    blk_out = blk(
                        tokens,
                        xpos,
                        adaln_rep,
                        **kv_kwargs,
                    )

            if return_kv_cache:
                tokens, blk_kv_out = blk_out
                kv_cache_out.append(blk_kv_out)
            else:
                tokens = blk_out

            if is_raywise:
                if num_time_int > 1:
                    tokens = tokens.reshape(
                        bs, num_time_int, npatch, num_layers_only, dim
                    )
                    tokens = tokens.permute(0, 1, 3, 2, 4).contiguous()
                    tokens = tokens.reshape(bs, nview, npatch, dim)
                else:
                    tokens = tokens.reshape(bs, npatch, nview, dim)
                    tokens = tokens.permute(0, 2, 1, 3)  # [B, L, P, D]

            if is_viewwise:
                # Unwrap [B*L*P, V, D] back to [B*V, L, P, D] so subsequent
                # blocks see the canonical [bs, nview, npatch, dim] layout.
                tokens = tokens.reshape(
                    bs_orig, num_layers_only, npatch, num_view_int, dim
                )
                tokens = tokens.permute(0, 3, 1, 2, 4).contiguous().reshape(
                    bs, nview, npatch, dim
                )

            if is_global and num_view_int > 1:
                # Restore [bs, nview, npatch, dim] layout after cross-view
                # global attention so subsequent blocks operate on the
                # familiar [B*V, L, P, D] structure.
                tokens = tokens.reshape(bs, nview, npatch, dim)

            if not use_kv_cache and is_global and ar_ca_layers is not None:
                if num_time_int > 1:
                    raise NotImplementedError(
                        "AR cross-attention is not yet supported with num_time>1."
                    )
                if ar_context_per_block is not None:
                    tokens = ar_ca_layers[ar_ca_idx](
                        tokens,
                        ar_context_per_block[ar_ca_idx],
                        qpos=xpos,
                        kvpos=ar_context_pos_per_block[ar_ca_idx],
                    )
                elif ar_context is not None:
                    tokens = ar_ca_layers[ar_ca_idx](
                        tokens, ar_context, qpos=xpos, kvpos=ar_context_pos
                    )

            if (
                is_global
                and num_time_int > 1
                and temporal_blocks is not None
                and ar_ca_idx in temporal_idx_map
            ):
                tokens = tokens.reshape(bs, nview, npatch, dim)
                tokens = temporal_blocks[temporal_idx_map[ar_ca_idx]](
                    tokens,
                    adaln_input,
                    num_time=num_time_int,
                    num_layers=num_layers_only,
                )

            if per_block_tokens_out is not None and is_global:
                per_block_tokens_out.append(tokens.reshape(bs, nview, npatch, dim))

            if is_global:
                ar_ca_idx += 1

        tokens = tokens.reshape(bs, nview, npatch, dim)
        if return_per_block_tokens:
            if return_kv_cache:
                return tokens, global_tokens, kv_cache_out, per_block_tokens_out
            return tokens, global_tokens, per_block_tokens_out
        if return_kv_cache:
            return tokens, global_tokens, kv_cache_out
        return tokens, global_tokens


class ARTokenContextBuilder(nn.Module):
    """Build AR context directly from decoder output tokens of previous layers.

    Replaces ARContextEncoder when ``ar_token_context=True``.  Instead of
    re-encoding pixel-space depth/mask through an independent encoder, this
    module simply applies a LayerNorm and constructs spatial RoPE positions
    for the downstream cross-attention layers (InputContextAttention and
    ARContextCrossAttention) — which remain unchanged.

    Each previous layer contributes ``2P`` tokens (P geo + P mask) from the
    decoder output.  For ``k`` context layers the output shape is
    ``[B, 2*k*P, D]`` — identical to ARContextEncoder's output.
    """

    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(embed_dim)
        self.position_getter = model_utils.PositionGetter()

    def forward(
        self,
        prev_tokens: list[torch.Tensor],
        noise_height: int,
        noise_width: int,
        ctx_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            prev_tokens: list of ``[B, 2P, D]`` decoder-output tensors, one
                per context layer (length ``k``).
            noise_height, noise_width: spatial dims used by RoPE position grid.
            ctx_valid: ``[B, k]`` bool — True for real context layers, False
                for padding (samples whose target layer < k).

        Returns:
            context:  ``[B, 2*k*P, D]`` or *None* when ``k == 0``.
            ctx_pos:  ``[B, 2*k*P, 2]`` or *None*.
        """
        k = len(prev_tokens)
        if k == 0:
            return None, None

        B = prev_tokens[0].shape[0]
        P2 = prev_tokens[0].shape[1]  # 2P

        context = torch.cat(prev_tokens, dim=1)  # [B, k*2P, D]

        if ctx_valid is not None:
            # Expand per-layer validity to per-token: [B, k] → [B, k*2P]
            token_valid = (
                ctx_valid.unsqueeze(-1)
                .expand(-1, -1, P2)
                .reshape(B, k * P2)
                .unsqueeze(-1)
                .float()
            )
            context = context * token_valid

        context = self.norm(context)

        if ctx_valid is not None:
            context = context * token_valid

        # Spatial positions: same (y, x) grid repeated for every token slot.
        # P2 may be 2P (split_token with mask stream) or P (depth_only).
        patch_h = noise_height // self.patch_size
        patch_w = noise_width // self.patch_size
        pos = self.position_getter(
            B, patch_h, patch_w, prev_tokens[0].device
        )  # [B, P, 2]
        num_repeats = (k * P2) // pos.shape[1]
        ctx_pos = pos.repeat(1, num_repeats, 1)  # [B, k*P2, 2]

        return context, ctx_pos


class ARContextEncoder(nn.Module):
    """Encode previous layers' depth and mask as AR context tokens.

    Depth and mask are patchified and projected independently, then tagged
    with per-layer embeddings and type embeddings (0=depth, 1=mask).
    Self-attention blocks with RoPE2D refine context tokens so they can
    reason about spatial relationships and cross-layer interactions.

    Output: ([B, 2*k*P, D], [B, 2*k*P, 2], [B, 2*k*P] | None)
    Returns (None, None, None) when k=0 (predicting layer 0, no context).
    """

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        max_layers: int,
        num_heads: int = 16,
        num_sa_blocks: int = 4,
    ):
        super().__init__()
        p2 = patch_size**2
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.depth_proj = nn.Sequential(
            nn.Linear(p2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mask_proj = nn.Sequential(
            nn.Linear(p2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.layer_embed = nn.Embedding(max_layers, embed_dim)
        self.type_embed = nn.Embedding(2, embed_dim)  # 0=depth, 1=mask

        self.position_getter = model_utils.PositionGetter()
        rope = model_blocks.RoPE2D(freq=100.0)
        blk = model_blocks.DecoderBlockSA(
            embed_dim,
            num_heads,
            norm_layer=nn.LayerNorm,
            use_qk_norm=True,
            rope=rope,
            init_values=0.1,
        )
        self.blocks = nn.ModuleList([deepcopy(blk) for _ in range(num_sa_blocks)])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        prev_depths: torch.Tensor,
        prev_masks: torch.Tensor,
        noise_height: int,
        noise_width: int,
        ctx_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            prev_depths: [B, k, H, W] normalized depth of context layers.
            prev_masks:  [B, k, H, W] binary float (0/1) of context layers.
            noise_height, noise_width: spatial dims for patchify.
            ctx_valid: [B, k] bool — True for real context layers, False for padding.

        Returns:
            context: [B, 2*k*P, D] context tokens, or None if k=0.
            ctx_pos: [B, 2*k*P, 2] spatial positions for RoPE, or None.
            ctx_token_valid: [B, 2*k*P] bool mask (True=valid), or None.
        """
        B, k = prev_depths.shape[:2]
        if k == 0:
            return None, None, None

        p = self.patch_size

        depth_flat = prev_depths.reshape(B * k, -1, 1)
        mask_flat = prev_masks.reshape(B * k, -1, 1)

        depth_patches = model_utils.patchify_image(
            depth_flat, noise_height, noise_width, p
        )[:, 0]  # [B*k, P, p²]
        mask_patches = model_utils.patchify_image(
            mask_flat, noise_height, noise_width, p
        )[:, 0]  # [B*k, P, p²]

        P = depth_patches.shape[1]

        depth_tokens = self.depth_proj(depth_patches)  # [B*k, P, D]
        mask_tokens = self.mask_proj(mask_patches)

        depth_tokens = depth_tokens.reshape(B, k, P, self.embed_dim)
        mask_tokens = mask_tokens.reshape(B, k, P, self.embed_dim)

        layer_ids = torch.arange(k, device=prev_depths.device)
        layer_emb = self.layer_embed(layer_ids)[None, :, None, :]  # [1, k, 1, D]
        depth_tokens = depth_tokens + layer_emb + self.type_embed.weight[0]
        mask_tokens = mask_tokens + layer_emb + self.type_embed.weight[1]

        # Build token-level valid mask [B, 2*k*P]
        if ctx_valid is not None:
            valid_4d = ctx_valid[:, :, None, None]  # [B, k, 1, 1] bool
            depth_tokens = depth_tokens * valid_4d
            mask_tokens = mask_tokens * valid_4d
            token_valid = ctx_valid.unsqueeze(-1).expand(-1, -1, P).reshape(B, k * P)
            token_valid = torch.cat([token_valid, token_valid], dim=1)  # [B, 2*k*P]
        else:
            token_valid = None

        depth_flat_out = depth_tokens.reshape(B, k * P, self.embed_dim)
        mask_flat_out = mask_tokens.reshape(B, k * P, self.embed_dim)
        context = torch.cat([depth_flat_out, mask_flat_out], dim=1)  # [B, 2*k*P, D]

        # Spatial positions: same (y, x) grid repeated for each layer and type
        patch_h = noise_height // p
        patch_w = noise_width // p
        pos = self.position_getter(B, patch_h, patch_w, prev_depths.device)  # [B, P, 2]
        ctx_pos = pos.repeat(1, 2 * k, 1)  # [B, 2*k*P, 2]

        # Self-attention refinement with RoPE; re-zero invalid tokens each block
        valid_mask_broad = (
            token_valid.unsqueeze(-1) if token_valid is not None else None
        )
        for blk in self.blocks:
            context = blk(context, ctx_pos)
            if valid_mask_broad is not None:
                context = context * valid_mask_broad

        context = self.final_norm(context)
        if valid_mask_broad is not None:
            context = context * valid_mask_broad

        return context, ctx_pos, token_valid


class ARContextCrossAttention(nn.Module):
    """Cross-attention from noise tokens to AR context tokens.

    Inserted after global (odd) decoder blocks. When context is None,
    this module is completely skipped — decoder behaves identically to non-AR.
    Each instance has its own ``kv_proj`` so different decoder layers extract
    different information from the same encoder output.  RoPE on Q/K enables
    spatially-aware cross-attention.
    """

    def __init__(self, dim: int, num_heads: int, ls_init: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.kv_proj = nn.Linear(dim, dim)
        self.cross_attn = model_blocks.CrossAttention(
            dim_q=dim,
            dim_kv=dim,
            num_heads=num_heads,
            use_qk_norm=True,
            rope=model_blocks.RoPE2D(freq=100.0),
        )
        self.ls = layer_scale.LayerScale(dim, init_values=ls_init)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        qpos: torch.Tensor | None = None,
        kvpos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [B, N, D], context: [B, K, D] → [B, N, D]"""
        kv = self.kv_proj(self.norm_kv(context))
        ca_out = self.cross_attn(self.norm_q(x), kv, qpos=qpos, kvpos=kvpos)
        return x + self.ls(ca_out)


class InputContextAttention(nn.Module):
    """Cross-attention to make image features context-aware before token construction.

    Applied once to feat (D/2) with AR context (D) as KV. When ar_context is
    None (layer 0) this module is skipped entirely at the call site.
    RoPE on Q/K enables spatially-aware cross-attention.
    """

    def __init__(
        self,
        feat_dim: int,
        ctx_dim: int,
        num_heads: int = 8,
        ls_init: float = 0.1,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_kv = nn.LayerNorm(ctx_dim)
        self.kv_proj = (
            nn.Linear(ctx_dim, feat_dim) if ctx_dim != feat_dim else nn.Identity()
        )
        self.cross_attn = model_blocks.CrossAttention(
            dim_q=feat_dim,
            dim_kv=feat_dim,
            num_heads=num_heads,
            use_qk_norm=True,
            rope=model_blocks.RoPE2D(freq=100.0),
        )
        self.ls = layer_scale.LayerScale(feat_dim, init_values=ls_init)

    def forward(
        self,
        feat: torch.Tensor,
        context: torch.Tensor,
        qpos: torch.Tensor | None = None,
        kvpos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """feat: [B, P, D/2], context: [B, K, D] → [B, P, D/2]"""
        kv = self.kv_proj(self.norm_kv(context))
        ca_out = self.cross_attn(self.norm_q(feat), kv, qpos=qpos, kvpos=kvpos)
        return feat + self.ls(ca_out)


class SplitLatentsProjection(nn.Module):
    """Drop-in replacement for MultilayerBackbone.latents_projection with separate heads.

    The original latents_projection is a single Linear(D, 4*P²) that predicts
    all 4 channels jointly.  This version uses independent linear heads for
    geometry (channels 0-2) and mask (channel 3), avoiding gradient interference
    between the continuous regression and binary classification tasks.

    The geo head always outputs 3 channels per pixel:
      - depth mode:  [depth, -, -] where channels 1-2 are not supervised
      - xyz mode:    [x, y, z]     where all 3 channels are supervised

    Usage: after constructing MultilayerBackbone, replace its latents_projection:
        model.net.latents_projection = SplitLatentsProjection(D, noise_patchify_size)
    """

    def __init__(self, decoder_embed_dim: int, noise_patchify_size: int):
        super().__init__()
        self.num_pixels = noise_patchify_size**2
        self.act = nn.SiLU()
        self.geo_linear = nn.Linear(decoder_embed_dim, 3 * self.num_pixels)
        self.mask_linear = nn.Linear(decoder_embed_dim, self.num_pixels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., P, D]  (decoder output tokens)
        Returns:
            [..., P, 4 * num_pixels]  matching MultilayerBackbone latents_projection output shape

        unpatchify_image rearranges the last dim as ``(ph pw d)`` where d=4,
        so channels must be interleaved per-pixel: [geo0, geo1, geo2, mask] for
        each spatial position within the patch.
        """
        x_act = self.act(x)
        geo = self.geo_linear(x_act)  # [..., P, 3 * num_pixels]
        mask = self.mask_linear(x_act)  # [..., P, num_pixels]
        geo = geo.unflatten(-1, (self.num_pixels, 3))  # [..., P, num_pixels, 3]
        mask = mask.unsqueeze(-1)  # [..., P, num_pixels, 1]
        stacked = torch.cat([geo, mask], dim=-1)  # [..., P, num_pixels, 4]
        return stacked.flatten(-2)


class SplitTransformerProjection(nn.Module):
    """Split geo/mask(/rgb) heads with alternating frame-wise and layer-wise attention.

    Each head (geo, mask, optionally rgb) has its own transformer blocks + final
    linear projection.  Blocks alternate between two attention patterns:
      - Even blocks (frame-wise): patches within each layer attend to each other
        [B*L, P, D] — spatial refinement
      - Odd blocks (layer-wise): layers at each patch position attend to each other
        [B*P, L, D] — cross-layer consistency (seq_len=L=8, near-zero cost)

    This mirrors the alternating attention used in the main decoder
    (decode_to_output_tokens), but applied independently per head to avoid
    gradient interference between continuous regression and mask classification.

    The geo head always outputs 3 channels per pixel:
      - depth mode:  [depth, -, -] where channels 1-2 are not supervised
      - xyz mode:    [x, y, z]     where all 3 channels are supervised

    When predict_color=True, an additional rgb head (3 channels) is added.
    Channel layout: [geo0,geo1,geo2, (rgb0,rgb1,rgb2,) mask]

    Drop-in replacement for latents_projection: forward(tokens_out) → [B,L,P,C*P²].
    """

    def __init__(
        self,
        decoder_embed_dim: int,
        num_decoder_heads: int,
        noise_patchify_size: int,
        num_head_blocks: int = 2,
        rope=None,
        init_values: float | None = None,
        predict_color: bool = False,
        head_film: bool = False,
        num_layers: int = 8,
        head_attn_mode: str = "layerwise",
        head_timestep: bool = False,
        model_task: str = "joint",
        geo_channels: int = 3,
        depth_only: bool = False,
        mask_only: bool = False,
        head_final_proj: str = "linear",
        convhead_dim_upsample: tuple[int, int, int] = (256, 128, 64),
        convhead_num_res_blocks: int = 2,
        # r81 raymap head -------------------------------------------------
        # When use_raymap=True, an additional RaymapHead (ConvHead-isomorphic,
        # input ch = num_layers * decoder_embed_dim, output 6 ch, zero-init
        # final 1x1) is added.  Its output is concatenated to geo_convhead
        # output (broadcasting raymap over the L axis) to form a 9-ch v_t.
        use_raymap: bool = False,
    ):
        super().__init__()
        self.num_pixels = noise_patchify_size**2
        self.noise_patchify_size = noise_patchify_size
        self.num_head_blocks = num_head_blocks
        self.predict_color = predict_color
        self.head_film = head_film
        self.head_attn_mode = head_attn_mode
        self.head_timestep = head_timestep
        self.model_task = model_task
        self.geo_channels = geo_channels
        self.depth_only = depth_only
        self.mask_only = mask_only
        self.has_rope = rope is not None
        self.head_final_proj = head_final_proj
        self.use_raymap = use_raymap
        self.num_layers = num_layers
        if self.has_rope:
            self.position_getter = model_utils.PositionGetter()
        if head_final_proj not in ("linear", "convhead"):
            raise ValueError(
                f"head_final_proj must be 'linear' or 'convhead', got {head_final_proj!r}"
            )

        blk_cls = (
            model_blocks.DecoderBlockDiT if head_timestep else model_blocks.DecoderBlockSA
        )

        def _make_blocks(n):
            blk_list = nn.ModuleList()
            for i in range(n):
                if head_attn_mode == "frame_ray_global":
                    attn_type = ("frame", "ray", "global")[i % 3]
                    blk_rope = None if attn_type == "ray" else rope
                elif head_attn_mode == "global":
                    blk_rope = rope
                else:
                    blk_rope = rope if i % 2 == 0 else None
                blk = blk_cls(
                    decoder_embed_dim,
                    num_decoder_heads,
                    norm_layer=nn.LayerNorm,
                    use_qk_norm=True,
                    rope=blk_rope,
                    init_values=init_values,
                )
                blk_list.append(blk)
            return blk_list

        def _make_film_proj():
            proj = nn.Sequential(
                nn.Linear(decoder_embed_dim, decoder_embed_dim),
                nn.GELU(),
                nn.Linear(decoder_embed_dim, 2 * decoder_embed_dim),
            )
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)
            return proj

        # depth_only=True: only geo head; mask_only=True: only mask head
        has_geo = model_task in ("joint", "geo", "split_token") and not mask_only
        has_mask = model_task in ("joint", "mask", "split_token") and not depth_only

        if has_geo:
            self.geo_blocks = _make_blocks(num_head_blocks)
            if head_final_proj == "convhead":
                # r80: MoGe-style ConvHead replaces the linear unpatchify.
                # The ConvHead does its own per-token rearrange and outputs
                # full-resolution per-pixel predictions, so we keep the
                # transformer head blocks unchanged but skip self.geo_proj.
                # Mask / RGB heads still use the legacy linear projection.
                from wt._core.arch.models.patchhead_utils import (
                    ConvHead as _ConvHead,
                    RaymapHead as _RaymapHead,
                )

                def _identity(x):
                    return x

                self.geo_convhead = _ConvHead(
                    dim_in=decoder_embed_dim,
                    dim_out=[geo_channels],
                    activations=[_identity],
                    pred_names=["geo"],
                    patch_size=noise_patchify_size,
                    dim_upsample=list(convhead_dim_upsample),
                    num_res_blocks=convhead_num_res_blocks,
                )
                logger.info(
                    f"Geo head uses ConvHead (MoGe-style upsample) "
                    f"with dim_upsample={list(convhead_dim_upsample)}, "
                    f"num_res_blocks={convhead_num_res_blocks}, "
                    f"output channels={geo_channels}."
                )
                # r81: parallel RaymapHead — same input tokens as ConvHead, but
                # collapses L into the channel axis and outputs a single per-view
                # 6-channel raymap (zero-init final 1x1).
                if use_raymap:
                    self.raymap_head = _RaymapHead(
                        dim_in=decoder_embed_dim,
                        num_layers=num_layers,
                        patch_size=noise_patchify_size,
                        dim_upsample=list(convhead_dim_upsample),
                        num_res_blocks=convhead_num_res_blocks,
                        out_channels=6,
                    )
                    logger.info(
                        f"RaymapHead enabled (r81): collapsed L*D input "
                        f"({num_layers}*{decoder_embed_dim} = "
                        f"{num_layers * decoder_embed_dim}), output 6 ch, "
                        f"final 1x1 zero-init."
                    )
            else:
                self.geo_proj = nn.Linear(
                    decoder_embed_dim, geo_channels * self.num_pixels
                )
                if use_raymap:
                    raise NotImplementedError(
                        "use_raymap=True requires head_final_proj='convhead' "
                        "(linear head + RaymapHead is not implemented; r81 "
                        "uses convhead by design)."
                    )

        if has_mask:
            self.mask_blocks = _make_blocks(num_head_blocks)
            self.mask_proj = nn.Linear(decoder_embed_dim, self.num_pixels)

        if predict_color:
            self.rgb_blocks = _make_blocks(num_head_blocks)
            self.rgb_proj = nn.Linear(decoder_embed_dim, 3 * self.num_pixels)

        if head_film:
            if has_geo:
                self.geo_layer_embed = nn.Embedding(num_layers, decoder_embed_dim)
                self.geo_film_proj = _make_film_proj()
            if has_mask:
                self.mask_layer_embed = nn.Embedding(num_layers, decoder_embed_dim)
                self.mask_film_proj = _make_film_proj()
            if predict_color:
                self.rgb_layer_embed = nn.Embedding(num_layers, decoder_embed_dim)
                self.rgb_film_proj = _make_film_proj()
            logger.info(
                f"Head FiLM enabled: independent layer embedding + FiLM projection per head "
                f"(num_layers={num_layers}, zero-init)."
            )

    def _get_frame_xpos(
        self, num_patches: int, batch_size: int, device
    ) -> torch.Tensor | None:
        """Get 2D patch positions for frame-wise attention (only when rope is enabled)."""
        if not self.has_rope:
            return None
        patch_h = patch_w = int(num_patches**0.5)
        return self.position_getter(batch_size, patch_h, patch_w, device)

    def _compute_film(
        self,
        layer_embed: nn.Embedding,
        film_proj: nn.Module,
        num_layers: int,
        device,
        target_layer_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute FiLM gamma/beta from a head's own layer embedding."""
        if target_layer_idx is not None:
            layer_emb = layer_embed(target_layer_idx)  # [B, D]
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                film_params = film_proj(layer_emb.float())  # [B, 2*D]
            gamma, beta = film_params.chunk(2, dim=-1)  # each [B, D]
            return gamma[:, None, None, :], beta[:, None, None, :]  # [B, 1, 1, D]
        layer_ids = torch.arange(num_layers, device=device)
        layer_emb = layer_embed(layer_ids)  # [L, D]
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            film_params = film_proj(layer_emb.float())  # [L, 2*D]
        gamma, beta = film_params.chunk(2, dim=-1)  # each [L, D]
        return gamma[None, :, None, :], beta[None, :, None, :]  # [1, L, 1, D]

    def _run_alternating_blocks(
        self,
        x: torch.Tensor,
        blk_list: nn.ModuleList,
        gamma: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        adaln_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run transformer blocks with alternating attention patterns.

        Supports 2-way (frame + layerwise/global) and 3-way (frame + ray + global)
        patterns controlled by ``self.head_attn_mode``.

        Args:
            x: [B, L, P, D]
            gamma: [1, L, 1, D] FiLM scale (optional)
            beta: [1, L, 1, D] FiLM shift (optional)
            adaln_input: [B, 1, 6, D] timestep AdaLN (only used when head_timestep=True)
        Returns:
            [B, L, P, D]
        """
        B, L, P, D = x.shape
        frame_xpos = self._get_frame_xpos(P, B * L, x.device)

        use_global = self.head_attn_mode in ("global", "frame_ray_global")
        use_3way = self.head_attn_mode == "frame_ray_global"
        if use_global and frame_xpos is not None:
            global_xpos = frame_xpos.reshape(B, L * P, 2)
        else:
            global_xpos = None

        def _get_head_attn_type(idx: int) -> str:
            if use_3way:
                return ("frame", "ray", "global")[idx % 3]
            return (
                "frame" if idx % 2 == 0 else ("global" if use_global else "layerwise")
            )

        for blk_idx, blk in enumerate(blk_list):
            attn_type = _get_head_attn_type(blk_idx)

            if attn_type == "frame":
                if gamma is not None:
                    with torch.autocast(device_type="cuda", enabled=False):
                        x = x.float() * (1 + gamma) + beta
                x_flat = x.reshape(B * L, P, D)
                if self.head_timestep:
                    adaln_rep = einops.repeat(adaln_input, "b ... -> (b m) ...", m=L)
                    x_flat = blk(x_flat, frame_xpos, adaln_rep)
                else:
                    x_flat = blk(x_flat, xpos=frame_xpos)
                x = x_flat.reshape(B, L, P, D)
            elif attn_type == "ray":
                x_t = x.permute(0, 2, 1, 3).reshape(B * P, L, D)
                if self.head_timestep:
                    adaln_rep = einops.repeat(adaln_input, "b ... -> (b m) ...", m=P)
                    x_t = blk(x_t, None, adaln_rep)
                else:
                    x_t = blk(x_t, xpos=None)
                x = x_t.reshape(B, P, L, D).permute(0, 2, 1, 3)
            elif attn_type == "global":
                x_flat = x.reshape(B, L * P, D)
                if self.head_timestep:
                    x_flat = blk(x_flat, global_xpos, adaln_input)
                else:
                    x_flat = blk(x_flat, xpos=global_xpos)
                x = x_flat.reshape(B, L, P, D)
            else:
                # layerwise (original 2-way non-global)
                x_t = x.permute(0, 2, 1, 3).reshape(B * P, L, D)
                if self.head_timestep:
                    adaln_rep = einops.repeat(adaln_input, "b ... -> (b m) ...", m=P)
                    x_t = blk(x_t, None, adaln_rep)
                else:
                    x_t = blk(x_t, xpos=None)
                x = x_t.reshape(B, P, L, D).permute(0, 2, 1, 3)
        return x

    def forward(
        self,
        x: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, P, D] decoder output tokens
            adaln_input: [B, 1, 6, D] timestep AdaLN (only when head_timestep=True)
        Returns:
            [B, L, P, C * num_pixels] interleaved per-pixel channels.
            model_task="joint": C=4 [geo0,geo1,geo2,mask] or C=7 with color
            model_task="mask":  C=1 [mask]
            model_task="geo":   C=geo_channels [depth] or [x,y,z]
        """
        if self.model_task == "split_token":
            raise RuntimeError(
                "SplitTransformerProjection.forward() must not be called in "
                "split_token mode — use forward_split() instead."
            )

        L = x.shape[1]
        device = x.device
        has_geo = self.model_task in ("joint", "geo")
        has_mask = self.model_task in ("joint", "mask")

        geo_gamma, geo_beta = None, None
        mask_gamma, mask_beta = None, None
        rgb_gamma, rgb_beta = None, None
        if self.head_film:
            if has_geo:
                geo_gamma, geo_beta = self._compute_film(
                    self.geo_layer_embed, self.geo_film_proj, L, device
                )
            if has_mask:
                mask_gamma, mask_beta = self._compute_film(
                    self.mask_layer_embed, self.mask_film_proj, L, device
                )
            if self.predict_color:
                rgb_gamma, rgb_beta = self._compute_film(
                    self.rgb_layer_embed, self.rgb_film_proj, L, device
                )

        if self.model_task == "mask":
            mask_out = self._run_alternating_blocks(
                x, self.mask_blocks, mask_gamma, mask_beta, adaln_input=adaln_input
            )
            return self.mask_proj(mask_out)  # [B, L, P, num_pixels]

        if self.model_task == "geo":
            geo_out = self._run_alternating_blocks(
                x, self.geo_blocks, geo_gamma, geo_beta, adaln_input=adaln_input
            )
            geo = self.geo_proj(geo_out)  # [B, L, P, geo_channels * num_pixels]
            if self.geo_channels == 1:
                return geo  # [B, L, P, num_pixels]
            geo = geo.unflatten(-1, (self.num_pixels, self.geo_channels))
            return geo.flatten(-2)

        # model_task == "joint" — original behaviour
        geo_out = self._run_alternating_blocks(
            x, self.geo_blocks, geo_gamma, geo_beta, adaln_input=adaln_input
        )
        geo = self.geo_proj(geo_out)

        mask_out = self._run_alternating_blocks(
            x, self.mask_blocks, mask_gamma, mask_beta, adaln_input=adaln_input
        )
        mask = self.mask_proj(mask_out)

        geo = geo.unflatten(-1, (self.num_pixels, 3))
        mask = mask.unsqueeze(-1)

        if self.predict_color:
            rgb_out = self._run_alternating_blocks(
                x, self.rgb_blocks, rgb_gamma, rgb_beta, adaln_input=adaln_input
            )
            rgb = self.rgb_proj(rgb_out)
            rgb = rgb.unflatten(-1, (self.num_pixels, 3))
            stacked = torch.cat([geo, rgb, mask], dim=-1)
        else:
            stacked = torch.cat([geo, mask], dim=-1)
        return stacked.flatten(-2)

    def forward_split(
        self,
        geo_tokens: torch.Tensor,
        mask_tokens: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        target_layer_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run separate geo / mask token streams through their own head blocks.

        Args:
            geo_tokens:  [B, L, P, D]
            mask_tokens: [B, L, P, D]
            adaln_input: [B, 1, 6, D] timestep AdaLN (only when head_timestep=True)
            target_layer_idx: [B] per-sample target layer (AR mode). When provided,
                head FiLM uses this index instead of arange(L).
        Returns:
            (geo_raw, mask_raw)  each [B, L, P, C * num_pixels]
        """
        L = geo_tokens.shape[1]
        device = geo_tokens.device

        geo_gamma, geo_beta = None, None
        mask_gamma, mask_beta = None, None
        if self.head_film:
            geo_gamma, geo_beta = self._compute_film(
                self.geo_layer_embed,
                self.geo_film_proj,
                L,
                device,
                target_layer_idx=target_layer_idx,
            )
            mask_gamma, mask_beta = self._compute_film(
                self.mask_layer_embed,
                self.mask_film_proj,
                L,
                device,
                target_layer_idx=target_layer_idx,
            )

        geo_out = self._run_alternating_blocks(
            geo_tokens, self.geo_blocks, geo_gamma, geo_beta, adaln_input=adaln_input
        )
        mask_out = self._run_alternating_blocks(
            mask_tokens,
            self.mask_blocks,
            mask_gamma,
            mask_beta,
            adaln_input=adaln_input,
        )

        return self.geo_proj(geo_out), self.mask_proj(mask_out)

    def forward_geo_only(
        self,
        geo_tokens: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        target_layer_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run only the geo head (depth_only mode, split_token).

        Args:
            geo_tokens:  [B, L, P, D]  — decoder output (no mask tokens present)
            adaln_input: [B, 1, 6, D]  (only when head_timestep=True)
            target_layer_idx: [B]      (AR mode, for head FiLM)
        Returns:
            head_final_proj == "linear":
                geo_raw [B, L, P, geo_channels * num_pixels]
                (caller calls _unpatchify_prediction → [B, S, C])
            head_final_proj == "convhead":
                geo_raw [B, S=L*H*W, geo_channels]
                (already unpatchified to image resolution; caller's
                 _unpatchify_prediction is a no-op for [B, S, C] shape).
        """
        device = geo_tokens.device
        B, L, P, D = geo_tokens.shape

        geo_gamma, geo_beta = None, None
        if self.head_film:
            geo_gamma, geo_beta = self._compute_film(
                self.geo_layer_embed,
                self.geo_film_proj,
                L,
                device,
                target_layer_idx=target_layer_idx,
            )

        geo_out = self._run_alternating_blocks(
            geo_tokens, self.geo_blocks, geo_gamma, geo_beta, adaln_input=adaln_input
        )

        if self.head_final_proj == "convhead":
            # ConvHead expects tokens shape [B, V*P, D] with V=num_view (=L here).
            # It outputs {pred_name: [B, V, H, W, C], ...}.  We rearrange to
            # [B, V*H*W, C] so it matches the contract of unpatchify_image.
            tokens_flat = geo_out.reshape(B, L * P, D)
            patch = self.noise_patchify_size
            patch_h = patch_w = int(P**0.5)
            assert patch_h * patch_w == P, (
                f"non-square patch grid: P={P}; ConvHead requires P=h*h"
            )
            img_h = patch_h * patch
            img_w = patch_w * patch
            out_dict = self.geo_convhead(
                tokens_flat, num_view=L, img_h=img_h, img_w=img_w
            )
            geo_img = out_dict["geo"]  # [B, L, H, W, geo_channels=3]

            # r81: parallel RaymapHead.  Output is [B, L, H, W, 6] with raymap
            # broadcast over the L axis (per-view-camera but all L layers see
            # the same raymap).  Concat to [B, L, H, W, 9].
            if self.use_raymap:
                raymap_img = self.raymap_head(
                    tokens_flat, num_view=L, img_h=img_h, img_w=img_w
                )                                            # [B, L, H, W, 6]
                # Sanity: shapes match modulo last dim
                assert geo_img.shape[:-1] == raymap_img.shape[:-1], (
                    f"geo {geo_img.shape} vs raymap {raymap_img.shape}"
                )
                geo_img = torch.cat([geo_img, raymap_img], dim=-1)  # [B, L, H, W, 9]

            return einops.rearrange(geo_img, "b v h w c -> b (v h w) c")

        return self.geo_proj(geo_out)  # [B, L, P, geo_channels * num_pixels]

    def forward_mask_only(
        self,
        mask_tokens: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        target_layer_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run only the mask head (mask_only mode, split_token).

        Args:
            mask_tokens: [B, L, P, D]  — decoder output (no geo tokens present)
            adaln_input: [B, 1, 6, D]  (only when head_timestep=True)
            target_layer_idx: [B]      (AR mode, for head FiLM)
        Returns:
            mask_raw: [B, L, P, num_pixels]
        """
        device = mask_tokens.device
        L = mask_tokens.shape[1]

        mask_gamma, mask_beta = None, None
        if self.head_film:
            mask_gamma, mask_beta = self._compute_film(
                self.mask_layer_embed,
                self.mask_film_proj,
                L,
                device,
                target_layer_idx=target_layer_idx,
            )

        mask_out = self._run_alternating_blocks(
            mask_tokens,
            self.mask_blocks,
            mask_gamma,
            mask_beta,
            adaln_input=adaln_input,
        )
        return self.mask_proj(mask_out)  # [B, L, P, num_pixels]


class MultilayerXYZModel(nn.Module):
    """
    Wrapper around :class:`MultilayerBackbone` for multilayer XYZ diffusion.

    The multilayer dimension is treated as the "view" dimension expected by
    MultilayerBackbone / FMLossWrapper. The input RGB is repeated across layers to match
    this dimension during training.
    """

    def __init__(
        self,
        num_layers: int = 16,
        decoder_embed_dim: int = 1024,
        num_decoder_blocks: int = 12,
        num_decoder_heads: int = 16,
        patch_size: int = 14,
        freeze_encoder: bool = True,
        img_token_scale: float = 0.3,
        use_x0_prediction: bool = True,
        use_split_head: bool = False,
        split_head_mode: str = "none",
        num_head_blocks: int = 2,
        use_rope: bool = False,
        ls_init_values: float | None = None,
        layer_embed_mode: str = "input_only",
        cfm_mask: bool = False,
        cfm_noise_type: str = "fixed_0.5",
        predict_color: bool = False,
        head_film: bool = False,
        head_attn_mode: str = "layerwise",
        head_timestep: bool = False,
        head_final_proj: str = "linear",
        use_pose_head: bool = False,
        pose_head_layers: int = 5,
        pose_head_hidden_dim: int | None = None,
        use_cross_attn: bool = False,
        cross_attn_ls_init: float = 0.1,
        model_task: str = "joint",
        output_mode: str = "depth",
        use_ar: bool = False,
        ar_ls_init: float = 0.1,
        ar_encoder_num_blocks: int = 4,
        ar_token_context: bool = False,
        ar_per_block_context: bool = False,
        depth_only: bool = False,
        mask_only: bool = False,
        attention_pattern: str = "frame_global",
        use_temporal_blocks: bool = False,
        temporal_ls_init: float = 1e-5,
        temporal_insert_indices: tuple[int, ...] | list[int] | None = None,
        temporal_mlp_ratio: float = 4.0,
        temporal_use_qk_norm: bool = True,
        head_activation_checkpoint: bool = False,
        # r81 raymap diffusion ---------------------------------------------
        # When True, the geo diffusion target widens 3 → 9 channels:
        #   ch 0..2  = XYZ (per-layer, same as r80)
        #   ch 3..5  = ray direction in world frame (broadcast over L)
        #   ch 6..8  = ray origin    in world frame (broadcast over L)
        # See auto-experiments/r81_raymap_diffusion/DESIGN.md for the full
        # rationale.  At use_raymap=False this entire code path is dormant
        # and the model is bit-equivalent to r80.
        use_raymap: bool = False,
        # Inference-only release: when True (default), skip downloading MoGe
        # encoder weights at model construction time -- the encoder state is
        # restored from the checkpoint anyway.  Flip to False only when
        # fine-tuning from a fresh ``MultilayerXYZModel`` (in that case set
        # the ``MOGE_LOCAL_ZOO`` env var to a directory containing the
        # official MoGe-ViT-L safetensors, or rely on
        # ``MoGeModel.from_pretrained`` to download them itself).
        inference_mode: bool = True,
        # Legacy args (ignored, for backward compatibility)
        hidden_dim: int | None = None,
        num_blocks: int | None = None,
        num_heads: int | None = None,
        use_pretrained_encoder: bool = True,
    ):
        super().__init__()

        if hidden_dim is not None:
            decoder_embed_dim = hidden_dim
        if num_blocks is not None:
            num_decoder_blocks = num_blocks
        if num_heads is not None:
            num_decoder_heads = num_heads

        # Backward compatibility: use_split_head=True maps to split_head_mode="linear"
        if use_split_head and split_head_mode == "none":
            split_head_mode = "linear"

        # r80: model_task="xyz_only" is shorthand for split_token + depth_only.
        # All downstream code paths still see model_task="split_token" with
        # depth_only=True (i.e. zero new code branches created for xyz_only).
        if model_task == "xyz_only":
            model_task = "split_token"
            depth_only = True
            output_mode = "xyz"
            logger.info(
                "model_task='xyz_only' expanded to "
                "model_task='split_token', depth_only=True, output_mode='xyz'."
            )

        if predict_color and split_head_mode != "transformer":
            raise ValueError(
                f"predict_color=True requires split_head_mode='transformer', "
                f"got '{split_head_mode}'"
            )

        if model_task != "joint" and split_head_mode != "transformer":
            raise ValueError(
                f"model_task='{model_task}' requires split_head_mode='transformer', "
                f"got '{split_head_mode}'"
            )

        if depth_only and mask_only:
            raise ValueError("depth_only and mask_only are mutually exclusive.")

        if attention_pattern == "frame_ray_global" and num_decoder_blocks % 3 != 0:
            raise ValueError(
                f"attention_pattern='frame_ray_global' requires num_decoder_blocks "
                f"divisible by 3, got {num_decoder_blocks}"
            )

        if (
            attention_pattern == "layer_ray_view_global"
            and num_decoder_blocks % 4 != 0
        ):
            raise ValueError(
                f"attention_pattern='layer_ray_view_global' requires num_decoder_blocks "
                f"divisible by 4, got {num_decoder_blocks}"
            )

        if head_attn_mode == "frame_ray_global" and num_head_blocks % 3 != 0:
            raise ValueError(
                f"head_attn_mode='frame_ray_global' requires num_head_blocks "
                f"divisible by 3, got {num_head_blocks}"
            )

        if use_cross_attn and attention_pattern in (
            "frame_ray_global",
            "layer_ray_view_global",
        ):
            raise ValueError(
                f"use_cross_attn=True is incompatible with attention_pattern='{attention_pattern}'. "
                "Cross-attn block replacement overwrites ray/view-wise blocks with RoPE-enabled blocks, "
                "causing assertion failures when xpos=None is passed."
            )

        if model_task == "split_token" and use_cross_attn:
            raise ValueError(
                "model_task='split_token' is not compatible with use_cross_attn=True."
            )

        self.num_layers = num_layers
        self.patch_size = patch_size
        self.predict_color = predict_color
        self.layer_embed_mode = layer_embed_mode
        self.use_cross_attn = use_cross_attn
        self._decoder_embed_dim = decoder_embed_dim
        self.model_task = model_task
        self.attention_pattern = attention_pattern

        # r81: validate use_raymap dependencies up-front
        if use_raymap and not (model_task == "split_token" and depth_only and output_mode == "xyz"):
            raise ValueError(
                "use_raymap=True requires model_task='split_token' (or 'xyz_only') "
                f"+ depth_only=True + output_mode='xyz'.  Got "
                f"model_task='{model_task}', depth_only={depth_only}, "
                f"output_mode='{output_mode}'."
            )
        self.use_raymap = use_raymap

        if model_task == "mask":
            noise_channel = 1
            cfm_mask = True
        elif model_task == "geo":
            noise_channel = 3 if output_mode == "xyz" else 1
            cfm_mask = False
        elif model_task == "split_token":
            # r81: nc_geo widens 3 → 9 when use_raymap.  XYZ keeps 3, raymap adds 6.
            nc_geo_xyz = 3 if output_mode == "xyz" else 1
            nc_geo = nc_geo_xyz + (6 if use_raymap else 0)
            if depth_only:
                noise_channel = nc_geo  # depth only, no mask channel
                cfm_mask = False
            elif mask_only:
                noise_channel = 1  # mask only, no geo channel
                cfm_mask = True
            else:
                noise_channel = nc_geo + 1
                cfm_mask = True
        else:
            noise_channel = 7 if predict_color else 4

        if model_task == "split_token":
            # XYZ ConvHead keeps producing 3 ch; raymap is a separate head
            # in r81 (Phase 3).  Phase 2 stub: geo_channels stays at the
            # XYZ ConvHead width for now.  When use_raymap is True the model
            # output is later concatenated with raymap_head output for a 9-ch
            # v_t in compute_diffusion_loss / inference.
            geo_channels = nc_geo_xyz
        elif model_task == "geo":
            geo_channels = noise_channel
        else:
            geo_channels = 3

        # Stash widths for downstream (loss/inference) introspection
        self._nc_geo_total = nc_geo            # 3 or 9
        self._nc_geo_xyz = nc_geo_xyz if model_task == "split_token" else None
        self._noise_channel = noise_channel

        self.cfm_mask = cfm_mask
        self.cfm_noise_type = cfm_noise_type

        encoder_model = "moge" if use_pretrained_encoder else "null"
        self.net = MultilayerBackbonePatched(
            model_type="diffusion",
            img_fusion_mode="group_concat",
            fuse_raw_rgb=True,
            encoder_model=encoder_model,
            patch_size=patch_size,
            num_decoder_blocks=num_decoder_blocks,
            num_decoder_heads=num_decoder_heads,
            decoder_embed_dim=decoder_embed_dim,
            img_token_scale=img_token_scale,
            positional_encoding="rope" if use_rope else "none",
            ls_init_values=ls_init_values,
            use_raymap=False,
            use_dense_raycond=False,
            freeze_encoder=freeze_encoder,
            noise_channel=noise_channel,
            noise_patchify_size=patch_size,
            use_x0_prediction=use_x0_prediction,
            use_activation_checkpoint=True,
            inference_mode=inference_mode,
        )

        if cfm_mask and not use_x0_prediction:
            raise ValueError(
                "cfm_mask requires use_x0_prediction=True (endpoint parameterization)"
            )
        self.cfm_mask = cfm_mask
        self.net.cfm_mask = cfm_mask
        self.net.cfm_noise_type = cfm_noise_type
        self.net.attention_pattern = attention_pattern
        if attention_pattern == "frame_ray_global":
            for i, blk in enumerate(self.net.decoder_blocks):
                if i % 3 == 1:
                    _unwrap_ac(blk).attn.rope = None
            logger.info(
                f"Decoder attention pattern: {attention_pattern} "
                f"(disabled RoPE on {num_decoder_blocks // 3} ray-wise blocks)"
            )
        elif attention_pattern == "layer_ray_view_global":
            # 4-way: blocks 4i+0=layer (RoPE), 4i+1=ray (no RoPE),
            #        4i+2=view (no RoPE, NEW), 4i+3=global (RoPE).
            # View blocks are zero-initialised (LayerScale=1e-5 if available)
            # so the first forward equals the per-view-independent baseline,
            # ensuring V=1 inference matches a 3-way model bit-equivalent and
            # warm-starting from a 3-way ckpt does not perturb predictions.
            view_block_indices = list(range(2, num_decoder_blocks, 4))
            ray_block_indices = list(range(1, num_decoder_blocks, 4))
            for i, blk in enumerate(self.net.decoder_blocks):
                if i in ray_block_indices or i in view_block_indices:
                    _unwrap_ac(blk).attn.rope = None
            for i in view_block_indices:
                blk_inner = _unwrap_ac(self.net.decoder_blocks[i])
                ls1 = getattr(blk_inner, "ls1", None)
                if ls1 is not None and hasattr(ls1, "gamma"):
                    nn.init.constant_(ls1.gamma, 1e-5)
                ls2 = getattr(blk_inner, "ls2", None)
                if ls2 is not None and hasattr(ls2, "gamma"):
                    nn.init.constant_(ls2.gamma, 1e-5)
            logger.info(
                f"Decoder attention pattern: {attention_pattern} "
                f"(disabled RoPE on {len(ray_block_indices)} ray + "
                f"{len(view_block_indices)} view blocks; "
                f"view blocks LayerScale init 1e-5)"
            )
        elif attention_pattern != "frame_global":
            logger.info(f"Decoder attention pattern: {attention_pattern}")
        if cfm_mask:
            logger.info(
                "CFM mask mode enabled: mask channel uses sigmoid endpoint parameterization."
            )

        # --- Split-token projections & type embedding ---
        if model_task == "split_token":
            p2 = patch_size**2
            self.net.use_split_token = True
            self.net.nc_geo = nc_geo
            self.net.depth_only = depth_only
            self.net.mask_only = mask_only
            if mask_only:
                mask_raw_dim = 1 * p2 + 3 * p2  # mask patches + raw RGB
                self.net.mask_noise_projection = nn.Sequential(
                    nn.Linear(mask_raw_dim, decoder_embed_dim // 2),
                )
                logger.info(
                    f"Split-token mask_only mode: mask_raw_dim={mask_raw_dim}, "
                    f"no geo_noise_projection/type_embed."
                )
            elif depth_only:
                geo_raw_dim = nc_geo * p2 + 3 * p2  # geo patches + raw RGB
                self.net.geo_noise_projection = nn.Sequential(
                    nn.Linear(geo_raw_dim, decoder_embed_dim // 2),
                )
                logger.info(
                    f"Split-token depth_only mode: geo_raw_dim={geo_raw_dim}, "
                    f"nc_geo={nc_geo}, no mask_noise_projection/type_embed."
                )
            else:
                geo_raw_dim = nc_geo * p2 + 3 * p2  # geo patches + raw RGB
                self.net.geo_noise_projection = nn.Sequential(
                    nn.Linear(geo_raw_dim, decoder_embed_dim // 2),
                )
                mask_raw_dim = 1 * p2 + 3 * p2  # mask patches + raw RGB
                self.net.mask_noise_projection = nn.Sequential(
                    nn.Linear(mask_raw_dim, decoder_embed_dim // 2),
                )
                self.net.type_embed = nn.Embedding(2, decoder_embed_dim)
                logger.info(
                    f"Split-token mode: geo_raw_dim={geo_raw_dim}, mask_raw_dim={mask_raw_dim}, "
                    f"nc_geo={nc_geo}, type_embed dim={decoder_embed_dim}."
                )
            # Remove base noise_projection (replaced by split geo/mask projections;
            # keeping it would cause DDP errors from unused parameters)
            if hasattr(self.net, "noise_projection"):
                del self.net.noise_projection

        if split_head_mode == "linear":
            self.net.latents_projection = SplitLatentsProjection(
                decoder_embed_dim=decoder_embed_dim,
                noise_patchify_size=patch_size,
            )
            logger.info("Using split depth/mask heads (SplitLatentsProjection).")
        elif split_head_mode == "transformer":
            self.net.latents_projection = SplitTransformerProjection(
                decoder_embed_dim=decoder_embed_dim,
                num_decoder_heads=num_decoder_heads,
                noise_patchify_size=patch_size,
                num_head_blocks=num_head_blocks,
                rope=self.net.rope,
                init_values=ls_init_values,
                predict_color=predict_color,
                head_film=head_film,
                num_layers=num_layers,
                head_attn_mode=head_attn_mode,
                head_timestep=head_timestep,
                model_task=model_task,
                geo_channels=geo_channels,
                depth_only=depth_only,
                mask_only=mask_only,
                head_final_proj=head_final_proj,
                use_raymap=use_raymap,
            )
            if model_task == "mask" or mask_only:
                head_desc = "mask-only"
            elif model_task == "geo" or depth_only:
                head_desc = f"geo-only({geo_channels}ch)"
            else:
                head_desc = "geo+rgb+mask" if predict_color else "geo+mask"
            film_desc = " + head_film" if head_film else ""
            attn_desc = f", attn={head_attn_mode}"
            ts_desc = " + timestep" if head_timestep else ""
            if head_attn_mode == "frame_ray_global":
                pattern_desc = "frame-wise/ray-wise/global"
            elif head_attn_mode == "global":
                pattern_desc = "frame-wise/global"
            else:
                pattern_desc = "frame-wise/layer-wise"
            logger.info(
                f"Using split transformer heads (SplitTransformerProjection, "
                f"{num_head_blocks} blocks, {head_desc}{film_desc}{attn_desc}{ts_desc}, "
                f"alternating {pattern_desc})."
            )

            # Full AC on every head block.  Head blocks run inside an fp32
            # autocast and see [B*T, L, P, D] which scales linearly with T,
            # so their activations dominate at clip_length>=8; AC is a pure
            # win when clip training is enabled.  Gated behind a flag so
            # the default behaviour (False) stays bit-for-bit identical to
            # the pre-patch model, and any smoke test can A/B compare.
            if head_activation_checkpoint:
                proj = self.net.latents_projection
                wrapped_names = []
                for blks_name in ("geo_blocks", "mask_blocks", "rgb_blocks"):
                    blks = getattr(proj, blks_name, None)
                    if blks is None:
                        continue
                    for i in range(len(blks)):
                        blks[i] = activation_checkpoint.apply_activation_checkpointing(
                            blks[i], mode="full"
                        )
                    wrapped_names.append(f"{blks_name}x{len(blks)}")
                # r80: also wrap ConvHead's upsample_blocks (the bf16 conv stack
                # that goes from [B*V, 1536, P, P] all the way to
                # [B*V, 64, 8P, 8P]).  At image_size=504, V=4 these
                # intermediate feature maps are the second-largest activation
                # source after the decoder blocks.  AC here adds another ~10%
                # step time but cuts ~5–8 GB of saved activations on H100.
                conv_head = getattr(proj, "geo_convhead", None)
                if conv_head is not None:
                    for i in range(len(conv_head.upsample_blocks)):
                        conv_head.upsample_blocks[i] = (
                            activation_checkpoint.apply_activation_checkpointing(
                                conv_head.upsample_blocks[i], mode="full"
                            )
                        )
                    wrapped_names.append(
                        f"geo_convhead.upsample_blocksx{len(conv_head.upsample_blocks)}"
                    )
                # r81: also wrap RaymapHead's upsample_blocks (same activation
                # cost regime as the geo_convhead upsample stack since the
                # collapsed-L feature map size matches at every scale).
                raymap_head = getattr(proj, "raymap_head", None)
                if raymap_head is not None:
                    for i in range(len(raymap_head.upsample_blocks)):
                        raymap_head.upsample_blocks[i] = (
                            activation_checkpoint.apply_activation_checkpointing(
                                raymap_head.upsample_blocks[i], mode="full"
                            )
                        )
                    wrapped_names.append(
                        f"raymap_head.upsample_blocksx{len(raymap_head.upsample_blocks)}"
                    )
                logger.info(
                    f"Head activation checkpointing enabled on "
                    f"SplitTransformerProjection: {', '.join(wrapped_names)}"
                )
        else:
            logger.info("Using shared latents_projection (original MultilayerBackbone head).")

        # --- Cross-attention mode: replace even decoder blocks, add noise MLP ---
        if use_cross_attn and layer_embed_mode == "input_only":
            logger.warning(
                "use_cross_attn=True with layer_embed_mode='input_only' means NO layer "
                "conditioning: the layer embedding is not added to image features (no "
                "img_tokens in this path) and no FiLM/AdaLN is used. Consider using "
                "layer_embed_mode='film' or 'film_adaln' for proper layer differentiation."
            )

        if use_cross_attn:
            self.net.use_cross_attn = True

            # Noise projection MLP: noise_channel * patch² → D
            noise_dim = noise_channel * patch_size**2
            self.net.noise_projection_ca = nn.Sequential(
                nn.Linear(noise_dim, decoder_embed_dim),
                nn.SiLU(),
                nn.Linear(decoder_embed_dim, decoder_embed_dim),
            )

            # LayerNorm for image context (KV)
            self.img_context_norm = nn.LayerNorm(decoder_embed_dim)

            # Replace even (frame-wise) decoder blocks with cross-attn variants
            rope = self.net.rope
            for i in range(0, num_decoder_blocks, 2):
                new_blk = DecoderBlockDiTCrossAttn(
                    dim=decoder_embed_dim,
                    num_heads=num_decoder_heads,
                    norm_layer=nn.LayerNorm,
                    use_qk_norm=True,
                    rope=rope,
                    init_values=ls_init_values,
                    cross_attn_ls_init=cross_attn_ls_init,
                )
                new_blk = activation_checkpoint.apply_activation_checkpointing(
                    new_blk, mode="full"
                )
                self.net.decoder_blocks[i] = new_blk

            logger.info(
                f"Cross-attention enabled: {num_decoder_blocks // 2} frame-wise blocks "
                f"upgraded to DecoderBlockDiTCrossAttn (LayerScale={cross_attn_ls_init}). "
                f"Noise projection: {noise_dim}→{decoder_embed_dim} MLP (bypasses concat fusion)."
            )

        self.layer_embed = nn.Embedding(num_layers, decoder_embed_dim)

        # --- Dense layer-embedding modules (zero-initialized for baseline equivalence) ---
        if layer_embed_mode == "adaln":
            proj = nn.Linear(decoder_embed_dim, decoder_embed_dim * 6)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.layer_adaln_proj = nn.Sequential(nn.SiLU(), proj)
            logger.info(
                "Layer embed mode: adaln (per-layer AdaLN in frame-wise blocks, zero-init)."
            )
        elif layer_embed_mode == "per_block":
            self.dense_layer_embeds = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(num_layers, decoder_embed_dim))
                    for _ in range(num_decoder_blocks)
                ]
            )
            logger.info(
                f"Layer embed mode: per_block ({num_decoder_blocks} independent embeddings, zero-init)."
            )
        elif layer_embed_mode == "global_only":
            _global_divisor = _attn_pattern_period(attention_pattern)
            num_global_blocks = num_decoder_blocks // _global_divisor
            self.dense_layer_embeds = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(num_layers, decoder_embed_dim))
                    for _ in range(num_global_blocks)
                ]
            )
            logger.info(
                f"Layer embed mode: global_only ({num_global_blocks} embeddings at global blocks, zero-init)."
            )
        elif layer_embed_mode == "film":
            film_proj = nn.Sequential(
                nn.Linear(decoder_embed_dim, decoder_embed_dim),
                nn.GELU(),
                nn.Linear(decoder_embed_dim, 2 * decoder_embed_dim),
            )
            nn.init.zeros_(film_proj[-1].weight)
            nn.init.zeros_(film_proj[-1].bias)
            self.layer_film_proj = film_proj
            logger.info(
                "Layer embed mode: film (per-layer FiLM scale/shift at frame-wise blocks, zero-init)."
            )
        elif layer_embed_mode == "film_adaln":
            film_proj = nn.Sequential(
                nn.Linear(decoder_embed_dim, decoder_embed_dim),
                nn.GELU(),
                nn.Linear(decoder_embed_dim, 2 * decoder_embed_dim),
            )
            nn.init.zeros_(film_proj[-1].weight)
            nn.init.zeros_(film_proj[-1].bias)
            self.layer_film_proj = film_proj

            adaln_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim * 6)
            nn.init.zeros_(adaln_proj.weight)
            nn.init.zeros_(adaln_proj.bias)
            self.layer_adaln_proj = nn.Sequential(nn.SiLU(), adaln_proj)
            logger.info(
                "Layer embed mode: film_adaln (FiLM + AdaLN at frame-wise blocks, zero-init)."
            )
        elif layer_embed_mode == "input_only":
            logger.info(
                "Layer embed mode: input_only (baseline, single addition before decoder)."
            )
        else:
            raise ValueError(f"Unknown layer_embed_mode: {layer_embed_mode!r}")

        # --- AR (Autoregressive) mode ---
        self.use_ar = use_ar
        self.ar_token_context = ar_token_context
        self.ar_per_block_context = ar_per_block_context
        self.mask_only = mask_only
        if use_ar:
            if model_task != "split_token":
                raise ValueError("use_ar=True requires model_task='split_token'")

            _global_div = _attn_pattern_period(attention_pattern)
            num_global_blocks = num_decoder_blocks // _global_div

            if ar_token_context:
                self.ar_token_context_builder = ARTokenContextBuilder(
                    embed_dim=decoder_embed_dim,
                    patch_size=patch_size,
                )
                logger.info(
                    "AR token context enabled: ARTokenContextBuilder (LayerNorm only, "
                    "replaces ARContextEncoder)."
                )
            else:
                self.ar_context_encoder = ARContextEncoder(
                    patch_size=patch_size,
                    embed_dim=decoder_embed_dim,
                    max_layers=num_layers,
                    num_heads=num_decoder_heads,
                    num_sa_blocks=ar_encoder_num_blocks,
                )

            self.net.ar_cross_attn_layers = nn.ModuleList(
                [
                    ARContextCrossAttention(
                        dim=decoder_embed_dim,
                        num_heads=num_decoder_heads,
                        ls_init=ar_ls_init,
                    )
                    for _ in range(num_global_blocks)
                ]
            )

            self.net.input_context_attn = InputContextAttention(
                feat_dim=decoder_embed_dim // 2,
                ctx_dim=decoder_embed_dim,
                num_heads=num_decoder_heads,
                ls_init=ar_ls_init,
            )

            if ar_per_block_context and not ar_token_context:
                raise ValueError(
                    "ar_per_block_context=True requires ar_token_context=True"
                )

            if not ar_token_context:
                logger.info(
                    f"AR mode enabled: context encoder ({ar_encoder_num_blocks} SA blocks, "
                    f"RoPE) + {num_global_blocks} decoder cross-attn (per-layer kv_proj, "
                    f"RoPE) + input context attn (RoPE) (LayerScale init={ar_ls_init})"
                )
            else:
                _pbc = " + per-block context" if ar_per_block_context else ""
                logger.info(
                    f"AR mode enabled: token context builder{_pbc} + "
                    f"{num_global_blocks} decoder cross-attn (per-layer kv_proj, "
                    f"RoPE) + input context attn (RoPE) (LayerScale init={ar_ls_init})"
                )

        self.use_temporal_blocks = use_temporal_blocks
        if use_temporal_blocks:
            _global_div_t = _attn_pattern_period(attention_pattern)
            num_global_blocks_t = num_decoder_blocks // _global_div_t
            # Default (temporal_insert_indices=None): insert one temporal block
            # after *every* global attention block.  This matches modern video
            # DiTs (CogVideoX / AnimateDiff / OpenSora) where spatial and
            # temporal attention are strictly interleaved, and it is "free" in
            # terms of correctness because every temporal block is LayerScale
            # zero-initialized (init=temporal_ls_init, default 1e-5) — so at
            # iter 0 every block is ≈ identity and the single-frame behaviour
            # of the warm-start checkpoint is exactly preserved.  Gradients
            # will automatically raise LS on blocks that help and leave the
            # rest near zero ("learned pruning").
            #
            # Callers can still override with e.g.
            #   temporal_insert_indices=range(num_global_blocks_t - 7,
            #                                 num_global_blocks_t)
            # to get the previous "last-7-only" behaviour (cheaper at the cost
            # of less temporal capacity in the feature-extraction half of the
            # decoder).
            if temporal_insert_indices is None:
                temporal_insert_indices = tuple(range(num_global_blocks_t))
            else:
                temporal_insert_indices = tuple(int(x) for x in temporal_insert_indices)
                for gi in temporal_insert_indices:
                    if gi < 0 or gi >= num_global_blocks_t:
                        raise ValueError(
                            f"temporal_insert_indices={temporal_insert_indices} "
                            f"contains out-of-range index {gi} "
                            f"(expected 0..{num_global_blocks_t - 1})"
                        )

            temporal_blocks = nn.ModuleList()
            for _ in temporal_insert_indices:
                tblk = TemporalAttentionBlock(
                    dim=decoder_embed_dim,
                    num_heads=num_decoder_heads,
                    mlp_ratio=temporal_mlp_ratio,
                    init_values=temporal_ls_init,
                    use_qk_norm=temporal_use_qk_norm,
                )
                tblk = activation_checkpoint.apply_activation_checkpointing(
                    tblk, mode="full"
                )
                temporal_blocks.append(tblk)
            self.temporal_blocks = temporal_blocks
            self.temporal_insert_indices = temporal_insert_indices
            _all_global = tuple(range(num_global_blocks_t))
            _coverage = (
                "ALL"
                if temporal_insert_indices == _all_global
                else f"{len(temporal_insert_indices)}/{num_global_blocks_t}"
            )
            logger.info(
                "Temporal blocks enabled",
                num_temporal_blocks=len(temporal_blocks),
                num_global_blocks=num_global_blocks_t,
                coverage=_coverage,
                insert_indices=list(temporal_insert_indices),
                ls_init=temporal_ls_init,
            )
        else:
            self.temporal_blocks = None
            self.temporal_insert_indices = None

        # r80: Pose head — 5-layer MLP from per-view pooled token to 9D
        # (6D continuous rotation [Zhou et al. CVPR'19] + 3D translation).
        # Reference-free; final 3×3 R is recovered via Gram-Schmidt
        # orthogonalisation of the 6D rep.  We previously emitted 12D
        # (9D mat + 3D t) and orthogonalised via SVD — but SVD's
        # backward is undefined for matrices with repeated singular
        # values (e.g. R == I at init), producing NaN gradients on the
        # very first step.  6D + Gram-Schmidt is the canonical fix and
        # has continuous gradients everywhere away from a measure-zero
        # set; combined with the identity-bias init below, the model
        # starts from R == I with a stable backward path.
        self.use_pose_head = use_pose_head
        if use_pose_head:
            hidden_dim = pose_head_hidden_dim or decoder_embed_dim
            ph_layers: list[nn.Module] = []
            in_dim = decoder_embed_dim
            for _ in range(max(1, pose_head_layers - 1)):
                ph_layers.append(nn.LayerNorm(in_dim))
                ph_layers.append(nn.Linear(in_dim, hidden_dim))
                ph_layers.append(nn.GELU())
                in_dim = hidden_dim
            # Final 9D projection (no activation).
            #
            # Initialisation strategy — *small Gaussian weight* + identity bias:
            #   weight ~ N(0, 1e-3),  bias = (1, 0, 0, 0, 1, 0, 0, 0, 0).
            #
            # Why NOT zero-init weight (the obvious choice)?  A zero matrix
            # in the *output* layer creates a "gradient bottleneck": forward
            # is fine (output = bias = identity rotation + zero translation),
            # but backward propagates dL/dx = (dL/dpose) @ W^T = 0 to all
            # upstream layers.  This means
            #   (a) the 4 hidden LayerNorm+Linear+GELU blocks of pose_head
            #       receive *zero* gradient and never learn, and
            #   (b) decoder backbone receives *zero* gradient from pose/world
            #       losses — making the cross-view supervision invisible.
            # We diagnosed exactly this on r80 production runs (loss_rot
            # stuck at ~1.6, rot_deg ~93° = uniform-random for hundreds
            # of iterations while loss_xyz fell normally).
            #
            # The small-Gaussian weight (sigma=1e-3) keeps forward output
            # essentially equal to the identity bias (||W·x|| ~ 1e-3·sqrt(D) ·
            # ||x|| ≈ 0.04 ≪ ||b|| = 1) while restoring a non-zero W^T for
            # the backward pass.  See IMPLEMENTATION.md §12 for the full
            # diagnosis + the smoke test that pins this behaviour.
            final_proj = nn.Linear(in_dim, 9)
            nn.init.normal_(final_proj.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final_proj.bias)
            with torch.no_grad():
                final_proj.bias[0] = 1.0  # b1.x = 1
                final_proj.bias[4] = 1.0  # a2.y = 1 (post-Gram-Schmidt: b2.y = 1)
            ph_layers.append(final_proj)
            self.pose_head = nn.Sequential(*ph_layers)
            logger.info(
                f"PoseHead enabled: {pose_head_layers}-layer MLP "
                f"(hidden={hidden_dim}, out=9 = 6D rot + 3D trans), "
                f"6D Gram-Schmidt orthogonalisation, "
                f"weight init N(0, 1e-3), bias init = identity rotation."
            )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "MultilayerXYZModel initialized",
            total_params=total_params / 1e6,
            trainable_params=trainable_params / 1e6,
        )

    def predict_pose(
        self,
        decoder_tokens: torch.Tensor,
        num_view: int,
    ) -> dict[str, torch.Tensor]:
        """Predict per-view camera extrinsics from decoder tokens.

        r80 Pi3-style reference-free pose prediction: each view independently
        predicts a 6D rotation (Gram-Schmidt-orthogonalised → 3×3) and a
        3D translation, both in the m_global-normalized frame.  Caller
        computes pairwise relative poses and the corresponding loss; no
        canonical "view-0 == identity" assumption is baked in here.

        Note on the 6D representation: previously we used 9D + SVD
        orthogonalisation (matching DA3 / VGGT / Pi3 baselines).  However
        ``torch.linalg.svd``'s backward is undefined for matrices with
        repeated singular values (e.g. R == I at init), producing NaN
        gradients.  6D Gram-Schmidt (Zhou et al., CVPR'19) is
        mathematically equivalent in coverage of SO(3) but has
        continuous gradients everywhere, and is the canonical fix for
        this class of bug.

        Args:
            decoder_tokens: [B*V, L, P, D] decoder output (the same tensor
                returned alongside ``v_t`` as ``decoder_tokens`` from the
                forward pass).
            num_view: V (views folded into batch).

        Returns:
            dict with:
              R_pred: [B, V, 3, 3]  — Gram-Schmidt-orthogonalised rotations
              t_pred: [B, V, 3]     — translation (m_global-normalized scale)
              raw_6d: [B, V, 6]     — un-orthogonalised 6D rep (for ablation)
        """
        if not self.use_pose_head:
            raise RuntimeError(
                "predict_pose called but use_pose_head=False; "
                "set use_pose_head=True in __init__."
            )
        BV, L, P, D = decoder_tokens.shape
        assert BV % num_view == 0, (
            f"decoder_tokens leading dim {BV} must be divisible by "
            f"num_view {num_view}"
        )
        B = BV // num_view

        # Per-view feature: mean-pool over (L, P).  This deliberately mixes
        # all depth layers for each view; cross-view interaction has already
        # happened inside the decoder via the view+global blocks.
        # The pose head runs in fp32 (autocast disabled) for SVD numerical
        # stability and to match the LayerNorm parameter dtype.
        feat = decoder_tokens.mean(dim=(1, 2))  # [B*V, D]
        with torch.autocast(device_type="cuda", enabled=False):
            feat_fp32 = feat.float()
            pose_9d = self.pose_head(feat_fp32)  # [B*V, 9]
        pose_9d = pose_9d.reshape(B, num_view, 9)

        rep_6d = pose_9d[..., :6]  # [B, V, 6] — 6D continuous rep (Zhou'19)
        t_pred = pose_9d[..., 6:9]  # [B, V, 3]

        # Gram-Schmidt orthogonalisation: stable backward at R == I (init),
        # unlike SVD whose backward NaNs out for repeated singular values.
        # The pose_head's final-layer bias is set to (1, 0, 0, 0, 1, 0) and
        # the final-layer weight is N(0, 1e-3) (see __init__), so the
        # forward decodes to ~identity rotation at iter 0 while preserving a
        # non-zero W^T for the backward pass.
        R_pred = _gram_schmidt_6d_to_R(rep_6d)  # [B, V, 3, 3]

        return {"R_pred": R_pred, "t_pred": t_pred, "raw_6d": rep_6d}

    def _add_layer_embedding(
        self,
        img_tokens: torch.Tensor,
        target_layer_idx: torch.Tensor | None = None,
        num_layers_per_frame: int | None = None,
        num_time: int = 1,
    ) -> torch.Tensor:
        """Add layer-position embedding to image tokens.

        - Legacy path (``num_time == 1``): ``img_tokens`` has shape ``[B, L, P, D]``
          and each of the ``L`` views receives a unique layer embedding (or
          per-sample embedding in AR mode).  Behaviour is bit-for-bit identical
          to the pre-clip-mode implementation.
        - Clip mode (``num_time > 1``): ``img_tokens`` has shape
          ``[B, T*L, P, D]`` where ``L == num_layers_per_frame``.  The same
          per-layer embedding is broadcast across all ``T`` frames, so every
          frame's tokens receive layer positions 0..L-1 (the same as a single
          frame would).
        """
        if target_layer_idx is not None:
            # AR mode: per-sample target layer index [B] → [B, 1, 1, D]
            layer_emb = self.layer_embed(target_layer_idx)[:, None, None, :]
            return img_tokens + layer_emb
        if num_time > 1:
            if num_layers_per_frame is None:
                raise ValueError(
                    "num_layers_per_frame is required when num_time > 1."
                )
            total = img_tokens.shape[1]
            if total != num_time * num_layers_per_frame:
                raise ValueError(
                    f"img_tokens shape {tuple(img_tokens.shape)} incompatible "
                    f"with num_time={num_time}, num_layers_per_frame="
                    f"{num_layers_per_frame}."
                )
            if num_layers_per_frame > self.layer_embed.num_embeddings:
                raise ValueError(
                    "num_layers_per_frame exceeds the configured layer "
                    "embedding size."
                )
            layer_ids = torch.arange(
                num_layers_per_frame, device=img_tokens.device
            )
            layer_emb = self.layer_embed(layer_ids)  # [L, D]
            # Broadcast [L, D] → [1, T*L, 1, D] (same L pattern for every frame)
            layer_emb_bcast = (
                layer_emb[None, :, None, :]
                .expand(num_time, -1, -1, -1)
                .reshape(1, total, 1, img_tokens.shape[-1])
            )
            return img_tokens + layer_emb_bcast
        num_layers = img_tokens.shape[1]
        if num_layers > self.layer_embed.num_embeddings:
            raise ValueError("num_layers exceeds the configured layer embedding size.")
        layer_ids = torch.arange(num_layers, device=img_tokens.device)
        layer_emb = self.layer_embed(layer_ids)[None, :, None, :]
        return img_tokens + layer_emb

    def forward(self, psi_t, t, conditioning):
        conditioning = dict(conditioning)
        rgb = conditioning.get("rgb")
        if rgb is None:
            raise ValueError("conditioning must include 'rgb' for layer embedding.")
        batch_size, num_layers = rgb.shape[0], rgb.shape[1]

        # --- Clip mode (T > 1) detection ---
        # When ``num_time > 1``, the incoming ``rgb`` has shape
        # ``[B, T*L_true, 3, H, W]`` where each consecutive block of
        # ``L_true`` entries along axis 1 shares the same per-frame image
        # (the outer training code replicates the frame RGB across the L
        # layer axis so the existing ``encode_conditioning`` / ``xpos``
        # machinery keeps working).  We use ``num_layers_true`` for layer
        # embeddings / FILM / AdaLN so those tensors have the per-frame
        # layer shape (broadcast across T happens inside
        # ``decode_to_output_tokens``).
        num_time_int = int(conditioning.get("num_time", 1))
        if num_time_int > 1:
            if num_layers % num_time_int != 0:
                raise ValueError(
                    f"num_layers ({num_layers}) must be divisible by "
                    f"num_time ({num_time_int}) in clip mode."
                )
            num_layers_true = num_layers // num_time_int
        else:
            num_layers_true = num_layers

        # --- AR context encoding ---
        target_layer_idx = conditioning.get("target_layer_idx", None)  # [B] or None
        ar_context = None
        ar_ctx_pos = None

        if self.use_ar and target_layer_idx is not None:
            # When _skip_input_ctx_attn is set (KV cache default), ar_context is
            # not needed — skip ARTokenContextBuilder / ARContextEncoder entirely.
            _skip_ctx_build = conditioning.get("_skip_input_ctx_attn", False)

            if not _skip_ctx_build:
                ar_per_block_ctx = conditioning.get("ar_per_block_context_tokens")
                ar_ctx_tokens = conditioning.get("ar_context_tokens")

                if ar_per_block_ctx is not None and len(ar_per_block_ctx) > 0:
                    # Per-block context: each AR CA layer gets context from the
                    # matching decoder block depth of previous layers.
                    ar_ctx_valid = conditioning.get("ar_context_valid")
                    num_global = len(ar_per_block_ctx[0])
                    per_block_ctx_list: list[torch.Tensor] = []
                    per_block_pos_list: list[torch.Tensor] = []
                    for gi in range(num_global):
                        block_gi_tokens = [
                            ar_per_block_ctx[j][gi]
                            for j in range(len(ar_per_block_ctx))
                        ]
                        ctx, pos = self.ar_token_context_builder(
                            block_gi_tokens,
                            noise_height=conditioning["noise_height"],
                            noise_width=conditioning["noise_width"],
                            ctx_valid=ar_ctx_valid,
                        )
                        per_block_ctx_list.append(ctx)
                        per_block_pos_list.append(pos)
                    # InputContextAttention uses last block's context
                    ar_context = per_block_ctx_list[-1]
                    ar_ctx_pos = per_block_pos_list[-1]
                    conditioning["ar_context_per_block"] = per_block_ctx_list
                    conditioning["ar_context_pos_per_block"] = per_block_pos_list
                elif ar_ctx_tokens is not None and len(ar_ctx_tokens) > 0:
                    ar_ctx_valid = conditioning.get("ar_context_valid")
                    ar_context, ar_ctx_pos = self.ar_token_context_builder(
                        ar_ctx_tokens,
                        noise_height=conditioning["noise_height"],
                        noise_width=conditioning["noise_width"],
                        ctx_valid=ar_ctx_valid,
                    )
                else:
                    # Legacy path: pixel-space depth/mask → ARContextEncoder
                    ar_ctx_depth = conditioning.get("ar_context_depth")
                    ar_ctx_mask = conditioning.get("ar_context_mask")
                    ar_ctx_valid = conditioning.get("ar_context_valid")

                    if ar_ctx_depth is not None and ar_ctx_depth.shape[1] > 0:
                        ar_context, ar_ctx_pos, _ar_ctx_token_valid = (
                            self.ar_context_encoder(
                                ar_ctx_depth,
                                ar_ctx_mask,
                                noise_height=conditioning["noise_height"],
                                noise_width=conditioning["noise_width"],
                                ctx_valid=ar_ctx_valid,
                            )
                        )

            conditioning["ar_context"] = ar_context
            conditioning["ar_context_pos"] = ar_ctx_pos

            # Auto-request per-block token collection for context-only forwards
            if self.ar_per_block_context and conditioning.get("_context_only", False):
                conditioning["_return_per_block_tokens"] = True

        # --- Image encoding ---
        if self.use_cross_attn:
            if "img_context" not in conditioning:
                if num_time_int > 1:
                    raise NotImplementedError(
                        "use_cross_attn is not supported with num_time > 1. "
                        "Use split-token mode (model_task='split_token') for "
                        "clip/video training."
                    )
                single_rgb = rgb[:, :1]
                single_tokens = self.net.encode_image(single_rgb)
                img_context = single_tokens[:, 0]
                img_context = self.img_context_norm(img_context)
                conditioning["img_context"] = img_context
        else:
            if "img_tokens" not in conditioning:
                if num_time_int > 1:
                    # Extract one RGB per frame (first layer of each T block)
                    # and encode the T unique frames with the frozen encoder.
                    # rgb is [B, T*L_true, 3, H, W] with per-frame replication.
                    frame_rgb = rgb[:, ::num_layers_true]  # [B, T, 3, H, W]
                    if frame_rgb.shape[1] != num_time_int:
                        raise RuntimeError(
                            f"Expected T={num_time_int} unique frames after "
                            f"striding but got {frame_rgb.shape[1]}."
                        )
                    frame_tokens = self.net.encode_image(
                        frame_rgb
                    )  # [B, T, P, D]
                    # Expand along L_true: [B, T, P, D] → [B, T, L, P, D]
                    # → [B, T*L, P, D].  Each L-block within a frame sees the
                    # same encoder features (same image) but distinct layer
                    # embeddings are added below.
                    n_patch = frame_tokens.shape[-2]
                    feat_dim = frame_tokens.shape[-1]
                    img_tokens = (
                        frame_tokens.unsqueeze(2)
                        .expand(-1, -1, num_layers_true, -1, -1)
                        .reshape(
                            batch_size,
                            num_time_int * num_layers_true,
                            n_patch,
                            feat_dim,
                        )
                        .contiguous()
                    )
                    img_tokens = self._add_layer_embedding(
                        img_tokens,
                        target_layer_idx=target_layer_idx,
                        num_layers_per_frame=num_layers_true,
                        num_time=num_time_int,
                    )
                else:
                    single_rgb = rgb[:, :1]
                    single_tokens = self.net.encode_image(single_rgb)
                    img_tokens = single_tokens.expand(
                        -1, num_layers, -1, -1
                    ).contiguous()
                    img_tokens = self._add_layer_embedding(
                        img_tokens, target_layer_idx=target_layer_idx
                    )
                conditioning["img_tokens"] = img_tokens

        # --- Dense layer-embedding info for the decoder ---
        # In AR mode, use per-sample target_layer_idx instead of arange(num_layers).
        # In clip mode, use arange(num_layers_true) so FILM/AdaLN produce
        # per-layer (not per-(T*L)) embeddings; the decoder broadcasts them
        # across T when applying attention patterns.
        if target_layer_idx is not None:
            layer_ids = target_layer_idx  # [B]
        else:
            layer_ids = torch.arange(num_layers_true, device=rgb.device)  # [L_true]

        _is_ar = target_layer_idx is not None

        if self.layer_embed_mode == "adaln":
            layer_emb = self.layer_embed(layer_ids)
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                layer_adaln = self.layer_adaln_proj(layer_emb.float())
            layer_adaln = layer_adaln.unflatten(-1, (6, self._decoder_embed_dim))
            if _is_ar:
                conditioning["layer_embed_info"] = {
                    "mode": "adaln",
                    "layer_adaln": layer_adaln[:, None, None, :, :],  # [B,1,1,6,D]
                }
            else:
                conditioning["layer_embed_info"] = {
                    "mode": "adaln",
                    "layer_adaln": layer_adaln[None, :, None, :, :],  # [1,L,1,6,D]
                }
        elif self.layer_embed_mode == "film":
            layer_emb = self.layer_embed(layer_ids)
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                film_params = self.layer_film_proj(layer_emb.float())
            gamma, beta = film_params.chunk(2, dim=-1)
            if _is_ar:
                conditioning["layer_embed_info"] = {
                    "mode": "film",
                    "gamma": gamma[:, None, None, :],  # [B,1,1,D]
                    "beta": beta[:, None, None, :],
                }
            else:
                conditioning["layer_embed_info"] = {
                    "mode": "film",
                    "gamma": gamma[None, :, None, :],  # [1,L,1,D]
                    "beta": beta[None, :, None, :],
                }
        elif self.layer_embed_mode == "film_adaln":
            layer_emb = self.layer_embed(layer_ids)
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                film_params = self.layer_film_proj(layer_emb.float())
                layer_adaln = self.layer_adaln_proj(layer_emb.float())
            gamma, beta = film_params.chunk(2, dim=-1)
            layer_adaln = layer_adaln.unflatten(-1, (6, self._decoder_embed_dim))
            if _is_ar:
                conditioning["layer_embed_info"] = {
                    "mode": "film_adaln",
                    "gamma": gamma[:, None, None, :],
                    "beta": beta[:, None, None, :],
                    "layer_adaln": layer_adaln[:, None, None, :, :],
                }
            else:
                conditioning["layer_embed_info"] = {
                    "mode": "film_adaln",
                    "gamma": gamma[None, :, None, :],
                    "beta": beta[None, :, None, :],
                    "layer_adaln": layer_adaln[None, :, None, :, :],
                }
        elif self.layer_embed_mode in ("per_block", "global_only"):
            if _is_ar:
                indexed_embeds = [
                    emb[target_layer_idx] for emb in self.dense_layer_embeds
                ]
                conditioning["layer_embed_info"] = {
                    "mode": self.layer_embed_mode,
                    "layer_embeds": indexed_embeds,
                    "ar_mode": True,
                }
            else:
                conditioning["layer_embed_info"] = {
                    "mode": self.layer_embed_mode,
                    "layer_embeds": self.dense_layer_embeds,
                }

        # --- Plumb temporal blocks through conditioning (clip mode only) ---
        # Temporal blocks live on MultilayerXYZModel (this class) while the
        # decoder lives on self.net (MultilayerBackbonePatched).  decode_to_output_tokens
        # reads them from conditioning["_temporal_blocks"] /
        # conditioning["_temporal_insert_indices"].  When num_time == 1 or
        # use_temporal_blocks is False, both remain None and the new decoder
        # path is a strict no-op (the BC fast-path is taken in
        # ``decode_to_output_tokens``).
        if num_time_int > 1 and self.temporal_blocks is not None:
            if "_temporal_blocks" not in conditioning:
                conditioning["_temporal_blocks"] = self.temporal_blocks
            if "_temporal_insert_indices" not in conditioning:
                conditioning["_temporal_insert_indices"] = list(
                    self.temporal_insert_indices
                )

        out = self.net(psi_t, t, conditioning)
        return out

    def get_additional_kwargs(self) -> dict:
        return self.net.get_additional_kwargs()

    def encode_image(self, *args, **kwargs):
        img_tokens = self.net.encode_image(*args, **kwargs)
        if self.use_cross_attn:
            return img_tokens
        return self._add_layer_embedding(img_tokens)
