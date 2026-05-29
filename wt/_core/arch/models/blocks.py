from copy import deepcopy

import einops
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from wt._core.vendor.vggt.layers import layer_scale
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

from wt._core.components import flash_attention, nn_layers
from wt._core.models.wan_video import layers as wan_video_layers
from wt._core.utils import torch_utils
from wt._core.arch.models import config as model_config
from wt._core.arch.models import model_utils

logger = structlog.get_logger(__name__)


class Attention(nn.Module):
    """
    Attention module from Croco with sdpa enabled.
    """

    def __init__(
        self,
        dim,
        rope=None,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        zero_init_last=False,
        use_qk_norm: bool = False,
        use_zero_attn: bool = False,
        scale_factor: float = -0.5,
        use_gated_attn: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**scale_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.zero_init_last = zero_init_last
        if zero_init_last:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.flex_attention = torch.compile(flex_attention, dynamic=False)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        self.use_zero_attn = use_zero_attn

        # apply gating to sdpa output: https://arxiv.org/pdf/2505.06708
        self.use_gated_attn = use_gated_attn
        if use_gated_attn:
            self.gate_proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(
        self,
        x,
        xpos=None,
        attn_mask=None,
        attn_temperature_tuning=False,
        use_fa_interface=False,
    ):
        bs, npatch, nchannel = x.shape

        qkv = self.qkv(x)
        if self.use_gated_attn:
            gate_score = self.gate_proj(x)
            gate_score = gate_score.reshape(
                bs, npatch, self.num_heads, nchannel // self.num_heads
            )
        qkv = qkv.reshape(bs, npatch, 3, self.num_heads, nchannel // self.num_heads)
        if use_fa_interface:
            # bs, npatch, nhead, nchannel
            pass
        else:
            # bs, nhead, npatch, nchannel
            qkv = qkv.transpose(1, 3)
        q, k, v = (qkv[:, :, i] for i in range(3))

        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
            q = q.to(v.dtype)
            k = k.to(v.dtype)

        if self.rope is None:
            pass
        else:
            assert xpos is not None
            q = self.rope(q, xpos, use_fa_interface=use_fa_interface)
            k = self.rope(k, xpos, use_fa_interface=use_fa_interface)

        if self.use_zero_attn:
            # add zero attn
            zero_attn_shape = (bs, self.num_heads, 1, self.head_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)],
                dim=2,
            )
            v = torch.cat(
                [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)],
                dim=2,
            )

        if attn_temperature_tuning:
            # TODO: remove hardcoded value
            floor_scale = 1536  # max training tokens
            attn_scale = 1.0  # lamma4 uses 0.1
            seq_positions = torch.arange(
                0, npatch, device=q.device, dtype=torch.float32
            )
            attn_scales = (
                torch.log(torch.floor((seq_positions + 1.0) / floor_scale) + 1.0)
                * attn_scale
                + 1.0
            )
            # reshape for broadcasting [seqlen] -> [1, seqlen, 1, 1]
            attn_scales = attn_scales.view(1, 1, npatch, 1)
            q = q * attn_scales

        assert attn_mask is None, "attn_mask is not supported for flash attention"
        if use_fa_interface:
            x = flash_attention.flash_attention(
                q, k, v, softmax_scale=self.scale, use_bshc=True
            )
            x = x.reshape(bs, npatch, nchannel)
        else:
            # use flex attention in training for attn_mask augmentation
            # use sdp in inference for speed
            if self.training:
                x = (
                    self.flex_attention(q, k, v, scale=self.scale, block_mask=attn_mask)
                    .transpose(1, 2)
                    .reshape(bs, npatch, nchannel)
                )
            else:
                with torch.backends.cuda.sdp_kernel(enable_flash=True):
                    x = (
                        F.scaled_dot_product_attention(
                            q, k, v, scale=self.scale, attn_mask=attn_mask
                        )
                        .transpose(1, 2)
                        .reshape(bs, npatch, nchannel)
                    )

        if self.use_gated_attn:
            # bs, npatch, (nhead * c)
            x = x * torch.sigmoid(gate_score.reshape(bs, npatch, -1))

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
        zero_init_last=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = model_utils.to_2tuple(bias)
        drop_probs = model_utils.to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        if zero_init_last:
            nn.init.zeros_(self.fc2.weight)
            if bias[1]:
                nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DecoderBlockSA(nn.Module):
    """
    Decoder block from Croco with cross attenation removed.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rope=None,
        zero_init_last=False,
        use_qk_norm: bool = False,
        init_values: float | None = None,
        use_zero_attn: bool = False,
        scale_factor: float = -0.5,
        use_gated_attn: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            zero_init_last=zero_init_last,
            use_qk_norm=use_qk_norm,
            use_zero_attn=use_zero_attn,
            scale_factor=scale_factor,
            use_gated_attn=use_gated_attn,
        )
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            zero_init_last=zero_init_last,
        )
        self.zero_init_last = zero_init_last

        self.ls1 = (
            layer_scale.LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.ls2 = (
            layer_scale.LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )

    def zero_init_last_all(self):
        nn.init.zeros_(self.attn.proj.weight)
        nn.init.zeros_(self.attn.proj.bias)
        nn.init.zeros_(self.mlp.fc2.weight)
        nn.init.zeros_(self.mlp.fc2.bias)

    def forward(self, x, xpos, attn_mask=None, use_fa_interface=True):
        x = x + self.ls1(
            self.attn(
                self.norm1(x),
                xpos,
                attn_mask=attn_mask,
                use_fa_interface=use_fa_interface,
            )
        )
        x = x + self.ls2(self.mlp(self.norm3(x)))
        return x


class HeadTransformer(nn.Module):
    """
    Tranformer for camera and scale heads.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: float | None = None,
        use_gated_attn: bool = False,
        rope=None,
        num_blocks: int = 4,
    ):
        super().__init__()
        blk = DecoderBlockSA(
            embed_dim,
            num_heads,
            norm_layer=norm_layer,
            use_qk_norm=True,
            init_values=init_values,
            use_gated_attn=use_gated_attn,
            rope=rope,
        )
        self.blocks = nn.Sequential(*[deepcopy(blk) for _ in range(num_blocks)])

    def forward(self, x, xpos):
        for blk in self.blocks:
            x = blk(x, xpos)
        return x


class DecoderBlockDiT(DecoderBlockSA):
    """
    Decoder block in DiT style.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # modulation
        self.modulation_shape = (1, 6, self.dim)
        self.modulation = nn.Parameter(
            torch.randn(self.modulation_shape).reshape(-1) / self.dim**0.5
        )

    def forward(
        self,
        x: Float[Tensor, "b s d"],
        xpos: Float[Tensor, "b s 2"],
        adaln_input: Float[Tensor, "b 1 6 d"],
    ):
        x_dtype = x.dtype
        shift_a, scale_a, dscale_a, shift_m, scale_m, dscale_m = (
            self.modulation.view(self.modulation_shape).unsqueeze(1)
            + adaln_input.float()
        ).unbind(-2)

        # attention
        y = self.ls1(
            self.attn(
                (self.norm1(x).float() * (1 + scale_a) + shift_a).to(x_dtype),
                xpos,
                use_fa_interface=True,
            )
        )
        x = (x + y * dscale_a).to(x_dtype)

        # FFN
        y = self.ls2(
            self.mlp((self.norm3(x).float() * (1 + scale_m) + shift_m).to(x_dtype))
        )
        x = (x + y * dscale_m).to(x_dtype)
        return x


class CrossAttention(nn.Module):
    """
    Cross-attention module: queries from x_q attend over keys/values from x_kv.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int | None = None,
        rope=None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        zero_init_last: bool = False,
        zero_bias_last: bool = False,
        use_qk_norm: bool = False,
        scale_factor: float = -0.5,
        use_gated_attn: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.dim_kv = dim_q if dim_kv is None else dim_kv
        assert self.dim_kv == self.dim_q, "For now we require dim_kv == dim_q"

        head_dim = dim_q // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**scale_factor

        # Separate projections for Q and KV (cross-attn)
        self.q_proj = nn_layers.Linear(self.dim_q, dim_q, bias=qkv_bias)
        self.kv_proj = nn_layers.Linear(self.dim_kv, self.dim_kv * 2, bias=qkv_bias)

        self.proj = nn_layers.Linear(self.dim_kv, self.dim_kv)
        self.zero_init_last = zero_init_last
        if zero_init_last:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

        if zero_bias_last:
            nn.init.zeros_(self.proj.bias)

        self.rope = rope  # will be applied to Q and K with their own positions

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn_layers.LayerNorm(self.head_dim)
            self.k_norm = nn_layers.LayerNorm(self.head_dim)

        # apply gating to sdpa output: https://arxiv.org/pdf/2505.06708
        self.use_gated_attn = use_gated_attn
        if use_gated_attn:
            self.gate_proj = nn_layers.Linear(self.dim_q, self.dim_q)

    def forward(
        self,
        x_q: Float[Tensor, "b lq dq"],  # queries
        x_kv: Float[Tensor, "b lk dv"],  # keys/values source
        qpos: Float[Tensor, "b lq 2"] | None = None,  # positions for Q
        kvpos: Float[Tensor, "b lk 2"] | None = None,  # positions for K
    ) -> Float[Tensor, "b lq dv"]:
        bq, nqpatch, _ = x_q.shape
        bk, _, _ = x_kv.shape
        assert bq == bk, "Batch size of query and kv must match."

        # Projections
        q = self.q_proj(x_q)  # [b, lq, dim_q]
        kv = self.kv_proj(x_kv)  # [b, lk, 2*dim_q]

        if self.use_gated_attn:
            gate_score = self.gate_proj(x_q)
            gate_score = gate_score.reshape(bq, nqpatch, self.num_heads, -1)

        # reshape to (b, seq, h, d)
        q = einops.rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k, v = kv[:, :, : self.dim_kv], kv[:, :, self.dim_kv :]
        k = einops.rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v = einops.rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        if self.use_qk_norm:
            # LayerNorm across the per-head channel dim
            q = self.q_norm(q)
            k = self.k_norm(k)
            # keep dtype consistent with v (important with AMP/bfloat16)
            q = q.to(v.dtype)
            k = k.to(v.dtype)

        # RoPE on Q/K with their respective positions
        if self.rope is not None:
            assert (qpos is not None) and (
                kvpos is not None
            ), "Provide qpos and kvpos for RoPE."
            q = self.rope(q, qpos, use_fa_interface=True)
            k = self.rope(k, kvpos, use_fa_interface=True)

        # Flash attention expects (b, seq, h, d) with use_bshc=True
        x = flash_attention.flash_attention(
            q, k, v, softmax_scale=self.scale, use_bshc=True
        )  # -> [b, lq, h, d]

        if self.use_gated_attn:
            x = x * torch.sigmoid(gate_score)

        x = einops.rearrange(x, "b l h d -> b l (h d)")
        x = self.proj(x)
        return x


class DecoderBlockCA(nn.Module):
    """
    Decoder block from Croco with cross attenation removed.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rope=None,
        zero_init_last=False,
        zero_bias_last=False,
        use_qk_norm: bool = False,
        init_values: float | None = None,
        scale_factor: float = -0.5,
        add_query_residual: bool = True,
        use_gated_attn: bool = False,
    ):
        """
        Args:
            add_query_residual: whether to add query residual to the output. For dense
              prediction, we set add_query_residual to False.
        """

        super().__init__()
        self.dim = dim
        self.add_query_residual = add_query_residual
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            zero_init_last=zero_init_last,
            zero_bias_last=zero_bias_last,
            use_qk_norm=use_qk_norm,
            scale_factor=scale_factor,
            use_gated_attn=use_gated_attn,
        )
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            zero_init_last=zero_init_last,
        )
        self.zero_init_last = zero_init_last

        self.ls1 = (
            layer_scale.LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.ls2 = (
            layer_scale.LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )

    def forward(
        self,
        x_q: Float[Tensor, "b lq dq"],
        x_kv: Float[Tensor, "b lk dv"],
        qpos: Float[Tensor, "b lq 2"] | None = None,
        kvpos: Float[Tensor, "b lk 2"] | None = None,
    ):
        attn_out = self.ls1(self.attn(self.norm1(x_q), self.norm2(x_kv), qpos, kvpos))
        if self.add_query_residual:
            x_q = x_q + attn_out
        else:
            x_q = attn_out
        x_q = x_q + self.ls2(self.mlp(self.norm3(x_q)))
        return x_q


class DecoderBlockCADiT(DecoderBlockCA):
    """
    Decoder block with cross attention from DiT style.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # modulation
        self.modulation_shape = (1, 6, self.dim)
        self.modulation = nn.Parameter(
            torch.randn(self.modulation_shape) / self.dim**0.5
        )
        # make torch.compile happy
        self.modulation.data = self.modulation.data.reshape(-1)

    def forward(
        self,
        x_q: Float[Tensor, "b lq dq"],
        x_kv: Float[Tensor, "b lk dv"],
        qpos: Float[Tensor, "b lq 2"] | None = None,
        kvpos: Float[Tensor, "b lk 2"] | None = None,
        adaln_input: Float[Tensor, "b 1 6 d"] | None = None,
    ) -> Float[Tensor, "b lq dv"]:
        x_dtype = x_q.dtype
        shift_a, scale_a, dscale_a, shift_m, scale_m, dscale_m = (
            self.modulation.view(self.modulation_shape).unsqueeze(1)
            + adaln_input.float()
        ).unbind(-2)

        # attention
        attn_out = self.ls1(
            self.attn(
                (self.norm1(x_q) * (1 + scale_a) + shift_a).to(x_dtype),
                self.norm2(x_kv),
                qpos,
                kvpos,
            )
        )
        if self.add_query_residual:
            x_q = (x_q + attn_out * dscale_a).to(x_dtype)
        else:
            x_q = (attn_out * dscale_a).to(x_dtype)

        # ffn
        ffn_out = self.ls2(
            self.mlp((self.norm3(x_q) * (1 + scale_m) + shift_m).to(x_dtype))
        )
        x_q = (x_q + ffn_out * dscale_m).to(x_dtype)
        return x_q


class MLPBlock(nn.Module):
    """
    MLP block with residual connection and normalization.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        zero_last: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)

        # initialze the last layer to 0
        if zero_last:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, xin: Float[Tensor, "b lq dv"]) -> Float[Tensor, "b lq dv"]:
        x = self.norm(xin)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + xin


class MLPBlockDiT(MLPBlock):
    """
    MLP block with residual connection and normalization from DiT style.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # modulation
        self.modulation_shape = (1, 3, self.dim)
        self.modulation = nn.Parameter(
            torch.randn(self.modulation_shape) / self.dim**0.5
        )

    def forward(
        self, xin: Float[Tensor, "b lq dv"], adaln_input: Float[Tensor, "b 1 3 d"]
    ) -> Float[Tensor, "b lq dv"]:
        x_dtype = xin.dtype
        shift, scale, dscale = (
            self.modulation.view(self.modulation_shape).unsqueeze(1)
            + adaln_input.float()
        ).unbind(-2)

        x1 = (self.norm(xin).float() * (1 + scale) + shift).to(x_dtype)
        x2 = self.act(self.fc1(x1))
        y = xin + (self.fc2(x2) * dscale).to(x_dtype)
        return y


class TimestepProjection(nn.Module):
    """
    Timestep projection module.
    """

    def __init__(
        self, t_embed_channels: int, decoder_embed_dim: int, adaln_channels: int = 6
    ):
        super().__init__()
        self.t_embed_channels = t_embed_channels
        self.decoder_embed_dim = decoder_embed_dim
        self.adaln_channels = adaln_channels

        self.time_embedding = nn.Sequential(
            nn_layers.Linear(self.t_embed_channels, self.decoder_embed_dim),
            nn.SiLU(),
            nn_layers.Linear(self.decoder_embed_dim, self.decoder_embed_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn_layers.Linear(
                self.decoder_embed_dim, self.decoder_embed_dim * self.adaln_channels
            ),
        )
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

    def forward(self, t: Float[Tensor, "b"]) -> Float[Tensor, "b 1 x d"]:
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
                -1, (self.adaln_channels, self.decoder_embed_dim)
            )
            if t.device.type == "cuda":
                assert (
                    t_embed.dtype == torch.float32
                    and adaln_input.dtype == torch.float32
                )
        return adaln_input


class RoPE2D(torch.nn.Module):
    """
    RoPE2D from Croco.
    """

    def __init__(self, freq=100.0):
        super().__init__()
        self.base = freq
        self.cache = {}

    def get_cos_sin(
        self, num_dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Float[Tensor, "F d"], Float[Tensor, "F d"]]:
        """
        Get the cosine and sine given the input frequency.
        """
        if (num_dim, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, num_dim, 2).float().to(device) / num_dim)
            )  # inv_freq from 1./base to 1; slowest moving component is seq_len / base
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos = angles.cos()  # (Seq, Dim)
            sin = angles.sin()
            self.cache[num_dim, seq_len, device, dtype] = (cos, sin)
        return self.cache[num_dim, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        """
        Construct the rotated results of x.
        [cos(x), -sin(x)] [xa]
        [sin(x),  cos(x)] [xb]
        =
        [xa cos(x) - xb sin(x)]
        [xb cos(x) + xa sin(x)]
        """
        xa, xb = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-xb, xa), dim=-1)

    def apply_rope1d(
        self,
        tokens: Float[Tensor, "b s h d"],
        pos1d: Float[Tensor, "b s"],
        cos: Float[Tensor, "F d"],
        sin: Float[Tensor, "F d"],
        use_fa_interface: bool = False,
    ) -> Float[Tensor, "b s h d"]:
        """
        Same as apply_rope1d but uses direct indexing instead of F.embedding.
        """
        assert pos1d.ndim == 2
        cos_indexed = cos[pos1d]  # (b, s, d)
        sin_indexed = sin[pos1d]  # (b, s, d)
        if use_fa_interface:
            # tokens: (b, s, h, d) -> need (b, s, 1, d)
            cos_indexed = cos_indexed[:, :, None, :]
            sin_indexed = sin_indexed[:, :, None, :]
        else:
            # tokens: (b, h, s, d) -> need (b, 1, s, d)
            cos_indexed = cos_indexed[:, None, :, :]
            sin_indexed = sin_indexed[:, None, :, :]
        return (tokens * cos_indexed) + (self.rotate_half(tokens) * sin_indexed)

    def forward(
        self,
        tokens: Float[Tensor, "b s h d"] | Float[Tensor, "b h s d"],
        positions: Float[Tensor, "b s 2"],
        use_fa_interface: bool = False,
    ) -> Float[Tensor, "b s h d"] | Float[Tensor, "b h s d"]:
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert (
            tokens.size(3) % 2 == 0
        ), "number of dimensions should be a multiple of two"
        num_dim = tokens.size(3) // 2
        assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2
        position_max = self.get_position_max(positions)
        # seq x dim
        # sin/cos are repeated like
        # [sin(w_1 x)... sin(w_n x), sin(w_1 x)... sin(w_n x)]
        # [cos(w_1 x)... cos(w_n x), cos(w_1 x)... cos(w_n x)]
        cos, sin = self.get_cos_sin(num_dim, position_max, tokens.device, tokens.dtype)
        # split features into two along the feature dimension,
        #  and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin, use_fa_interface)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin, use_fa_interface)
        tokens = torch.cat((y, x), dim=-1)
        return tokens

    @torch.compiler.disable
    def get_position_max(self, positions):
        return int(positions.max()) + 1


class RoPEND(RoPE2D):
    """
    RoPEND: Generalized N-dimensional RoPE.
    """

    def forward(
        self,
        tokens: Float[Tensor, "b s h d"] | Float[Tensor, "b h s d"],
        positions: Float[Tensor, "b s n"],
        use_fa_interface: bool = False,
    ) -> Float[Tensor, "b s h d"] | Float[Tensor, "b h s d"]:
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x n
              (position of each token in n dimensions)
        output:
            * tokens after applying RoPEND (batch_size x nheads x ntokens x dim)
        """
        n_dims = positions.shape[-1]
        assert positions.ndim == 3  # Batch, Seq, N

        # pad token channels to multiple of 2*n_dims to ensure each chunk is even
        # (required by rotate_half which splits each chunk in half)
        divisor = 2 * n_dims
        remainder = tokens.size(3) % divisor
        pad_size = (divisor - remainder) % divisor
        if pad_size > 0:
            tokens = torch.nn.functional.pad(tokens, (0, pad_size))

        num_dim = tokens.size(3) // n_dims
        position_max = self.get_position_max(positions)
        cos, sin = self.get_cos_sin(num_dim, position_max, tokens.device, tokens.dtype)

        # split features into n_dims chunks along the feature dimension
        chunks = tokens.chunk(n_dims, dim=-1)
        rotated_chunks = []
        for i, chunk in enumerate(chunks):
            rotated_chunk = self.apply_rope1d(
                chunk, positions[:, :, i], cos, sin, use_fa_interface
            )
            rotated_chunks.append(rotated_chunk)
        tokens = torch.cat(rotated_chunks, dim=-1)

        # remove the padded channels
        if pad_size > 0:
            tokens = tokens[:, :, :, :-pad_size]
        return tokens


class PoPEND(torch.nn.Module):
    """
    PoPEND: Generalized N-dimensional PoPE (Polar Coordinate Positional Embedding)
    from arXiv:2509.10534.

    Key differences from RoPE:
    - Uses softplus(x) as magnitude to encode content (the "what")
    - Uses pure position-dependent phase rotation (the "where")
    - No interaction term between content and position (decoupled)
    - Each input element produces 2 outputs (real + imag), so output dim = 2 * input dim

    The attention computation remains standard dot product: for q_pope and k_pope
    stacked as [real, imag], the dot product computes Re(q^H @ k) correctly.

    Output dimension is 2 * input dimension.
    """

    def __init__(self, freq: float = 100.0):
        super().__init__()
        self.base = freq
        self.cache = {}

    def get_cos_sin(
        self, num_dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Float[Tensor, "F d"], Float[Tensor, "F d"]]:
        """
        Get cos/sin for position encoding.
        Uses frequencies θ_c = 1 / (base^(c/d)) for c = 0,...,d-1
        """
        if (num_dim, seq_len, device, dtype) not in self.cache:
            # Frequencies: same as RoPE for consistency
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, num_dim, 2).float().to(device) / num_dim)
            )
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.outer(t, inv_freq).to(dtype)  # [seq_len, num_dim/2]
            cos = angles.cos()
            sin = angles.sin()
            self.cache[num_dim, seq_len, device, dtype] = (cos, sin)
        return self.cache[num_dim, seq_len, device, dtype]

    def apply_pope1d(
        self,
        tokens: Float[Tensor, "b s h d"] | Float[Tensor, "b h s d"],
        pos1d: Float[Tensor, "b s"],
        cos: Float[Tensor, "F d2"],
        sin: Float[Tensor, "F d2"],
        use_fa_interface: bool = False,
    ) -> Float[Tensor, "b s h d"] | Float[Tensor, "b h s d"]:
        """
        Apply PoPE to tokens for 1D positions.

        To maintain d -> d dimension (like RoPE), we:
        1. Merge adjacent pairs to get magnitude: d -> d/2
        2. Apply position-dependent rotation: d/2 -> d (real + imag)

        Input: tokens [..., d]
        Output: [..., d]
        """
        assert pos1d.ndim == 2
        assert tokens.size(-1) % 2 == 0, "dimension must be even for pair merging"

        # Merge adjacent pairs to get magnitude (d -> d/2)
        # Using L2 norm of pairs, then softplus for non-negativity
        a, b = tokens[..., 0::2], tokens[..., 1::2]  # each [..., d/2]
        mu = F.softplus(torch.sqrt(a * a + b * b + 1e-8))  # [..., d/2]

        # Get cos/sin for each position
        cos_indexed = cos[pos1d]  # (b, s, d/2)
        sin_indexed = sin[pos1d]  # (b, s, d/2)

        if use_fa_interface:
            # tokens: (b, s, h, d) -> need (b, s, 1, d/2)
            cos_indexed = cos_indexed[:, :, None, :]
            sin_indexed = sin_indexed[:, :, None, :]
        else:
            # tokens: (b, h, s, d) -> need (b, 1, s, d/2)
            cos_indexed = cos_indexed[:, None, :, :]
            sin_indexed = sin_indexed[:, None, :, :]

        # Complex output in Cartesian form: μ * e^(i*θ) = (μ*cos(θ), μ*sin(θ))
        real = mu * cos_indexed  # [..., d/2]
        imag = mu * sin_indexed  # [..., d/2]

        # Interleave [real, imag] to get output dimension d
        # This gives [r0, i0, r1, i1, ...] matching RoPE's pair structure
        out = torch.stack([real, imag], dim=-1)  # [..., d/2, 2]
        out = out.flatten(-2)  # [..., d]
        return out

    @torch.compiler.disable
    def get_position_max(self, positions):
        return int(positions.max()) + 1

    def forward(
        self,
        tokens: Float[Tensor, "b s h d"] | Float[Tensor, "b h s d"],
        positions: Float[Tensor, "b s n"],
        use_fa_interface: bool = False,
    ) -> Float[Tensor, "b s h d"] | Float[Tensor, "b h s d"]:
        """
        Input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x n (position in n dimensions)
        Output:
            * tokens after applying PoPEND: same dimension as input
        """
        n_dims = positions.shape[-1]
        assert positions.ndim == 3  # Batch, Seq, N

        d = tokens.size(-1)
        # Pad to ensure even split across n_dims AND each chunk is even
        # (for pair merging)
        divisor = 2 * n_dims
        remainder = d % divisor
        pad_size = (divisor - remainder) % divisor
        if pad_size > 0:
            tokens = torch.nn.functional.pad(tokens, (0, pad_size))

        num_dim = tokens.size(-1) // n_dims
        position_max = self.get_position_max(positions)
        cos, sin = self.get_cos_sin(num_dim, position_max, tokens.device, tokens.dtype)

        # Split features into n_dims chunks
        chunks = tokens.chunk(n_dims, dim=-1)
        pope_chunks = []
        for i, chunk in enumerate(chunks):
            pope_chunk = self.apply_pope1d(
                chunk, positions[:, :, i], cos, sin, use_fa_interface
            )
            pope_chunks.append(pope_chunk)

        # Concatenate all PoPE outputs
        tokens = torch.cat(pope_chunks, dim=-1)

        # Remove padding to restore original dimension
        if pad_size > 0:
            tokens = tokens[..., :-pad_size]
        return tokens
