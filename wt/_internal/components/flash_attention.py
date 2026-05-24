"""Flash attention API.

This is a wrapper around the flash attention library.  When neither
``flash_attn_interface`` (FA3) nor ``flash_attn`` (FA2) is installed we fall
back to PyTorch's ``F.scaled_dot_product_attention`` (which dispatches to
the fastest available kernel on the current device).  The fallback covers
the non-varlen call sites used by the published inference release.
"""

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def _sdpa_fallback_varlen(
    q, k, v, q_lens, k_lens,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
):
    """PyTorch fallback for the non-varlen case (all sequences same length).

    Re-pads ``q``/``k``/``v`` from the FA "flattened" layout back into
    ``[B, H, S, C]`` and dispatches to
    :func:`torch.nn.functional.scaled_dot_product_attention`.  When
    ``q_lens`` / ``k_lens`` are all equal (the common case at inference) the
    output is identical (up to FP error) to the FA varlen path.
    """
    from torch.nn import functional as F

    b = q_lens.shape[0]
    lq = int(q_lens.max().item())
    lk = int(k_lens.max().item())
    hq = q.shape[1]
    hk = k.shape[1]
    c = q.shape[2]

    if not (torch.all(q_lens == lq) and torch.all(k_lens == lk)):
        raise NotImplementedError(
            "SDPA fallback only supports the fixed-length case; install "
            "flash-attn or flash_attn_interface to use variable-length "
            "attention."
        )

    q = q.view(b, lq, hq, c).transpose(1, 2).contiguous()  # [B, Hq, Lq, C]
    k = k.view(b, lk, hk, c).transpose(1, 2).contiguous()
    v = v.view(b, lk, hk, v.shape[2]).transpose(1, 2).contiguous()

    if hk != hq:
        rep = hq // hk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    out = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=dropout_p,
        scale=softmax_scale,
        is_causal=causal,
    )
    out = out.transpose(1, 2).contiguous().reshape(b * lq, hq, v.shape[3])
    return out


def flash_sdpa_flops(batch_size: int, q_len: int, kv_len: int, dim: int) -> int:
    """See :func:`attention.sdpa_flops` for more details."""
    sdp_num_elems = batch_size * q_len * kv_len
    sdp_flops_per_elem = 4 * dim
    return int(sdp_flops_per_elem * sdp_num_elems)


def flash_attention(
    q: Float[Tensor, "b hq sq c1"],
    k: Float[Tensor, "b hk sk c1"],
    v: Float[Tensor, "b hk sk c2"],
    q_lens: Int[Tensor, "b"] | None = None,
    k_lens: Int[Tensor, "b"] | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
    use_bshc: bool = False,
) -> Float[Tensor, "b hq sq c1"]:
    """
    Flash attention API.

    Note that flash attention requires qkv of shape `batch seqlen num_heads head_dim`.
    It is different from pytorch sdpa which requires `batch num_heads seqlen head_dim`.

    Args:
        q: The query tensor.
        k: The key tensor.
        v: The value tensor.
        q_lens: The length of the query tensor. If None, the length is the same as the
            q.size(1).
        k_lens: The length of the key tensor. If None, the length is the same as the
            k.size(1).
        dropout_p: The dropout probability.
        softmax_scale: The scaling of QK^T before applying softmax.
        causal: Whether to apply causal attention mask.
        window_size: (left, right). If not (-1, -1), apply sliding window local
            attention.
        deterministic: Whether to use deterministic version. If True, slightly slower
            and uses more memory.
        dtype: The target dtype of qkv. Cast qkv to this dtype when their dtype is not
            float16/bfloat16.
        use_bshc: Whether to use the BSHC format, which is the default format in
            flash attention. Defaults to `False` which is consistent with the PyTorch
            conventions.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    if dtype not in half_dtypes:
        raise ValueError(f"Expected {dtype=} to be one of {half_dtypes=}.")
    if q.device.type != "cuda":
        raise ValueError(f"Expected {q.device.type=} to be 'cuda'.")
    if q.size(-1) > 256:
        raise ValueError(f"Expected {q.size(-1)=} to be <= 256.")

    if not use_bshc:
        q = einops.rearrange(q, "b hq sq c -> b sq hq c")
        k = einops.rearrange(k, "b hk sk c -> b sk hk c")
        v = einops.rearrange(v, "b hk sk c -> b sk hk c")

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        # apply attention
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    else:
        # PyTorch SDPA fallback (no flash_attn / flash_attn_interface installed).
        x = _sdpa_fallback_varlen(
            q, k, v, q_lens, k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        ).unflatten(0, (b, lq))

    if not use_bshc:
        x = einops.rearrange(x, "b sq hq c -> b hq sq c")

    return x.type(out_dtype)


def flash_attention_with_lse(
    q: Float[Tensor, "b hq sq c1"],
    k: Float[Tensor, "b hk sk c1"],
    v: Float[Tensor, "b hk sk c2"],
    q_lens: Int[Tensor, "b"] | None = None,
    k_lens: Int[Tensor, "b"] | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
    use_bshc: bool = False,
) -> tuple[Float[Tensor, "b hq sq c1"], Float[Tensor, "b hq sq"]]:
    """Flash attention returning both output and log-sum-exp for online softmax merging.

    Same interface as :func:`flash_attention` but additionally returns
    the softmax LSE tensor needed by ring attention to combine partial
    attention results across sequence shards.

    Returns:
        (output, lse) where output has the same shape as the query and
        lse has shape ``[B, H, S_q]``.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    if dtype not in half_dtypes:
        raise ValueError(f"Expected {dtype=} to be one of {half_dtypes=}.")
    if q.device.type != "cuda":
        raise ValueError(f"Expected {q.device.type=} to be 'cuda'.")
    if q.size(-1) > 256:
        raise ValueError(f"Expected {q.size(-1)=} to be <= 256.")

    if not use_bshc:
        q = einops.rearrange(q, "b hq sq c -> b sq hq c")
        k = einops.rearrange(k, "b hk sk c -> b sk hk c")
        v = einops.rearrange(v, "b hk sk c -> b sk hk c")

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    cu_seqlens_q = (
        torch.cat([q_lens.new_zeros([1]), q_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True)
    )
    cu_seqlens_k = (
        torch.cat([k_lens.new_zeros([1]), k_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True)
    )

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        out, softmax_lse = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        x = out.unflatten(0, (b, lq))
    else:
        out, softmax_lse, *_ = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        x = out.unflatten(0, (b, lq))

    # FA varlen returns LSE as [H, total_q]; reshape to [B, H, S_q].
    softmax_lse = softmax_lse.unflatten(1, (b, lq)).transpose(0, 1).contiguous()

    if not use_bshc:
        x = einops.rearrange(x, "b sq hq c -> b hq sq c")

    return x.type(out_dtype), softmax_lse


def flash_attention_varlen(
    q: Float[Tensor, "sq hq c1"],
    k: Float[Tensor, "sk hk c1"],
    v: Float[Tensor, "sk kh c2"],
    q_cumlens: Int[Tensor, "b+1"],
    k_cumlens: Int[Tensor, "b+1"],
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    deterministic: bool = False,
    version: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Float[Tensor, "hq sq c2"]:
    """
    Flash Attention API for variable-length sequences.

    In this API the batch dimension and the sequence dimension are merged in a single
    axis; a batch of variable-length sequences can be packed together.

    Args:
        q: Query vectors
        k: Key vectors
        v: Value vectors
        q_cumlens: Tensor of shape batch_size + 1 where the start and end index of the
            ith batch element in q spans indices [q_cumlens[i], q_cumlens[i + 1])
        k_cumlens: Tensor of shape batch_size + 1 where the start and end index of the
            ith batch elements in k and v span indicies [k_cumlens[i], k_cumlens[i + 1])
        max_seqlen_q: The length of the longest sequence in q. If not provided this will
            be computed from q_cumlens; but this will cause an extra reduction and
            device sync so it is more efficient to pass it in explicitly if possible.
        max_seqlen_k: The length of the longest sequence in k. If not provided this will
            be computed from k_cumlens; but this will cause an extra reduction and
            device sync so it is more efficient to pass it in explicitly if possible.
        softmax_scale: The scaling of QK^T before applying softmax.
        causal: Whether to apply causal attention mask.
        deterministic: Whether to use deterministic version. If True, slightly slower
            and uses more memory.
        version: Which version of flash attention to use. Defaults to FA3 if possible.
        dtype: The target dtype of q, k, and v. They will be cast to this dtype if they
            do not already have it. Must be torch.bfloat16 or torch.float16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    if dtype not in half_dtypes:
        raise ValueError(f"Expected {dtype=} to be one of {half_dtypes=}.")
    if q.device.type != "cuda":
        raise ValueError(f"Expected {q.device.type=} to be 'cuda'.")
    if q.size(-1) > 256:
        raise ValueError(f"Expected {q.size(-1)=} to be <= 256.")

    if max_seqlen_q is None:
        max_seqlen_q = int((q_cumlens[1:] - q_cumlens[:-1]).max().item())
    if max_seqlen_k is None:
        max_seqlen_k = int((k_cumlens[1:] - k_cumlens[:-1]).max().item())

    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q = half(q)
    k = half(k)
    v = half(v)
    q_cumlens = q_cumlens.to(dtype=torch.int32, device=q.device)
    k_cumlens = k_cumlens.to(dtype=torch.int32, device=q.device)

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=q_cumlens,
            cu_seqlens_k=k_cumlens,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
    else:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=q_cumlens,
            cu_seqlens_k=k_cumlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )

    return x.type(out_dtype)


class FlashAttention(nn.Module):
    """
    Flash attention nn.Modulewrapper.

    This is a wrapper around the flash_attention function that allows for
    more flexible usage of the flash attention API with things like context parallel.
    """

    def __init__(
        self,
        dropout_p: float = 0.0,
        attn_scale: float | None = None,  # Named this "attn_scale" to match prev API.
        causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        version: int | None = None,
    ):
        super().__init__()
        self.dropout_p = dropout_p
        self.softmax_scale = attn_scale
        self.causal = causal
        self.window_size = window_size
        self.deterministic = deterministic
        if version == 3:
            assert FLASH_ATTN_3_AVAILABLE, "FlashAttention3 is not available!"
            assert dropout_p == 0.0, "Dropout is not supported in FlashAttention3!"
            assert window_size == (
                -1,
                -1,
            ), "Window size is not supported in FlashAttention3!"
        self.version = version

    def forward(
        self,
        q: Float[Tensor, "b hq sq c1"],
        k: Float[Tensor, "b hk sk c1"],
        v: Float[Tensor, "b hk sk c2"],
        q_lens: Int[Tensor, "b"] | None = None,
        k_lens: Int[Tensor, "b"] | None = None,
        q_scale: float | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> Float[Tensor, "b nq sq c1"]:
        """
        Note that FlashAttention does not support the following kwargs:
            block_offset, block_size, q_offset.
        """
        unsupported_kwargs = ["block_offset", "block_size", "q_offset"]
        if any([kwargs.get(key) is not None for key in unsupported_kwargs]):
            raise ValueError(
                f"FlashAttention does not support the following kwargs: "
                f"{unsupported_kwargs}. Received {kwargs.keys()=}."
            )

        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            q_scale=q_scale,
            causal=self.causal,
            window_size=self.window_size,
            deterministic=self.deterministic,
            dtype=dtype,
            version=self.version,
        )

    def model_flops(
        self,
        q: Float[Tensor, "b hq sq c1"],
        k: Float[Tensor, "b hk sk c1"],
        v: Float[Tensor, "b hk sk c2"],
        **kwargs,
    ) -> int:
        del kwargs, v
        batch_size, _, q_len, q_dim = q.shape
        _, _, k_len, _ = k.shape
        return flash_sdpa_flops(
            batch_size=batch_size,
            q_len=q_len,
            kv_len=k_len,
            dim=q_dim,
        )
