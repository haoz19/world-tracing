"""
Utilities to enable selective ops-based activation checkpointing in PyTorch.

This module uses several protoype/beta functions and undocumented code from
PyTorch. Some parts can be replaced with core PyTorch implementation after their
API has been stabilized in subsequent versions.
"""

import types
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any, Literal

import structlog
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

logger = structlog.get_logger(__name__)

_CHECKPOINT_PREFIX = "_checkpoint_wrapped_module."


SupportedACModes = Literal["none", "full", "ops", "layer"]
# pylint: disable=protected-access
# Following options will be saved instead of recomputing.
DEFAULT_SAVE_LIST = (
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
)
# pylint: enable=protected-access


def _apply_selective_ops_ac(
    module: nn.Module,
    save_ops_list: Sequence[Any] = DEFAULT_SAVE_LIST,
    save_matmul_frequency: int | None = None,
) -> nn.Module:
    """
    Apply operator-based selective activation checkpointing to a module.

    Args:
        module: The module that will be applied ops-based activation checkpointing.
            Can be transformer or transformer block.
        save_ops_list: List of operations that will be saved instead of recomputing.
        save_matmul_frequency: The frequency to save `torch.ops.aten.mm`. e.g. 2
            means every other matmul will be saved, 1 means all matmul will be saved.
            If `None`, all matmul will be recomputed.

    Returns:
        Wrapped module.
    """

    def _get_custom_policy(meta):
        # pylint: disable=unused-argument
        def _custom_policy(ctx, func, *args, **kwargs):
            """
            The policy function should accept a `SelectiveCheckpointContext`, the
            function for op, args and kwargs to the op, and return a `CheckpointPolicy`
            enum value indicating whether the execution of the op should be recomputed.
            """
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                meta[mm_count_key] += 1
            to_save = func in save_ops_list
            if func == torch.ops.aten.mm.default:
                if save_matmul_frequency is None:
                    to_save = False
                else:
                    to_save = meta[mm_count_key] % save_matmul_frequency == 0

            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def _selective_checkpointing_context_fn():
        meta = defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return ptd_checkpoint_wrapper(
        module,
        context_fn=_selective_checkpointing_context_fn,
        # torchtitan set to False, but does not matter if we use `torch.compile`
        preserve_rng_state=False,
    )


def _apply_full_ac(module: nn.Module) -> nn.Module:
    """Apply activation checkpointing to the full module."""
    return ptd_checkpoint_wrapper(module, preserve_rng_state=False)


def _apply_selective_layer_ac(module: nn.Module, ac_freq: int = 2) -> nn.Module:
    """
    Apply layer-based selective activation checkpointing to a module.

    Args:
        module: The module that will be applied layer-based activation checkpointing.
            Typically a transformer block in a Transformer.
        ac_freq: The number of frequency to apply activation checkpointing. E.g 1 means
            all transformer blocks will be checkpointed (same as full). 2 means every
            other transformer blocks will be checkpointed.

    Returns:
        Wrapped module.
    """
    # pylint: disable=protected-access
    ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
    ptd_checkpoint_wrapper._count += 1
    if ptd_checkpoint_wrapper._count % ac_freq == 0:
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
    else:
        return module


def apply_activation_checkpointing(
    module: nn.Module,
    mode: SupportedACModes = "none",
    **kwargs,
) -> nn.Module:
    """
    Apply activation checkpointing to a module.

    Args:
        module: The module that will be applied activation checkpointing.
        mode: The supported modes of activation checkpointing.
        kwargs: Additional parameters passed into `_apply_selective_ops_ac` and
            `_apply_selective_layer_ac`.
    """
    if mode == "full":
        return _apply_full_ac(module)
    elif mode == "ops":
        return _apply_selective_ops_ac(
            module,
            save_ops_list=kwargs.get("save_ops_list", DEFAULT_SAVE_LIST),
            save_matmul_frequency=kwargs.get("save_matmul_frequency"),
        )
    elif mode == "layer":
        return _apply_selective_layer_ac(module, ac_freq=kwargs.get("ac_freq", 2))
    else:
        return module


def monkey_patch_named_parameters(model: nn.Module) -> nn.Module:
    """
    Monkey patch `named_parameters()` to remove `_CHECKPOINT_PREFIX` for a given model.

    For modules that are passed into `ptd_checkpoint_wrapper`, their
    `named_parameters()` will be overwritten to remove `_CHECKPOINT_PREFIX`. However, if
    they have a parent module, the `named_parameters()` of the parent module still has
    `_CHECKPOINT_PREFIX` in the parameter names. This is because the implementation
    of `named_parameters()` does not rely on `named_parameters()` of submodules. For
    example, if we apply ac to transformer blocks, the transformer still got
    `_CHECKPOINT_PREFIX` when we call its `named_parameters()`. This function can help
    to solve this issue.

    Args:
        model: A model whose submodules have been applied activation checkpointing.

    Returns:
        The same model whose named_parameters has been updated to remove
        `_CHECKPOINT_PREFIX` from parameter names.
    """

    def custom_named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        # Use nn.Module.named_parameters directly instead of super(type(self), self)
        # to avoid dynamo-hostile dynamic super resolution.
        for param_name, param in nn.Module.named_parameters(self, *args, **kwargs):
            yield param_name.replace(_CHECKPOINT_PREFIX, ""), param

    model.named_parameters = types.MethodType(custom_named_parameters, model)

    return model
