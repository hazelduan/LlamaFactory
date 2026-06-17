# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from collections.abc import Mapping
from functools import wraps
from typing import Any, List, Optional

import torch

from fsdp_turbo.fsdp_turbo_config import ChunkBatchPlanConfig
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match

logger = logging.getLogger(__name__)


def chunk_batch_modules(model, plan: ChunkBatchPlanConfig):
    """Apply chunked micro-batch execution to all modules selected by the plan.

    This function is intentionally a thin orchestration layer:
    1. find modules by their qualified names, for example ``model.layers.{*}``;
    2. wrap each matched module's ``forward`` with ``chunk_mbs_forward``;
    3. return the same model object with patched forwards.

    Example:
        If a Qwen-style model has ``model.layers.0`` ... ``model.layers.47`` and
        ``plan.apply_modules == ["model.layers.{*}"]``, all decoder layer
        forwards will be patched while ``model.embed_tokens`` and ``lm_head``
        remain unchanged.
    """
    chunk_mbs_modules = get_chunkmbs_modules(model, plan.apply_modules)
    apply_chunkmbs_module(chunk_mbs_modules, plan)
    return model


def get_chunkmbs_modules(modules, plan):
    """Return ``(name, module)`` pairs selected by extended module patterns.

    ``module_name_match`` supports numeric wildcards such as ``{*}`` and numeric
    ranges such as ``{0-3}``.

    Examples:
        ``model.layers.{*}`` matches ``model.layers.0`` and ``model.layers.47``.
        ``model.layers.{0-3}`` matches only the first four decoder layers.
        ``model.layers.{*}.mlp`` matches the MLP submodule of every decoder layer.
    """
    matched_modules = []
    for plan_name in plan:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                matched_modules.append((name, module))
    if len(matched_modules) == 0:
        raise RuntimeError(f'[ChunkMBS] No module named {plan}.')
    return matched_modules


def apply_chunkmbs_module(chunk_mbs_modules, chunkmbs_cfg):
    """Patch every selected module by replacing its ``forward`` method.

    The patch is done at the module level rather than by adding a wrapper module,
    because replacing the module object itself can interfere with parameter
    loading or FSDP traversal. Keeping the same module instance preserves its
    parameters, buffers, and qualified name.
    """
    for name, module in chunk_mbs_modules:
        print_rank(logger.info, f'Applying chunkmbs to module: {name}')
        module.forward = chunk_mbs_forward(
            chunk_mbs=chunkmbs_cfg.chunk_mbs,
            batch_dim=chunkmbs_cfg.batch_dim,
            chunk_arg_indexs=chunkmbs_cfg.chunk_arg_indexs,
            chunk_kwarg_names=chunkmbs_cfg.chunk_kwarg_names,
        )(module.forward)


def _slice_batch_recursive(data: Any, start: int, end: int, batch_dim: int = 0) -> Any:
    """Slice tensors inside nested input structures on ``batch_dim``.

    Supported structures are tensors, tuples, lists, and mappings. Non-tensor
    values are returned unchanged so flags, enums, strings, cache objects, or
    ``None`` can pass through the wrapped forward safely.

    Example:
        For ``hidden_states.shape == [4, 2048, 4096]`` and
        ``attention_mask.shape == [4, 1, 2048, 2048]``, calling
        ``_slice_batch_recursive(value, 1, 2, batch_dim=0)`` produces tensors
        with shapes ``[1, 2048, 4096]`` and ``[1, 1, 2048, 2048]``.
    """
    if isinstance(data, torch.Tensor):
        slices = [slice(None)] * data.ndim
        slices[batch_dim] = slice(start, end)
        return data[tuple(slices)]
    if isinstance(data, tuple):
        return tuple(_slice_batch_recursive(item, start, end, batch_dim) for item in data)
    if isinstance(data, list):
        return [_slice_batch_recursive(item, start, end, batch_dim) for item in data]
    if isinstance(data, Mapping):
        return {key: _slice_batch_recursive(value, start, end, batch_dim) for key, value in data.items()}
    return data


def chunk_mbs_forward(
    chunk_mbs: int = 1,
    batch_dim: int = 0,
    chunk_arg_indexs: Optional[List[int]] = None,
    chunk_kwarg_names: Optional[List[str]] = None,
):
    """Build a decorator that runs a module forward by batch chunks.

    ``chunk_mbs`` means "micro batch size per chunk", not "number of chunks".
    The number of chunks is ``ceil(full_batch_size / chunk_mbs)``.

    Example:
        If ``full_batch_size == 4`` and ``chunk_mbs == 1``, the wrapped forward
        runs four times with slices ``[0:1]``, ``[1:2]``, ``[2:3]``, and
        ``[3:4]``. The outputs are concatenated back to batch size 4.

        If ``full_batch_size == 4`` and ``chunk_mbs == 4``, no chunking is
        needed and the original forward is called once.

    ``chunk_arg_indexs`` and ``chunk_kwarg_names`` define which forward inputs
    must be sliced. For a HuggingFace decoder layer, ``hidden_states`` and
    ``attention_mask`` often both carry a batch dimension, so they must be
    sliced consistently. Slicing only ``hidden_states`` while keeping the full
    ``attention_mask`` can produce a shape mismatch such as batch 1 vs mask 4.
    """
    chunk_arg_indexs = [] if chunk_arg_indexs is None else chunk_arg_indexs
    chunk_kwarg_names = [] if chunk_kwarg_names is None else chunk_kwarg_names

    def decorator(forward_func):
        @wraps(forward_func)
        def wrapper(*args, **kwargs):
            # Infer the full batch size from the first configured tensor input.
            # Positional inputs are checked first to support calls like
            # layer(hidden_states, attention_mask=...), then keyword inputs are
            # used for calls like layer(hidden_states=hidden_states, ...).
            if chunk_arg_indexs and len(chunk_arg_indexs) > 0 and len(args) > chunk_arg_indexs[0]:
                full_batch_size = args[chunk_arg_indexs[0]].shape[batch_dim]
            elif chunk_kwarg_names and len(chunk_kwarg_names) > 0 and chunk_kwarg_names[0] in kwargs:
                full_batch_size = kwargs[chunk_kwarg_names[0]].shape[batch_dim]
            else:
                raise ValueError("No tensor input found to infer batch size.")

            # When the configured micro-batch is not smaller than the incoming
            # batch, chunking would only add overhead and is skipped.
            if full_batch_size <= chunk_mbs:
                return forward_func(*args, **kwargs)

            num_micros = (full_batch_size + chunk_mbs - 1) // chunk_mbs
            outputs = []
            for idx in range(num_micros):
                start = idx * chunk_mbs
                end = min(start + chunk_mbs, full_batch_size)

                # Slice only the configured positional arguments. Other
                # arguments may be scalar metadata, cache objects, or tensors
                # without a batch dimension, so they are forwarded unchanged.
                micro_args = []
                for arg_idx, arg in enumerate(args):
                    if arg_idx in chunk_arg_indexs:
                        micro_args.append(_slice_batch_recursive(arg, start, end, batch_dim))
                    else:
                        micro_args.append(arg)

                # Slice only the configured keyword arguments. For attention
                # layers, ``attention_mask`` must be sliced together with
                # ``hidden_states`` when it has the full batch dimension.
                micro_kwargs = {}
                for kwarg_name, kwarg_value in kwargs.items():
                    if kwarg_name in chunk_kwarg_names:
                        micro_kwargs[kwarg_name] = _slice_batch_recursive(kwarg_value, start, end, batch_dim)
                    else:
                        micro_kwargs[kwarg_name] = kwarg_value

                outputs.append(forward_func(*micro_args, **micro_kwargs))

            # Most transformer blocks return either a tensor or a tuple/list
            # whose first element is the hidden states. Concatenate each output
            # field on the same batch dimension to restore the public contract.
            if isinstance(outputs[0], torch.Tensor):
                return torch.cat(outputs, dim=batch_dim)
            if isinstance(outputs[0], (tuple, list)):
                return type(outputs[0])(
                    torch.cat([out[idx] for out in outputs], dim=batch_dim)
                    for idx in range(len(outputs[0]))
                )
            raise TypeError(f"Unsupported output type: {type(outputs[0])}")

        return wrapper

    return decorator
