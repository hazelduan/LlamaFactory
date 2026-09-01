# Copyright 2025 the LlamaFactory team.
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

import sys
from functools import partial, wraps

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers

from ....accelerator.interface import Dim, DistributedInterface
from ....utils import logging
from ....utils.plugin import BasePlugin
from ....utils.types import ModelOutput
from .gdn_attention import _get_gdn_module, gdn_forward_with_cp, is_gdn_layer
from .ulysses import (
    UlyssesAttention,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    set_ulysses_sequence_parallel_group,
)


logger = logging.get_logger(__name__)


class SequenceParallelModelPlugin(BasePlugin):
    def __call__(self, model, cp_size: int):
        return super().__call__(model, cp_size)


class SequenceParallelLossPlugin(BasePlugin):
    def __call__(self, model, inputs, *args, **kwargs):
        return super().__call__(model, inputs, *args, **kwargs)


class _DeepseekV4AttentionModuleProxy:
    """Expose CP-local attention metadata without mutating the model module."""

    def __init__(self, module, sinks: torch.Tensor, num_key_value_groups: int):
        self._module = module
        self.sinks = sinks
        self.num_key_value_groups = num_key_value_groups

    def __getattr__(self, name):
        return getattr(self._module, name)


def _expand_deepseek_v4_attention_mask(module, query, key, attention_mask, group):
    """Rebuild the full sliding-window mask after Ulysses gathers the sequence."""
    cp_size = dist.get_world_size(group)
    full_query_length = query.shape[2]
    full_key_length = key.shape[2]
    if full_query_length % cp_size != 0:
        raise ValueError(f"DeepSeek V4 query length {full_query_length} is not divisible by CP size {cp_size}.")
    if full_key_length < full_query_length:
        raise ValueError("DeepSeek V4 compressed KV length cannot be shorter than its query length.")

    if attention_mask is not None and attention_mask.shape[-2:] == (full_query_length, full_key_length):
        return attention_mask

    local_query_length = full_query_length // cp_size
    batch_size = query.shape[0]
    if attention_mask is None:
        local_key_valid = torch.ones((batch_size, 1, local_query_length), device=query.device, dtype=torch.int32)
    else:
        if attention_mask.ndim != 4 or attention_mask.shape[-2] != local_query_length:
            raise ValueError(
                "DeepSeek V4 Ulysses expects a local 4D attention mask before the sequence exchange, "
                f"got shape {tuple(attention_mask.shape)}."
            )
        local_sliding_mask = attention_mask[..., :local_query_length]
        local_allowed = local_sliding_mask if local_sliding_mask.dtype == torch.bool else local_sliding_mask == 0
        local_key_valid = torch.diagonal(local_allowed, dim1=-2, dim2=-1).to(torch.int32)

    gathered_key_valid = [torch.empty_like(local_key_valid) for _ in range(cp_size)]
    dist.all_gather(gathered_key_valid, local_key_valid.contiguous(), group=group)
    key_valid = torch.cat(gathered_key_valid, dim=-1).bool()

    positions = torch.arange(full_query_length, device=query.device)
    query_positions = positions.view(-1, 1)
    key_positions = positions.view(1, -1)
    sliding_window = int(getattr(module, "sliding_window", module.config.sliding_window))
    allowed = (key_positions <= query_positions) & (key_positions > query_positions - sliding_window)
    allowed = allowed.view(1, 1, full_query_length, full_query_length) & key_valid.unsqueeze(-2)

    full_mask = torch.zeros(
        (batch_size, 1, full_query_length, full_key_length), device=query.device, dtype=query.dtype
    )
    full_mask[..., :full_query_length].masked_fill_(~allowed, torch.finfo(query.dtype).min)
    return full_mask


def _get_deepseek_v4_eager_attention_fn(fn, group):
    """Adapt the official eager callable to FSDPTurbo's post-Ulysses tensors."""

    @wraps(fn)
    def wrapper(module, query, key, value, attention_mask=None, *args, **kwargs):
        cp_size = dist.get_world_size(group)
        cp_rank = dist.get_rank(group)
        local_heads = query.shape[1]
        sinks = module.sinks
        if sinks.shape[0] == local_heads:
            local_sinks = sinks
        elif sinks.shape[0] == local_heads * cp_size:
            local_sinks = sinks.narrow(0, cp_rank * local_heads, local_heads)
        else:
            raise ValueError(
                f"DeepSeek V4 has {sinks.shape[0]} attention sinks, expected {local_heads} or "
                f"{local_heads * cp_size} for CP size {cp_size}."
            )

        if query.shape[1] % key.shape[1] != 0:
            raise ValueError("DeepSeek V4 query heads must be divisible by the post-Ulysses KV heads.")
        module_proxy = _DeepseekV4AttentionModuleProxy(
            module,
            sinks=local_sinks,
            num_key_value_groups=query.shape[1] // key.shape[1],
        )
        attention_mask = _expand_deepseek_v4_attention_mask(module, query, key, attention_mask, group)
        kwargs["s_aux"] = local_sinks
        return fn(module_proxy, query, key, value, attention_mask, *args, **kwargs)

    return wrapper


def _apply_deepseek_v4_sequence_parallel(model, group):
    from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_context_parallel import (
        ulysses_context_parallelize_modules,
    )
    from fsdp_turbo.fsdp_turbo_config import CPPlanConfig
    from fsdp_turbo.utils.patch import patch_namespace_members, resolve_callable

    target = "transformers.models.deepseek_v4.modeling_deepseek_v4.eager_attention_forward"
    original = resolve_callable(target)
    patch_namespace_members(target, _get_deepseek_v4_eager_attention_fn(original, group))
    cp_plan = CPPlanConfig(
        ulysses_function_patches=[
            {
                "target_functions": [target],
                "type": "compressor_full_attention_transposed",
            }
        ]
    )
    ulysses_context_parallelize_modules(model, group, cp_plan)
    logger.info_rank0("Enabled DeepSeek V4 eager attention with FSDPTurbo Ulysses context parallelism.")


def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    sequence_parallel_size=1,
    dropout=0,
    deterministic=False,
    is_causal=True,
    group=None,
    mode="ulysses",
    attn_fn=None,
    target_dtype=None,
    **kwargs,
):
    if mode == "ulysses":
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_fn=attn_fn)
        attn_output = dist_attn(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length=query_states.shape[1] * sequence_parallel_size,
            deterministic=deterministic,
            dropout_p=dropout,
            causal=is_causal,
            position_ids=kwargs.get("position_ids", None),
            target_dtype=target_dtype,
        )
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


@SequenceParallelModelPlugin("ulysses").register()
def apply_sequence_parallel(model, cp_size: int):
    # Replace _flash_attention_forward with new_flash_attn_forward
    module = sys.modules[model.__module__]

    set_ulysses_sequence_parallel_group(DistributedInterface().get_group(Dim.CP))
    sequence_parallel_group = get_ulysses_sequence_parallel_group()

    if getattr(model.config, "model_type", None) == "deepseek_v4":
        if getattr(model.config, "_attn_implementation", None) != "eager":
            raise ValueError("DeepSeek V4 sequence parallelism requires its upstream eager attention implementation.")
        _apply_deepseek_v4_sequence_parallel(model, sequence_parallel_group)
        return

    if getattr(model.config, "_attn_implementation", None) != "flash_attention_2":
        raise ValueError("Sequence parallelism requires flash attention. Please set `flash_attn: flash_attention_2`.")

    try:
        num_attention_heads, num_key_value_heads = (
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
        )
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,
            model.config.text_config.num_key_value_heads,
        )

    assert num_attention_heads % cp_size == 0, "num_attention_heads must be divisible by cp_size"
    assert num_key_value_heads % cp_size == 0 or cp_size % num_key_value_heads == 0, (
        "num_key_value_heads must be divisible by cp_size"
    )

    origin_attn = transformers.modeling_flash_attention_utils._flash_attention_forward
    new_flash_attention_forward = partial(
        new_flash_attn_forward,
        group=sequence_parallel_group,
        mode="ulysses",
        attn_fn=origin_attn,
        sequence_parallel_size=cp_size,
    )

    for module_name, module in list(sys.modules.items()):
        try:
            if (
                hasattr(module, "__file__")
                and "transformers" in module.__file__
                and getattr(module._flash_attention_forward, "__name__", "") == "_flash_attention_forward"
            ):
                module._flash_attention_forward = new_flash_attention_forward
                logger.info_rank0(
                    f"Replaced _flash_attention_forward in module {module_name} with new_flash_attn_forward for sequence parallel."
                )
        except (AttributeError, TypeError):
            continue

    # Register GDN forward for CP support
    if cp_size > 1:
        replaced_modules = set()
        for name, module in model.named_modules():
            if is_gdn_layer(module):
                gdn_module = _get_gdn_module(module)
                if id(gdn_module) in replaced_modules:
                    continue
                replaced_modules.add(id(gdn_module))
                gdn_module.original_forward = gdn_module.forward
                gdn_module.forward = gdn_forward_with_cp.__get__(gdn_module, type(gdn_module))
                gdn_name = name if gdn_module is module else f"{name}.linear_attn"
                logger.info_rank0(f"Replaced GDN forward in {gdn_name} with gdn_forward_with_cp for context parallel.")


def padding_and_split_data(data, device_mesh=None):
    if device_mesh is not None:
        cp_size = device_mesh["cp"].size()
        cp_rank = device_mesh["cp"].get_local_rank()
        cp_group = device_mesh["cp"].get_group()
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.ndim > 1:
                data_len = torch.tensor(v.shape[-1], device=v.device, dtype=torch.int64)
                global_data_len = [torch.empty_like(data_len) for _ in range(cp_size)]
                dist.all_gather(global_data_len, data_len, group=cp_group)
                max_data_len = max(global_data_len)
                pad_size = max_data_len - v.shape[-1] + (cp_size - max_data_len % cp_size) % cp_size
                if k == "labels":
                    pad_value = -100
                elif k == "loss_weights":
                    pad_value = 0.0
                else:
                    pad_value = 0
                pad_data = F.pad(v, (0, pad_size), value=pad_value)
                data[k] = torch.chunk(pad_data, chunks=cp_size, dim=-1)[cp_rank].contiguous()
    return data


@SequenceParallelLossPlugin("sequence_parallel_loss").register()
def sequence_parallel_loss(model, model_inputs):
    device_mesh = DistributedInterface().get_device_mesh(Dim.CP)

    model_inputs = {
        k: v.to(dist.get_rank(), non_blocking=True) for k, v in model_inputs.items() if isinstance(v, torch.Tensor)
    }

    model_inputs = padding_and_split_data(model_inputs, device_mesh)

    batch_size, _ = model_inputs["labels"].shape

    outputs: ModelOutput = model(**model_inputs)

    logits = outputs.logits.float()

    labels = model_inputs["labels"]

    cp_group = get_ulysses_sequence_parallel_group()
    cp_world_size = get_ulysses_sequence_parallel_world_size(cp_group)
    cp_rank = get_ulysses_sequence_parallel_rank(cp_group)

    # use all_gather to collect labels from all sequence parallel processes
    global_labels = [torch.empty_like(labels) for _ in range(cp_world_size)]
    dist.all_gather(global_labels, labels, group=cp_group)
    labels = torch.cat(global_labels, dim=1).contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = F.pad(shift_labels, (0, 1), value=-100)
    shift_labels = torch.chunk(shift_labels, chunks=cp_world_size, dim=1)[cp_rank].contiguous()

    # use all_gather to collect loss_weights from all sequence parallel processes
    loss_weights = model_inputs["loss_weights"]
    global_loss_weights = [torch.empty_like(loss_weights) for _ in range(cp_world_size)]
    dist.all_gather(global_loss_weights, loss_weights, group=cp_group)
    shift_loss_weights = torch.cat(global_loss_weights, dim=1).contiguous()
    shift_loss_weights = shift_loss_weights[..., 1:].contiguous()

    shift_logits = logits.view(-1, logits.size(-1)).contiguous()
    shift_labels = shift_labels.view(-1).contiguous()

    # use all_gather to collect log_probs from all sequence parallel processes
    log_probs = -F.cross_entropy(shift_logits, shift_labels, reduction="none").view(batch_size, -1)
    global_log_probs = dist.nn.all_gather(log_probs, group=cp_group)
    global_log_probs = torch.cat(global_log_probs, dim=1).contiguous()
    log_probs = global_log_probs[..., :-1].contiguous()

    loss = (-log_probs * shift_loss_weights).sum() / (shift_loss_weights.sum() + 1e-6)

    return loss
