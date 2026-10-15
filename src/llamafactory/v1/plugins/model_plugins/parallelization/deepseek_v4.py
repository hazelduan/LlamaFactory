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

from functools import wraps

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

from ....utils import logging


logger = logging.get_logger(__name__)


class _DeepseekV4AttentionModuleProxy:
    """Override HF eager metadata after Ulysses has expanded KV and sharded heads.

    Keep the original module untouched for recomputation and other CP calls;
    the narrowed sinks remain a differentiable view of the original parameter.
    """

    def __init__(self, module, sinks: torch.Tensor, num_key_value_groups: int):
        self._module = module
        self.sinks = sinks
        self.num_key_value_groups = num_key_value_groups

    def __getattr__(self, name):
        return getattr(self._module, name)


def _expand_deepseek_v4_attention_mask(module, query, key, attention_mask, group):
    """Rebuild the ordinary causal sliding-window mask after sequence all-to-all.

    FSDPTurbo exchanges Q/K/V but forwards the pre-exchange local mask unchanged.
    This reconstruction does not preserve arbitrary custom/packed mask semantics.
    """
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
        return fn(module_proxy, query, key, value, attention_mask, *args, **kwargs)

    return wrapper


def _prepare_deepseek_v4_compressors(model, group):
    """Preserve global compression windows, positions and indexer candidates."""
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4CSACompressor,
        DeepseekV4HCACompressor,
    )

    cp_size = dist.get_world_size(group)
    cp_rank = dist.get_rank(group)

    def gather_inputs(module, args):
        hidden_states, q_residual, position_ids, cache, layer_idx = args
        if cache is not None:
            raise ValueError("DeepSeek V4 CP compression does not support a generation cache.")
        hidden_states = torch.cat(all_gather(hidden_states.contiguous(), group=group), dim=1)
        q_residual = torch.cat(all_gather(q_residual.contiguous(), group=group), dim=1)
        position_ids = torch.cat(all_gather(position_ids.contiguous(), group=group), dim=-1)
        return hidden_states, q_residual, position_ids, cache, layer_idx

    def split_output(module, args, output):
        if output.shape[2] % cp_size:
            raise ValueError(
                "DeepSeek V4 compressed KV length must be divisible by CP size; "
                "use a sequence length divisible by CP size times the compression rate."
            )
        return output.chunk(cp_size, dim=2)[cp_rank].contiguous()

    for module in model.modules():
        if isinstance(module, (DeepseekV4CSACompressor, DeepseekV4HCACompressor)):
            if getattr(module, "_lf_cp_compressor_hooks", None) is not None:
                raise ValueError("DeepSeek V4 CP compression hooks are already installed.")
            module._lf_cp_compressor_hooks = (
                module.register_forward_pre_hook(gather_inputs),
                module.register_forward_hook(split_output),
            )


def apply_deepseek_v4_sequence_parallel(model, group):
    from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_context_parallel import (
        ulysses_context_parallelize_modules,
    )
    from fsdp_turbo.fsdp_turbo_config import CPPlanConfig
    from fsdp_turbo.utils.patch import patch_namespace_members, resolve_callable

    _prepare_deepseek_v4_compressors(model, group)
    target = "transformers.models.deepseek_v4.modeling_deepseek_v4.eager_attention_forward"
    original = resolve_callable(target)
    patch_namespace_members(target, _get_deepseek_v4_eager_attention_fn(original, group))
    cp_plan = CPPlanConfig(
        ulysses_core_attention_function=[target],
        ulysses_attention_type="with_compressor",
        ulysses_attn_output_transposed=True,
    )
    ulysses_context_parallelize_modules(model, group, cp_plan)
    logger.info_rank0("Enabled DeepSeek V4 eager attention with FSDPTurbo Ulysses context parallelism.")
