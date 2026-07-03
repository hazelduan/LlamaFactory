# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch import nn


logger = logging.getLogger(__name__)


def current_accelerator_device() -> str:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if hasattr(torch, "npu") and torch.npu.is_available():
        return f"npu:{local_rank}"
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def _find_submodule(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    pieces = name.split(".")
    for piece in pieces[:-1]:
        module = getattr(module, piece)
    return module, pieces[-1]


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, inputs, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        if torch.distributed.get_world_size(group=group) == 1:
            return inputs

        inputs = inputs.contiguous()
        if output_split_sizes is None:
            output = torch.empty_like(inputs)
        else:
            output_split_sizes = [int(x) for x in output_split_sizes]
            input_split_sizes = [int(x) for x in input_split_sizes]
            output = inputs.new_empty([sum(output_split_sizes), *inputs.size()[1:]])
        torch.distributed.all_to_all_single(
            output,
            inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, _AllToAll.apply(ctx.group, grad_output, ctx.input_split_sizes, ctx.output_split_sizes), None, None


def _all_to_all(group, inputs, output_split_sizes=None, input_split_sizes=None):
    return _AllToAll.apply(group, inputs, output_split_sizes, input_split_sizes)


def _current_stream_synchronize() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.current_stream().synchronize()
    elif torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()


def _get_experts_forward_fn(ep_group, fused):
    from fsdp_turbo.distributed.dist_ops import gather_along_first_dim_expert_parallel
    from fsdp_turbo.distributed.expert_parallel.utils import normalize_expert_args
    from fsdp_turbo.ops.moe import grouped_matmul, permute, unpermute
    from torch.distributed.tensor import DTensor

    def dispatch_preprocess(top_k_index, num_global_experts, expert_ids_per_ep_rank):
        ep_size = torch.distributed.get_world_size(ep_group)
        ep_rank = torch.distributed.get_rank(ep_group)
        num_local_experts = num_global_experts // ep_size
        local_start = num_local_experts * ep_rank
        local_end = local_start + num_local_experts

        num_local_tokens_per_expert = torch.bincount(top_k_index.view(-1), minlength=num_global_experts)
        num_global_tokens_per_expert, _ = gather_along_first_dim_expert_parallel(num_local_tokens_per_expert, ep_group)
        num_global_tokens_per_local_expert = num_global_tokens_per_expert.reshape(ep_size, num_global_experts)[
            :, local_start:local_end
        ]
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
        input_split = (
            num_local_tokens_per_expert.reshape(ep_size, num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        output_splits = num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True)
        global_indices = torch.repeat_interleave(expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
        return (num_tokens_per_local_expert, global_indices), (input_split, output_splits)

    def alltoall_dispatch(hidden_states, top_k_index, indices, split_sizes, use_fused):
        local_indices, global_indices = indices
        input_split, output_splits = split_sizes
        hidden_states, unpermute_indices1 = permute(hidden_states, top_k_index, use_eager=not use_fused)
        _current_stream_synchronize()
        hidden_states = _all_to_all(ep_group, hidden_states, output_splits.tolist(), input_split.tolist())
        hidden_states, unpermute_indices2 = permute(hidden_states, global_indices, use_eager=not use_fused)
        return hidden_states, (unpermute_indices1, unpermute_indices2), local_indices

    def alltoall_combine(hidden_states, top_k_weights, unpermute_indices, split_sizes, use_fused):
        unpermute_indices1, unpermute_indices2 = unpermute_indices
        input_split, output_splits = split_sizes
        hidden_states = unpermute(hidden_states, unpermute_indices2, use_eager=not use_fused)
        hidden_states = _all_to_all(ep_group, hidden_states, input_split.tolist(), output_splits.tolist())
        hidden_states = unpermute(hidden_states, unpermute_indices1, top_k_weights, use_eager=not use_fused)
        return hidden_states

    def experts_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        top_k_index, top_k_weights = normalize_expert_args(top_k_index, top_k_weights)
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
        act_limit = self.limit if hasattr(self, "limit") else None

        permute_indices, split_sizes = dispatch_preprocess(
            top_k_index, self.num_global_experts, self.expert_ids_per_ep_rank
        )
        use_fused = fused
        if use_fused and (permute_indices[0] == 0).any():
            use_fused = False
        hidden_states, unpermute_indices, local_indices = alltoall_dispatch(
            hidden_states, top_k_index, permute_indices, split_sizes, use_fused
        )
        gate, up = grouped_matmul(hidden_states, local_indices, gate_up_proj, use_eager=not use_fused).chunk(2, dim=-1)
        if act_limit is not None:
            gate = gate.clamp(max=act_limit)
            up = up.clamp(min=-act_limit, max=act_limit)
        hidden_states = grouped_matmul(self.act_fn(gate) * up, local_indices, down_proj, use_eager=not use_fused)
        hidden_states = alltoall_combine(hidden_states, top_k_weights, unpermute_indices, split_sizes, use_fused)
        return hidden_states.view(*hidden_states_shape)

    return experts_forward


def _expert_parallelize_modules(modules: nn.Module, ep_mesh, plan):
    import types

    from fsdp_turbo.distributed.expert_parallel.expert_parallel import distribute_experts_module, get_ep_modules

    ep_modules = get_ep_modules(modules, plan)
    ep_group = ep_mesh.get_group()
    ep_rank = torch.distributed.get_rank(ep_group)
    ep_size = torch.distributed.get_world_size(ep_group)

    for module in ep_modules:
        module.num_global_experts = len(module) if not hasattr(module, "num_experts") else module.num_experts
        if module.num_global_experts % ep_size != 0:
            raise AssertionError(
                f"Number of experts({module.num_global_experts}) is not divisible by ep size({ep_size})."
            )
        module.num_local_experts = module.num_global_experts // ep_size
        local_start = ep_rank * module.num_local_experts
        module.local_expert_indices = [local_start + i for i in range(module.num_local_experts)]
        if module.num_local_experts > 1:
            module.expert_ids_per_ep_rank = torch.tensor(
                [i % module.num_local_experts for i in range(module.num_global_experts)],
                dtype=torch.int32,
                device=current_accelerator_device(),
            )

        if not getattr(module, "_fsdp_turbo_ep_pre_sharded", False):
            distribute_experts_module(module, ep_mesh)

        dispatcher = plan.dispatcher
        if dispatcher == "eager":
            forward_fn = _get_experts_forward_fn(ep_group, fused=False)
        elif dispatcher == "fused":
            forward_fn = _get_experts_forward_fn(ep_group, fused=True)
        else:
            from fsdp_turbo.distributed.expert_parallel.expert_parallel import get_dispatcher_fn

            forward_fn = get_dispatcher_fn(dispatcher, ep_group, fixed_router=getattr(plan, "fixed_router", False))
        module.forward = types.MethodType(forward_fn, module)
    return modules


class _WeightLoader:
    @staticmethod
    def _get_state_dict_files(weights_path: str) -> list[str]:
        import glob
        import json

        index_file = os.path.join(weights_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)
            files = set(index["weight_map"].values())
            return [os.path.join(weights_path, f) for f in sorted(files)]

        for name in ("model.safetensors",):
            path = os.path.join(weights_path, name)
            if os.path.exists(path):
                return [path]
        for pattern in ("*.safetensors", "*.bin", "*.pt"):
            files = sorted(glob.glob(os.path.join(weights_path, pattern)))
            if files:
                return files
        raise FileNotFoundError(f"No weight files found in {weights_path}")

    @staticmethod
    def preload_metadata(weights_path: str) -> dict[str, dict]:
        metadata = {}
        for filepath in _WeightLoader._get_state_dict_files(weights_path):
            if filepath.endswith(".safetensors"):
                from safetensors import safe_open

                with safe_open(filepath, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        metadata[key] = {
                            "file": filepath,
                            "key": key,
                            "dtype": tensor.dtype,
                            "shape": tuple(tensor.shape),
                        }
            else:
                state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
                for key, tensor in state_dict.items():
                    metadata[key] = {"file": filepath, "key": key, "dtype": tensor.dtype, "shape": tuple(tensor.shape)}
        logger.info("> Pre-scanned %s parameter keys from checkpoint", len(metadata))
        return metadata

    @staticmethod
    def load_single_tensor(info: dict) -> torch.Tensor:
        filepath = info["file"]
        if filepath.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(filepath, framework="pt", device="cpu") as f:
                return f.get_tensor(info["key"])
        return torch.load(filepath, map_location="cpu", weights_only=True)[info["key"]]

    @staticmethod
    def create_param_init_fn(model: nn.Module, metadata: dict[str, dict], device: str):
        module_prefix = {mod: name for name, mod in model.named_modules()}
        metadata_keys = set(metadata.keys())

        def param_init_fn(module: nn.Module) -> None:
            if not any(param.device.type == "meta" for param in module.parameters()):
                return
            prefix = module_prefix.get(module, "")
            module.to_empty(device=device)
            for local_name, param in module.named_parameters():
                full_name = f"{prefix}.{local_name}" if prefix else local_name
                if full_name in metadata_keys:
                    weight = _WeightLoader.load_single_tensor(metadata[full_name])
                    param.data.copy_(weight.to(device=param.device, dtype=param.dtype))

        return param_init_fn


def _fully_shard_parallel_modules(model: nn.Module, fsdp_mesh, fsdp_plan):
    from fsdp_turbo.distributed.fine_grained_fully_shard import get_fsdp_strategy
    from fsdp_turbo.distributed.fully_shard_parallel.fully_shard_parallel import (
        find_hook_module,
        get_fsdp_hook_modules,
        get_fsdp_modules,
        get_ignored_modules,
        get_mixprecision_policy,
        set_modules_to_prefetch,
    )

    ignored_modules, ignored_params = get_ignored_modules(model, fsdp_plan)
    fsdp_modules = get_fsdp_modules(model, fsdp_plan, ignored_modules)
    hook_modules = get_fsdp_hook_modules(model, fsdp_plan)
    config = {
        "mesh": fsdp_mesh,
        "ignored_params": ignored_params,
        "mp_policy": get_mixprecision_policy(fsdp_plan),
        "reshard_after_forward": getattr(fsdp_plan, "reshard_after_forward", True),
    }
    fully_shard_fn = get_fsdp_strategy(fsdp_plan.fsdp_implementation).get_unified_fully_shard_fn()

    for module, plan in fsdp_modules.items():
        module_config = config.copy()
        module_config.update(plan)
        if getattr(fsdp_plan, "param_init_fn", None) is not None:
            fsdp_plan.param_init_fn(module)
        hook_module = find_hook_module(module, hook_modules)
        fully_shard_fn(module, hook_module=hook_module, **module_config)

    _move_cpu_params_to_device(model, fsdp_mesh)
    set_modules_to_prefetch(model, fsdp_modules, fsdp_plan)
    return model


def _move_cpu_params_to_device(model: nn.Module, fsdp_mesh) -> None:
    device_type = fsdp_mesh.device_type
    if device_type not in ("npu", "cuda"):
        return
    target_device = torch.device(current_accelerator_device())
    for param in set(model.parameters()):
        if param.device.type == "cpu" and type(param).__name__ not in ("FSDPParam", "FSDPParameter"):
            param.data = param.data.to(target_device)
    for buffer in set(model.buffers()):
        if buffer.device.type == "cpu":
            buffer.data = buffer.data.to(target_device)


class LlamaFactoryFSDPTurbo(nn.Module):
    def __init__(self, config, model: nn.Module, init_device: str = "cpu", weights_path: Optional[str] = None):
        super().__init__()
        from fsdp_turbo.distributed.parallel_state import init_parallel_state

        self.config = config
        self.model = model
        self.init_device = init_device
        self.weights_path = weights_path
        self.parallel_state = init_parallel_state(self.config)
        if self.init_device == "meta":
            self._init_meta_path()
        else:
            self._init_cpu_path()

    def _init_meta_path(self) -> None:
        device = current_accelerator_device()
        ep_pre_sharded = False
        if self.config.distributed.fully_shard_parallel_size > 1:
            if self.config.distributed.expert_parallel_size > 1:
                ep_pre_sharded = self._materialize_ep_modules_before_fsdp(device)
            self._materialize_non_fsdp_modules(device)
            buffer_backup = {
                name: buffer.clone() for name, buffer in self.model.named_buffers() if buffer.device.type != "meta"
            }
            metadata = _WeightLoader.preload_metadata(self.weights_path) if self.weights_path is not None else {}
            self.config.distributed.fsdp_plan.param_init_fn = _WeightLoader.create_param_init_fn(
                self.model, metadata, device
            )
            self.apply_fsdp_modules()
            self._restore_buffers(buffer_backup)
        else:
            raise RuntimeError("FSDPTurbo meta init with fsdp_size=1 is not supported in the LlamaFactory adapter.")

        self._validate_no_meta_params()
        self._post_process_meta()
        self.apply_tp_modules()
        self.apply_ep_modules(skip_expert_parallel=ep_pre_sharded)
        self.apply_recompute_modules()
        self.apply_quantization_modules()

    def _init_cpu_path(self) -> None:
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        wrapped = FSDPTurbo(self.config, self.model)
        self.model = wrapped.model

    def _restore_buffers(self, buffer_backup: dict[str, torch.Tensor]) -> None:
        for name, buffer in buffer_backup.items():
            try:
                module, local_name = _find_submodule(self.model, name)
                model_buffer = dict(module.named_buffers(recurse=False))[local_name]
                model_buffer.copy_(buffer.to(device=model_buffer.device, dtype=model_buffer.dtype))
            except Exception as exc:
                logger.warning("> Failed to restore buffer %s: %s", name, exc)

    def _post_process_meta(self) -> None:
        if hasattr(self.model, "config") and getattr(self.model.config, "tie_word_embeddings", True):
            try:
                input_embeddings = self.model.get_input_embeddings()
                output_embeddings = self.model.get_output_embeddings()
                if output_embeddings is not None and input_embeddings is not None:
                    output_embeddings.weight = input_embeddings.weight
                    logger.info("> Tied input/output embeddings")
            except Exception as exc:
                logger.warning("> Failed to tie embeddings: %s", exc)

    def _materialize_non_fsdp_modules(self, device: str) -> None:
        from fsdp_turbo.utils.str_match import module_name_match

        fsdp_patterns = (
            set(self.config.distributed.fsdp_plan.apply_modules.keys())
            if self.config.distributed.fsdp_plan.apply_modules
            else set()
        )
        fsdp_covered = set()
        for full_name, _ in self.model.named_modules():
            if any(module_name_match(pattern, full_name) for pattern in fsdp_patterns):
                fsdp_covered.add(full_name)

        modules_to_materialize = {}
        for full_name, module in self.model.named_modules():
            if not any(param.device.type == "meta" for param in module.parameters(recurse=False)):
                continue
            parts = full_name.split(".")
            if any(".".join(parts[: i + 1]) in fsdp_covered for i in range(len(parts))):
                continue
            modules_to_materialize[full_name] = module

        if not modules_to_materialize:
            return

        metadata = _WeightLoader.preload_metadata(self.weights_path) if self.weights_path is not None else {}
        for module_path, module in modules_to_materialize.items():
            module.to_empty(device=device)
            for local_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_path}.{local_name}" if module_path else local_name
                if full_name in metadata:
                    weight = _WeightLoader.load_single_tensor(metadata[full_name])
                    param.data.copy_(weight.to(device=param.device, dtype=param.dtype))

    @torch.no_grad()
    def _materialize_ep_modules_before_fsdp(self, device: str) -> bool:
        if self.weights_path is None:
            return False
        ep_plan = self.config.distributed.ep_plan
        if not ep_plan.apply_modules:
            return False

        from fsdp_turbo.distributed.expert_parallel.expert_parallel import get_ep_modules

        ep_modules = get_ep_modules(self.model, ep_plan)
        if not ep_modules:
            return False

        ep_mesh = self.parallel_state.get_ep_device_mesh()
        ep_group = ep_mesh.get_group()
        ep_rank = torch.distributed.get_rank(ep_group)
        ep_size = torch.distributed.get_world_size(ep_group)
        module_prefix = {module: name for name, module in self.model.named_modules()}
        metadata = _WeightLoader.preload_metadata(self.weights_path)

        materialized = 0
        for module in ep_modules:
            num_global_experts = getattr(module, "num_experts", None) or len(module)
            if num_global_experts % ep_size != 0:
                raise AssertionError(
                    f"Number of experts({num_global_experts}) is not divisible by ep size({ep_size})."
                )
            num_local_experts = num_global_experts // ep_size
            local_start = ep_rank * num_local_experts
            local_end = local_start + num_local_experts
            module.num_global_experts = num_global_experts
            module.num_local_experts = num_local_experts
            module.local_expert_indices = list(range(local_start, local_end))
            if num_local_experts > 1:
                module.expert_ids_per_ep_rank = torch.tensor(
                    [i % num_local_experts for i in range(num_global_experts)], dtype=torch.int32, device=device
                )

            prefix = module_prefix[module]
            loaded_for_module = 0
            for local_name, param in list(module.named_parameters(recurse=True)):
                full_name = f"{prefix}.{local_name}" if prefix else local_name
                if full_name not in metadata:
                    continue
                weight = _WeightLoader.load_single_tensor(metadata[full_name])
                if weight.dim() > 0 and weight.shape[0] == num_global_experts:
                    weight = weight[local_start:local_end].contiguous()
                parent = module
                pieces = local_name.split(".")
                for piece in pieces[:-1]:
                    parent = getattr(parent, piece)
                setattr(
                    parent,
                    pieces[-1],
                    nn.Parameter(weight.to(device=device, dtype=param.dtype), requires_grad=param.requires_grad),
                )
                loaded_for_module += 1

            if loaded_for_module == 0:
                raise RuntimeError(f"No expert parameters loaded for EP module '{prefix}'.")
            module._fsdp_turbo_ep_pre_sharded = True
            materialized += loaded_for_module

        logger.info(
            "> Pre-sharded %s EP modules before FSDP (rank %s/%s, loaded %s tensors)",
            len(ep_modules),
            ep_rank,
            ep_size,
            materialized,
        )
        self.model = _expert_parallelize_modules(self.model, ep_mesh, ep_plan)
        return True

    def _validate_no_meta_params(self) -> None:
        remaining = [name for name, param in self.model.named_parameters() if param.device.type == "meta"]
        if remaining:
            raise RuntimeError(f"Found {len(remaining)} parameters still on meta device. First 10: {remaining[:10]}")

    def apply_fsdp_modules(self) -> None:
        if self.config.distributed.fully_shard_parallel_size == 1:
            return
        self.model = _fully_shard_parallel_modules(
            self.model, self.parallel_state.get_fsdp_device_mesh(), self.config.distributed.fsdp_plan
        )

    def apply_tp_modules(self) -> None:
        if self.config.distributed.tensor_parallel_size == 1:
            return
        from fsdp_turbo.distributed.tensor_parallel.tensor_parallel import tensor_parallel_modules

        self.model = tensor_parallel_modules(
            self.model, self.parallel_state.get_tp_device_mesh(), self.config.distributed.tp_plan
        )

    def apply_ep_modules(self, skip_expert_parallel: bool = False) -> None:
        from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules

        if self.config.distributed.expert_parallel_size > 1 and not skip_expert_parallel:
            self.model = _expert_parallelize_modules(
                self.model, self.parallel_state.get_ep_device_mesh(), self.config.distributed.ep_plan
            )
        if self.config.distributed.expert_fully_shard_parallel_size > 1:
            self.model = expert_fully_shard_modules(
                self.model,
                self.parallel_state.get_efsdp_device_mesh(),
                self.config.distributed.ep_plan,
                self.config.distributed.fsdp_plan,
            )

    def apply_recompute_modules(self) -> None:
        if not self.config.memory.recompute:
            return
        from fsdp_turbo.memory.recompute.recompute import recompute_modules

        self.model = recompute_modules(self.model, self.config.memory.recompute_plan)

    def apply_quantization_modules(self) -> None:
        quant_plan = self.config.quantization.quantization_plan
        if not getattr(quant_plan, "quant_recipe", None):
            return
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        FSDPTurbo.apply_quantization_modules(self)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
