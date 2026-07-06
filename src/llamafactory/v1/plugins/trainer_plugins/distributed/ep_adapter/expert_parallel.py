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

import re
import types
from collections.abc import Callable

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard, distribute_module, distribute_tensor

from .....accelerator.interface import DistributedInterface
from .....utils.logging import get_logger


logger = get_logger(__name__)


def _compile_extended_pattern(pattern: str):
    specs = []
    placeholder = "__NUM__"

    def replace_brace(match):
        inner = match.group(1)
        if inner == "*":
            specs.append(None)
            return placeholder
        if "-" in inner:
            parts = inner.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid brace pattern: {inner}")
            low, high = int(parts[0]), int(parts[1])
            if low > high:
                raise ValueError(f"Invalid range: {inner} (low > high)")
            specs.append((low, high))
            return placeholder
        raise ValueError(f"Unrecognized brace pattern: {inner}")

    temp_pattern = re.sub(r"\{([^}]*)\}", replace_brace, pattern)
    regex_parts = []
    i = 0
    while i < len(temp_pattern):
        if temp_pattern.startswith(placeholder, i):
            regex_parts.append(r"(\d+)")
            i += len(placeholder)
            continue

        char = temp_pattern[i]
        if char == "*":
            regex_parts.append(r".*")
        elif char == "?":
            regex_parts.append(r".")
        elif char == ".":
            regex_parts.append(r"\.")
        else:
            regex_parts.append(re.escape(char))
        i += 1

    return re.compile("^" + "".join(regex_parts) + "$"), specs


def module_name_match(pattern: str, string: str) -> bool:
    regex, specs = _compile_extended_pattern(pattern)
    match = regex.match(string)
    if not match:
        return False

    for num_str, spec in zip(match.groups(), specs):
        if spec is None:
            continue
        num = int(num_str)
        low, high = spec
        if not low <= num <= high:
            return False

    return True


def get_modules_by_patterns(model: torch.nn.Module, patterns: list[str]) -> list[torch.nn.Module]:
    modules = []
    for pattern in patterns:
        for name, module in model.named_modules():
            if module_name_match(pattern, name):
                logger.debug(f"[EP] Matched module <{name}> by pattern <{pattern}>.")
                modules.append(module)

    return modules


def collect_ignored_params(model: torch.nn.Module, patterns: list[str]) -> set[torch.nn.Parameter]:
    ignored_params = set()
    for name, module in model.named_modules():
        for pattern in patterns:
            if module_name_match(pattern, name):
                ignored_params.update(list(module.parameters(recurse=True)))

    return ignored_params


def distribute_expert_weight(module_name: str, module: torch.nn.Module, ep_mesh: DeviceMesh) -> None:
    for name, param in module.named_parameters(recurse=False):
        dist_param = torch.nn.Parameter(distribute_tensor(param, ep_mesh, [Shard(0)]))
        module.register_parameter(name, dist_param)

    for child_name, child_module in module.named_children():
        distribute_expert_weight(child_name, child_module, ep_mesh)


def distribute_experts_module(module: torch.nn.Module, ep_mesh: DeviceMesh) -> torch.nn.Module:
    return distribute_module(module=module, device_mesh=ep_mesh, partition_fn=distribute_expert_weight)


def apply_expert_parallel(
    model: torch.nn.Module,
    ep_mesh: DeviceMesh,
    ep_modules: list[str],
    dispatcher_factory: Callable,
    fixed_router: bool = False,
) -> torch.nn.Module:
    modules = get_modules_by_patterns(model, ep_modules)
    if not modules:
        raise RuntimeError(f"[EP] No module matched by patterns {ep_modules}.")

    ep_group = ep_mesh.get_group()
    ep_rank = torch.distributed.get_rank(ep_group)
    ep_size = torch.distributed.get_world_size(ep_group)
    experts_forward_fn = dispatcher_factory(ep_group, fixed_router=fixed_router)

    for module in modules:
        module.num_global_experts = len(module) if not hasattr(module, "num_experts") else module.num_experts
        if module.num_global_experts % ep_size != 0:
            raise AssertionError(
                f"Number of experts({module.num_global_experts}) is not divisible by ep size({ep_size})."
            )

        module.num_local_experts = module.num_global_experts // ep_size
        local_expert_indices_offset = ep_rank * module.num_local_experts
        module.local_expert_indices = [
            local_expert_indices_offset + i for i in range(module.num_local_experts)
        ]
        if module.num_local_experts > 1:
            module.expert_ids_per_ep_rank = torch.tensor(
                [i % module.num_local_experts for i in range(module.num_global_experts)],
                dtype=torch.int32,
                device=DistributedInterface().current_device,
            )

        distribute_experts_module(module, ep_mesh)
        module.forward = types.MethodType(experts_forward_fn, module)

    return model


def set_gradient_divide_factor(module: torch.nn.Module, factor: float) -> None:
    if hasattr(module, "set_gradient_divide_factor"):
        module.set_gradient_divide_factor(factor)
    else:
        module.set_reduce_scatter_divide_factor(factor)


def apply_expert_fsdp(
    model: torch.nn.Module,
    efsdp_mesh: DeviceMesh,
    ep_fsdp_modules: list[str],
    gradient_divide_factor: float,
) -> torch.nn.Module:
    modules = get_modules_by_patterns(model, ep_fsdp_modules)
    if not modules:
        raise RuntimeError(f"[EFSDP] No module matched by patterns {ep_fsdp_modules}.")

    config = {
        "mesh": efsdp_mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        "shard_placement_fn": lambda x: Shard(1) if x.dim() > 1 else Shard(0),
    }

    for module in modules:
        if isinstance(module, torch.nn.ModuleList):
            for expert in module:
                fully_shard(expert, **config)
                set_gradient_divide_factor(expert, gradient_divide_factor)
        else:
            fully_shard(module, **config)
            set_gradient_divide_factor(module, gradient_divide_factor)

    return model
