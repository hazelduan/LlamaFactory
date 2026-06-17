# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from typing import Set, List, Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from fsdp_turbo.fsdp_turbo_config import FSDPPlanConfig
from fsdp_turbo.distributed.fine_grained_fully_shard import get_fsdp_strategy
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match
from fsdp_turbo.utils.dtype import get_dtype


logger = logging.getLogger(__name__)


def fully_shard_parallel_modules(model: torch.nn.Module, fsdp_mesh: DeviceMesh, fsdp_plan: FSDPPlanConfig):
    ignored_modules, ignored_params = get_ignored_modules(model, fsdp_plan)
    fsdp_modules = get_fsdp_modules(model, fsdp_plan, ignored_modules)
    hook_modules = get_fsdp_hook_modules(model, fsdp_plan)

    has_param_init_fn = (
        hasattr(fsdp_plan, 'param_init_fn')
        and fsdp_plan.param_init_fn is not None
    )

    mp_policy = get_mixprecision_policy(fsdp_plan)

    config = {
        'mesh': fsdp_mesh,
        'ignored_params': ignored_params,
        'mp_policy': mp_policy,
        'reshard_after_forward': fsdp_plan.reshard_after_forward,
    }

    strategy = get_fsdp_strategy(fsdp_plan.fsdp_implementation)
    fully_shard_fn = strategy.get_unified_fully_shard_fn()

    for module, plan in fsdp_modules.items():
        module_config = config.copy()
        module_config.update(plan)

        if has_param_init_fn:
            fsdp_plan.param_init_fn(module)

        hook_module = find_hook_module(module, hook_modules)
        fully_shard_fn(module, hook_module=hook_module, **module_config)

    is_root_sharded = _is_fsdp_managed(model)

    # Avoid root wrapping after explicit module wrapping. Root FSDP can re-shard
    # ignored expert parameters before EP has a chance to distribute them.
    if not is_root_sharded and len(fsdp_modules) == 0:
        fully_shard_fn(model, **config)

    _move_cpu_params_to_device(model, fsdp_mesh)
    set_modules_to_prefetch(model, fsdp_modules, fsdp_plan)
    return model


def _is_fsdp_managed(module: torch.nn.Module) -> bool:
    return hasattr(module, "_fsdp_state") or hasattr(module, "_is_fully_sharded")


def _move_cpu_params_to_device(model: torch.nn.Module, fsdp_mesh: DeviceMesh):
    device_type = fsdp_mesh.device_type
    if device_type not in ("npu", "cuda"):
        return

    try:
        if device_type == "npu":
            target_device = torch.device(f"npu:{torch.npu.current_device()}")
        else:
            target_device = torch.device(f"cuda:{torch.cuda.current_device()}")

        for p in set(model.parameters()):
            if p.device.type == "cpu":
                if type(p).__name__ in ('FSDPParam', 'FSDPParameter'):
                    continue
                p.data = p.data.to(target_device)

        for b in set(model.buffers()):
            if b.device.type == "cpu":
                b.data = b.data.to(target_device)

    except Exception as e:
        logger.warning(f"Failed to move parameters to {device_type}: {e}")


def set_modules_to_prefetch(model: torch.nn.Module, fsdp_modules: list[torch.nn.Module], fsdp_plan: FSDPPlanConfig):
    """Configure forward and backward prefetching."""
    wrapped_modules_in_order: list[torch.nn.Module] = []
    for sub_module in model.modules():  # pre-order
        if any(sub_module is target_module for target_module in fsdp_modules):
            wrapped_modules_in_order.append(sub_module)

    if fsdp_plan.num_to_forward_prefetch > 0:
        for i, layer in enumerate(wrapped_modules_in_order):
            j_end = min(len(wrapped_modules_in_order), i + 1 + fsdp_plan.num_to_forward_prefetch)
            layers_to_prefetch = wrapped_modules_in_order[i + 1:j_end]
            if layers_to_prefetch:
                layer.set_modules_to_forward_prefetch(layers_to_prefetch)

    if fsdp_plan.num_to_backward_prefetch > 0:
        rev_wrapped_modules_in_order = list(reversed(wrapped_modules_in_order))
        for i, layer in enumerate(rev_wrapped_modules_in_order):
            j_end = min(len(rev_wrapped_modules_in_order), i + 1 + fsdp_plan.num_to_backward_prefetch)
            layers_to_prefetch = rev_wrapped_modules_in_order[i + 1:j_end]
            if layers_to_prefetch:
                layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def get_mixprecision_policy(fsdp_plan: FSDPPlanConfig):
    """Construct the MixedPrecisionPolicy object."""
    param_dtype = get_dtype(fsdp_plan.param_dtype) if fsdp_plan.param_dtype else None
    reduce_dtype = get_dtype(fsdp_plan.reduce_dtype) if fsdp_plan.reduce_dtype else None
    output_dtype = get_dtype(fsdp_plan.output_dtype) if fsdp_plan.output_dtype else None

    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        cast_forward_inputs=fsdp_plan.cast_forward_inputs
    )


def get_fsdp_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig, ignored_modules: Set[str]) -> dict[Any, Any]:
    fsdp_modules = {}
    for name, module in model.named_modules():
        for pattern, plan in fsdp_plan.apply_modules.items():
            if module_name_match(pattern, name) and name not in ignored_modules:
                print_rank(logger.debug, f'[FSDP2]: Apply fsdp2 to module <{name}>')
                if module not in fsdp_modules:
                    fsdp_modules[module] = {}
                fsdp_modules.get(module).update(plan)
    if len(fsdp_modules) == 0:
        raise RuntimeError(f'[FSDP2] No module named {fsdp_plan.apply_modules.keys()}.')
    return fsdp_modules


def _post_order_traverse(model: torch.nn.Module, parent_path: str = ""):
    """
    Perform post-order traversal of model submodules.

    Post-order traversal ensures child modules are visited before their parents,
    which is important for FSDP to properly handle nested modules.

    Args:
        model: The model to traverse.
        parent_path: The path to the current module in the hierarchy.

    Yields:
        Tuple of (module_path, module) for each module in the model.
    """
    for name, child in model.named_children():
        child_path = f"{parent_path}.{name}" if parent_path else name
        yield from _post_order_traverse(child, child_path)
    yield parent_path, model


def get_fsdp_hook_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig) -> List[Any]:
    fsdp_hook_modules = []
    if fsdp_plan.apply_modules is None:
        return fsdp_hook_modules
    # Traverse all modules in the model
    if fsdp_plan.hook_modules:
        for name, module in _post_order_traverse(model):
            # Check if module matches any pattern in the FSDP plan
            for pattern in fsdp_plan.hook_modules:
                if module_name_match(pattern, name):
                    print_rank(logger.debug, f'[FSDP2]: Apply fsdp2 hook to hook_module <{name}>')
                    fsdp_hook_modules.append(module)
        # Ensure at least one module matches the FSDP plan
        if len(fsdp_hook_modules) == 0:
            raise RuntimeError(f'[FSDP2] No module named {fsdp_plan.hook_modules}.')

    return fsdp_hook_modules


def find_hook_module(target_module: torch.nn.Module, hook_module_list: List[torch.nn.Module]) -> Optional[
    torch.nn.Module]:
    for hook_module in hook_module_list:
        for _, sub_mod in hook_module.named_modules():
            if sub_mod is target_module:
                return hook_module
    return None


def get_ignored_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig):
    ignored_modules = set()
    ignored_params = set()
    for name, module in model.named_modules():
        for pattern in fsdp_plan.ignored_modules:
            if module_name_match(pattern, name):
                print_rank(logger.debug, f'[FSDP2]: Ignored module to apply fsdp2 <{name}>')
                ignored_modules.add(name)
                ignored_params.update(list(module.parameters(recurse=True)))
    return ignored_modules, ignored_params
