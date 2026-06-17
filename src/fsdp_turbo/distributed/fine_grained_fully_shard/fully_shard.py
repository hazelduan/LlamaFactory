# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import weakref
from typing import Optional, Callable, Union

import torch.nn as nn
from torch.distributed._composable import contract
from torch.distributed.tensor import DeviceMesh, Shard
from torch.distributed.utils import _get_root_modules

from torch.distributed.fsdp._fully_shard._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    HSDPMeshInfo,
)
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _get_device_from_mesh,
    _get_managed_modules,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _move_states_to_device,
)
from torch.distributed.fsdp._fully_shard._fully_shard import _unimplemented_deepcopy, FSDPModule

from fsdp_turbo.distributed.fine_grained_fully_shard.state import FSDPState
from fsdp_turbo.distributed.fine_grained_fully_shard.param_group import FSDPParamGroup


# Mapping from original module class to the dynamically created FSDP-wrapped class
_cls_to_fsdp_cls: dict[type, type] = {}

# Tracks the number of communication contexts assigned to each hook module.
# Key: hook_module, Value: count of contexts used (used to generate next index)
HOOK_MODULE_COMM_CTX_COUNT: weakref.WeakKeyDictionary[nn.Module, int] = weakref.WeakKeyDictionary()


@contract(state_cls=FSDPState)  # type: ignore[misc] # see [1]
def fully_shard(
    module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    ignored_params: Optional[set[nn.Parameter]] = None,
    hook_module: Optional[nn.Module] = None,
):
    """
    Applies Fully Sharded Data Parallel (FSDP2) to a module with custom hook and stream management.

    Args:
         module: The module to shard.
         mesh: The device mesh for sharding. If None, a default 1D mesh is created.
         reshard_after_forward: Whether to reshard parameters after forward pass.
         shard_placement_fn: Custom function to determine shard placement.
         mp_policy: Mixed precision policy.
         offload_policy: CPU offload policy.
         ignored_params: Set of parameters to ignore during sharding.
         hook_module:
             The specific module to register forward/pre-forward hooks on.
             If None, hooks are registered on the 'module' itself.
             This allows grouping multiple FSDP units under a single logical layer hook.
    Returns:
     The sharded module.
    """

    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    mesh = mesh or _init_default_fully_shard_mesh()
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    elif mesh.ndim == 1:
        mesh_info = FSDPMeshInfo(mesh, shard_mesh_dim=0)
    else:
        if mesh.mesh_dim_names is None:
            raise AssertionError(
                "Please init the 2D mesh for HSDP with mesh_dim_names specified"
            )
        mesh_info = HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    device = _get_device_from_mesh(mesh)
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward, mesh_info
    )

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )
    state = fully_shard.state(modules[0])  # type: ignore[attr-defined] # see [1]

    # Determine hook_module
    if hook_module:
        _hook_module = hook_module
    else:
        _hook_module = (modules[0] if len(modules) > 0 else modules)

    # Auto-increment comm_ctx_index
    if _hook_module not in HOOK_MODULE_COMM_CTX_COUNT:
        HOOK_MODULE_COMM_CTX_COUNT[_hook_module] = 0
    comm_ctx_index = HOOK_MODULE_COMM_CTX_COUNT.get(_hook_module)
    HOOK_MODULE_COMM_CTX_COUNT[_hook_module] = comm_ctx_index + 1

    # Initialize state with custom parameters
    state.init(
        modules, device, mp_policy,
        hook_module=hook_module,
        comm_ctx_index=comm_ctx_index,
    )

    managed_modules = _get_managed_modules(modules, ignored_params)
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    _move_states_to_device(params, buffers, device)
    if params:
        state._fsdp_param_group = FSDPParamGroup(
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            shard_placement_fn,
            mp_policy,
            offload_policy,
        )

    # For Dynamo
    for managed_module in managed_modules:
        managed_module._is_fsdp_managed_module = True  # type: ignore[assignment]
        managed_module._fsdp_use_orig_params = True  # type: ignore[assignment]

    # Place FSDP leftmost for highest priority in the method resolution order
    for module in modules:
        cls = module.__class__
        new_cls = _cls_to_fsdp_cls.get(cls, None)
        if not new_cls:
            dct = {"__deepcopy__": _unimplemented_deepcopy}
            new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct)
            _cls_to_fsdp_cls[cls] = new_cls
        module.__class__ = new_cls
    return arg_module
