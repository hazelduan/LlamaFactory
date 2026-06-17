# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import weakref
import functools
from typing import Any, Optional, NamedTuple

import torch
import torch.nn as nn
from torch.distributed._composable_state import _insert_module_state
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    TrainingState,
    compiled_autograd_enabled,
    _cast_fp_tensor,
)
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext
from torch.distributed.fsdp._fully_shard._fsdp_state import (
    FSDPState as PyTorchFSDPState,
    disable_if_config_true,
    logger,
    _register_group_forward_hooks,
)
from torch.distributed.fsdp._fully_shard._fsdp_api import MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_collectives import AllGatherResult
from torch.utils._pytree import tree_map

from fsdp_turbo.distributed.fine_grained_fully_shard.utils import copy_fsdp_comm_ctx


class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.Event
    hook_module: nn.Module


class ReduceScatterState(NamedTuple):
    reduce_scatter_input: torch.Tensor
    event: torch.Event
    hook_module: nn.Module


class FSDPState(PyTorchFSDPState):
    """
    FSDPTurbo Extended FSDP State Management

    Compared to PyTorch's native FSDPState, the new features include:
    - hook_module: Supports registering forward hooks on specified modules
    - comm_ctx_index: Supports multiple communication context indices
    - global_comm_ctx: Global communication context list
    """
    
    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        hook_module: Optional[nn.Module] = None,
        comm_ctx_index: int = 0,
    ) -> None:
        """
        Custom initialization for FSDPState.

        Extends the default init to:
        1. Register hooks on a specific 'hook_module' (if provided) instead of the first managed module.
        2. Store the 'comm_ctx_index' for multi-stream management.
        """

        for module in modules:
            _insert_module_state(module, self)
        self._modules = modules
        self._device = device
        self._device_handle = _get_device_handle(device.type)
        self._mp_policy = mp_policy
        self.comm_ctx_index = comm_ctx_index

        # Register Hooks
        if hook_module:
            # Register hooks on the user-specified hook_module
            self._pre_forward_hook_handle = hook_module.register_forward_pre_hook(
                self._pre_forward, prepend=True, with_kwargs=True
            )
            self._post_forward_hook_handle = hook_module.register_forward_hook(
                self._post_forward, prepend=False
            )
            self.hook_module = weakref.ref(hook_module)
        else:
            # Fallback to default behavior if no hook_module is specified
            if len(modules) == 1:
                self._pre_forward_hook_handle = modules[0].register_forward_pre_hook(
                    self._pre_forward, prepend=True, with_kwargs=True
                )
                self._post_forward_hook_handle = modules[0].register_forward_hook(
                    self._post_forward, prepend=False
                )
            else:
                hook_handle = _register_group_forward_hooks(
                    modules,
                    self._pre_forward,
                    self._post_forward,
                    self._modules_to_run_forward,
                )
                self._pre_forward_hook_handle = hook_handle
                self._post_forward_hook_handle = hook_handle
            self.hook_module = weakref.ref(modules[0])
    
    def _init_shared_state(self) -> None:
        """
        Initializes shared state across all FSDP states in the context.

        Creates a global list of communication contexts (global_comm_ctx) to manage
        multiple streams. It ensures that every unique comm_ctx_index used by any
        state in the group has a corresponding initialized FSDPCommContext.
        """

        self._comm_ctx.lazy_init(self._device)
        if not hasattr(self, "global_comm_ctx"):
            self.global_comm_ctx = [self._comm_ctx]

        # Collect all unique comm_ctx_indices used in this state context
        global_comm_ctx_list = [0]
        for state in self._state_ctx.all_states:
            if state.comm_ctx_index not in global_comm_ctx_list:
                global_comm_ctx_list.append(state.comm_ctx_index)
                new_comm_ctx = FSDPCommContext()
                new_comm_ctx = copy_fsdp_comm_ctx(new_comm_ctx, self._comm_ctx)
                self.global_comm_ctx.append(new_comm_ctx)

        # Assign the correct comm_ctx to each state based on its index
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx

            # set comm_ctx_index
            _comm_ctx = self.global_comm_ctx[global_comm_ctx_list.index(state.comm_ctx_index)]

            setattr(state, "global_comm_ctx", self.global_comm_ctx)

            state._comm_ctx = _comm_ctx
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.comm_ctx = _comm_ctx
                setattr(fsdp_param_group, "hook_module", state.hook_module)
                setattr(fsdp_param_group, "global_comm_ctx", self.global_comm_ctx)
    
    @disable_if_config_true
    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        """
        Custom post-forward hook.

        Waits for all ALL-Gather operations from ALL communication contexts to complete
        and frees their events before proceeding. This prevents memory leaks and ensures
        data readiness for subsequent operations.
        """

        if self._training_state == TrainingState.PRE_BACKWARD:
            return output
        if self._fsdp_param_group:
            output = self._fsdp_param_group.post_forward(module, input, output)
        output = self._register_pre_backward_hook(output)
        self._training_state = TrainingState.IDLE
        # Wait and free ALL-Gather states for ALL global communication contexts
        if self._state_ctx.iter_forward_root is self:
            for comm_ctx in self.global_comm_ctx:
                # Free the last all-gather result if needed; refer to
                # [Note: Overlapping all-gather copy-in and all-gather]
                if comm_ctx.all_gather_state:
                    # Wait for the copy-in and main all-gather streams
                    self._comm_ctx.all_gather_copy_in_stream.wait_event(comm_ctx.all_gather_state.event)
                    self._comm_ctx.all_gather_stream.wait_event(comm_ctx.all_gather_state.event)
                # Free the all-gather result
                comm_ctx.all_gather_state = None
            self._state_ctx.iter_forward_root = None
        if self._mp_policy.output_dtype is not None:
            with torch.profiler.record_function("FSDP::cast_forward_outputs"):
                output = tree_map(
                    functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
                    output,
                )
        return output
    
    def _root_post_backward_final_callback(self) -> None:
        """
        Custom callback executed after the final backward pass.

        Ensures that the main stream waits for ALL reduce-scatter events from
        ALL communication contexts (global_comm_ctx) to complete before finishing.
        This is crucial for correctness when using multiple streams.
        """

        if not compiled_autograd_enabled():
            logger.debug("FSDP::root_post_backward")
        with torch.profiler.record_function("FSDP::root_post_backward_callback"):
            for state in self._state_ctx.all_states:
                fsdp_param_group = state._fsdp_param_group
                if (
                    fsdp_param_group
                    and fsdp_param_group._training_state != TrainingState.POST_BACKWARD
                ):
                    # Run post-backward in case forward inputs did not require
                    # gradient so the autograd backward did not run
                    fsdp_param_group.post_backward()
                state._training_state = TrainingState.IDLE
                if fsdp_param_group:
                    fsdp_param_group._training_state = TrainingState.IDLE
                if self._state_ctx.is_last_backward:
                    state._finalize_backward()
            if self._state_ctx.is_last_backward:
                self._comm_ctx.post_forward_order.clear()
                if self._comm_ctx.reduce_scatter_state is not None:
                    self._device_handle.current_stream().wait_event(
                        self._comm_ctx.reduce_scatter_state.event
                    )
                    self._comm_ctx.reduce_scatter_state = None

                # WAIT FOR ALL GLOBAL COMM CONTEXTS
                # This ensures synchronization across all custom streams
                if hasattr(self, "global_comm_ctx"):
                    for _comm_ctx in self.global_comm_ctx:
                        _comm_ctx.post_forward_order.clear()
                        if _comm_ctx.reduce_scatter_state is not None:
                            self._device_handle.current_stream().wait_event(
                                _comm_ctx.reduce_scatter_state.event
                            )
                        _comm_ctx.reduce_scatter_state = None

            self._state_ctx.post_backward_final_callback_queued = False
