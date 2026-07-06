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

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from ....accelerator.interface import Dim, DistributedInterface
from ....utils.logging import get_logger
from ....utils.types import HFModel
from .ep_adapter import EPModelSpec, get_model_type
from .ep_adapter.expert_parallel import apply_expert_fsdp, apply_expert_parallel, collect_ignored_params
from .ep_adapter.fsdp_turbo_dispatcher import apply_hccl_premul_sum_patch, get_dispatcher_factory
from .fsdp2 import FSDP2Engine


logger = get_logger(__name__)


class FSDPTurboDim:
    """Backend-local mesh dimension names."""

    EDP = "edp"
    EFSDP = "efsdp"
    EP = "ep"
    CP = "expert_cp"


def _grad_to_local_fp32(grad: torch.Tensor) -> torch.Tensor:
    from torch.distributed._tensor import DTensor

    local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
    return local_grad.detach().to(torch.float32)


def _local_pth_sum(parameters: list[torch.nn.Parameter], norm_type: float, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for param in parameters:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        total = total + torch.norm(_grad_to_local_fp32(grad), p=norm_type).pow(norm_type)
    return total


def _allreduce_sum_(value: torch.Tensor, groups: list[object]) -> torch.Tensor:
    import torch.distributed as dist

    for group in groups:
        if group is not None:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    return value


def clip_grad_norm_(model: HFModel, max_norm: float, **kwargs) -> float:
    """Clip gradients across FSDPTurbo EP/EFSDP/FSDP meshes without all-gathering full grads."""
    from torch.distributed._tensor import DTensor

    def _get_dtensor_group(grad: torch.Tensor, dim_name: str):
        if not isinstance(grad, DTensor):
            return None

        mesh = grad.device_mesh
        mesh_names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
        if dim_name not in mesh_names:
            return None

        return mesh.get_group(mesh_names.index(dim_name))

    norm_type = float(kwargs.get("norm_type", 2.0))
    dist_interface = DistributedInterface()
    device = dist_interface.current_device
    dp_group = dist_interface.get_group(Dim.DP)
    cp_group = dist_interface.get_group(Dim.CP) if dist_interface.strategy.cp_size > 1 else None

    ep_params: list[torch.nn.Parameter] = []
    non_ep_params: list[torch.nn.Parameter] = []
    ep_group = None
    efsdp_group = None
    for param in model.parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue

        mesh_names = set(getattr(getattr(grad, "device_mesh", None), "mesh_dim_names", ()) or ())
        is_ep_side = isinstance(grad, DTensor) and bool(mesh_names & {FSDPTurboDim.EP, FSDPTurboDim.EFSDP})
        if is_ep_side:
            ep_group = ep_group or _get_dtensor_group(grad, FSDPTurboDim.EP)
            efsdp_group = efsdp_group or _get_dtensor_group(grad, FSDPTurboDim.EFSDP)
            ep_params.append(param)
        else:
            non_ep_params.append(param)

    if not ep_params and not non_ep_params:
        return 0.0

    total_pth = torch.zeros((), device=device, dtype=torch.float32)
    if non_ep_params:
        non_ep_pth = _local_pth_sum(non_ep_params, norm_type, device)
        total_pth = total_pth + _allreduce_sum_(non_ep_pth, [dp_group, cp_group])
    if ep_params:
        ep_pth = _local_pth_sum(ep_params, norm_type, device)
        total_pth = total_pth + _allreduce_sum_(ep_pth, [efsdp_group, ep_group, cp_group])

    total_norm = total_pth.pow(1.0 / norm_type)
    clip_coef = min(max_norm / (float(total_norm.item()) + 1e-6), 1.0)
    if clip_coef < 1.0:
        for param in ep_params + non_ep_params:
            grad = getattr(param, "grad", None)
            if grad is not None:
                grad.detach().mul_(clip_coef)

    return float(total_norm.item())


class FSDPTurboEngine(FSDP2Engine):
    """FSDPTurbo backend that keeps LlamaFactory's training/load/checkpoint lifecycle."""

    def __init__(self, dist_config: dict, bf16: bool = False):
        super().__init__(dist_config, bf16=bf16)
        self.dist_config = dist_config
        self.ep_size = int(self.dist_config.get("ep_size", 1))
        self.ep_fsdp_size = int(self.dist_config.get("ep_fsdp_size", 1))
        self._expert_ep_mesh: DeviceMesh | None = None
        self._expert_efsdp_mesh: DeviceMesh | None = None
        self.mixed_precision = "bf16" if bf16 else "fp32"
        self._init_backend_meshes()

    def _init_backend_meshes(self) -> None:
        if self.ep_size <= 1:
            return

        dp_size = self.dist_interface.get_world_size(Dim.DP)
        cp_size = self.dist_interface.get_world_size(Dim.CP)
        if dp_size % (self.ep_size * self.ep_fsdp_size) != 0:
            raise ValueError(
                f"dp_size must be divisible by ep_size * ep_fsdp_size, got "
                f"{dp_size} % ({self.ep_size} * {self.ep_fsdp_size}) != 0."
            )

        edp_size = dp_size // (self.ep_size * self.ep_fsdp_size)
        expert_mesh = init_device_mesh(
            device_type=self.dist_interface.current_device.type,
            mesh_shape=(edp_size, self.ep_fsdp_size, self.ep_size, cp_size),
            mesh_dim_names=(FSDPTurboDim.EDP, FSDPTurboDim.EFSDP, FSDPTurboDim.EP, FSDPTurboDim.CP),
        )
        self._expert_ep_mesh = expert_mesh[FSDPTurboDim.EP]
        self._expert_efsdp_mesh = expert_mesh[FSDPTurboDim.EFSDP]

    def _get_expert_meshes(self) -> tuple[DeviceMesh | None, DeviceMesh | None]:
        return self._expert_ep_mesh, self._expert_efsdp_mesh

    def _get_spec(self, model: HFModel) -> EPModelSpec | None:
        return EPModelSpec.get(model)

    def _get_ep_modules(self, model: HFModel) -> list[str]:
        ep_modules = self.dist_config.get("ep_modules")
        if ep_modules is not None:
            return ep_modules

        spec = self._get_spec(model)
        if spec is not None:
            return spec.ep_modules

        return []

    def _get_ep_fsdp_modules(self, model: HFModel, ep_modules: list[str]) -> list[str]:
        ep_fsdp_modules = self.dist_config.get("ep_fsdp_modules")
        if ep_fsdp_modules is not None:
            return ep_fsdp_modules

        spec = self._get_spec(model)
        if spec is not None and spec.ep_fsdp_modules is not None:
            return spec.ep_fsdp_modules

        return [module.removesuffix(".experts") if module.endswith(".experts") else module for module in ep_modules]

    def prepare_model_ep(self, model: HFModel) -> tuple[HFModel, set]:
        """Apply LF-owned EP/EFSDP and return parameters that outer FSDP should ignore."""
        ep_modules = self._get_ep_modules(model)
        if self.ep_size > 1 and not ep_modules:
            raise ValueError(
                f"`ep_modules` is not specified and no built-in FSDPTurbo EP spec is registered for "
                f"model_type={get_model_type(model)}."
            )

        spec = self._get_spec(model)
        if spec is not None:
            model = spec.prepare(model)

        if self.ep_size > 1:
            ep_mesh, efsdp_mesh = self._get_expert_meshes()
            if ep_mesh is None:
                raise RuntimeError("FSDPTurbo EP mesh is not initialized.")

            dispatcher = self.dist_config.get("ep_dispatcher", "eager")
            fixed_router = self.dist_config.get("fixed_router", False)
            dispatcher_factory = get_dispatcher_factory(dispatcher)
            gradient_divide_factor = float(self.ep_size * self.ep_fsdp_size)

            logger.info_rank0(f"Applying LlamaFactory EP with mesh: {ep_mesh}")
            logger.info_rank0(f"EP apply patterns: {ep_modules}")
            logger.info_rank0(f"FSDPTurbo dispatcher: {dispatcher}")
            logger.info_rank0(f"EP gradient divide factor: {gradient_divide_factor}")
            model = apply_expert_parallel(
                model,
                ep_mesh,
                ep_modules,
                dispatcher_factory,
                fixed_router=fixed_router,
            )

            if self.ep_fsdp_size > 1:
                if efsdp_mesh is None:
                    raise RuntimeError("FSDPTurbo EFSDP mesh is not initialized.")

                if self.dist_interface.current_device.type == "npu":
                    apply_hccl_premul_sum_patch()

                ep_fsdp_modules = self._get_ep_fsdp_modules(model, ep_modules)
                logger.info_rank0(f"Applying LlamaFactory EFSDP with mesh: {efsdp_mesh}")
                logger.info_rank0(f"EFSDP apply patterns: {ep_fsdp_modules}")
                model = apply_expert_fsdp(model, efsdp_mesh, ep_fsdp_modules, gradient_divide_factor)

        fsdp_ignored_modules = list(self.dist_config.get("fsdp_ignored_modules", []))
        if self.ep_size > 1:
            fsdp_ignored_modules.extend(ep_modules)

        ignored_params = collect_ignored_params(model, fsdp_ignored_modules)

        if ignored_params:
            logger.info_rank0(f"FSDPTurbo FSDP2: Ignoring {len(ignored_params)} EP parameters from outer FSDP.")

        return model, ignored_params

    def prepare_model(self, model: HFModel) -> HFModel:
        model, ignored_params = self.prepare_model_ep(model)
        return super().prepare_model(model, ignored_params=ignored_params)

    def _warmup_grad_norm(self, model: HFModel) -> None:
        if self.fsdp_mesh is None:
            return

        logger.info_rank0("Warming up FSDPTurbo grad norm computation...")
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

        with torch.no_grad():
            clip_grad_norm_(model, 1.0)

        for param in model.parameters():
            if param.requires_grad:
                param.grad = None

        logger.info_rank0("FSDPTurbo grad norm warmup completed.")

    def _copy_weights(self, param, loaded_tensor):
        from torch.distributed._tensor import DTensor, Shard

        if loaded_tensor.dtype != param.dtype:
            loaded_tensor = loaded_tensor.to(param.dtype)

        if isinstance(param, DTensor):
            local_tensor = param.to_local()
            shard_placements = [
                (i, placement) for i, placement in enumerate(param.placements) if isinstance(placement, Shard)
            ]

            if not shard_placements:
                local_tensor.copy_(loaded_tensor)
                return

            mesh = param.device_mesh
            my_coordinate = mesh.get_coordinate()
            if my_coordinate is None:
                return

            sliced_tensor = loaded_tensor
            for mesh_dim, shard_placement in shard_placements:
                dim = shard_placement.dim
                rank_in_dim = my_coordinate[mesh_dim]
                world_size_in_dim = mesh.size(mesh_dim)
                full_size = sliced_tensor.shape[dim]
                chunk_size = (full_size + world_size_in_dim - 1) // world_size_in_dim
                start = rank_in_dim * chunk_size
                end = min(start + chunk_size, full_size)

                if start >= full_size:
                    return

                sliced_tensor = sliced_tensor.narrow(dim, start, end - start)

            slices = [slice(None)] * local_tensor.ndim
            for _, shard_placement in shard_placements:
                dim = shard_placement.dim
                slices[dim] = slice(0, sliced_tensor.shape[dim])
            local_tensor[tuple(slices)].copy_(sliced_tensor)
            return

        param.data.copy_(loaded_tensor)
