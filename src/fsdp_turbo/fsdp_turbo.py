# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional

import torch
from torch.distributed.tensor import DTensor

from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules
from fsdp_turbo.distributed.fully_shard_parallel.fully_shard_parallel import \
    fully_shard_parallel_modules
from fsdp_turbo.distributed.parallel_state import init_parallel_state
from fsdp_turbo.distributed.tensor_parallel.tensor_parallel import tensor_parallel_modules
from fsdp_turbo.memory.chunk_batch.chunk_batch import chunk_batch_modules
from fsdp_turbo.memory.recompute.recompute import recompute_modules
from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig
from fsdp_turbo.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
from fsdp_turbo.model_loader.model_loader import WeightLoader
from fsdp_turbo.model_loader.device_manager import get_device_type
from fsdp_turbo.utils.log import print_rank

import logging
import os
logger = logging.getLogger(__name__)


class FSDPTurbo(torch.nn.Module):
    def __init__(self, config: FSDPTurboConfig, model: torch.nn.Module,
                 init_device: str = "cpu", weights_path: Optional[str] = None,
                 seed: Optional[int] = None):
        super(FSDPTurbo, self).__init__()
        self.config = config
        self.model = model
        self.init_device = init_device
        self.weights_path = weights_path
        self.seed = seed

        self.parallel_state = init_parallel_state(self.config)

        if self.init_device == "meta":
            self._init_meta_path()
        else:
            self._init_cpu_path()

    def _init_meta_path(self):
        """
        Industry-standard meta device initialization flow.

        Flow:
        1. Materialize non-FSDP modules (norm, lm_head, etc.) BEFORE FSDP
        2. Save buffer values (computed during model __init__ on CPU)
        3. Apply FSDP with param_init_fn -> materializes directly to NPU
        4. Restore buffer values (destroyed by to_empty() in param_init_fn)
        5. Post-process (embedding tying)
        6. Apply TP, EP, recompute, quantization
        """
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device = f'npu:{local_rank}'

        if self.config.distributed.fully_shard_parallel_size > 1:
            ep_pre_sharded = False
            if self.config.distributed.expert_parallel_size > 1:
                ep_pre_sharded = self._materialize_ep_modules_before_fsdp(device)

            self._materialize_non_fsdp_modules(device)

            buffer_backup = {}
            for name, buffer in self.model.named_buffers():
                if buffer.device.type != "meta":
                    buffer_backup[name] = buffer.clone()
            logger.info(f"> Saved {len(buffer_backup)} buffers before FSDP wrapping")

            param_init_fn = self._create_param_init_fn(device)
            self.config.distributed.fsdp_plan.param_init_fn = param_init_fn
            logger.info("> Applying FSDP with param_init_fn (lazy materialization)...")
            self.apply_fsdp_modules()

            self._restore_buffers(buffer_backup)
        else:
            logger.info(f"> FSDP size=1, materializing directly to {device}...")
            WeightLoader.load(
                model=self.model,
                weights_path=self.weights_path,
                device=device,
                seed=self.seed,
            )

        self._validate_no_meta_params()

        self._post_process_meta()

        self.apply_tp_modules()
        self.apply_ep_modules(skip_expert_parallel=ep_pre_sharded)
        self.apply_recompute_modules()
        self.apply_quantization_modules()

    def _create_param_init_fn(self, device: str):
        """Create param_init_fn for FSDP lazy initialization.

        Returns a param_init_fn that either loads from checkpoint or does
        random initialization, depending on whether weights_path is provided.
        """
        if self.weights_path is not None:
            logger.info("> Pre-scanning checkpoint metadata for lazy loading...")
            metadata = WeightLoader.preload_metadata(self.weights_path)
            return WeightLoader.create_param_init_fn(
                self.model, metadata, device=device, seed=self.seed
            )
        else:
            logger.info("> Creating random init param_init_fn for training from scratch...")
            return WeightLoader.create_random_init_fn(
                self.model, device=device, seed=self.seed
            )

    def _post_process_meta(self):
        """Post-process after meta device materialization.

        Random init is handled by param_init_fn during FSDP wrapping.
        Only embedding tying needs to be done here.
        """
        if hasattr(self.model, 'config') and getattr(self.model.config, "tie_word_embeddings", True):
            try:
                input_embeddings = self.model.get_input_embeddings()
                output_embeddings = self.model.get_output_embeddings()
                if output_embeddings is not None and input_embeddings is not None:
                    output_embeddings.weight = input_embeddings.weight
                    logger.info("> Tied input/output embeddings")
            except Exception as e:
                logger.warning(f"> Failed to tie embeddings: {e}")

    def _init_cpu_path(self):
        """Traditional CPU initialization flow."""
        has_real_params = any(
            p.device.type != 'meta' for p in self.model.parameters()
        )
        if self.weights_path is not None or not has_real_params:
            logger.info("> Loading weights before parallel wrapping...")
            WeightLoader.load(
                model=self.model,
                weights_path=self.weights_path,
                device="cpu",
                seed=self.seed,
            )
        else:
            logger.info("> Model already has real parameters, skipping weight loading")

        self.apply_quantization_modules()
        self.apply_tp_modules()
        self._capture_tp_info_on_mx_linear()
        self.apply_ep_modules()
        self.apply_recompute_modules()
        self.apply_chunk_batch_modules()
        self.apply_fsdp_modules()

        if self.config.distributed.fully_shard_parallel_size == 1:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            device = torch.device(f'npu:{local_rank}')
            self.model = self.model.to(device)
            logger.info(f"> Moved model to {device} (fsdp_size=1)")

    def _restore_buffers(self, buffer_backup: dict):
        """Restore buffer values that were destroyed by to_empty() during param_init_fn."""
        if not buffer_backup:
            return

        from fsdp_turbo.model_loader.model_loader import _find_submodule

        restored = 0
        for name, buffer in buffer_backup.items():
            try:
                module, local_name = _find_submodule(self.model, name)
                model_buffer = dict(module.named_buffers(recurse=False))[local_name]
                model_buffer.copy_(buffer.to(device=model_buffer.device, dtype=model_buffer.dtype))
                restored += 1
            except Exception as e:
                logger.warning(f"> Failed to restore buffer {name}: {e}")

        logger.info(f"> Restored {restored}/{len(buffer_backup)} buffers after FSDP wrapping")

    def _materialize_non_fsdp_modules(self, device: str):
        """Materialize modules not covered by FSDP plan BEFORE FSDP wrapping.

        Industry-standard approach: non-sharded modules (norm, lm_head, etc.)
        are materialized first as regular torch.Tensor. FSDP is then applied
        only to sharded modules via param_init_fn. This avoids DTensor conflicts.
        """
        from fsdp_turbo.utils.str_match import module_name_match

        fsdp_patterns = set(self.config.distributed.fsdp_plan.apply_modules.keys()) if self.config.distributed.fsdp_plan.apply_modules else set()

        fsdp_covered = set()
        for full_name, _ in self.model.named_modules():
            for pattern in fsdp_patterns:
                if module_name_match(pattern, full_name):
                    fsdp_covered.add(full_name)
                    break

        modules_to_materialize = {}
        for full_name, module in self.model.named_modules():
            has_meta = any(
                p.device.type == 'meta' for p in module.parameters(recurse=False)
            )
            if not has_meta:
                continue

            parts = full_name.split(".")
            is_covered = False
            for i in range(len(parts)):
                ancestor = ".".join(parts[:i + 1])
                if ancestor in fsdp_covered:
                    is_covered = True
                    break
            if is_covered:
                continue

            modules_to_materialize[full_name] = module

        if not modules_to_materialize:
            logger.info("> No non-FSDP modules to materialize")
            return

        logger.info(
            f"> Materializing {len(modules_to_materialize)} non-FSDP modules "
            f"BEFORE FSDP wrapping: {list(modules_to_materialize.keys())}"
        )

        if self.weights_path is not None:
            metadata = WeightLoader.preload_metadata(self.weights_path)
            metadata_keys = set(metadata.keys())
        else:
            metadata = {}
            metadata_keys = set()

        for module_path, module in modules_to_materialize.items():
            module.to_empty(device=device)

            loaded_count = 0
            for local_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_path}.{local_name}" if module_path else local_name
                if full_name in metadata_keys:
                    info = metadata[full_name]
                    weight = WeightLoader._load_single_tensor(info)
                    param.data.copy_(weight.to(device=param.device, dtype=param.dtype))
                    loaded_count += 1

            if loaded_count > 0:
                logger.debug(f"> Loaded {loaded_count} weights for '{module_path}'")

        logger.info("> Non-FSDP modules materialized successfully")

    @torch.no_grad()
    def _materialize_ep_modules_before_fsdp(self, device: str) -> bool:
        """Load only the local EP shard for expert modules before FSDP wrapping."""
        if self.weights_path is None:
            return False

        ep_plan = self.config.distributed.ep_plan
        if not ep_plan.apply_modules:
            return False

        from fsdp_turbo.distributed.expert_parallel.expert_parallel import expert_parallelize_modules, get_ep_modules
        from fsdp_turbo.model_loader.model_loader import WeightLoader

        ep_modules = get_ep_modules(self.model, ep_plan)
        if not ep_modules:
            return False

        ep_mesh = self.parallel_state.get_ep_device_mesh()
        ep_group = ep_mesh.get_group()
        ep_rank = torch.distributed.get_rank(ep_group)
        ep_size = torch.distributed.get_world_size(ep_group)
        module_prefix = {module: name for name, module in self.model.named_modules()}
        metadata = WeightLoader.preload_metadata(self.weights_path)

        materialized = 0
        for module in ep_modules:
            num_global_experts = getattr(module, "num_experts", None)
            if num_global_experts is None:
                num_global_experts = len(module)
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
                    [i % num_local_experts for i in range(num_global_experts)],
                    dtype=torch.int32,
                    device=torch.accelerator.current_device_index(),
                )

            prefix = module_prefix[module]
            loaded_for_module = 0
            for local_name, param in list(module.named_parameters(recurse=True)):
                full_name = f"{prefix}.{local_name}" if prefix else local_name
                if full_name not in metadata:
                    continue

                weight = WeightLoader._load_single_tensor(metadata[full_name])
                if weight.dim() > 0 and weight.shape[0] == num_global_experts:
                    weight = weight[local_start:local_end].contiguous()

                parent = module
                pieces = local_name.split(".")
                for piece in pieces[:-1]:
                    parent = getattr(parent, piece)
                setattr(
                    parent,
                    pieces[-1],
                    torch.nn.Parameter(weight.to(device=device, dtype=param.dtype), requires_grad=param.requires_grad),
                )
                loaded_for_module += 1

            if loaded_for_module == 0:
                raise RuntimeError(f"No expert parameters loaded for EP module '{prefix}'.")

            module._fsdp_turbo_ep_pre_sharded = True
            materialized += loaded_for_module

        logger.info(
            f"> Pre-sharded {len(ep_modules)} EP modules before FSDP "
            f"(rank {ep_rank}/{ep_size}, loaded {materialized} tensors)"
        )
        self.model = expert_parallelize_modules(self.model, ep_mesh, ep_plan)
        return True

    def _validate_no_meta_params(self):
        """Validate that no parameters remain on meta device."""
        remaining = [
            full_name for full_name, param in self.model.named_parameters()
            if param.device.type == 'meta'
        ]
        if remaining:
            raise RuntimeError(
                f"Found {len(remaining)} parameters still on meta device. "
                f"First 10: {remaining[:10]}"
            )
        logger.info("> Verified: no parameters remain on meta device")

    def apply_fsdp_modules(self):
        if self.config.distributed.fully_shard_parallel_size == 1:
            return
        self.model = fully_shard_parallel_modules(self.model, self.parallel_state.get_fsdp_device_mesh(), self.config.distributed.fsdp_plan)

    def apply_tp_modules(self):
        if self.config.distributed.tensor_parallel_size == 1:
            return
        self.model = tensor_parallel_modules(self.model, self.parallel_state.get_tp_device_mesh(), self.config.distributed.tp_plan)

    def apply_ep_modules(self, skip_expert_parallel: bool = False):
        if self.config.distributed.expert_parallel_size > 1 and not skip_expert_parallel:
            self.model = expert_parallelize_modules(self.model, self.parallel_state.get_ep_device_mesh(), self.config.distributed.ep_plan)
        if self.config.distributed.expert_fully_shard_parallel_size > 1:
            self.model = expert_fully_shard_modules(
                self.model,
                self.parallel_state.get_efsdp_device_mesh(),
                self.config.distributed.ep_plan,
                self.config.distributed.fsdp_plan,
            )

    def apply_recompute_modules(self):
        if not self.config.memory.recompute:
            return
        self.model = recompute_modules(self.model, self.config.memory.recompute_plan)

    def _capture_tp_info_on_mx_linear(self):
        if self.config.distributed.tensor_parallel_size <= 1:
            return
        from fsdp_turbo.quantization.mx_formats.mx_linear import MXLinear
        from torch.distributed.tensor import Shard, Partial
        from fsdp_turbo.utils.str_match import module_name_match
        tp_mesh = self.parallel_state.get_device_mesh('tp')
        colwise = self.config.distributed.tp_plan.colwise_parallel or []
        rowwise = self.config.distributed.tp_plan.rowwise_parallel or []
        for name, mod in self.model.named_modules():
            if not isinstance(mod, MXLinear):
                continue
            if any(module_name_match(p, name) for p in colwise):
                mod._tp_mesh = tp_mesh
                mod._tp_output_placements = [Shard(-1)]
            elif any(module_name_match(p, name) for p in rowwise):
                mod._tp_mesh = tp_mesh
                mod._tp_output_placements = [Partial()]

    def apply_chunk_batch_modules(self):
        if not self.config.memory.chunk_batch:
            return
        self.model = chunk_batch_modules(self.model, self.config.memory.chunk_batch_plan)

    def apply_quantization_modules(self):
        """Apply quantization based on quantization_format + quantization_recipe."""
        if not self.config.quantization.quantization_plan.quant_recipe:
            return
        try:
            # When recompute is enabled, forward may be called during backward.
            # Force "all" mode so both fwd and bwd quantized tensors are available.
            if self.config.memory.recompute:
                self.config.quantization.quantization_plan.fsdp_low_precision_all_gather_mode = "all"

            from fsdp_turbo.quantization.converter.model_converter import build_model_converter

            model_converters = build_model_converter(self.config.quantization.quantization_plan)
            model_converters.convert(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to convert quantization plan") from e

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save_checkpoint(self, save_path: str, rank: int = 0) -> None:
        """
        Save model checkpoint.
        All ranks must participate in DTensor gathering, but only rank 0 saves to disk.

        Args:
            save_path: Directory path to save checkpoint
            rank: Current rank, only rank 0 will save to disk
        """
        os.makedirs(save_path, exist_ok=True)

        state_dict = {}
        for name, param in self.model.named_parameters():
            if isinstance(param.data, DTensor):
                full_tensor = param.data.full_tensor().cpu()
            else:
                full_tensor = param.data.cpu()

            if rank == 0:
                state_dict[name] = full_tensor

        for name, buffer in self.model.named_buffers():
            if isinstance(buffer.data, DTensor):
                full_tensor = buffer.data.full_tensor().cpu()
            else:
                full_tensor = buffer.data.cpu()

            if rank == 0:
                state_dict[name] = full_tensor

        if rank == 0:
            checkpoint_file = os.path.join(save_path, "model.safetensors")
            from safetensors.torch import save_file
            save_file(state_dict, checkpoint_file)
            logger.info(f"> Saved checkpoint to {checkpoint_file} with {len(state_dict)} tensors")
