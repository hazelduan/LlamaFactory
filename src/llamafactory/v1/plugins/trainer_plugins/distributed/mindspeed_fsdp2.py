# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import torch

from ....accelerator.interface import DistributedInterface
from ....utils.logging import get_logger
from ....utils.types import HFModel


logger = get_logger(__name__)


_FSDP2_NPU_COMM_PATCHED = False


def _patch_fsdp2_comm_context_lazy_init() -> None:
    """Initialize native FSDP2 communication streams for non-root-wrapped modules."""

    global _FSDP2_NPU_COMM_PATCHED
    if _FSDP2_NPU_COMM_PATCHED:
        return

    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext, FSDPParamGroup
    except Exception as exc:  # pragma: no cover - depends on torch internals
        logger.warning_rank0("Unable to patch native FSDP2 communication context: %s", exc)
        return

    original_lazy_init = FSDPParamGroup.lazy_init
    original_get_all_gather_streams = FSDPCommContext.get_all_gather_streams
    original_prefetch_unshard = FSDPParamGroup._prefetch_unshard

    def lazy_init_with_comm_streams(self):
        # FSDPTurbo may wrap only leaf modules to keep expert parameters local;
        # each module then needs its own fully initialized communication context.
        if not hasattr(self.comm_ctx, "all_gather_copy_in_stream"):
            self.comm_ctx.lazy_init(self.device)
        return original_lazy_init(self)

    def get_all_gather_streams_with_fallback(self, async_op, training_state):
        if hasattr(self, "all_gather_copy_in_stream") and hasattr(self, "all_gather_stream"):
            return original_get_all_gather_streams(self, async_op, training_state)
        if hasattr(self, "device_handle"):
            current_stream = self.device_handle.current_stream()
        else:
            current_stream = torch.npu.current_stream() if hasattr(torch, "npu") else torch.cuda.current_stream()
        return current_stream, current_stream

    def prefetch_unshard_with_lazy_init(target_fsdp_param_group, pass_type):
        target_fsdp_param_group.lazy_init()
        return original_prefetch_unshard(target_fsdp_param_group, pass_type)

    FSDPParamGroup.lazy_init = lazy_init_with_comm_streams
    FSDPCommContext.get_all_gather_streams = get_all_gather_streams_with_fallback
    FSDPParamGroup._prefetch_unshard = staticmethod(prefetch_unshard_with_lazy_init)
    _FSDP2_NPU_COMM_PATCHED = True


def _dtype_name(bf16: bool) -> str:
    return "bf16" if bf16 else "fp32"


class MindSpeedFSDP2Engine:
    """FSDPTurbo-backed FSDP2 + expert-parallel engine for V1 training."""

    def __init__(self, dist_config: dict, bf16: bool = False):
        self.dist_config = dist_config
        self.bf16 = bf16
        self.dist_interface = DistributedInterface()
        self.rank = self.dist_interface.get_rank()
        self.world_size = self.dist_interface.get_world_size()

    def _build_turbo_config(self):
        from fsdp_turbo.fsdp_turbo_config import (
            DistributedConfig,
            EPPlanConfig,
            FSDPPlanConfig,
            FSDPTurboConfig,
            MemoryConfig,
            QuantizationConfig,
            QuantizeConfig,
            TPPlanConfig,
        )

        dtype = _dtype_name(self.bf16)
        fsdp_size = int(self.dist_config.get("fsdp_size", self.dist_config.get("fully_shard_parallel_size", self.world_size)))
        ep_size = int(self.dist_config.get("ep_size", self.dist_config.get("expert_parallel_size", 1)))
        efsdp_size = int(
            self.dist_config.get(
                "ep_fsdp_size",
                self.dist_config.get("expert_fully_shard_parallel_size", self.dist_config.get("efsdp_size", 1)),
            )
        )

        fsdp_plan = FSDPPlanConfig(
            apply_modules=self.dist_config.get(
                "fsdp_modules",
                {
                    "model.language_model.layers.{*}.linear_attn.conv1d": {},
                    "model.language_model.layers.{*}.linear_attn.out_proj": {},
                    "model.language_model.layers.{*}.linear_attn.in_proj_qkv": {},
                    "model.language_model.layers.{*}.linear_attn.in_proj_z": {},
                    "model.language_model.layers.{*}.linear_attn.in_proj_b": {},
                    "model.language_model.layers.{*}.linear_attn.in_proj_a": {},
                    "model.language_model.layers.{*}.self_attn.q_proj": {},
                    "model.language_model.layers.{*}.self_attn.k_proj": {},
                    "model.language_model.layers.{*}.self_attn.v_proj": {},
                    "model.language_model.layers.{*}.self_attn.o_proj": {},
                    "model.language_model.layers.{*}.mlp.gate": {},
                    "model.language_model.layers.{*}.mlp.shared_expert.gate_proj": {},
                    "model.language_model.layers.{*}.mlp.shared_expert.up_proj": {},
                    "model.language_model.layers.{*}.mlp.shared_expert.down_proj": {},
                    "model.language_model.layers.{*}.mlp.shared_expert_gate": {},
                    "model.language_model.embed_tokens": {},
                    "lm_head": {},
                },
            ),
            param_dtype=dtype,
            reduce_dtype=self.dist_config.get("reduce_dtype", "fp32"),
            output_dtype=dtype,
            cast_forward_inputs=True,
            reshard_after_forward=self.dist_config.get("reshard_after_forward", True),
            num_to_forward_prefetch=self.dist_config.get("num_to_forward_prefetch", 0),
            num_to_backward_prefetch=self.dist_config.get("num_to_backward_prefetch", 0),
            hook_modules=self.dist_config.get("hook_modules", ["model.language_model.layers.{*}"]),
            fsdp_implementation=self.dist_config.get("fsdp_implementation", "native"),
        )
        tp_plan = TPPlanConfig(
            colwise_parallel=self.dist_config.get("tp_colwise_modules", []),
            rowwise_parallel=self.dist_config.get("tp_rowwise_modules", []),
            sequence_parallel=self.dist_config.get("tp_sequence_modules", []),
        )
        ep_plan = EPPlanConfig(
            apply_modules=self.dist_config.get("ep_modules", []),
            dispatcher=self.dist_config.get("ep_dispatcher", "eager"),
            apply_efsdp_modules=self.dist_config.get("ep_fsdp_modules", []),
        )
        quant_plan = QuantizeConfig() if self.dist_config.get("quantization_plan") is None else self.dist_config.get("quantization_plan")

        return FSDPTurboConfig(
            distributed=DistributedConfig(
                fully_shard_parallel_size=fsdp_size,
                tensor_parallel_size=int(self.dist_config.get("tp_size", self.dist_config.get("tensor_parallel_size", 1))),
                context_parallel_size=int(self.dist_config.get("cp_size", self.dist_config.get("context_parallel_size", 1))),
                ulysses_parallel_size=int(self.dist_config.get("ulysses_parallel_size", 1)),
                expert_parallel_size=ep_size,
                expert_fully_shard_parallel_size=efsdp_size,
                fsdp_plan=fsdp_plan,
                tp_plan=tp_plan,
                ep_plan=ep_plan,
            ),
            memory=MemoryConfig(
                recompute=bool(self.dist_config.get("recompute", True)),
                recompute_plan=self.dist_config.get("recompute_modules", ["model.language_model.layers.{*}"]),
            ),
            quantization=QuantizationConfig(quantization_plan=quant_plan),
        )

    def shard_model(self, model: HFModel) -> HFModel:
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        _patch_fsdp2_comm_context_lazy_init()

        config = self._build_turbo_config()
        init_mode = getattr(model, "_init_mode", "init_on_default")
        init_device = "meta" if init_mode == "init_on_meta" else "cpu"
        weights_path = getattr(model.config, "name_or_path", None)

        if self.rank == 0:
            logger.info(
                "Using MindSpeed/FSDPTurbo FSDP2 path: fsdp_size=%s, ep_size=%s, efsdp_size=%s, dispatcher=%s.",
                config.distributed.fully_shard_parallel_size,
                config.distributed.expert_parallel_size,
                config.distributed.expert_fully_shard_parallel_size,
                config.distributed.ep_plan.dispatcher,
            )

        turbo_model = FSDPTurbo(config, model, init_device=init_device, weights_path=weights_path)
        if getattr(model, "is_gradient_checkpointing", False) and hasattr(turbo_model.model, "enable_input_require_grads"):
            turbo_model.model.enable_input_require_grads()
        return turbo_model
