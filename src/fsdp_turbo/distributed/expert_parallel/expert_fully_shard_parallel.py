import logging

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import Shard

from fsdp_turbo.distributed.fully_shard_parallel.fully_shard_parallel import get_fsdp_hook_modules, find_hook_module
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, FSDPPlanConfig
from fsdp_turbo.distributed.fine_grained_fully_shard import get_fsdp_strategy
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match
from fsdp_turbo.utils.torch_patch import apply_hccl_premul_sum_patch

logger = logging.getLogger(__name__)


def expert_fully_shard_modules(model: torch.nn.Module, efsdp_mesh, ep_plan: EPPlanConfig, fsdp_plan: FSDPPlanConfig) -> torch.nn.Module:
    efsdp_modules = get_efsdp_modules(model, ep_plan)
    efsdp_hook_modules = get_fsdp_hook_modules(model, fsdp_plan)
    config = {'mesh': efsdp_mesh,
              'mp_policy': MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
              'shard_placement_fn': lambda x: Shard(1) if x.dim() > 1 else Shard(0)}

    apply_hccl_premul_sum_patch()
    strategy = get_fsdp_strategy(fsdp_plan.fsdp_implementation)
    fully_shard_fn = strategy.get_unified_fully_shard_fn()

    for experts in efsdp_modules:
        hook_module = find_hook_module(experts, efsdp_hook_modules)
        if isinstance(experts, torch.nn.ModuleList):
            for expert in experts:
                fully_shard_fn(expert, hook_module=hook_module, **config)
                set_gradient_divide_factor(expert, ep_plan._gradient_divide_factor)
        else:
            fully_shard_fn(experts, hook_module=hook_module, **config)
            set_gradient_divide_factor(experts, ep_plan._gradient_divide_factor)

    return model


def get_efsdp_modules(modules: torch.nn.Module, plan: EPPlanConfig):
    efsdp_modules = []
    for plan_name in plan.apply_efsdp_modules:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                print_rank(logger.debug, f'[Expert Fully Shard]: Apply efsdp to module <{name}>')
                efsdp_modules.append(module)
    if len(efsdp_modules) == 0:
        raise RuntimeError(f'[Expert Fully Shard] No module named {plan} or not be ModuleList')
    return efsdp_modules


def set_gradient_divide_factor(module, factor):
    if hasattr(module, 'set_gradient_divide_factor'):
        module.set_gradient_divide_factor(factor)
    else:
        module.set_reduce_scatter_divide_factor(factor)
