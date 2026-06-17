# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch.nn as nn


def create_fully_shard_wrapper(
    fully_shard_fn: Callable, 
    supports_hook_module: bool
) -> Callable:
    """
    Create a unified fully_shard wrapper to handle parameter differences.
    
    Args:
        fully_shard_fn: The original fully_shard function.
        supports_hook_module: Whether the function supports hook_module parameter.
    
    Returns:
        A wrapped fully_shard function that uniformly accepts hook_module parameter.
    """
    def wrapper(
        module: nn.Module,
        hook_module: Optional[nn.Module] = None,
        **kwargs
    ):
        if supports_hook_module:
            return fully_shard_fn(module, hook_module=hook_module, **kwargs)
        else:
            return fully_shard_fn(module, **kwargs)
    
    return wrapper


class FSDPStrategy(ABC):
    """Base class for FSDP implementation strategies."""
    
    @abstractmethod
    def get_fully_shard_fn(self) -> Callable:
        """Return the original fully_shard function."""
        pass
    
    @property
    @abstractmethod
    def supports_hook_module(self) -> bool:
        """Whether the implementation supports hook_module parameter."""
        pass
    
    def get_unified_fully_shard_fn(self) -> Callable:
        """Return a unified fully_shard wrapper function."""
        return create_fully_shard_wrapper(
            self.get_fully_shard_fn(),
            self.supports_hook_module
        )


class CustomFSDPStrategy(FSDPStrategy):
    """FSDPTurbo custom FSDP implementation."""
    
    def get_fully_shard_fn(self) -> Callable:
        from fsdp_turbo.distributed.fine_grained_fully_shard import fully_shard
        return fully_shard
    
    @property
    def supports_hook_module(self) -> bool:
        return True


class NativeFSDPStrategy(FSDPStrategy):
    """PyTorch native FSDP implementation."""
    
    def get_fully_shard_fn(self) -> Callable:
        from torch.distributed.fsdp import fully_shard
        return fully_shard
    
    @property
    def supports_hook_module(self) -> bool:
        return False


def get_fsdp_strategy(implementation: str) -> FSDPStrategy:
    """
    Get FSDP strategy based on configuration.
    
    Args:
        implementation: Implementation type, either 'custom' or 'native'.
    
    Returns:
        Corresponding FSDP strategy instance.
    """
    strategies = {
        'custom': CustomFSDPStrategy(),
        'native': NativeFSDPStrategy(),
    }
    if implementation not in strategies:
        raise ValueError(
            f"Unknown FSDP implementation: {implementation}. "
            f"Supported: {list(strategies.keys())}"
        )
    return strategies[implementation]
