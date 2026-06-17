# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


@register_op('permute', 'npu')
def permute_npu(tokens, indices):
    """
    NPU implementation of token permutation for MoE (Mixture of Experts).
    Uses optimized NPU kernel for better performance.
    
    Args:
        tokens: Input tokens tensor [num_tokens, hidden_dim]
        indices: Expert indices tensor [num_tokens] or [num_tokens, topk]
    
    Returns:
        permuted_tokens: Permuted tokens
        sorted_indices: Indices for unpermutation
    """
    return torch_npu.npu_moe_token_permute(tokens, indices)


@register_op('unpermute', 'npu')
def unpermute_npu(permuted_tokens, sorted_indices, probs=None):
    """
    NPU implementation of token unpermutation for MoE (Mixture of Experts).
    Uses optimized NPU kernel for better performance.
    
    Args:
        permuted_tokens: Permuted tokens tensor
        sorted_indices: Indices from permutation operation
        probs: Optional probability weights [num_tokens, topk]
    
    Returns:
        unpermuted_tokens: Unpermuted tokens
    """
    if probs is not None:
        permuted_tokens = permuted_tokens.to(probs.dtype)
    return torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs)
