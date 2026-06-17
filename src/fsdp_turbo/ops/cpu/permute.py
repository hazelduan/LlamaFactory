# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from fsdp_turbo.ops.registry import register_op


@register_op('permute', 'cpu')
def permute_cpu(tokens, indices):
    """
    CPU implementation of token permutation for MoE (Mixture of Experts).
    
    Args:
        tokens: Input tokens tensor [num_tokens, hidden_dim]
        indices: Expert indices tensor [num_tokens] or [num_tokens, topk]
    
    Returns:
        permuted_tokens: Permuted tokens
        sorted_indices: Indices for unpermutation
    """
    topk = 1 if indices.dim() == 1 else indices.size(1)
    indices_dtype = indices.dtype
    sorted_indices = torch.argsort(indices.float().view(-1), stable=True).to(indices_dtype)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


@register_op('unpermute', 'cpu')
def unpermute_cpu(permuted_tokens, sorted_indices, probs=None):
    """
    CPU implementation of token unpermutation for MoE (Mixture of Experts).
    
    Args:
        permuted_tokens: Permuted tokens tensor
        sorted_indices: Indices from permutation operation
        probs: Optional probability weights [num_tokens, topk]
    
    Returns:
        unpermuted_tokens: Unpermuted tokens
    """
    num_tokens, topk = (permuted_tokens.size(0), 1) if probs is None else (probs.numel(), probs.size(1))
    unpermuted_tokens = torch.zeros([num_tokens, permuted_tokens.shape[-1]], dtype=permuted_tokens.dtype,
                                    device=permuted_tokens.device)
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens *= probs.unsqueeze(-1)
    return unpermuted_tokens.sum(dim=1)
