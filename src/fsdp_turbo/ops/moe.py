# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from fsdp_turbo.ops import dispatch_op


def grouped_matmul(inputs, m_split, weights, use_eager=False):
    """
    Grouped matrix multiplication with automatic device dispatch.
    
    Args:
        inputs: Input tensor
        m_split: Group sizes
        weights: Weight tensor
        use_eager: If False, use fused (optimized) implementation for current device.
                   If True, use CPU implementation.
    
    Returns:
        Result of grouped matrix multiplication
    """
    device_type = 'cpu' if use_eager else None  # Use CPU implementation if not fused
    return dispatch_op('grouped_matmul', inputs, m_split, weights, device_type=device_type)


def all2all_grouped_matmul(inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None, use_eager=False):
    """
    All-to-all communication followed by grouped matrix multiplication.

    Args:
        inputs: Input tensor
        weights: Weight tensor
        group: Process group for distributed communication
        send_counts: Counts of elements to send to each rank
        recv_counts: Counts of elements to receive from each rank
        shared_inputs: Optional shared expert inputs
        shared_weight: Optional shared expert weight
        use_eager: If False, use fused (optimized) implementation for current device.
                   If True, use CPU implementation.

    Returns:
        Output tensor(s) - expert output and optionally shared expert output
    """
    device_type = 'cpu' if use_eager else None  # Use CPU implementation if not fused
    return dispatch_op('all2all_grouped_matmul', inputs, weights, group, send_counts, recv_counts,
                      shared_inputs, shared_weight, device_type=device_type)


def grouped_matmul_all2all(inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None, use_eager=False):
    """
    Grouped matrix multiplication followed by all-to-all communication.

    Args:
        inputs: Input tensor
        weights: Weight tensor
        group: Process group for distributed communication
        send_counts: Counts of elements to send to each rank
        recv_counts: Counts of elements to receive from each rank
        shared_inputs: Optional shared expert inputs
        shared_weight: Optional shared expert weight
        use_eager: If False, use fused (optimized) implementation for current device.
                   If True, use CPU implementation.

    Returns:
        Output tensor(s) - expert output and optionally shared expert output
    """
    device_type = 'cpu' if use_eager else None  # Use CPU implementation if not fused
    return dispatch_op('grouped_matmul_all2all', inputs, weights, group, send_counts, recv_counts,
                      shared_inputs, shared_weight, device_type=device_type)


def permute(tokens, indices, use_eager=False):
    """
    Token permutation for MoE (Mixture of Experts).

    Args:
        tokens: Input tokens tensor [num_tokens, hidden_dim]
        indices: Expert indices tensor [num_tokens] or [num_tokens, topk]
        use_eager: If False, use fused (optimized) implementation for current device.
                   If True, use CPU implementation.

    Returns:
        permuted_tokens: Permuted tokens
        sorted_indices: Indices for unpermutation
    """
    device_type = 'cpu' if use_eager else None  # Use CPU implementation if not fused
    return dispatch_op('permute', tokens, indices, device_type=device_type)


def unpermute(permuted_tokens, sorted_indices, probs=None, use_eager=False):
    """
    Token unpermutation for MoE (Mixture of Experts).

    Args:
        permuted_tokens: Permuted tokens tensor
        sorted_indices: Indices from permutation operation
        probs: Optional probability weights [num_tokens, topk]
        use_eager: If False, use fused (optimized) implementation for current device.
                   If True, use CPU implementation.

    Returns:
        unpermuted_tokens: Unpermuted tokens
    """
    device_type = 'cpu' if use_eager else None  # Use CPU implementation if not fused
    if permuted_tokens.size(0) != sorted_indices.numel():
        raise AssertionError(
            f'permuted tokens({permuted_tokens.size(0)}) != sorted indices({sorted_indices.size()})'
        )
    return dispatch_op('unpermute', permuted_tokens, sorted_indices, probs, device_type=device_type)
