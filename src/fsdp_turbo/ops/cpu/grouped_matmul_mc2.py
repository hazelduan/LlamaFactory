# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from fsdp_turbo.ops.registry import register_op


@register_op('all2all_grouped_matmul', 'cpu')
def all2all_grouped_matmul_cpu(inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None):
    """
    CPU implementation of all2all_grouped_matmul.
    
    This performs all-to-all communication followed by grouped matrix multiplication.
    For CPU, we simulate the all-to-all operation using standard distributed operations.
    
    Args:
        inputs: Input tensor
        weights: Weight tensor
        group: Process group for distributed communication
        send_counts: Counts of elements to send to each rank
        recv_counts: Counts of elements to receive from each rank
        shared_inputs: Optional shared expert inputs
        shared_weight: Optional shared expert weight
    
    Returns:
        Output tensor(s) - expert output and optionally shared expert output
    """
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    
    # Perform all-to-all communication
    # For CPU, we use all_to_all operation
    send_tensor = inputs
    recv_tensor = torch.empty_like(send_tensor)
    
    # Convert send_counts and recv_counts to lists if they are tensors
    if isinstance(send_counts, torch.Tensor):
        send_counts_list = send_counts.tolist()
    else:
        send_counts_list = send_counts
        
    if isinstance(recv_counts, torch.Tensor):
        recv_counts_list = recv_counts.tolist()
    else:
        recv_counts_list = recv_counts
    
    # Perform all-to-all
    torch.distributed.all_to_all_single(recv_tensor, send_tensor, recv_counts_list, send_counts_list, group=group)
    
    # Perform grouped matrix multiplication on received data
    # Calculate group sizes from recv_counts
    if isinstance(recv_counts, torch.Tensor):
        group_sizes = recv_counts.reshape(world_size, -1).sum(dim=0)
    else:
        # Assume recv_counts is already flattened
        group_sizes = torch.tensor(recv_counts_list).reshape(world_size, -1).sum(dim=0)
    
    # Split received tensor into groups
    output_chunks = []
    start_idx = 0
    for i, size in enumerate(group_sizes.tolist()):
        end_idx = start_idx + size
        chunk = recv_tensor[start_idx:end_idx]
        # Perform matrix multiplication with corresponding weight
        output_chunk = torch.matmul(chunk, weights[i])
        output_chunks.append(output_chunk)
        start_idx = end_idx
    
    # Concatenate results
    output = torch.cat(output_chunks, dim=0)
    
    # Handle shared expert if provided
    shared_output = None
    if shared_inputs is not None and shared_weight is not None:
        shared_output = torch.matmul(shared_inputs, shared_weight)
    
    if shared_inputs is not None:
        return output, shared_output
    return output


@register_op('grouped_matmul_all2all', 'cpu')
def grouped_matmul_all2all_cpu(inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None):
    """
    CPU implementation of grouped_matmul_all2all.
    
    This performs grouped matrix multiplication followed by all-to-all communication.
    For CPU, we simulate the operation using standard distributed operations.
    
    Args:
        inputs: Input tensor
        weights: Weight tensor
        group: Process group for distributed communication
        send_counts: Counts of elements to send to each rank
        recv_counts: Counts of elements to receive from each rank
        shared_inputs: Optional shared expert inputs
        shared_weight: Optional shared expert weight
    
    Returns:
        Output tensor(s) - expert output and optionally shared expert output
    """
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    
    # Convert send_counts and recv_counts to lists if they are tensors
    if isinstance(send_counts, torch.Tensor):
        send_counts_list = send_counts.tolist()
        group_sizes = send_counts.reshape(world_size, -1).sum(dim=0)
    else:
        send_counts_list = send_counts
        group_sizes = torch.tensor(send_counts_list).reshape(world_size, -1).sum(dim=0)
        
    if isinstance(recv_counts, torch.Tensor):
        recv_counts_list = recv_counts.tolist()
    else:
        recv_counts_list = recv_counts
    
    # Perform grouped matrix multiplication first
    output_chunks = []
    start_idx = 0
    for i, size in enumerate(group_sizes.tolist()):
        end_idx = start_idx + size
        chunk = inputs[start_idx:end_idx]
        # Perform matrix multiplication with corresponding weight
        output_chunk = torch.matmul(chunk, weights[i])
        output_chunks.append(output_chunk)
        start_idx = end_idx
    
    # Concatenate results
    gmm_output = torch.cat(output_chunks, dim=0)
    
    # Perform all-to-all communication
    send_tensor = gmm_output
    recv_tensor = torch.empty_like(send_tensor)
    
    # Perform all-to-all
    torch.distributed.all_to_all_single(recv_tensor, send_tensor, recv_counts_list, send_counts_list, group=group)
    
    output = recv_tensor
    
    # Handle shared expert if provided
    shared_output = None
    if shared_inputs is not None and shared_weight is not None:
        shared_output = torch.matmul(shared_inputs, shared_weight)
    
    if shared_inputs is not None:
        return output, shared_output
    return output
