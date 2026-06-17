# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from fsdp_turbo.ops.registry import register_op


@register_op('grouped_matmul', 'cpu')
def grouped_matmul_cpu(inputs, m_split, weights):
    """
    Grouped matrix multiplication that handles two weight tensor formats.

    Args:
        inputs: Tensor of shape [batch_size, input_dim]
        m_split: Tensor of group sizes that sum to batch_size
        weights: Weight tensor of either:
                 Format 1: [num_groups, input_dim, output_dim] - ready for matmul
                 Format 2: [num_groups, output_dim, input_dim] - needs transpose

    Returns:
        Tensor of shape [batch_size, output_dim]
    """
    batch_size, input_dim = inputs.shape

    # Automatically detect and adjust weight format
    # Check if second dimension matches input dimension (Format 1)
    if weights.shape[1] == input_dim:
        # Format 1: [num_groups, input_dim, output_dim]
        output_dim = weights.shape[2]
        # No transformation needed - weights are already in correct format
    else:
        # Format 2: [num_groups, output_dim, input_dim]
        # Transpose to convert to Format 1: [num_groups, input_dim, output_dim]
        output_dim = weights.shape[1]
        weights = weights.transpose(1, 2)

    # Initialize output tensor
    output_shape = (batch_size, output_dim)
    final_hidden_states = torch.zeros(output_shape, dtype=inputs.dtype, device=inputs.device)

    # Calculate group boundaries from cumulative sum

    group_list = [0] + torch.cumsum(m_split, dim=0).tolist()

    # Process each group separately
    for i in range(len(group_list) - 1):
        start_idx = group_list[i]
        end_idx = group_list[i + 1]

        # Matrix multiplication for current group
        # inputs[start_idx:end_idx, :] has shape [group_size, input_dim]
        # weights[i] has shape [input_dim, output_dim] (after format normalization)
        final_hidden_states[start_idx:end_idx, :] = torch.matmul(
            inputs[start_idx:end_idx, :],
            weights[i]
        )

    return final_hidden_states
