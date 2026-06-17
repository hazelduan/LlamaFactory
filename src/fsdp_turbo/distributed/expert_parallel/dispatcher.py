# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from torch.distributed.tensor import DTensor
from fsdp_turbo.distributed.expert_parallel.utils import normalize_expert_args
from fsdp_turbo.distributed.dist_ops import all_to_all, gather_along_first_dim_expert_parallel
from fsdp_turbo.ops.moe import grouped_matmul, permute, unpermute


def get_experts_forward_fn(ep_group, fused):
    def experts_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        # Ensure correct parameter order and dimensions
        top_k_index, top_k_weights = normalize_expert_args(top_k_index, top_k_weights)
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        gate_up_proj = (self.gate_up_proj.to_local()
                        if isinstance(self.gate_up_proj, DTensor)
                        else self.gate_up_proj)
        down_proj = (self.down_proj.to_local()
                     if isinstance(self.down_proj, DTensor)
                     else self.down_proj)

        weights = (gate_up_proj, down_proj)
        act_fn = self.act_fn
        num_global_experts = self.num_global_experts
        expert_ids_per_ep_rank = self.expert_ids_per_ep_rank
        act_limit = self.limit if hasattr(self, 'limit') else None

        hidden_states = dispatch_mlp_combine(ep_group, fused, hidden_states, top_k_index, top_k_weights, weights, act_fn,
                                             num_global_experts, expert_ids_per_ep_rank, act_limit)

        return hidden_states.view(*hidden_states_shape)
    return experts_forward


def dispatch_mlp_combine(ep_group, fused, hidden_states, top_k_index, top_k_weights, weights, act_fn,
                         num_global_experts, expert_ids_per_ep_rank, act_limit):
    gate_up_weights, down_weights = weights
    # MoE preprocess to get local/global indices and AllToAll split sizes
    permute_indices, split_sizes = dispatch_preprocess(ep_group, top_k_index, num_global_experts,
                                                       expert_ids_per_ep_rank)
    if fused and (permute_indices[0] == 0).any():
        fused = False
    # AllToAll dispatch --> MLP computation --> AllToAll combine
    hidden_states, unpermute_indices = alltoall_dispatch(ep_group, hidden_states, top_k_index, permute_indices,
                                                         split_sizes, fused)
    hidden_states = experts_computation(hidden_states, permute_indices[0], gate_up_weights, down_weights,
                                        act_fn, act_limit, fused)
    hidden_states = alltoall_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes, fused)
    return hidden_states


def dispatch_preprocess(ep_group, top_k_index, num_global_experts, expert_ids_per_ep_rank):
    ep_size = torch.distributed.get_world_size(ep_group)
    ep_rank = torch.distributed.get_rank(ep_group)
    num_local_experts = num_global_experts // ep_size
    local_experts_start_id = num_local_experts * ep_rank
    local_experts_end_id = local_experts_start_id + num_local_experts

    # [B*S, K] --> [E]
    num_local_tokens_per_expert = torch.bincount(top_k_index.view(-1), minlength=num_global_experts)
    # [E] --> [EP*E]
    num_global_tokens_per_expert, _ = gather_along_first_dim_expert_parallel(num_local_tokens_per_expert, ep_group)
    # [EP*E] --> [EP, local_E]
    num_global_tokens_per_local_expert = num_global_tokens_per_expert.reshape(ep_size, num_global_experts)[:,
                                         local_experts_start_id: local_experts_end_id]
    # [EP, local_E] --> [local_E]
    num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
    # [E] --> [EP, local_E] --> [EP]
    input_split = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(axis=1).to(torch.device("cpu"),
                                                                                                 non_blocking=True)
    # [EP, local_E] --> [EP]
    output_splits = num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True)
    # [EP, local_E] --> [E*select]
    global_indices = torch.repeat_interleave(expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
    return (num_tokens_per_local_expert, global_indices), (input_split, output_splits)


def alltoall_dispatch(ep_group, hidden_states, top_k_index, indices, split_sizes, fused):
    local_indices, global_indices = indices
    input_split, output_splits = split_sizes

    hidden_states, unpermute_indices1 = permute(hidden_states, top_k_index, use_eager=not fused)
    torch.npu.current_stream().synchronize()
    hidden_states = all_to_all(ep_group, hidden_states, output_splits.numpy(), input_split.numpy())
    hidden_states, unpermute_indices2 = permute(hidden_states, global_indices, use_eager=not fused)

    return hidden_states, (unpermute_indices1, unpermute_indices2)


def experts_computation(hidden_states, split_list, gate_up_weights, down_weights, act_fn, act_limit, fused):
    gate, up = grouped_matmul(hidden_states, split_list, gate_up_weights, use_eager=not fused).chunk(2, dim=-1)
    if act_limit is not None:
        gate = gate.clamp(max=act_limit)
        up = up.clamp(min=-act_limit, max=act_limit)
    act = act_fn(gate) * up
    hidden_states = grouped_matmul(act, split_list, down_weights, use_eager=not fused)
    return hidden_states


def alltoall_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes, fused):
    unpermute_indices1, unpermute_indices2 = unpermute_indices
    input_split, output_splits = split_sizes
    hidden_states = unpermute(hidden_states, unpermute_indices2, use_eager=not fused)
    hidden_states = all_to_all(ep_group, hidden_states, input_split.numpy(), output_splits.numpy())
    hidden_states = unpermute(hidden_states, unpermute_indices1, top_k_weights, use_eager=not fused)
    return hidden_states
