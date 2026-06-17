import torch
from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


class GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weights, weights_bias, m_split, group_list_type) -> torch.Tensor:
        # Due to ascend gmm kernel k split limitations, we need a tensor m_split, not a tensor List.
        if not isinstance(m_split, torch.Tensor):
            ctx.group_list = torch.tensor(m_split, device='npu', dtype=torch.int64)
        else:
            ctx.group_list = m_split

        ctx.group_list_type = group_list_type

        # Get weight chunks
        weight_chunks = [w[0] for w in weights.chunk(weights.shape[0], dim=0)]

        # Determine if weight matrices need to be transposed
        # Matrix multiplication requires: last dimension of input_tensor must equal first dimension of weight matrix
        # input_tensor.shape = [M, K], weight matrix should be [K, N] for M@K * K@N = M@N
        weight_shape = weight_chunks[0].shape
        input_last_dim = input_tensor.shape[-1]

        if weight_shape[0] == input_last_dim:
            # Weight matrix shape is [K, N], compatible with input matrix [M, K], no transpose needed
            ctx.needs_transpose = False
            weights_for_matmul = weight_chunks
        else:
            # Assume weight matrix needs to be transposed
            ctx.needs_transpose = True
            weights_for_matmul = [w.T for w in weight_chunks]

        ctx.save_for_backward(input_tensor, weights)

        fwd_output = torch_npu.npu_grouped_matmul([input_tensor], weights_for_matmul, bias=weights_bias,
                                                  group_list=ctx.group_list, split_item=2, group_type=0,
                                                  group_list_type=ctx.group_list_type)[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        group_list = ctx.group_list
        inp, weights = ctx.saved_tensors
        group_list_type = ctx.group_list_type
        needs_transpose = ctx.needs_transpose

        # Get weight chunks
        weight_chunks = [w[0] for w in weights.chunk(weights.shape[0], dim=0)]

        # Calculate input gradient: grad_output @ weight^T or grad_output @ weight
        # If weights were transposed in forward (needs_transpose=True), use original weights
        # If weights were not transposed in forward (needs_transpose=False), transpose weights
        if needs_transpose:
            weights_for_grad = weight_chunks
        else:
            weights_for_grad = [w.T for w in weight_chunks]

        # Calculate input gradient
        grad = torch_npu.npu_grouped_matmul([grad_output], weights_for_grad, bias=None,
                                            group_list=group_list, split_item=2, group_type=0,
                                            group_list_type=group_list_type)[0]

        # Calculate weight gradient (K split gmm): grad_weight = inp^T @ grad_output
        grad_weight = torch_npu.npu_grouped_matmul([inp.T], [grad_output], bias=None,
                                                   group_list=group_list, split_item=3,
                                                   group_type=2, group_list_type=group_list_type)[0]

        # Adjust weight gradient orientation
        # When forward uses transposed weights, transpose the computed gradient back
        grad_weight_chunks = [w.T if needs_transpose else w for w in grad_weight]

        return grad, torch.stack(grad_weight_chunks), None, None, None


@register_op('grouped_matmul', 'npu')
def grouped_matmul_npu(inputs, m_split, weights):
    return GroupedMatmul.apply(inputs, weights, None, m_split, 1)
