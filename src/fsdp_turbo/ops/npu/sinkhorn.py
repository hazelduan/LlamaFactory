# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.ops.triton.sinkhorn_kernel import hc_split_sinkhorn as _hc_split_sinkhorn_kernel
from fsdp_turbo.ops.triton.sinkhorn_kernel import hc_split_sinkhorn_backward as _hc_split_sinkhorn_backward_kernel
from mindspeed.lite.ops.triton.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class HcSplitSinkhornFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6
    ):
        pre, post, comb = _hc_split_sinkhorn_kernel(
            mixes,
            hc_scale,
            hc_base,
            hc_mult,
            sinkhorn_iters,
            eps
        )

        ctx.save_for_backward(mixes, hc_scale, hc_base)
        ctx.hc_mult = hc_mult
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.eps = eps

        return pre, post, comb

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        grad_pre: torch.Tensor,
        grad_post: torch.Tensor,
        grad_comb: torch.Tensor,
    ):
        mixes, hc_scale, hc_base = ctx.saved_tensors
        hc_mult = ctx.hc_mult
        sinkhorn_iters = ctx.sinkhorn_iters
        eps = ctx.eps

        grad_mixes, grad_scale, grad_base = _hc_split_sinkhorn_backward_kernel(
            grad_pre, grad_post, grad_comb,
            mixes, hc_scale, hc_base,
            hc_mult, sinkhorn_iters, eps
        )

        return grad_mixes, grad_scale, grad_base, None, None, None


@torch.compiler.disable
def _hc_split_sinkhorn_triton(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre, post, comb = HcSplitSinkhornFunction.apply(
        mixes,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_iters,
        eps
    )
    return pre, post, comb


@register_op('hc_split_sinkhorn', 'npu')
def hc_split_sinkhorn_npu(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    return _hc_split_sinkhorn_triton(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
