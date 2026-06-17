# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from fsdp_turbo.ops import dispatch_op


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6, use_eager=False):
    """
    HC-Split Sinkhorn algorithm for MoE routing with automatic device dispatch.

    Args:
        mixes: Input tensor [batch_size, seq_len, (2+hc_mult)*hc_mult]
        hc_scale: Scale parameters [3] (pre, post, comb)
        hc_base: Base parameters [(2+hc_mult)*hc_mult]
        hc_mult: HC multiplier dimension (default=4)
        sinkhorn_iters: Sinkhorn normalization iterations (default=20)
        eps: Epsilon for numerical stability (default=1e-6)
        use_eager: If False, use fused implementation for current device.
                   If True, use CPU implementation.

    Returns:
        tuple: (pre, post, comb)
            - pre: [batch_size, seq_len, hc_mult]
            - post: [batch_size, seq_len, hc_mult]
            - comb: [batch_size, seq_len, hc_mult, hc_mult]
    """
    device_type = 'cpu' if use_eager else None
    return dispatch_op('hc_split_sinkhorn', mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps,
                       device_type=device_type)
