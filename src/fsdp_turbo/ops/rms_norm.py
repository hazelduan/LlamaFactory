# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from fsdp_turbo.ops import dispatch_op


def rms_norm(x, weight, eps, use_eager=False):
    """
    Root Mean Square Layer Normalization with automatic device dispatch.

    Args:
        x: Input tensor
        weight: Scale parameter
        eps: Epsilon for numerical stability
        use_eager: If False, use fused (optimized) implementation for current device.
                   If True, use CPU implementation.

    Returns:
        Normalized output tensor
    """
    device_type = 'cpu' if use_eager else None
    return dispatch_op('rms_norm', x, weight, eps, device_type=device_type)
