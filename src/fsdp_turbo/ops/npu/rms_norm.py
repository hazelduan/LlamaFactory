# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


@register_op('rms_norm', 'npu')
def rms_norm_npu(x, weight, eps):
    return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]
