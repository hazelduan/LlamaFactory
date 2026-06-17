# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Any
import torch

IS_NPU_AVAILABLE = False
IS_CUDA_AVAILABLE = torch.cuda.is_available()
try:
    if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
        IS_NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    pass


def get_device_type() -> str:
    """Get device type based on current machine, currently only support CPU, CUDA, NPU."""
    if IS_NPU_AVAILABLE:
        device = "npu"
    elif IS_CUDA_AVAILABLE:
        device = "cuda"
    else:
        device = "cpu"

    return device


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda or torch.npu"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        return torch.cuda
