# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import sys
import types

import torch
import torch_npu


def accelerator_getattr(module, fallback_module):
    def __getattr__(name):
        if hasattr(fallback_module, name):
            attr = getattr(fallback_module, name)
            setattr(module, name, attr)
            return attr
        else:
            raise AttributeError(f'module {module} and {fallback_module} has no attribute {name}.')

    return __getattr__


def set_accelerator_compatible():
    """Set up ``torch.accelerator`` to delegate to the detected device backend.

    Automatically detects the available accelerator in the following order:
    1. NPU (Huawei Ascend) — if ``torch_npu`` is importable
    2. CUDA (NVIDIA GPU) — if ``torch.cuda.is_available()``

    Returns:
        str: The distributed backend name (``"hccl"`` for NPU, ``"nccl"`` for CUDA).

    Raises:
        RuntimeError: If no supported accelerator is found.
    """
    try:
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
        fallback_module = torch.npu
        backend = "hccl"
    except ImportError:
        if torch.cuda.is_available():
            fallback_module = torch.cuda
            backend = "nccl"
        else:
            raise RuntimeError("No available accelerator (NPU/CUDA).")

    accelerator_module = types.ModuleType('torch.accelerator')
    accelerator_module.__doc__ = f'Fallback accelerator module that delegates to {fallback_module}'
    for attr in dir(torch.accelerator):
        if attr.startswith('__'):
            continue
        setattr(accelerator_module, attr, getattr(torch.accelerator, attr))

    accelerator_module.__getattr__ = accelerator_getattr(accelerator_module, fallback_module)
    torch.accelerator = accelerator_module
    sys.modules['torch.accelerator'] = accelerator_module

    return backend
