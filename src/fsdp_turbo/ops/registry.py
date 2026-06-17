import logging

import torch

from fsdp_turbo.utils.log import log_warning_once

_OPERATOR_REGISTRY = {}
logger = logging.getLogger(__name__)


def register_op(op_name: str, device_type: str):
    """
    Decorator for registering operator implementations for specific device types.

    Args:
        op_name: Name of the operator (e.g., 'grouped_matmul')
        device_type: Device type (e.g., 'cpu', 'npu', 'cuda')

    Returns:
        Decorator function that registers the operator implementation

    Example:
        @register_op('grouped_matmul', 'cpu')
        def grouped_matmul_cpu(inputs, m_split, weights):
            # CPU implementation
            pass
    """

    def decorator(func):
        if op_name not in _OPERATOR_REGISTRY:
            _OPERATOR_REGISTRY[op_name] = {}

        _OPERATOR_REGISTRY[op_name][device_type] = func

        return func

    return decorator


def get_op(op_name: str, device_type: str = None):
    """
    Get operator implementation for a specific device type.

    Args:
        op_name: Name of the operator
        device_type: Device type. If None, uses current accelerator type.

    Returns:
        Operator implementation function

    Raises:
        KeyError: If operator not found for the device type
    """
    if device_type is None:
        device_type = torch.accelerator.current_accelerator().type

    if op_name not in _OPERATOR_REGISTRY:
        raise KeyError(f"Operator '{op_name}' is not registered")

    if device_type in _OPERATOR_REGISTRY[op_name]:
        return _OPERATOR_REGISTRY[op_name][device_type]

    if 'cpu' in _OPERATOR_REGISTRY[op_name]:
        log_warning_once(
            logger,
            f"Operator '{op_name}' not available for device '{device_type}', "
            f"falling back to CPU implementation"
        )
        return _OPERATOR_REGISTRY[op_name]['cpu']

    raise KeyError(
        f"Operator '{op_name}' not available for device '{device_type}' "
        f"and no CPU fallback exists"
    )


def dispatch_op(op_name: str, *args, device_type: str = None, **kwargs):
    """
    Dispatch operator call to the appropriate implementation based on device type.

    Args:
        op_name: Name of the operator
        device_type: Device type. If None, uses current accelerator type.
        *args, **kwargs: Arguments to pass to the operator

    Returns:
        Result of the operator call
    """
    op_func = get_op(op_name, device_type)
    return op_func(*args, **kwargs)


def get_device_ops_module():
    """Get the appropriate ops module for the current device."""
    device_type = torch.accelerator.current_accelerator().type
    try:
        if device_type == 'npu':
            import fsdp_turbo.ops.npu
            return fsdp_turbo.ops.npu
        elif device_type == 'cuda':
            import fsdp_turbo.ops.cuda
            return fsdp_turbo.ops.cuda
        else:
            raise ImportError
    except ImportError:
        log_warning_once(logger, f'[] Could not import fsdp_turbo.ops.{device_type} fallback to cpu ops!')
        return None
