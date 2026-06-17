# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext


def copy_fsdp_comm_ctx(new_comm_ctx: FSDPCommContext, comm_ctx: FSDPCommContext) -> FSDPCommContext:
    """
    Copies critical stream and state attributes from one communication context to another.
    Used to initialize additional global communication contexts based on the root context.
    """

    new_comm_ctx.device_handle = comm_ctx.device_handle

    # Copy streams
    new_comm_ctx.all_gather_copy_in_stream = comm_ctx.all_gather_copy_in_stream
    new_comm_ctx.all_gather_stream = comm_ctx.all_gather_stream
    new_comm_ctx.reduce_scatter_stream = comm_ctx.reduce_scatter_stream
    new_comm_ctx.all_reduce_stream = comm_ctx.all_reduce_stream

    # Copy state placeholders
    new_comm_ctx.all_gather_state = comm_ctx.all_gather_state
    new_comm_ctx.reduce_scatter_state = comm_ctx.reduce_scatter_state
    new_comm_ctx.post_forward_order = comm_ctx.post_forward_order

    return new_comm_ctx
