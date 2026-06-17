# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch_npu
import triton
import triton.language as tl
import torch.nn.functional as F
import triton.language.extra.cann.extension as extension


@triton.jit
def _hc_split_sinkhorn_kernel_part1(
    mixes_ptr, hc_scale_ptr, hc_base_ptr,
    pre_ptr, post_ptr, comb_ptr,
    batch_seq_size,
    eps: tl.constexpr,
    feat_dim: tl.constexpr,
    hc_mult: tl.constexpr,
    group: tl.constexpr,
):
    ar4 = tl.arange(0, hc_mult)
    arange_val = tl.arange(0, hc_mult * hc_mult)

    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)
    pid_mask = pids < batch_seq_size

    pid_comb_off = pids[:, None] * hc_mult * hc_mult
    pid_feat_off = pids[:, None] * feat_dim
    pid_hc_off = pids[:, None] * hc_mult

    scale_pre = tl.load(hc_scale_ptr + 0)
    scale_post = tl.load(hc_scale_ptr + 1)
    scale_comb = tl.load(hc_scale_ptr + 2)

    base_pre = tl.load(hc_base_ptr + ar4)
    base_post = tl.load(hc_base_ptr + hc_mult + ar4)
    base_comb = tl.load(hc_base_ptr + 2 * hc_mult + arange_val)

    mixes_pre = tl.load(
        mixes_ptr + pid_feat_off + ar4[None, :],
        mask=pid_mask[:, None],
        other=0.0
    )
    mixes_post = tl.load(
        mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :],
        mask=pid_mask[:, None],
        other=0.0
    )
    mixes_comb = tl.load(
        mixes_ptr + pid_feat_off[:, :, None] + (2 * hc_mult + arange_val)[None, :],
        mask=pid_mask[:, None, None]
    )

    pre = tl.sigmoid(mixes_pre * scale_pre + base_pre[None, :]) + eps
    tl.store(
        pre_ptr + pid_hc_off + ar4[None, :],
        pre,
        mask=pid_mask[:, None]
    )

    post = 2.0 * tl.sigmoid(mixes_post * scale_post + base_post[None, :])
    tl.store(
        post_ptr + pid_hc_off + ar4[None, :],
        post,
        mask=pid_mask[:, None]
    )

    comb = mixes_comb * scale_comb + base_comb[None, :, :]
    comb_flat = tl.reshape(comb, (group, hc_mult * hc_mult))
    tl.store(
        comb_ptr + pid_comb_off + arange_val[None, :],
        comb_flat,
        mask=pid_mask[:, None]
    )


@triton.jit
def _hc_split_sinkhorn_kernel_part2(
    comb_tmp_ptr,
    comb_ptr,
    batch_seq_size,
    hc_mult: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    eps: tl.constexpr,
    group: tl.constexpr,
    BLOCK_ALIGN: tl.constexpr = 8
):
    lin = tl.arange(0, hc_mult * BLOCK_ALIGN)

    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)
    pid_mask = pids < batch_seq_size

    pid_comb_off = pids[:, None] * (hc_mult * BLOCK_ALIGN)

    comb = tl.load(
        comb_tmp_ptr + pid_comb_off + lin[None, :],
        mask=pid_mask[:, None]
    )
    comb = comb.reshape(group, hc_mult, BLOCK_ALIGN)

    row_max = tl.max(comb, axis=2)
    comb = tl.exp(comb - row_max[:, :, None])

    for _ in range(sinkhorn_iters):
        row_sum = tl.sum(comb, axis=2)
        comb = comb / (row_sum[:, :, None] + eps)

        col_sum = tl.sum(comb, axis=1)
        comb = comb / (col_sum[:, None, :] + eps)

    comb_flat = tl.reshape(comb, (group, hc_mult * BLOCK_ALIGN))
    tl.store(
        comb_ptr + pid_comb_off + lin[None, :],
        comb_flat,
        mask=pid_mask[:, None]
    )


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    origin_dtype = mixes.dtype
    mixes = mixes.to(dtype=torch.float32)
    hc_scale = hc_scale.to(dtype=torch.float32)
    hc_base = hc_base.to(dtype=torch.float32)

    b, s, _ = mixes.shape
    feat_dim = (2 + hc_mult) * hc_mult
    batch_seq_size = b * s
    mixes_flat = mixes.view(-1, feat_dim).contiguous()

    pre_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)
    post_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)
    comb_tmp = torch.empty((batch_seq_size, hc_mult, hc_mult), dtype=mixes.dtype, device=mixes.device)

    BLOCK_ALIGN = 8
    group_part1 = 64
    group_part2 = 32

    _hc_split_sinkhorn_kernel_part1[(triton.cdiv(batch_seq_size, group_part1),)](
        mixes_flat, hc_scale, hc_base,
        pre_flat, post_flat, comb_tmp,
        batch_seq_size,
        eps, feat_dim, hc_mult,
        group_part1
    )

    comb_tmp_padded = F.pad(comb_tmp, pad=(0, BLOCK_ALIGN - hc_mult), mode="constant", value=float('-inf'))
    comb_flat_padded = torch.empty((batch_seq_size, hc_mult * BLOCK_ALIGN), dtype=mixes.dtype, device=mixes.device)

    _hc_split_sinkhorn_kernel_part2[(triton.cdiv(batch_seq_size, group_part2),)](
        comb_tmp_padded,
        comb_flat_padded,
        batch_seq_size, hc_mult, sinkhorn_iters,
        eps, group_part2,
        BLOCK_ALIGN=BLOCK_ALIGN,
    )

    pre = pre_flat.view(b, s, hc_mult).to(dtype=origin_dtype)
    post = post_flat.view(b, s, hc_mult).to(dtype=origin_dtype)
    comb = comb_flat_padded.view(b, s, hc_mult, BLOCK_ALIGN)[:, :, :, :hc_mult].to(dtype=origin_dtype)

    return pre, post, comb


@triton.jit
def hc_split_sinkhorn_backward_kernel_part1(
    grad_pre_ptr,
    grad_post_ptr,
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    comb_tmp_ptr,
    grad_mixes_ptr,
    grad_hc_scale_ptr,
    grad_hc_base_ptr,
    batch_seq_size,
    hc_mult: tl.constexpr = 4,
    group: tl.constexpr = 32,
):
    feat_dim = (2 + hc_mult) * hc_mult
    arange_val = tl.arange(0, hc_mult * hc_mult)

    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)
    pid_mask = pids < batch_seq_size

    pid_comb_off = pids[:, None] * hc_mult * hc_mult
    pid_feat_off = pids[:, None] * feat_dim
    pid_hc_off = pids[:, None] * hc_mult
    ar4 = tl.arange(0, hc_mult)

    scale_pre = tl.load(hc_scale_ptr + 0)
    scale_post = tl.load(hc_scale_ptr + 1)
    scale_comb = tl.load(hc_scale_ptr + 2)

    pre_slice = tl.load(mixes_ptr + pid_feat_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
    post_slice = tl.load(mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :], mask=pid_mask[:, None], other=0.0)
    comb_slice = tl.load(mixes_ptr + pid_feat_off + (2 * hc_mult + arange_val)[None, :], mask=pid_mask[:, None], other=0.0)

    base_pre = tl.load(hc_base_ptr + ar4)
    base_post = tl.load(hc_base_ptr + hc_mult + ar4)
    base_comb = tl.load(hc_base_ptr + 2 * hc_mult + arange_val)

    pre_input = pre_slice * scale_pre + base_pre[None, :]
    sigmoid_pre = tl.sigmoid(pre_input)
    sigmoid_deriv = sigmoid_pre * (1.0 - sigmoid_pre)
    grad_pre = tl.load(grad_pre_ptr + pid_hc_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
    grad_pre_input = grad_pre * sigmoid_deriv

    tl.store(grad_mixes_ptr + pid_feat_off + ar4[None, :], grad_pre_input * scale_pre, mask=pid_mask[:, None])

    tl.atomic_add(grad_hc_scale_ptr + 0, tl.sum(grad_pre_input * pre_slice))
    grad_pre_input_sum = tl.sum(grad_pre_input, axis=0)
    tl.atomic_add(grad_hc_base_ptr + ar4, grad_pre_input_sum)

    post_input = post_slice * scale_post + base_post[None, :]
    sigmoid_post = tl.sigmoid(post_input)
    sigmoid_deriv_post = sigmoid_post * (1.0 - sigmoid_post)
    grad_post = tl.load(grad_post_ptr + pid_hc_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
    grad_post_input = grad_post * 2.0 * sigmoid_deriv_post

    tl.store(
        grad_mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :],
        grad_post_input * scale_post,
        mask=pid_mask[:, None]
    )

    tl.atomic_add(grad_hc_scale_ptr + 1, tl.sum(grad_post_input * post_slice))
    grad_post_input_sum = tl.sum(grad_post_input, axis=0)
    tl.atomic_add(grad_hc_base_ptr + hc_mult + ar4, grad_post_input_sum)

    comb = comb_slice * scale_comb + base_comb[None, :, :]
    comb_flat = tl.reshape(comb, (group, hc_mult * hc_mult))
    tl.store(comb_tmp_ptr + pid_comb_off + arange_val[None, :], comb_flat, mask=pid_mask[:, None])


@triton.jit
def hc_split_sinkhorn_backward_kernel_part2(
    grad_comb_ptr,
    mixes_ptr,
    hc_scale_ptr,
    comb_tmp_ptr,
    grad_mixes_ptr,
    grad_hc_scale_ptr,
    grad_hc_base_ptr,
    batch_seq_size,
    hc_mult: tl.constexpr = 4,
    sinkhorn_iters: tl.constexpr = 20,
    eps: tl.constexpr = 1e-6,
    BLOCK_ALIGN: tl.constexpr = 8,
    group: tl.constexpr = 32,
):
    arange_val = tl.arange(0, hc_mult * BLOCK_ALIGN)
    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)
    pid_mask = pids < batch_seq_size

    c = tl.arange(0, BLOCK_ALIGN)[None, :]
    col_mask = c < hc_mult
    mask_val = col_mask[None, :, :]
    pid_feat_off = pids[:, None] * hc_mult * BLOCK_ALIGN

    comb_slice_flat = tl.load(mixes_ptr + pid_feat_off + arange_val)
    comb_slice = comb_slice_flat.reshape(group, hc_mult, BLOCK_ALIGN)

    scale_comb = tl.load(hc_scale_ptr + 2)

    comb_init = tl.load(comb_tmp_ptr + pid_feat_off + arange_val)
    comb_init = comb_init.reshape(group, hc_mult, BLOCK_ALIGN)

    row_max = tl.max(comb_init, axis=2).reshape(group, hc_mult, 1)
    exp_comb = tl.exp(comb_init - row_max)

    row_sum_list = tl.full((sinkhorn_iters, group, hc_mult, 1), 0.0, dtype=tl.float32)
    col_sum_list = tl.full((sinkhorn_iters, group, 1, BLOCK_ALIGN), 0.0, dtype=tl.float32)
    K = exp_comb

    for i in range(sinkhorn_iters):
        row_sum = tl.sum(K, axis=2).reshape(group, hc_mult, 1)
        K_row = K / (row_sum + eps)

        col_sum = tl.sum(K_row, axis=1).reshape(group, 1, BLOCK_ALIGN)
        K_col = K_row / (col_sum + eps)

        row_sum_list = extension.insert_slice(
            ful=row_sum_list,
            sub=row_sum[None, :, :, :],
            offsets=[i, 0, 0, 0],
            sizes=[1, group, hc_mult, 1],
            strides=[1, 1, 1, 1],
        )
        col_sum_list = extension.insert_slice(
            ful=col_sum_list,
            sub=col_sum[None, :, :, :],
            offsets=[i, 0, 0, 0],
            sizes=[1, group, 1, BLOCK_ALIGN],
            strides=[1, 1, 1, 1],
        )
        K = K_col

    grad_comb_flat = tl.load(grad_comb_ptr + pid_feat_off + arange_val)
    dK = grad_comb_flat.reshape(group, hc_mult, BLOCK_ALIGN)

    for j in range(sinkhorn_iters):
        i = sinkhorn_iters - j - 1

        row_sum = extension.extract_slice(
            row_sum_list,
            [i, 0, 0, 0],
            [1, group, hc_mult, 1],
            [1, 1, 1, 1],
        )
        col_sum = extension.extract_slice(
            col_sum_list,
            [i, 0, 0, 0],
            [1, group, 1, BLOCK_ALIGN],
            [1, 1, 1, 1],
        )

        col_sum = col_sum.reshape(group, 1, BLOCK_ALIGN) + eps
        row_sum = row_sum.reshape(group, hc_mult, 1) + eps
        K_col = K * col_sum

        grad_direct = dK / col_sum
        d_col_sum_compressed = -tl.sum(dK * K_col / (col_sum * col_sum), axis=-2)
        dK_row = grad_direct + d_col_sum_compressed[:, None, :]

        K_row = K_col * row_sum
        K = K_row

        grad_direct_row = dK_row / row_sum
        d_row_sum_compressed = -tl.sum(dK_row * K_row / (row_sum * row_sum), axis=-1)
        dK = grad_direct_row + d_row_sum_compressed[:, :, None]

        dK = dK * mask_val

    d_exp_comb = dK
    d_comb_before_exp = d_exp_comb * exp_comb

    max_mask = tl.where(comb_init == row_max, 1.0, 0.0)
    max_count = tl.sum(max_mask, axis=-1).reshape(group, hc_mult, 1) + eps
    row_sum_d_before_exp = tl.sum(d_comb_before_exp, axis=-1).reshape(group, hc_mult, 1)
    d_comb_init = d_comb_before_exp - (row_sum_d_before_exp * max_mask / max_count)

    grad_comb_slice_flat = d_comb_init * scale_comb

    tl.store(
        grad_mixes_ptr + pid_feat_off + arange_val[None, :],
        grad_comb_slice_flat.reshape(group, hc_mult * BLOCK_ALIGN),
        mask=pid_mask[:, None]
    )

    tmp_res = d_comb_init * comb_slice
    tmp_res = tl.where(pid_mask[:, None, None], tmp_res, 0.0)
    d_comb_init = tl.where(pid_mask[:, None, None], d_comb_init, 0.0)

    tl.atomic_add(grad_hc_scale_ptr + 2, tl.sum(tmp_res))
    d_comb_init_sum = tl.sum(d_comb_init, axis=0)
    tl.atomic_add(grad_hc_base_ptr + arange_val, d_comb_init_sum.reshape(hc_mult * BLOCK_ALIGN))


def hc_split_sinkhorn_backward(
    grad_pre: torch.Tensor,
    grad_post: torch.Tensor,
    grad_comb: torch.Tensor,
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, _ = mixes.shape
    batch_seq_size = b * s

    origin_dtype = mixes.dtype
    mixes = mixes.to(dtype=torch.float32)
    hc_scale = hc_scale.to(dtype=torch.float32)
    hc_base = hc_base.to(dtype=torch.float32)
    grad_pre = grad_pre.to(dtype=torch.float32)
    grad_post = grad_post.to(dtype=torch.float32)
    grad_comb = grad_comb.to(dtype=torch.float32)

    grad_mixes = torch.zeros_like(mixes, device=mixes.device)
    grad_hc_scale = torch.zeros_like(hc_scale, device=hc_scale.device)
    grad_hc_base = torch.zeros_like(hc_base, device=hc_base.device)
    comb_tmp = torch.empty((batch_seq_size, hc_mult, hc_mult), dtype=mixes.dtype, device=mixes.device)

    grad_pre_flat = grad_pre.reshape(-1, hc_mult)
    grad_post_flat = grad_post.reshape(-1, hc_mult)

    BLOCK_ALIGN = 8
    group_part1 = 64
    group_part2 = 32

    hc_split_sinkhorn_backward_kernel_part1[(triton.cdiv(batch_seq_size, group_part1),)](
        grad_pre_flat,
        grad_post_flat,
        mixes,
        hc_scale,
        hc_base,
        comb_tmp,
        grad_mixes,
        grad_hc_scale,
        grad_hc_base,
        batch_seq_size,
        hc_mult=hc_mult,
        group=group_part1
    )

    mixes_flat = mixes.view(-1, (2 + hc_mult) * hc_mult)
    mixes_slice = mixes_flat[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult)
    mixes_pad = F.pad(mixes_slice, (0, BLOCK_ALIGN - hc_mult), mode="constant", value=0.0)

    grad_mixes_pad = torch.zeros(
        (batch_seq_size, hc_mult, BLOCK_ALIGN),
        dtype=grad_mixes.dtype,
        device=grad_mixes.device,
    )
    grad_hc_base_pad = torch.zeros(
        (hc_mult, BLOCK_ALIGN), dtype=grad_hc_base.dtype, device=grad_hc_base.device
    )

    grad_comb_flat = grad_comb.reshape(-1, hc_mult, hc_mult)
    grad_comb_flat_pad = F.pad(
        grad_comb_flat, (0, BLOCK_ALIGN - hc_mult), mode="constant", value=0.0
    )
    comb_tmp_padded = F.pad(comb_tmp, pad=(0, BLOCK_ALIGN - hc_mult), mode="constant", value=float('-inf'))

    hc_split_sinkhorn_backward_kernel_part2[(triton.cdiv(batch_seq_size, group_part2),)](
        grad_comb_flat_pad,
        mixes_pad,
        hc_scale,
        comb_tmp_padded,
        grad_mixes_pad,
        grad_hc_scale,
        grad_hc_base_pad,
        batch_seq_size,
        hc_mult,
        sinkhorn_iters,
        eps,
        BLOCK_ALIGN=BLOCK_ALIGN,
        group=group_part2
    )

    grad_mixes_slice = grad_mixes_pad[:, :, :hc_mult].reshape(b, s, hc_mult * hc_mult)
    grad_hc_base_slice = grad_hc_base_pad[:, :hc_mult].reshape(hc_mult * hc_mult)

    grad_mixes[:, :, 2 * hc_mult:] = grad_mixes_slice
    grad_hc_base[2 * hc_mult:] = grad_hc_base_slice

    grad_mixes = grad_mixes.to(dtype=origin_dtype)
    grad_hc_scale = grad_hc_scale.to(dtype=origin_dtype)
    grad_hc_base = grad_hc_base.to(dtype=origin_dtype)

    return grad_mixes, grad_hc_scale, grad_hc_base
