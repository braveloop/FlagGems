# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unregistered BI native-INT8 prototype; reference math is not claimed novel."""
import torch
import triton
import triton.language as tl


@triton.jit
def quantize_token_kernel(X, Q, Scale, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    keys = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + keys, keys < K, 0).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    scale = tl.div_rn(maximum, 127.0)
    safe_max = tl.where(maximum != 0, maximum, 1.0)
    inverse = tl.where(maximum != 0, tl.div_rn(127.0, safe_max), 0.0)
    value = x * inverse
    lower = tl.floor(value)
    fraction = value - lower
    odd = tl.abs(lower - 2.0 * tl.floor(lower * 0.5)) > 0.5
    rounded = tl.where((fraction > 0.5) | ((fraction == 0.5) & odd), lower + 1.0, lower)
    code = tl.minimum(tl.maximum(rounded, -128.0), 127.0).to(tl.int8)
    tl.store(Q + row * K + keys, code, keys < K)
    tl.store(Scale + row, scale)


@triton.jit
def native_w8a8_kernel(
    A, W, ScaleA, ScaleW, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUPED: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m, grid_n = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    if GROUPED:
        width = 8 * grid_n
        first_m = (pid // width) * 8
        group_size = tl.minimum(grid_m - first_m, 8)
        local = pid % width
        pm, pn = first_m + local % group_size, local // group_size
    else:
        pm, pn = pid // grid_n, pid % grid_n
    rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    channels = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
    for step in range(0, tl.cdiv(K, BLOCK_K)):
        keys = step * BLOCK_K + rk
        a = tl.load(A + rows[:, None] * K + keys[None, :],
                    (rows[:, None] < M) & (keys[None, :] < K), 0)
        w = tl.load(W + channels[None, :] * K + keys[:, None],
                    (channels[None, :] < N) & (keys[:, None] < K), 0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)
    sa = tl.load(ScaleA + rows, rows < M, 0)
    sw = tl.load(ScaleW + channels, channels < N, 0)
    value = acc.to(tl.float32) * sa[:, None]
    value = value * sw[None, :]
    tl.store(Out + rows[:, None] * N + channels[None, :], value.to(tl.bfloat16),
             (rows[:, None] < M) & (channels[None, :] < N))


def quantize_per_token_out(x, q, scale):
    m, k = x.shape
    quantize_token_kernel[(m,)](x, q, scale, k, triton.next_power_of_2(k),
                                num_warps=4, num_stages=1, enable_fp_fusion=False)
    return q, scale


def dynamic_w8a8_out(x, weight, weight_scale, *, out, BLOCK_M=64, BLOCK_N=256,
                      BLOCK_K=64, num_warps=16, num_stages=1, GROUPED=False):
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError('expected X[M,K], raw signed weight[N,K]')
    m, k = x.shape
    n, wk = weight.shape
    if (x.dtype != torch.bfloat16 or weight.dtype != torch.int8 or k != wk
            or weight_scale.dtype != torch.float32 or weight_scale.shape != (n, 1)
            or out.dtype != torch.bfloat16 or out.shape != (m, n)
            or x.stride() != (k, 1) or weight.stride() != (k, 1)
            or weight_scale.stride() != (1, 1) or out.stride() != (n, 1)
            or not (x.device == weight.device == weight_scale.device == out.device)):
        raise ValueError('unsupported isolated W8A8 contract')
    if any(t.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
           for t in (x, weight, weight_scale)):
        raise ValueError('output overlaps input storage')
    # Allocations are ordinary capture-compatible workspaces, not an input cache.
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    quantize_per_token_out(x, q, scale)
    native_w8a8_kernel[(triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)](
        q, weight, scale, weight_scale, out, m, n, k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUPED=GROUPED,
        num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=False)
    return out
