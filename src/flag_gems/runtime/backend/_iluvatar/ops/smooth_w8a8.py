# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BI W8A8 candidate: fuse SmoothQuant inverse scales into A8 and tail fill.

SmoothQuant mathematics is reused, not claimed novel. The native INT8 loading
path and fused activation preparation are target-specific implementations.
This module is not enabled by the production selector until validation.
"""
import torch
import triton
import triton.language as tl
from ixformer.inference.functions.w8a8 import w8a8 as vendor_w8a8


@triton.jit(do_not_specialize=['ACTUAL_M'])
def quantize_token_kernel(X, InvS, Q, Scale, K: tl.constexpr, BLOCK: tl.constexpr,
                          ACTUAL_M=0, PAD_ROWS: tl.constexpr = False):
    row = tl.program_id(0)
    keys = tl.arange(0, BLOCK)
    if PAD_ROWS:
        x = tl.load(X + row * K + keys, (row < ACTUAL_M) & (keys < K), 0).to(tl.float32)
    else:
        x = tl.load(X + row * K + keys, keys < K, 0).to(tl.float32)
    inv_s = tl.load(InvS + keys, keys < K, 1.0)
    x = x * inv_s
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


@triton.jit(do_not_specialize=['M'])
def native_w8a8_kernel(
    A, W, ScaleA, ScaleW, Out, M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUPED: tl.constexpr, PADDED_A: tl.constexpr = False,
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
        if PADDED_A:
            # Aq physically contains every launched row; the fused quantizer
            # wrote zero codes/scales for rows beyond logical M.
            a = tl.load(A + rows[:, None] * K + keys[None, :],
                        keys[None, :] < K, 0)
        else:
            a = tl.load(A + rows[:, None] * K + keys[None, :],
                        (rows[:, None] < M) & (keys[None, :] < K), 0)
        w = tl.load(W + channels[None, :] * K + keys[:, None],
                    (channels[None, :] < N) & (keys[:, None] < K), 0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)
    if PADDED_A:
        sa = tl.load(ScaleA + rows)
    else:
        sa = tl.load(ScaleA + rows, rows < M, 0)
    sw = tl.load(ScaleW + channels, channels < N, 0)
    value = acc.to(tl.float32) * sa[:, None]
    value = value * sw[None, :]
    tl.store(Out + rows[:, None] * N + channels[None, :], value.to(tl.bfloat16),
             (rows[:, None] < M) & (channels[None, :] < N))


def quantize_per_token_out(x, inverse_s, q, scale):
    m, k = x.shape
    physical_m = q.shape[0]
    if (q.shape[1:] != (k,) or physical_m < m
            or scale.shape != (physical_m, 1)):
        raise ValueError('invalid physical A8 workspace capacity')
    if physical_m != m:
        quantize_token_kernel[(physical_m,)](
            x, inverse_s, q, scale, k, triton.next_power_of_2(k), ACTUAL_M=m, PAD_ROWS=True,
            num_warps=4, num_stages=1, enable_fp_fusion=False)
    else:
        quantize_token_kernel[(m,)](x, inverse_s, q, scale, k, triton.next_power_of_2(k),
                                    num_warps=4, num_stages=1, enable_fp_fusion=False)
    return q, scale


def smooth_w8a8_out(x, weight, weight_scale, inverse_s, *, out):
    # Fixed policies from the existing native INT8 measurements. Generic shapes
    # retain the same quantized ABI and use a conservative masked implementation.
    k, n = x.shape[-1], weight.shape[0]
    fast = k % 64 == 0 and n % 256 == 0 and k <= 8192
    BLOCK_M = (256 if x.shape[0] >= 256 else 64) if fast else 32
    BLOCK_N, BLOCK_K = (256, 64) if fast else (64, 32)
    num_warps = 16 if fast else 4
    num_stages = 2 if fast and BLOCK_M == 256 else 1
    GROUPED = False
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError('expected X[M,K], raw signed weight[N,K]')
    m, k = x.shape
    n, wk = weight.shape
    if (x.dtype != torch.bfloat16 or weight.dtype != torch.int8 or k != wk
            or not (0 < k <= 131071)
            or inverse_s.dtype != torch.float32 or inverse_s.shape != (k,)
            or inverse_s.stride() != (1,)
            or weight_scale.dtype != torch.float32 or weight_scale.shape != (n, 1)
            or out.dtype != torch.bfloat16 or out.shape != (m, n)
            or x.stride() != (k, 1) or weight.stride() != (k, 1)
            or weight_scale.stride() != (1, 1) or out.stride() != (n, 1)
            or not (x.device == weight.device == weight_scale.device == inverse_s.device == out.device)):
        raise ValueError('unsupported isolated W8A8 contract')
    if any(t.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
           for t in (x, weight, weight_scale, inverse_s)):
        raise ValueError('output overlaps input storage')
    if m == 0 or n == 0:
        return out
    # The calibrated BI Linear shapes use the installed TN implementation.
    # Quantization remains our fused inverse-s/RNE kernel; vendor GEMM is
    # explicitly attributed to ixformer. No weight repacking or BF16 mirror.
    if k == 2048 and n in (2560, 12288):
        q = torch.empty_like(x, dtype=torch.int8)
        scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
        quantize_per_token_out(x, inverse_s, q, scale)
        vendor_w8a8(q, weight, scale.view(-1), weight_scale.view(-1),
                    output=out, format="TN")
        return out
    # Allocations are ordinary capture-compatible workspaces, not an input cache.
    padded = m % BLOCK_M != 0
    physical_m = triton.cdiv(m, BLOCK_M) * BLOCK_M if padded else m
    q = (torch.empty((physical_m, k), dtype=torch.int8, device=x.device)
         if padded else torch.empty_like(x, dtype=torch.int8))
    scale = torch.empty((physical_m, 1), dtype=torch.float32, device=x.device)
    quantize_per_token_out(x, inverse_s, q, scale)
    native_w8a8_kernel[(triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)](
        q, weight, scale, weight_scale, out, m, n, k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUPED=GROUPED,
        # Aq has ceil(M/BM)*BM initialized rows even when M is tile-aligned.
        # Always expose that physical extent, so runtime M only masks stores
        # and never disables the native A-side SME load through a row predicate.
        PADDED_A=True,
        num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=False)
    return out


@triton.jit
def quantize_weight_kernel(W, S, Q, Scale, K: tl.constexpr, BLOCK: tl.constexpr):
    """One row per program: avoid full FP32 scaled/abs/rounded weight copies."""
    row = tl.program_id(0)
    keys = tl.arange(0, BLOCK)
    value = tl.load(W + row * K + keys, keys < K, 0).to(tl.float32)
    smooth = tl.load(S + keys, keys < K, 1.0)
    value = value * smooth
    maximum = tl.max(tl.abs(value), 0)
    scale = tl.div_rn(maximum, 127.0)
    inverse = tl.where(maximum > 0, tl.div_rn(127.0, tl.where(maximum > 0, maximum, 1.0)), 0.0)
    value = value * inverse
    lower = tl.floor(value)
    fraction = value - lower
    odd = tl.abs(lower - 2.0 * tl.floor(lower * 0.5)) > 0.5
    rounded = tl.where((fraction > 0.5) | ((fraction == 0.5) & odd), lower + 1.0, lower)
    tl.store(Q + row * K + keys, tl.minimum(tl.maximum(rounded, -128.0), 127.0).to(tl.int8), keys < K)
    tl.store(Scale + row, scale)


def quantize_weight_out(weight, smooth, *, out, scale):
    n, k = weight.shape
    if (weight.dtype != torch.bfloat16 or weight.stride() != (k, 1)
            or smooth.dtype != torch.float32 or smooth.shape != (k,) or smooth.stride() != (1,)
            or out.dtype != torch.int8 or out.shape != (n, k) or out.stride() != (k, 1)
            or scale.dtype != torch.float32 or scale.shape != (n, 1)
            or not (weight.device == smooth.device == out.device == scale.device)):
        raise ValueError('unsupported load-time smooth weight contract')
    quantize_weight_kernel[(n,)](weight, smooth, out, scale, k, triton.next_power_of_2(k),
                                num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out, scale

