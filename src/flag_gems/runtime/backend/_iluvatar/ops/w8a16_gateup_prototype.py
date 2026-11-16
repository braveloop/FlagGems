# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unregistered finite-input W8A16 prototype, using the existing W8A8 raw ABI.

This is the same class of math as the official weight_int8pack_mm, not a claim
of new quantization mathematics. Weights/scales must come from the independent
W8A8 adapter; this module deliberately contains no second weight quantizer.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def w8a16_kernel(
    A, W, Scale, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, GROUPED: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m, grid_n = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    if GROUPED:
        width = GROUP_M * grid_n
        first_m = (pid // width) * GROUP_M
        group_size = tl.minimum(grid_m - first_m, GROUP_M)
        local = pid % width
        pid_m = first_m + local % group_size
        pid_n = local // group_size
    else:
        pid_m, pid_n = pid // grid_n, pid % grid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    channels = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(0, tl.cdiv(K, BLOCK_K)):
        keys = start * BLOCK_K + rk
        a = tl.load(A + rows[:, None] * K + keys[None, :],
                    (rows[:, None] < M) & (keys[None, :] < K), 0)
        # W stays ordinary [N,K] row-major. Only this tile is cast, never a
        # full-weight BF16 expansion, and no physical transpose or repacking.
        b = tl.load(W + channels[None, :] * K + keys[:, None],
                    (channels[None, :] < N) & (keys[:, None] < K), 0).to(tl.bfloat16)
        acc = tl.dot(a, b, acc)
    scale = tl.load(Scale + channels, channels < N, 0)
    result = acc * scale[None, :]
    tl.store(Out + rows[:, None] * N + channels[None, :], result.to(tl.bfloat16),
             (rows[:, None] < M) & (channels[None, :] < N))


def w8a16_mm_out(a, weight, scale, *, out, BLOCK_M, BLOCK_N, BLOCK_K,
                 num_warps, num_stages, GROUP_M=8, GROUPED=False):
    """Caller-owned output; scale is the W8A8 FP32[N,1] viewed as [N].

    Only finite BF16 activations and the already-validated finite raw W8A8
    checkpoint representation are in scope. No runtime whole-weight cast.
    Configuration is explicit and frozen by the isolated caller, not autotuned.
    """
    if a.ndim != 2 or weight.ndim != 2:
        raise ValueError("expected A[M,K], W[N,K]")
    m, k = a.shape
    n, wk = weight.shape
    if (k != wk or min(n, k) <= 0 or a.dtype != torch.bfloat16
            or weight.dtype != torch.int8 or scale.dtype != torch.float32
            or scale.shape != (n,) or scale.stride() != (1,)
            or out.dtype != torch.bfloat16 or out.shape != (m, n)
            or a.stride() != (k, 1) or weight.stride() != (k, 1)
            or out.stride() != (n, 1)
            or a.device.type != "cuda"
            or not (a.device == weight.device == scale.device == out.device)):
        raise ValueError("unsupported W8A16 prototype contract")
    storage = out.untyped_storage().data_ptr()
    if any(t.untyped_storage().data_ptr() == storage for t in (a, weight, scale)):
        raise ValueError("output must not overlap input storage")
    if m:
        w8a16_kernel[(triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)](
            a, weight, scale, out, m, n, k,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M, GROUPED=GROUPED,
            num_warps=num_warps, num_stages=num_stages,
        )
    return out
