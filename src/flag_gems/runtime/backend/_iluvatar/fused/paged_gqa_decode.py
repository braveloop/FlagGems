# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Direct sequence-indexed paged GQA Decode using caller-owned split buffers."""

import triton
import triton.language as tl

from .paged_gqa_prefill import _paged_gqa_tile_update


@triton.jit
def _paged_gqa_decode_segment(
    q, K, V, Table, seq, kv_head, segment, positions, valid_rows, seq_len, Scale,
    BLOCK_SIZE: tl.constexpr, BLOCKS: tl.constexpr,
    TABLE_COLS: tl.constexpr, TABLE_STRIDE: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
    KV_TILE: tl.constexpr,
):
    tiles_per_segment = tl.cdiv(seq_len, 16 * KV_TILE)
    begin = segment * tiles_per_segment * KV_TILE
    stop = tl.minimum(seq_len, begin + tiles_per_segment * KV_TILE)
    full_end = stop // KV_TILE * KV_TILE
    m = tl.full((16,), -float("inf"), tl.float32)
    l = tl.full((16,), 0.0, tl.float32)
    acc = tl.full((16, 128), 0.0, tl.float32)
    for tile in range(begin, full_end, KV_TILE):
        m, l, acc = _paged_gqa_tile_update(
            q, K, V, Table, seq, kv_head, positions, valid_rows, seq_len - 1,
            tile, seq_len, Scale, m, l, acc,
            BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
            KS0, KS1, KS2, VS0, VS1, VS2, False, KV_TILE, KV_TILE == 64,
        )
    for tile in range(tl.maximum(begin, full_end), stop, KV_TILE):
        m, l, acc = _paged_gqa_tile_update(
            q, K, V, Table, seq, kv_head, positions, valid_rows, seq_len - 1,
            tile, seq_len, Scale, m, l, acc,
            BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
            KS0, KS1, KS2, VS0, VS1, VS2, True, KV_TILE, KV_TILE == 64,
        )
    return m, l, acc


@triton.jit(do_not_specialize=["TOKENS"])
def iluvatar_paged_gqa_decode_partial_kernel(
    Q, K, V, QStart, SeqLen, Table, Partial, Max, Sum, Scale,
    TOKENS, BLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    TABLE_COLS: tl.constexpr, TABLE_STRIDE: tl.constexpr,
    QS0: tl.constexpr, QS1: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
):
    seq, kv_head, segment = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q_start = tl.load(QStart + seq).to(tl.int64)
    q_end = tl.load(QStart + seq + 1).to(tl.int64)
    # FULL graphs may contain holes or tail padding; sequence is not token index.
    if (q_end - q_start == 1) & (q_start >= 0) & (q_start < TOKENS):
        r = tl.arange(0, 16)
        d = tl.arange(0, 128)
        heads = kv_head * 8 + r % 8
        valid_rows = r < 8
        positions = r // 8
        q = tl.load(Q + q_start * QS0 + heads[:, None] * QS1 + d[None, :],
                    valid_rows[:, None], 0)
        seq_len = tl.maximum(tl.load(SeqLen + seq).to(tl.int64), 0)
        # Below two wide tiles, use narrow tiles to avoid padded key work.
        if seq_len < 2 * 64:
            m, l, acc = _paged_gqa_decode_segment(
                q, K, V, Table, seq, kv_head, segment, positions, valid_rows, seq_len, Scale,
                BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                KS0, KS1, KS2, VS0, VS1, VS2, 32,
            )
        else:
            m, l, acc = _paged_gqa_decode_segment(
                q, K, V, Table, seq, kv_head, segment, positions, valid_rows, seq_len, Scale,
                BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                KS0, KS1, KS2, VS0, VS1, VS2, 64,
            )
        # Write every split for valid queries, including empty splits on replay.
        state_offset = seq * 16 * 16 + heads * 16 + segment
        tl.store(Max + state_offset, m, valid_rows)
        tl.store(Sum + state_offset, l, valid_rows)
        tl.store(Partial + state_offset[:, None] * 128 + d[None, :], acc,
                 valid_rows[:, None])


@triton.jit(do_not_specialize=["TOKENS"])
def iluvatar_paged_gqa_decode_merge_kernel(
    QStart, Partial, Max, Sum, O,
    TOKENS, OS0: tl.constexpr, OS1: tl.constexpr,
):
    seq, head = tl.program_id(0), tl.program_id(1)
    q_start = tl.load(QStart + seq).to(tl.int64)
    q_end = tl.load(QStart + seq + 1).to(tl.int64)
    if (q_end - q_start == 1) & (q_start >= 0) & (q_start < TOKENS):
        segments = tl.arange(0, 16)
        d = tl.arange(0, 128)
        state_offset = seq * 16 * 16 + head * 16 + segments
        m = tl.load(Max + state_offset)
        l = tl.load(Sum + state_offset)
        maximum = tl.max(m, 0)
        safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
        weights = tl.where(l > 0, tl.exp(m - safe_maximum), 0.0)
        mass = tl.sum(l * weights, 0)
        partial = tl.load(Partial + state_offset[:, None] * 128 + d[None, :])
        result = tl.sum(partial * weights[:, None], 0) / tl.where(mass > 0, mass, 1.0)
        tl.store(O + q_start * OS0 + head * OS1 + d, result)


def paged_gqa_decode(query, key_cache, value_cache, output, query_start_loc,
                     seq_lens, block_table, softmax_scale, partial, maximum, expsum):
    """Fill output using the original metadata's contiguous FP32 workspaces."""
    tokens, sequences = query.shape[0], seq_lens.numel()
    if not sequences or not tokens:
        return output
    iluvatar_paged_gqa_decode_partial_kernel[(sequences, 2, 16)](
        query, key_cache, value_cache, query_start_loc, seq_lens, block_table,
        partial, maximum, expsum, softmax_scale,
        tokens, key_cache.shape[0], key_cache.shape[1],
        block_table.shape[1], block_table.stride(0),
        query.stride(0), query.stride(1),
        *key_cache.stride()[:3], *value_cache.stride()[:3],
        num_warps=4, num_stages=1,
    )
    iluvatar_paged_gqa_decode_merge_kernel[(sequences, 16)](
        query_start_loc, partial, maximum, expsum, output,
        tokens, output.stride(0), output.stride(1), num_warps=4,
    )
    return output
