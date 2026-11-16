# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paged BF16 GQA: reuse each KV tile across eight queries and eight heads."""

import triton
import triton.language as tl


@triton.jit
def _paged_gqa_tile_update(
    q, K, V, Table, seq, kv_head, positions, valid_rows, context,
    begin, seq_len, scale, m, l, acc,
    BLOCK_SIZE: tl.constexpr, BLOCKS: tl.constexpr, TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr,
    KS2: tl.constexpr, VS0: tl.constexpr, VS1: tl.constexpr,
    VS2: tl.constexpr, CAUSAL_BOUNDARY: tl.constexpr,
    KV_TILE: tl.constexpr = 32, DEFER_V: tl.constexpr = False,
):
    lanes = tl.arange(0, KV_TILE)
    j = begin + lanes
    logical = j // BLOCK_SIZE
    page_valid = logical < TABLE_COLS
    if CAUSAL_BOUNDARY:
        page_valid = page_valid & (j < seq_len)
    # Read page IDs at page granularity for aligned 32/64/128-key tiles.
    # Load page IDs once at page granularity, not once for every token lane.
    if KV_TILE == 128 and (BLOCK_SIZE == 16 or BLOCK_SIZE == 32 or BLOCK_SIZE == 64):
        first_page = begin // BLOCK_SIZE
        physical = tl.full((KV_TILE,), 0, tl.int64)
        for page_offset in tl.static_range(0, KV_TILE // BLOCK_SIZE):
            page = first_page + page_offset
            page_id = tl.load(Table + seq * TABLE_STRIDE + page,
                              (page < TABLE_COLS) & (begin + page_offset * BLOCK_SIZE < seq_len), 0)
            physical = tl.where(lanes // BLOCK_SIZE == page_offset,
                                page_id.to(tl.int64), physical)
    elif BLOCK_SIZE == 16:
        first_page = begin // 16
        p0 = tl.load(Table + seq * TABLE_STRIDE + first_page,
                     (first_page < TABLE_COLS) & (begin < seq_len), 0)
        p1 = tl.load(Table + seq * TABLE_STRIDE + first_page + 1,
                     (first_page + 1 < TABLE_COLS) & (begin + 16 < seq_len), 0)
        if KV_TILE == 64:
            p2 = tl.load(Table + seq * TABLE_STRIDE + first_page + 2,
                         (first_page + 2 < TABLE_COLS) & (begin + 32 < seq_len), 0)
            p3 = tl.load(Table + seq * TABLE_STRIDE + first_page + 3,
                         (first_page + 3 < TABLE_COLS) & (begin + 48 < seq_len), 0)
            physical = tl.where(lanes < 32, tl.where(lanes < 16, p0, p1),
                                tl.where(lanes < 48, p2, p3)).to(tl.int64)
        else:
            physical = tl.where(lanes < 16, p0, p1).to(tl.int64)
    elif BLOCK_SIZE == 32 and KV_TILE == 64:
        first_page = begin // 32
        p0 = tl.load(Table + seq * TABLE_STRIDE + first_page,
                     (first_page < TABLE_COLS) & (begin < seq_len), 0)
        p1 = tl.load(Table + seq * TABLE_STRIDE + first_page + 1,
                     (first_page + 1 < TABLE_COLS) & (begin + 32 < seq_len), 0)
        physical = tl.where(lanes < 32, p0, p1).to(tl.int64)
    elif BLOCK_SIZE % KV_TILE == 0:
        first_page = begin // BLOCK_SIZE
        physical = tl.load(Table + seq * TABLE_STRIDE + first_page,
                           (first_page < TABLE_COLS) & (begin < seq_len), 0).to(tl.int64)
        physical = physical + tl.zeros((KV_TILE,), tl.int64)
    else:
        physical = tl.load(Table + seq * TABLE_STRIDE + logical, page_valid, 0).to(tl.int64)
    page_valid = page_valid & (physical >= 0) & (physical < BLOCKS)
    d = tl.arange(0, 128)
    k_address = physical[None, :] * KS0 + (j % BLOCK_SIZE)[None, :] * KS1 + kv_head * KS2 + d[:, None]
    k = tl.load(K + k_address, page_valid[None, :], 0)
    if not DEFER_V:
        v_address = physical[:, None] * VS0 + (j % BLOCK_SIZE)[:, None] * VS1 + kv_head * VS2 + d[None, :]
        v = tl.load(V + v_address, page_valid[:, None], 0)
    scores = tl.dot(q, k).to(tl.float32) * scale
    allowed = page_valid[None, :]
    if CAUSAL_BOUNDARY:
        allowed = allowed & valid_rows[:, None] & (j[None, :] <= context + positions[:, None])
    scores = tl.where(allowed, scores, -float("inf"))
    m_new = tl.maximum(m, tl.max(scores, 1))
    safe_max = tl.where(m_new == -float("inf"), 0.0, m_new)
    p = tl.exp(scores - safe_max[:, None])
    alpha = tl.exp(m - safe_max)
    l = l * alpha + tl.sum(p, 1)
    if DEFER_V:
        v_address = physical[:, None] * VS0 + (j % BLOCK_SIZE)[:, None] * VS1 + kv_head * VS2 + d[None, :]
        v = tl.load(V + v_address, page_valid[:, None], 0)
    acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
    return m_new, l, acc


@triton.jit(do_not_specialize=["TOKENS", "SEQS"])
def iluvatar_paged_gqa_prefill_kernel(
    Q, K, V, O, QStart, SeqLen, Table, Scale,
    TOKENS, SEQS, BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr, QS0: tl.constexpr, QS1: tl.constexpr,
    OS0: tl.constexpr, OS1: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
    QUERY_TILE: tl.constexpr, SKIP_Q1: tl.constexpr = False,
):
    tile, kv_head = tl.program_id(0), tl.program_id(1)
    # Strictly increasing prefixes also allocate harmless slots for empty queries.
    lo, hi = 0, SEQS
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        prefix = tl.load(QStart + mid) // QUERY_TILE + mid
        if tile >= prefix:
            lo = mid
        else:
            hi = mid
    seq = lo
    q_start = tl.load(QStart + seq).to(tl.int64)
    q_end = tl.load(QStart + seq + 1).to(tl.int64)
    local_tile = tile - (q_start // QUERY_TILE + seq)
    first_query = local_tile * QUERY_TILE
    q_len = q_end - q_start
    if (first_query < q_len) & ((q_len != 1) | (not SKIP_Q1)):
        M: tl.constexpr = QUERY_TILE * 8
        r = tl.arange(0, M)
        positions = first_query + r // 8
        heads = kv_head * 8 + r % 8
        d = tl.arange(0, 128)
        valid_rows = (positions < q_len) & (q_start + positions < TOKENS)
        q = tl.load(Q + (q_start + positions[:, None]) * QS0 + heads[:, None] * QS1 + d[None, :], valid_rows[:, None], 0)
        seq_len = tl.load(SeqLen + seq).to(tl.int64)
        context = seq_len - q_len
        m = tl.full((M,), -float("inf"), tl.float32)
        # Empty softmax has no mass; do not rely on a first update cancelling
        # an artificial nonzero normalizer initializer.
        l = tl.full((M,), 0.0, tl.float32)
        acc = tl.full((M, 128), 0.0, tl.float32)
        # The first query sees this entire prefix, so all later rows do too.
        full_end = tl.maximum(0, (context + first_query + 1) // 32 * 32)
        stop = tl.minimum(seq_len, context + tl.minimum(first_query + QUERY_TILE, q_len))
        # Q8/Q16 use wider tiles only in the prefix visible to every query row.
        wide_end = 0
        if QUERY_TILE >= 8:
            wide128_end = full_end // 128 * 128
            for begin in range(0, wide128_end, 128):
                m, l, acc = _paged_gqa_tile_update(
                    q, K, V, Table, seq, kv_head, positions, valid_rows, context,
                    begin, seq_len, Scale, m, l, acc,
                    BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                    KS0, KS1, KS2, VS0, VS1, VS2, False, 128, True,
                )
            wide_end = full_end // 64 * 64
            for begin in range(wide128_end, wide_end, 64):
                m, l, acc = _paged_gqa_tile_update(
                    q, K, V, Table, seq, kv_head, positions, valid_rows, context,
                    begin, seq_len, Scale, m, l, acc,
                    BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                    KS0, KS1, KS2, VS0, VS1, VS2, False, 64, True,
                )
        for begin in range(wide_end, full_end, 32):
            m, l, acc = _paged_gqa_tile_update(
                q, K, V, Table, seq, kv_head, positions, valid_rows, context,
                begin, seq_len, Scale, m, l, acc,
                BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                KS0, KS1, KS2, VS0, VS1, VS2, False,
            )
        for begin in range(full_end, stop, 32):
            m, l, acc = _paged_gqa_tile_update(
                q, K, V, Table, seq, kv_head, positions, valid_rows, context,
                begin, seq_len, Scale, m, l, acc,
                BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                KS0, KS1, KS2, VS0, VS1, VS2, True,
            )
        result = acc / tl.where(l > 0, l, 1.0)[:, None]
        tl.store(O + (q_start + positions[:, None]) * OS0 + heads[:, None] * OS1 + d[None, :], result, valid_rows[:, None])


def paged_gqa_prefill(query, key_cache, value_cache, output, query_start_loc, seq_lens, block_table, softmax_scale, *, query_tile=8, skip_q1=False):
    """Fill caller-owned output; bridge owns feature/shape compatibility guards."""
    tokens, sequences = query.shape[0], seq_lens.numel()
    if type(query_tile) is not int or query_tile not in (2, 8, 16, 32):
        raise ValueError("query_tile must be 2, 8, 16 or 32")
    if type(skip_q1) is not bool:
        raise ValueError("skip_q1 must be a bool")
    if not sequences or not tokens:
        return output
    iluvatar_paged_gqa_prefill_kernel[(tokens // query_tile + sequences, 2)](
        query, key_cache, value_cache, output, query_start_loc, seq_lens,
        block_table, softmax_scale,
        tokens, sequences, key_cache.shape[0], key_cache.shape[1],
        block_table.shape[1], block_table.stride(0),
        query.stride(0), query.stride(1), output.stride(0), output.stride(1),
        *key_cache.stride()[:3], *value_cache.stride()[:3],
        query_tile, skip_q1,
        num_warps=16 if query_tile == 32 else (8 if query_tile >= 8 else 4), num_stages=1,
    )
    return output
