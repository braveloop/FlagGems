# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Target-card differential tests and small paired timing for own paged GQA."""
import json
import hashlib
import importlib
import math
import statistics
import sys
import time
import types

import pytest
import torch
import triton
import triton.language as tl

if __name__ == "__main__":
    torch.cuda.init()

import flag_gems
from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_prefill import paged_gqa_prefill
from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_decode import paged_gqa_decode
from vllm.v1.attention.ops.triton_attention_helpers import softmax_step as native_softmax_step

pytestmark = pytest.mark.skipif(flag_gems.vendor_name != "iluvatar", reason="Iluvatar kernel")


@triton.jit
def _native_softmax_probe(Out, N: tl.constexpr):
    columns = tl.arange(0, 32)
    S = tl.where(columns[None, :] < N, tl.full((16, 32), 0.0, tl.float32), -float("inf"))
    M = tl.full((16,), -float("inf"), tl.float32)
    L = tl.full((16,), 1.0, tl.float32)
    M, L, P, alpha = native_softmax_step(S, M, L)
    tl.store(Out, tl.sum(alpha, 0) / 16)
    tl.store(Out + 1, tl.sum(L, 0) / 16)
    tl.store(Out + 2, tl.sum(tl.sum(P, 1), 0) / 16)
    tl.store(Out + 3, tl.exp(-float("inf")))


def probe_native_initial_normalizer():
    """Single-point diagnostic; never changes correctness assertions/oracle."""
    for visible in (1, 2, 8):
        query, cache, storage, starts, seq_lens, table = make_case([1], [visible - 1], kind="zero")
        cache[:, 1].fill_(1)
        out = storage[:1]
        native(query, cache, out, starts, seq_lens, table, 2, visible)
        probe = torch.empty(4, device="cuda", dtype=torch.float32)
        _native_softmax_probe[(1,)](probe, visible)
        torch.cuda.synchronize()
        print(json.dumps({"diagnostic": "native initial normalizer", "visible_keys": visible,
                          "native_min": out.min().item(), "native_max": out.max().item(),
                          "mathematical_expected": 1, "alpha_L_sumP_exp_neginf": probe.cpu().tolist(),
                          "probe_expected": [0, visible, visible, 0]}), flush=True)


def make_case(lengths, contexts, page=16, padded=False, kind="random"):
    generator = torch.Generator().manual_seed(42)
    seq_sizes = [q + c for q, c in zip(lengths, contexts)]
    columns = max(1, math.ceil(max(seq_sizes) / page))
    blocks = sum(math.ceil(s / page) for s in seq_sizes) + 3
    raw_cache = torch.randn((blocks, 2, page, 2, 128), generator=generator, dtype=torch.bfloat16)
    if padded:
        # HND-backed storage retains the public NHD logical shape.
        raw_cache = raw_cache.permute(0, 1, 3, 2, 4).contiguous().permute(0, 1, 3, 2, 4)
    cache = raw_cache.cuda()
    permutation = torch.randperm(blocks, generator=generator)
    table = torch.full((len(lengths), columns), -1, dtype=torch.int32)
    offset = 0
    for seq, size in enumerate(seq_sizes):
        needed = math.ceil(size / page)
        table[seq, :needed] = permutation[offset:offset + needed].to(torch.int32)
        offset += needed
    tokens = sum(lengths)
    q_storage = torch.randn((tokens + 3, 20 if padded else 16, 128), generator=generator, dtype=torch.bfloat16).cuda()
    query = q_storage[:tokens, :16]
    if kind == "zero":
        query.zero_()
        cache[:, 0].zero_()
    elif kind == "large":
        query.mul_(8)
        cache[:, 0].mul_(8)
    out_storage = torch.full((tokens + 3, 16, 128), 137, device="cuda", dtype=torch.bfloat16)
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(seq_sizes, dtype=torch.int32, device="cuda")
    return query, cache, out_storage, starts, seq_lens, table.cuda()


def explicit_reference(query, cache, starts, seq_lens, table):
    output = torch.empty_like(query, dtype=torch.float32)
    offsets = starts.cpu().tolist()
    sizes = seq_lens.cpu().tolist()
    for seq, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if start == end:
            continue
        positions = torch.arange(sizes[seq], device="cuda")
        physical = table[seq, positions // cache.shape[2]].long()
        k = cache[physical, 0, positions % cache.shape[2]].float().repeat_interleave(8, dim=1)
        v = cache[physical, 1, positions % cache.shape[2]].float().repeat_interleave(8, dim=1)
        q = query[start:end].float().transpose(0, 1)
        scores = torch.matmul(q, k.transpose(0, 1).transpose(1, 2)) / math.sqrt(128)
        valid = positions[None, :] <= (sizes[seq] - (end - start) + torch.arange(end - start, device="cuda"))[:, None]
        scores.masked_fill_(~valid[None], -float("inf"))
        output[start:end] = torch.matmul(scores.softmax(-1), v.transpose(0, 1)).transpose(0, 1)
    return output


def native(query, cache, output, starts, seq_lens, table, max_query, max_seq):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    k, v = cache.unbind(1)
    unified_attention(q=query, k=k, v=v, out=output, cu_seqlens_q=starts,
                      max_seqlen_q=max_query, seqused_k=seq_lens, max_seqlen_k=max_seq,
                      softmax_scale=1 / math.sqrt(128), causal=True,
                      window_size=(-1, -1), block_table=table, softcap=0,
                      q_descale=None, k_descale=None, v_descale=None)
    return output


def check_case(lengths, contexts, page=16, padded=False, kind="random", *, query_tile=8):
    query, cache, storage, starts, seq_lens, table = make_case(lengths, contexts, page, padded, kind)
    original_query, original_cache = query.clone(), cache.clone()
    output = storage[:query.shape[0]]
    k, v = cache.unbind(1)
    returned = paged_gqa_prefill(query, k, v, output, starts, seq_lens, table, 1 / math.sqrt(128), query_tile=query_tile)
    assert returned is output and torch.isfinite(output).all()
    expected = explicit_reference(query, cache, starts, seq_lens, table)
    difference = output.float() - expected
    record = {"lengths": lengths, "contexts": contexts, "page": page, "padded": padded, "kind": kind, "query_tile": query_tile,
              "max_abs_fp32": difference.abs().max().item() if difference.numel() else 0,
              "rms_fp32": difference.square().mean().sqrt().item() if difference.numel() else 0}
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
    # Force the original 2D native path for this standalone mathematical comparison.
    reference = torch.empty_like(output)
    native(query, cache, reference, starts, seq_lens, table, max(2, max(lengths)), max(q + c for q, c in zip(lengths, contexts)))
    record["max_abs_native"] = (output.float() - reference.float()).abs().max().item()
    record["rms_native"] = (output.float() - reference.float()).square().mean().sqrt().item()
    native_difference = reference.float() - expected
    record["native_fp32_max_abs"] = native_difference.abs().max().item()
    record["native_fp32_rms"] = native_difference.square().mean().sqrt().item()
    record["native_parity"] = bool(torch.allclose(output, reference, atol=2e-2, rtol=2e-2))
    record["native_comparison"] = "diagnostic only; normalization anomaly observed, cause unproven"
    assert torch.equal(query, original_query) and torch.equal(cache, original_cache)
    assert torch.all(storage[query.shape[0]:] == 137)
    print(json.dumps({"fp32_correctness": "PASS", **record}), flush=True)


def test_constant_value_identity():
    for visible in (1, 2, 8, 18, 32, 33, 64, 128):
        query, cache, storage, starts, seq_lens, table = make_case([1], [visible - 1], kind="zero")
        cache[:, 1].fill_(1)
        out = storage[:1]
        k, v = cache.unbind(1)
        paged_gqa_prefill(query, k, v, out, starts, seq_lens, table, 1 / math.sqrt(128))
        torch.testing.assert_close(out, torch.ones_like(out), atol=2e-2, rtol=2e-2)
        original = torch.empty_like(out)
        native(query, cache, original, starts, seq_lens, table, 2, visible)
        print(json.dumps({"constant_value_identity": "PASS", "visible_keys": visible,
                          "own_min": out.min().item(), "own_max": out.max().item(),
                          "native_min": original.min().item(), "native_max": original.max().item(),
                          "native_comparison": "diagnostic only; not an equality oracle"}), flush=True)


CASES = [([q], [c], 16, False, "random") for q in (1, 2, 7, 8, 9, 31, 33, 65) for c in (0, 1, 17, 127, 2049)]
CASES += [([1, 7, 33, 0], [17, 127, 1, 0], p, True, "random") for p in (16, 32, 48, 64)]
CASES += [([9], [127], 32, True, k) for k in ("zero", "large")]
CASES += [([31], [16385], 16, True, "random")]


@pytest.mark.parametrize("case", CASES)
def test_paged_gqa_prefill(case):
    check_case(*case)


def test_short_query_tiles():
    cases = (([2], [17], 2, 16), ([3], [17], 8, 32),
             ([4], [2049], 8, 16), ([5], [17], 8, 32),
             ([1, 2, 0], [17, 2049, 0], 2, 32),
             ([1, 3, 4, 0], [17, 1, 127, 0], 8, 16))
    for lengths, contexts, tile, page in cases:
        check_case(lengths, contexts, page, True, query_tile=tile)


def test_q32_query_boundaries():
    for page in (16, 48, 64):
        check_case([31, 32, 33, 0], [127, 128, 129, 0], page, True, query_tile=32)


def timed(operation, repeats=7):
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    host, device = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        begin.record()
        started = time.perf_counter()
        operation()
        end.record()
        end.synchronize()
        host.append((time.perf_counter() - started) * 1000)
        device.append(begin.elapsed_time(end))
    return {label: {"all_repeats_ms": values, "mean_ms": statistics.mean(values), "sd_ms": statistics.stdev(values), "cv_percent": statistics.stdev(values) / statistics.mean(values) * 100} for label, values in (("host", host), ("event_interval", device))}


def microbench():
    for lengths, contexts in (([2048], [0]), ([128], [8192]), ([1] * 7 + [1024], [8192] * 7 + [3072])):
        query, cache, storage, starts, seq_lens, table = make_case(lengths, contexts)
        out = storage[:query.shape[0]]
        k, v = cache.unbind(1)
        original = lambda: native(query, cache, out, starts, seq_lens, table, max(lengths), max(q + c for q, c in zip(lengths, contexts)))
        own = lambda: paged_gqa_prefill(query, k, v, out, starts, seq_lens, table, 1 / math.sqrt(128))
        results = {"native_before": timed(original), "candidate": timed(own), "native_restore": timed(original)}
        print(json.dumps({"micro": "diagnostic only, not official score", "lengths": lengths, "contexts": contexts, "timing": results}), flush=True)


def decode_workspaces(sequences):
    return (torch.full((sequences, 16, 16, 128), float("nan"), device="cuda"),
            torch.full((sequences, 16, 16), float("nan"), device="cuda"),
            torch.full((sequences, 16, 16), float("nan"), device="cuda"))


def native_decode_3d(query, cache, output, starts, seq_lens, table, workspace, max_seq):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    k, v = cache.unbind(1)
    unified_attention(q=query, k=k, v=v, out=output, cu_seqlens_q=starts,
                      max_seqlen_q=1, seqused_k=seq_lens, max_seqlen_k=max_seq,
                      softmax_scale=128 ** -.5, causal=True, window_size=(-1, -1),
                      block_table=table, softcap=0, q_descale=None, k_descale=None,
                      v_descale=None, seq_threshold_3D=64, num_par_softmax_segments=16,
                      softmax_segm_output=workspace[0], softmax_segm_max=workspace[1],
                      softmax_segm_expsum=workspace[2])
    return output


def capture_operation(operation):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def check_decode_case(lengths, contexts, page=16, padded=False, kind="random"):
    query, cache, storage, starts, seq_lens, table = make_case(lengths, contexts, page, padded, kind)
    original_query, original_cache = query.clone(), cache.clone()
    output = storage[:query.shape[0]]
    workspace = decode_workspaces(len(lengths))
    k, v = cache.unbind(1)
    returned = paged_gqa_decode(query, k, v, output, starts, seq_lens, table, 128 ** -.5, *workspace)
    expected = explicit_reference(query, cache, starts, seq_lens, table)
    assert returned is output and torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
    for seq, (active, size) in enumerate(zip(lengths, seq_lens.cpu().tolist())):
        if not active:
            continue
        tile = 32 if size < 128 else 64
        tiles = math.ceil(size / (16 * tile))
        for split in range(16):
            assert torch.isfinite(workspace[0][seq, :, split]).all()
            assert torch.isfinite(workspace[2][seq, :, split]).all()
            if split * tiles * tile >= size:
                assert torch.isneginf(workspace[1][seq, :, split]).all()
                assert not workspace[0][seq, :, split].any()
                assert not workspace[2][seq, :, split].any()
    assert torch.equal(query, original_query) and torch.equal(cache, original_cache)
    assert torch.all(storage[query.shape[0]:] == 137)
    print(json.dumps({"decode_fp32_correctness": "PASS", "batch": len(lengths),
                      "active": sum(lengths), "page": page, "padded": padded, "kind": kind,
                      "max_abs_fp32": (output.float() - expected).abs().max().item()}), flush=True)


def test_decode_dynamic_graph():
    sequences = 64
    query, cache, storage, starts, seq_lens, table = make_case([1] * sequences, [4096] * sequences, 48, True)
    output, workspace = storage[:sequences], decode_workspaces(sequences)
    k, v = cache.unbind(1)
    original_query, original_cache = query.clone(), cache.clone()
    operation = lambda: paged_gqa_decode(query, k, v, output, starts, seq_lens, table, 128 ** -.5, *workspace)
    addresses = [tensor.data_ptr() for tensor in (query, cache, starts, seq_lens, table, *workspace)]
    graph = capture_operation(operation)
    # Replay long -> short, with holes and tail padding, then grow active rows again.
    for active in ([1] * 64, [1 if i < 50 and i % 3 else 0 for i in range(64)],
                   [1] * 63 + [0], [0] * 64, [1] + [0] * 63):
        counts = torch.tensor(active, dtype=torch.int32)
        starts.copy_(torch.tensor([0, *counts.cumsum(0).tolist()], device="cuda", dtype=torch.int32))
        sizes = [min(4097, (1, 2, 17, 33, 511, 1025)[i % 6]) if value else 0 for i, value in enumerate(active)]
        if sum(active) == 64:
            sizes = [4097] * 64
        seq_lens.copy_(torch.tensor(sizes, device="cuda", dtype=torch.int32))
        storage.fill_(137)
        graph.replay()
        torch.cuda.synchronize()
        count = sum(active)
        expected = explicit_reference(query, cache, starts, seq_lens, table)
        torch.testing.assert_close(output[:count].float(), expected[:count], atol=2e-2, rtol=2e-2)
        assert torch.all(storage[count:] == 137)
        assert addresses == [tensor.data_ptr() for tensor in (query, cache, starts, seq_lens, table, *workspace)]
        print(json.dumps({"decode_dynamic_graph": "PASS", "sequences": sequences,
                          "active_queries": count, "max_keys": max(sizes), "holes": active.count(0)}), flush=True)
    # A valid query with no keys must overwrite every stale split and yield finite zero.
    seq_lens.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert not output[:1].any() and torch.isfinite(output[:1]).all()
    assert torch.isneginf(workspace[1][0]).all()
    assert not workspace[0][0].any() and not workspace[2][0].any()
    assert torch.equal(query, original_query) and torch.equal(cache, original_cache)
    print(json.dumps({"decode_empty_keys_graph": "PASS"}), flush=True)


def test_decode_constant_value_identity():
    for keys in (1, 2, 8, 18, 32, 33, 64, 128, 512, 513, 16385):
        query, cache, storage, starts, seq_lens, table = make_case([1], [keys - 1], 48, True, "zero")
        cache[:, 1].fill_(1)
        k, v = cache.unbind(1)
        output = storage[:1]
        paged_gqa_decode(query, k, v, output, starts, seq_lens, table, 128 ** -.5, *decode_workspaces(1))
        torch.testing.assert_close(output, torch.ones_like(output), atol=2e-2, rtol=2e-2)
        assert torch.all(storage[1:] == 137)
        print(json.dumps({"decode_constant_value_identity": "PASS", "visible_keys": keys}), flush=True)
    query, cache, storage, starts, seq_lens, table = make_case([0, 0], [0, 0])
    workspace = decode_workspaces(2)
    output = storage[:0]
    k, v = cache.unbind(1)
    assert paged_gqa_decode(query, k, v, output, starts, seq_lens, table, 128 ** -.5, *workspace) is output
    assert torch.all(storage == 137) and all(torch.isnan(t).all() for t in workspace)
    print(json.dumps({"decode_zero_query": "PASS"}), flush=True)


DECODE_CASES = [([1] * batch, [context] * batch, page, padded, "random")
                for batch, context, page, padded in ((1, 0, 16, False), (16, 32, 32, True),
                                                     (63, 511, 48, True), (64, 1024, 64, False))]
DECODE_CASES += [([0, 1, 0, 1, 1, 0], [0, 4096, 0, 16, 128, 0], page, True, "random")
                 for page in (16, 32, 48, 64)]
DECODE_CASES += [([1, 1], [0, 32], 16, False, kind) for kind in ("zero", "large")]
DECODE_CASES += [([1], [keys - 1], 16, False, "random") for keys in (32769, 131072)]
DECODE_CASES += [([1], [keys - 1], 16, False, "random") for keys in (63, 64, 65, 1024, 1025)]


@pytest.mark.parametrize("case", DECODE_CASES)
def test_paged_gqa_decode(case):
    check_decode_case(*case)


def test_t64_prefill_default():
    # Shared helper defaults must preserve P1's 32-key history and mixed tails.
    check_case([8], [128], 16)
    check_case([1, 7, 33, 0], [17, 127, 1, 0], 48, True)


def test_prefill_p64h():
    for context in (62, 63, 64):
        check_case([9], [context], 16, True)
    for page in (16, 32, 48, 64):
        check_case([1, 7, 9, 0], [8192, 63, 62, 0], page, True)
    check_case([8], [16385], 16, True)
    check_case([9], [64], 48, True, "large")
    check_case([2], [127], 32, True, query_tile=2)
    query, cache, storage, starts, seq_lens, table = make_case([9], [64], 48, True, "zero")
    cache[:, 1].fill_(1)
    original_query, original_cache = query.clone(), cache.clone()
    output = storage[:9]
    k, v = cache.unbind(1)
    assert paged_gqa_prefill(query, k, v, output, starts, seq_lens, table, 128 ** -.5) is output
    torch.testing.assert_close(output, torch.ones_like(output), atol=2e-2, rtol=2e-2)
    assert torch.equal(query, original_query) and torch.equal(cache, original_cache)
    assert torch.all(storage[9:] == 137)
    print(json.dumps({"p64h_constant_value": "PASS", "queries": 9, "context": 64}), flush=True)
    query, cache, storage, starts, seq_lens, table = make_case([0, 0], [0, 0])
    output = storage[:0]
    k, v = cache.unbind(1)
    assert paged_gqa_prefill(query, k, v, output, starts, seq_lens, table, 128 ** -.5) is output
    assert torch.all(storage == 137)
    print(json.dumps({"p64h_zero_query": "PASS"}), flush=True)


def compiled_resources(jit_function, label):
    caches = getattr(jit_function, "device_caches", {})
    for device_cache in caches.values():
        cache = device_cache[0]
        for kernel in cache.values():
            metadata = kernel.metadata
            print(json.dumps({"compiled_resources": label,
                              "shared": getattr(metadata, "shared", None),
                              "num_warps": getattr(metadata, "num_warps", None),
                              "warp_size": getattr(metadata, "warp_size", None),
                              "n_regs": getattr(kernel, "n_regs", None),
                              "n_spills": getattr(kernel, "n_spills", None)}), flush=True)


def test_decode_shared_default_graph():
    query, cache, storage, starts, seq_lens, table = make_case([1] * 16, [64] * 16, 16, True)
    output, workspace = storage[:16], decode_workspaces(16)
    k, v = cache.unbind(1)
    original_query, original_cache = query.clone(), cache.clone()
    graph = capture_operation(lambda: paged_gqa_decode(query, k, v, output, starts, seq_lens, table, 128 ** -.5, *workspace))
    expected = explicit_reference(query, cache, starts, seq_lens, table)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
    assert torch.equal(query, original_query) and torch.equal(cache, original_cache)
    assert torch.all(storage[16:] == 137)
    assert torch.isneginf(workspace[1][:, :, 3:]).all()
    assert not workspace[0][:, :, 3:].any() and not workspace[2][:, :, 3:].any()
    print(json.dumps({"decode_shared_default_graph": "PASS", "sequences": 16,
                      "keys": 65, "partial_tile": 32, "helper_defer_v": False}), flush=True)



def test_prefill_runtime_counts():
    module = sys.modules[paged_gqa_prefill.__module__]
    for query_tile, page in ((8, 16), (2, 48)):
        query, cache, storage, starts, seq_lens, table = make_case([9, 7, 1, 0], [2049] * 4, page, True)
        k, v = cache.unbind(1)
        original_query, original_cache = query.clone(), cache.clone()
        cases = (([9], [62]), ([9, 7], [63, 64]),
                 ([1, 0, 7, 1], [65, 0, 127, 128]), ([9, 7, 1, 0], [64, 62, 63, 0]))
        if query_tile == 2:
            cases = (([2], [62]), ([2, 1], [63, 64]),
                     ([1, 0, 2, 1], [65, 0, 127, 128]), ([2, 2, 0, 1], [64, 62, 63, 0]))
        counts, bodies = [], []
        addresses = [tensor.data_ptr() for tensor in (query, cache, storage, starts, seq_lens, table)]
        for lengths, contexts in cases:
            sequences, tokens = len(lengths), sum(lengths)
            starts[:sequences + 1].copy_(torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32))
            seq_lens[:sequences].copy_(torch.tensor([q + c for q, c in zip(lengths, contexts)], device="cuda", dtype=torch.int32))
            storage.fill_(137)
            output = storage[:tokens]
            assert paged_gqa_prefill(query[:tokens], k, v, output, starts[:sequences + 1], seq_lens[:sequences], table[:sequences], 128 ** -.5, query_tile=query_tile) is output
            expected = explicit_reference(query[:tokens], cache, starts[:sequences + 1], seq_lens[:sequences], table[:sequences])
            torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
            assert torch.equal(query, original_query) and torch.equal(cache, original_cache)
            assert torch.all(storage[tokens:] == 137)
            kernels = [kernel for device in module.iluvatar_paged_gqa_prefill_kernel.device_caches.values() for kernel in device[0].values()]
            fingerprints = tuple(sorted(hashlib.sha256(kernel.asm["ttir"].encode()).hexdigest() for kernel in kernels))
            count = len(kernels)
            counts.append(count)
            bodies.append(fingerprints)
            assert addresses == [tensor.data_ptr() for tensor in (query, cache, storage, starts, seq_lens, table)]
            print(json.dumps({"prefill_runtime_count": "PASS", "query_tile": query_tile, "page": page, "TOKENS": tokens, "SEQS": sequences, "compiled_bodies": count, "compiled_body_ttir_sha256": fingerprints, "fixed_storage_addresses": addresses, "max_abs_fp32": (output.float() - expected).abs().max().item()}), flush=True)
        assert len(set(counts)) == 1, ("counts caused new compiled bodies", counts)
        assert len(set(bodies)) == 1, ("counts changed compiled body", bodies)
        query.zero_()
        cache[:, 0].zero_()
        cache[:, 1].fill_(1)
        output = storage[:tokens]
        paged_gqa_prefill(query[:tokens], k, v, output, starts[:sequences + 1], seq_lens[:sequences], table[:sequences], 128 ** -.5, query_tile=query_tile)
        torch.testing.assert_close(output, torch.ones_like(output), atol=2e-2, rtol=2e-2)
        assert torch.all(storage[tokens:] == 137)
        print(json.dumps({"prefill_runtime_constant_value": "PASS", "query_tile": query_tile}), flush=True)
    compiled_resources(module.iluvatar_paged_gqa_prefill_kernel, "runtime TOKENS/SEQS")

def prefill_p64h_paired(frozen_fused_directory, candidate_label="p64h"):
    package = types.ModuleType("_paged_gqa_p1_control")
    package.__path__ = [frozen_fused_directory]
    sys.modules[package.__name__] = package
    control_module = importlib.import_module(package.__name__ + ".paged_gqa_prefill")
    for lengths, contexts in (([2048], [0]), ([128], [8192]),
                              ([1] * 7 + [1024], [8192] * 7 + [3072])):
        query, cache, storage, starts, seq_lens, table = make_case(lengths, contexts)
        output = storage[:query.shape[0]]
        k, v = cache.unbind(1)
        reference = lambda: native(query, cache, output, starts, seq_lens, table, max(lengths), max(q + c for q, c in zip(lengths, contexts)))
        original = lambda: control_module.paged_gqa_prefill(query, k, v, output, starts, seq_lens, table, 128 ** -.5)
        own = lambda: paged_gqa_prefill(query, k, v, output, starts, seq_lens, table, 128 ** -.5)
        expected = explicit_reference(query, cache, starts, seq_lens, table)
        for operation in (original, own):
            operation()
            torch.cuda.synchronize()
            torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
        results = {"native_2d": timed(reference), "p1_before": timed(original),
                   candidate_label: timed(own), "p1_restore": timed(original)}
        print(json.dumps({"prefill_p64h_paired": "diagnostic only, not end-to-end score", "candidate_label": candidate_label,
                          "lengths": lengths, "contexts": contexts,
                          "control_module_file": control_module.__file__,
                          "control_helper_module": control_module._paged_gqa_tile_update.__module__,
                          "control_helper_file": sys.modules[control_module._paged_gqa_tile_update.__module__].__file__,
                          "candidate_module_file": sys.modules[paged_gqa_prefill.__module__].__file__,
                          "candidate_helper_file": sys.modules[paged_gqa_prefill.__module__].__file__,
                          "timing": results}), flush=True)
    compiled_resources(control_module.iluvatar_paged_gqa_prefill_kernel, "frozen control "+frozen_fused_directory)
    compiled_resources(sys.modules[paged_gqa_prefill.__module__].iluvatar_paged_gqa_prefill_kernel, candidate_label)


def decode_t64_paired(frozen_fused_directory, label="t64"):
    package = types.ModuleType("_paged_gqa_d1_control")
    package.__path__ = [frozen_fused_directory]
    sys.modules[package.__name__] = package
    control_module = importlib.import_module(package.__name__ + ".paged_gqa_decode")
    control_helper = importlib.import_module(package.__name__ + ".paged_gqa_prefill")
    control = control_module.paged_gqa_decode
    for sequences, keys in ((64, 4097), (16, 16385), (64, 65)):
        query, cache, storage, starts, seq_lens, table = make_case([1] * sequences, [keys - 1] * sequences)
        output, workspace = storage[:sequences], decode_workspaces(sequences)
        k, v = cache.unbind(1)
        original = lambda: control(query, k, v, output, starts, seq_lens, table, 128 ** -.5, *workspace)
        own = lambda: paged_gqa_decode(query, k, v, output, starts, seq_lens, table, 128 ** -.5, *workspace)
        control_graph, own_graph = capture_operation(original), capture_operation(own)
        expected = explicit_reference(query, cache, starts, seq_lens, table)
        for graph in (control_graph, own_graph):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
        results = {"d1_before": timed(original), label: timed(own), "d1_restore": timed(original)}
        graph_results = {name: timed(lambda: [graph.replay() for _ in range(32)])
                         for name, graph in (("d1_before", control_graph), (label, own_graph),
                                             ("d1_restore", control_graph))}
        print(json.dumps({"decode_paired_candidate": label, "decode_t64_paired": "diagnostic only; graph event values cover 32 replays",
                          "sequences": sequences, "keys": keys, "segments": 16,
                          "control_source": frozen_fused_directory,
                          "control_decode_file": control_module.__file__,
                          "control_helper_file": control_helper.__file__,
                          "eager_timing": results, "graph_timing": graph_results}), flush=True)
    compiled_resources(control_module.iluvatar_paged_gqa_decode_partial_kernel, "frozen D1")
    compiled_resources(sys.modules[paged_gqa_decode.__module__].iluvatar_paged_gqa_decode_partial_kernel, label)


def test_decode_adaptive_graph():
    query, cache, storage, starts, seq_lens, table = make_case([1] * 64, [4096] * 64, 48, True)
    output, workspace = storage[:64], decode_workspaces(64)
    k, v = cache.unbind(1)
    query_before, cache_before = query.clone(), cache.clone()
    tensors = (query, cache, output, starts, seq_lens, table, *workspace)
    addresses = [tensor.data_ptr() for tensor in tensors]
    graph = capture_operation(lambda: paged_gqa_decode(query, k, v, output, starts, seq_lens, table, 128 ** -.5, *workspace))
    stages = [( [1] * 64, [65] * 64),
              ([1] * 64, [65, 127, 128, 129, 1025, 4097] * 10 + [65, 127, 128, 129]),
              ([1] * 64, [4097] * 64),
              ([1 if i % 3 else 0 for i in range(64)], [65] * 64),
              ([0] * 64, [0] * 64), ([1] + [0] * 63, [0] * 64)]
    for active, sizes in stages:
        sizes = [size if valid else 0 for size, valid in zip(sizes, active)]
        starts.copy_(torch.tensor([0, *torch.tensor(active).cumsum(0).tolist()], device="cuda", dtype=torch.int32))
        seq_lens.copy_(torch.tensor(sizes, device="cuda", dtype=torch.int32))
        storage.fill_(137)
        graph.replay()
        torch.cuda.synchronize()
        count = sum(active)
        expected = explicit_reference(query, cache, starts, seq_lens, table)
        torch.testing.assert_close(output[:count].float(), expected[:count], atol=2e-2, rtol=2e-2)
        assert torch.all(storage[count:] == 137)
        for seq, (valid, size) in enumerate(zip(active, sizes)):
            if not valid:
                continue
            tile = 32 if size < 128 else 64
            tiles = math.ceil(size / (16 * tile))
            for split in range(16):
                if split * tiles * tile >= size:
                    assert torch.isneginf(workspace[1][seq, :, split]).all()
                    assert not workspace[0][seq, :, split].any() and not workspace[2][seq, :, split].any()
        assert addresses == [tensor.data_ptr() for tensor in tensors]
        assert torch.equal(query, query_before) and torch.equal(cache, cache_before)
        print(json.dumps({"decode_adaptive_graph": "PASS", "active": count, "sizes": sorted(set(sizes)), "holes": active.count(0)}), flush=True)


def decode_microbench():
    from vllm.v1.attention.ops.triton_unified_attention import is_batch_invariant

    assert not is_batch_invariant, "native control must enter the original 3D path"
    for sequences, keys in ((64, 4097), (16, 16385)):
        query, cache, storage, starts, seq_lens, table = make_case([1] * sequences, [keys - 1] * sequences)
        output, workspace = storage[:sequences], decode_workspaces(sequences)
        k, v = cache.unbind(1)
        original = lambda: native_decode_3d(query, cache, output, starts, seq_lens, table, workspace, keys)
        own = lambda: paged_gqa_decode(query, k, v, output, starts, seq_lens, table, 128 ** -.5, *workspace)
        native_graph, own_graph = capture_operation(original), capture_operation(own)
        own_graph.replay()
        torch.cuda.synchronize()
        expected = explicit_reference(query, cache, starts, seq_lens, table)
        torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
        results = {"native_3d_before": timed(original), "candidate": timed(own), "native_3d_restore": timed(original)}
        graph_results = {}
        for name, graph in (("native_3d_before", native_graph), ("candidate", own_graph), ("native_3d_restore", native_graph)):
            graph_results[name] = timed(lambda: [graph.replay() for _ in range(32)])
        print(json.dumps({"decode_micro": "diagnostic only; graph event values cover 32 replays",
                          "sequences": sequences, "keys": keys, "segments": 16,
                          "native_workspaces": [list(t.shape) for t in workspace],
                          "native_main_grid": [sequences // 2 + sequences, 2, 16],
                          "own_main_grid": [sequences, 2, 16],
                          "eager_timing": results, "graph_timing": graph_results}), flush=True)
        if sequences == 64:
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as profile:
                    original()
                    native_graph.replay()
                    torch.cuda.synchronize()
                profile.export_chrome_trace("decode_native_trace.json")
                print(json.dumps({"native_3d_profile": "captured", "operators": [event.key for event in profile.key_averages()]}), flush=True)
            except RuntimeError as error:
                print(json.dumps({"native_3d_profile": "unavailable", "error": str(error)}), flush=True)


if __name__ == "__main__":
    assert flag_gems.vendor_name == "iluvatar"
    if "--prefill-runtime-paired" in sys.argv:
        prefill_p64h_paired(sys.argv[sys.argv.index("--prefill-runtime-paired") + 1], "rseq")
    elif "--prefill-runtime-counts" in sys.argv:
        test_prefill_runtime_counts()
    elif "--decode-adaptive-paired" in sys.argv:
        decode_t64_paired(sys.argv[sys.argv.index("--decode-adaptive-paired") + 1], "adaptive")
    elif "--decode-adaptive" in sys.argv:
        for keys in (127, 128, 129):
            check_decode_case([1] * 16, [keys - 1] * 16, 16, True)
        check_decode_case([1] * 6, [64, 126, 127, 128, 1024, 4096], 48, True)
        test_decode_adaptive_graph()
    elif "--decode-shared-default-graph" in sys.argv:
        test_decode_shared_default_graph()
    elif "--prefill-p64h-paired" in sys.argv:
        prefill_p64h_paired(sys.argv[sys.argv.index("--prefill-p64h-paired") + 1])
    elif "--prefill-p64h" in sys.argv:
        test_prefill_p64h()
    elif "--decode-t64-paired" in sys.argv:
        decode_t64_paired(sys.argv[sys.argv.index("--decode-t64-paired") + 1])
    elif "--decode-native" in sys.argv:
        decode_microbench()
    elif "--decode" in sys.argv:
        test_decode_constant_value_identity()
        for case in DECODE_CASES:
            check_decode_case(*case)
        test_decode_dynamic_graph()
        test_t64_prefill_default()
    else:
        test_constant_value_identity()
        for case in CASES:
            check_case(*case)
        microbench()
