# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.sort
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize(
    "hiddensize", [1, 256, 2048, 9333, 65536, 32768, 128 * 1024, 256 * 1024]
)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
def test_sort(batch_size, hiddensize, descending, dtype, dim):
    if dtype in utils.BOOL_TYPES:
        y = torch.randint(
            0, 2, (batch_size, hiddensize), dtype=dtype, device=flag_gems.device
        )
    elif dtype in utils.ALL_INT_DTYPES:
        min_v, max_v = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        y = torch.randint(
            min_v, max_v, (batch_size, hiddensize), dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    else:
        y = torch.randn((batch_size, hiddensize), dtype=dtype, device=flag_gems.device)

    ref_y = utils.to_reference(y)
    # we only implement stable sort, non-stable sort is undefined
    ref_value, ref_index = torch.sort(
        ref_y, dim=dim, stable=True, descending=descending
    )

    with flag_gems.use_gems():
        res_value, res_index = torch.sort(
            y, dim=dim, stable=True, descending=descending
        )

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)


@pytest.mark.sort_stable
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize(
    "hiddensize", [1, 256, 2048, 9333, 65536, 32768, 128 * 1024, 256 * 1024]
)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
def test_sort_stable(batch_size, hiddensize, descending, dtype, dim):
    if dtype in utils.BOOL_TYPES:
        y = torch.randint(
            0, 2, (batch_size, hiddensize), dtype=dtype, device=flag_gems.device
        )
    elif dtype in utils.ALL_INT_DTYPES:
        min_v, max_v = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        y = torch.randint(
            min_v, max_v, (batch_size, hiddensize), dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    else:
        y = torch.randn((batch_size, hiddensize), dtype=dtype, device=flag_gems.device)

    ref_y = utils.to_reference(y)
    ref_value, ref_index = torch.sort(
        ref_y, dim=dim, stable=True, descending=descending
    )

    with flag_gems.use_gems():
        res_value, res_index = torch.sort(
            y, dim=dim, stable=True, descending=descending
        )

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)


def _metax_radix_sort_candidate_and_original(y, descending):
    sort_module = importlib.import_module("flag_gems.ops.sort")
    original_selector = sort_module._select_global_hist_kernel
    with flag_gems.use_gems():
        candidate = sort_module.radix_sort(y, k_bits=4, descending=descending)
        sort_module._select_global_hist_kernel = (
            lambda dtype, k_bits, n: sort_module.compute_global_hist_kernel
        )
        try:
            original = sort_module.radix_sort(y, k_bits=4, descending=descending)
        finally:
            sort_module._select_global_hist_kernel = original_selector
    return candidate, original


@pytest.mark.sort
@pytest.mark.skipif(
    flag_gems.vendor_name != "metax",
    reason="The FP32 radix histogram specialization is MetaX-only",
)
@pytest.mark.parametrize(
    "hiddensize", [1023, 1024, 1025, 8191, 8192, 8193, 130560]
)
@pytest.mark.parametrize("descending", [False, True])
def test_sort_metax_fp32_radix_histogram_boundaries(hiddensize, descending):
    """Check the specialized path against both the generic path and torch."""
    sort_module = importlib.import_module("flag_gems.ops.sort")
    torch.manual_seed(20261002)
    y = torch.randn((1, hiddensize), dtype=torch.float32, device=flag_gems.device)
    ref_value, ref_index = torch.sort(
        utils.to_reference(y), dim=-1, stable=True, descending=descending
    )
    expected_kernel = (
        sort_module.compute_global_hist_kernel
        if hiddensize < sort_module.GLOBAL_HISTOGRAM_MIN_N
        else sort_module.compute_global_hist_kernel_metax_fp32_k4
    )
    assert (
        sort_module._select_global_hist_kernel(torch.float32, 4, hiddensize)
        is expected_kernel
    )

    (candidate_value, candidate_index), (original_value, original_index) = (
        _metax_radix_sort_candidate_and_original(y, descending)
    )

    candidate_bits = candidate_value.view(torch.int32)
    original_bits = original_value.view(torch.int32)
    ref_bits = ref_value.view(torch.int32)
    utils.gems_assert_equal(candidate_bits, original_bits)
    utils.gems_assert_equal(candidate_index, original_index)
    utils.gems_assert_equal(candidate_bits, ref_bits)
    utils.gems_assert_equal(candidate_index, ref_index)


@pytest.mark.sort
@pytest.mark.skipif(
    flag_gems.vendor_name != "metax",
    reason="The FP32 radix histogram specialization is MetaX-only",
)
@pytest.mark.parametrize(
    "hiddensize", [1023, 1024, 1025, 8191, 8192, 8193, 130560]
)
@pytest.mark.parametrize("descending", [False, True])
def test_sort_metax_fp32_radix_histogram_special_bits(hiddensize, descending):
    """Preserve special FP32 payloads relative to the generic radix path."""
    torch.manual_seed(20261002)
    y = torch.randn((1, hiddensize), dtype=torch.float32, device=flag_gems.device)
    special_bits = torch.tensor(
        [
            0,
            -2147483648,
            2139095040,
            -8388608,
            2143289345,
            -4194303,
            1065353216,
            -1082130432,
        ],
        dtype=torch.int32,
    )
    y[0, : special_bits.numel()] = special_bits.view(torch.float32).to(
        flag_gems.device
    )

    (candidate_value, candidate_index), (original_value, original_index) = (
        _metax_radix_sort_candidate_and_original(y, descending)
    )

    utils.gems_assert_equal(
        candidate_value.view(torch.int32), original_value.view(torch.int32)
    )
    utils.gems_assert_equal(candidate_index, original_index)
