# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License").
"""Process-local NT config reuse without synthetic exact-shape measurements."""

from functools import lru_cache
from threading import Lock

import torch

from flag_gems import runtime
from flag_gems.utils.libentry import LibTuner, LibTunerRunMode


class _GeometryRequest(tuple):
    """Immutable (original exact key, all-scalar geometry identity)."""


def _config_fingerprint(config):
    fields = tuple(sorted(config.all_kwargs().items()))
    if any(type(value) not in (int, float, bool, str, type(None)) for _, value in fields):
        raise TypeError("Unknown config metadata")
    return fields


class _GeometryCache:
    def __init__(self, exact):
        self.exact = exact
        self._memo = {}
        self._lock = Lock()
        self.reuse_count = 0

    def _seed(self, geometry, config):
        try:
            fingerprint = _config_fingerprint(config)
        except (AttributeError, TypeError):
            return
        if not any(fingerprint == item[0] for item in geometry[-2]):
            return
        with self._lock:
            if geometry in self._memo:
                previous = self._memo[geometry]
                if previous is not None and previous[0] != fingerprint:
                    self._memo[geometry] = None  # A conflicting class stays disabled.
            elif len(self._memo) < 256:
                self._memo[geometry] = (fingerprint, config)

    def get(self, key):
        if type(key) is not _GeometryRequest:
            return self.exact.get(key)
        raw, geometry = key
        config = self.exact.get(raw)  # Never hold the memo lock across SQL.
        if config is not None:
            self._seed(geometry, config)
            return config
        with self._lock:
            item = self._memo.get(geometry)
            if item is not None:
                self.reuse_count += 1
        if item is None:
            return None
        fingerprint, config = item
        try:
            unchanged = _config_fingerprint(config) == fingerprint
        except (AttributeError, TypeError):
            unchanged = False
        if not unchanged:
            return None
        return config

    def set(self, key, config):
        if type(key) is not _GeometryRequest:
            return self.exact.set(key, config)
        raw, geometry = key
        result = self.exact.set(raw, config)  # Only real policy results reach set.
        self._seed(geometry, config)
        return result

    def __contains__(self, key):
        return self.get(key) is not None

    def __getitem__(self, key):
        config = self.get(key)
        if config is None:
            raise KeyError(key)
        return config

    def __setitem__(self, key, config):
        self.set(key, config)


@lru_cache(maxsize=8)
def _device_name(index):
    return torch.cuda.get_device_name(index)


class MetaXNTGeometryTuner(LibTuner.get("default")):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = _GeometryCache(self.cache)

    def _set_configs_and_strategy(self, configs, strategy, *, mode=None):
        super()._set_configs_and_strategy(configs, strategy, mode=mode)
        self.cache = _GeometryCache(self.cache)

    def get_key(self, args):
        exact = super().get_key(args)
        try:
            geometry = self._geometry(args, exact)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            geometry = None
        return exact if geometry is None else _GeometryRequest((exact, geometry))

    def _geometry(self, args, exact):
        if (
            not isinstance(self.cache, _GeometryCache)
            or runtime.device.vendor_name != "metax"
            or self.__name__ != "mm_kernel_nt"
            or tuple(self.keys) != ("M", "N", "K")
            or self._flagtune_mode != runtime.TuningMode.DEFAULT
            or self._run_mode != LibTunerRunMode.NORMAL
            or self.perf_model is not None
            or self.early_config_prune is None
        ):
            return None
        m, n, k = (args[name] for name in ("M", "N", "K"))
        if any(type(value) is not int or value <= 0 for value in (m, n, k)):
            return None
        if m < 1024 or n < 128 or args.get("GROUP_M") != 8:
            return None
        if exact[:3] != (m, n, k):
            return None
        a, b, c = (args[name] for name in ("A", "B", "C"))
        if not all(isinstance(tensor, torch.Tensor) for tensor in (a, b, c)):
            return None
        if not all(tensor.dtype == torch.bfloat16 for tensor in (a, b, c)):
            return None
        if a.device.type != "cuda" or not (a.device == b.device == c.device):
            return None
        index = a.device.index
        name = _device_name(index)
        if "C500" not in name.upper():
            return None
        if (
            a.shape != (m, k) or b.shape != (k, n) or c.shape != (m, n)
            or a.stride() != (k, 1) or b.stride() != (1, k)
            or c.stride() != (n, 1)
            or max(a.numel(), b.numel(), c.numel()) >= 2**31
        ):
            return None
        configs = list(self.early_config_prune(self.configs, args))
        if not configs:
            return None
        fingerprints = tuple(
            (_config_fingerprint(config), id(config.pre_hook) if config.pre_hook else 0)
            for config in configs
        )
        vector = []
        for config in configs:
            bm, bn, bk = (config.kwargs[key] for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
            if any(type(block) is not int or block <= 0 for block in (bm, bn, bk)):
                return None
            vector.append(((m + bm - 1) // bm, m % bm == 0,
                           n % bn == 0, k % bk == 0))
        return (
            "metax_nt_geometry_v1", index, name, self.config_table_name,
            self.kernel_hash, self.configs_hash, tuple(self._benchmark_protocol),
            id(self.early_config_prune), n, k,
            tuple(str(tensor.dtype) for tensor in (a, b, c)),
            (a.stride(), b.stride(), c.stride()), args["GROUP_M"],
            fingerprints, tuple(vector),
        )
