# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DS V3.2 fused_norm_rope Triton vs Helion.

This is intentionally a source-file benchmark: it stubs the small vLLM modules
needed by the two kernel files so the benchmark can run without building the
compiled vLLM extension.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _install_stubs() -> None:
    vllm_mod = types.ModuleType("vllm")
    triton_utils_mod = types.ModuleType("vllm.triton_utils")
    triton_utils_mod.triton = triton
    triton_utils_mod.tl = tl

    class _CurrentPlatform:
        @staticmethod
        def fp8_dtype():
            return torch.float8_e4m3fn

    platforms_mod = types.ModuleType("vllm.platforms")
    platforms_mod.current_platform = _CurrentPlatform()

    import_utils_mod = types.ModuleType("vllm.utils.import_utils")
    import_utils_mod.has_helion = lambda: True
    utils_mod = types.ModuleType("vllm.utils")
    utils_mod.import_utils = import_utils_mod

    sys.modules.setdefault("vllm", vllm_mod)
    sys.modules["vllm.triton_utils"] = triton_utils_mod
    sys.modules["vllm.platforms"] = platforms_mod
    sys.modules.setdefault("vllm.utils", utils_mod)
    sys.modules["vllm.utils.import_utils"] = import_utils_mod


def _make_inputs(num_tokens: int, topk: int = 2048):
    device = "cuda"
    dtype = torch.bfloat16
    block_size = 64
    num_blocks = triton.cdiv(num_tokens, block_size) + 1
    max_pos = max(4096, num_tokens + 16)

    positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
    q_c = torch.randn(num_tokens, 1536, device=device, dtype=dtype)
    q_w = torch.randn(1536, device=device, dtype=dtype)
    kv_c = torch.randn(num_tokens, 512, device=device, dtype=dtype)
    kv_w = torch.randn(512, device=device, dtype=dtype)
    k_pe = torch.randn(num_tokens, 64, device=device, dtype=dtype)
    k_rope_cache = torch.randn(max_pos, 64, device=device, dtype=dtype)
    index_k = torch.randn(num_tokens, 128, device=device, dtype=dtype)
    index_w = torch.randn(128, device=device, dtype=dtype)
    index_b = torch.randn(128, device=device, dtype=dtype)
    index_rope_cache = torch.randn(max_pos, 128, device=device, dtype=dtype)
    topk_buf = torch.empty(num_tokens, topk, device=device, dtype=torch.int32)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)
    indexer_k_cache = torch.empty(
        num_blocks, block_size, 132, device=device, dtype=torch.uint8
    )
    mla_kv_cache = torch.empty(
        num_blocks, block_size, 576, device=device, dtype=dtype
    )
    mla_k_scale = torch.ones(1, device=device, dtype=torch.float32)

    return {
        "positions": positions,
        "q_c": q_c,
        "q_w": q_w,
        "kv_c": kv_c,
        "kv_w": kv_w,
        "k_pe": k_pe,
        "k_rope_cache": k_rope_cache,
        "index_k": index_k,
        "index_w": index_w,
        "index_b": index_b,
        "index_rope_cache": index_rope_cache,
        "topk_buf": topk_buf,
        "slot_mapping": slot_mapping,
        "indexer_k_cache": indexer_k_cache,
        "mla_kv_cache": mla_kv_cache,
        "mla_k_scale": mla_k_scale,
    }


def _clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in inputs.items()}


def _run_triton(triton_kernels, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return triton_kernels.fused_norm_rope(
        inputs["positions"],
        inputs["q_c"],
        inputs["q_w"],
        1e-6,
        inputs["kv_c"],
        inputs["kv_w"],
        1e-6,
        inputs["k_pe"],
        inputs["k_rope_cache"],
        inputs["index_k"],
        inputs["index_w"],
        inputs["index_b"],
        1e-6,
        inputs["index_rope_cache"],
        inputs["topk_buf"],
        slot_mapping=inputs["slot_mapping"],
        indexer_k_cache=inputs["indexer_k_cache"],
        mla_kv_cache=inputs["mla_kv_cache"],
        mla_kv_cache_dtype="auto",
        mla_k_scale=inputs["mla_k_scale"],
    )


def _run_helion(helion_kernels, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return helion_kernels.fused_norm_rope_helion(
        inputs["positions"],
        inputs["q_c"],
        inputs["q_w"],
        1e-6,
        inputs["kv_c"],
        inputs["kv_w"],
        1e-6,
        inputs["k_pe"],
        inputs["k_rope_cache"],
        inputs["index_k"],
        inputs["index_w"],
        inputs["index_b"],
        1e-6,
        inputs["index_rope_cache"],
        inputs["topk_buf"],
        slot_mapping=inputs["slot_mapping"],
        indexer_k_cache=inputs["indexer_k_cache"],
        mla_kv_cache=inputs["mla_kv_cache"],
        mla_kv_cache_dtype="auto",
        mla_k_scale=inputs["mla_k_scale"],
    )


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()


def _check_correctness(triton_kernels, helion_kernels, base_inputs) -> str:
    triton_inputs = _clone_inputs(base_inputs)
    helion_inputs = _clone_inputs(base_inputs)

    q_triton = _run_triton(triton_kernels, triton_inputs)
    q_helion = _run_helion(helion_kernels, helion_inputs)
    torch.cuda.synchronize()
    num_tokens = base_inputs["positions"].shape[0]
    triton_mla_slots = triton_inputs["mla_kv_cache"].view(-1, 576)[:num_tokens]
    helion_mla_slots = helion_inputs["mla_kv_cache"].view(-1, 576)[:num_tokens]

    checks = [
        (
            "q",
            torch.allclose(q_triton, q_helion, rtol=1e-2, atol=1e-2),
            _max_diff(q_triton, q_helion),
        ),
        (
            "mla",
            torch.allclose(
                triton_mla_slots,
                helion_mla_slots,
                rtol=1e-2,
                atol=1e-2,
            ),
            _max_diff(triton_mla_slots, helion_mla_slots),
        ),
        (
            "topk",
            torch.equal(triton_inputs["topk_buf"], helion_inputs["topk_buf"]),
            0.0,
        ),
        (
            "indexer",
            torch.equal(
                triton_inputs["indexer_k_cache"], helion_inputs["indexer_k_cache"]
            ),
            0.0,
        ),
    ]
    failed = [f"{name}:max_diff={diff:.6g}" for name, ok, diff in checks if not ok]
    if failed:
        return "FAIL(" + ";".join(failed) + ")"
    return "OK"


def _bench(fn, warmup: int = 10, repeat: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def main() -> None:
    _install_stubs()
    triton_kernels = _load_module(
        "ds32_triton_kernels",
        ROOT / "vllm/model_executor/specialized_models/deepseek_v3_2_nvfp4/kernels.py",
    )
    helion_kernels = _load_module(
        "ds32_helion_kernels",
        ROOT
        / "vllm/model_executor/specialized_models/deepseek_v3_2_nvfp4/helion_kernels.py",
    )

    print("GPU:", torch.cuda.get_device_name())
    print("tokens,correctness,triton_ms,helion_ms,helion/triton")

    for num_tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        base_inputs = _make_inputs(num_tokens)
        correctness = _check_correctness(triton_kernels, helion_kernels, base_inputs)
        inputs = _clone_inputs(base_inputs)

        def run_triton():
            return _run_triton(triton_kernels, inputs)

        def run_helion():
            return _run_helion(helion_kernels, inputs)

        triton_ms = _bench(run_triton)
        try:
            helion_ms = _bench(run_helion)
            ratio = helion_ms / triton_ms
            print(
                f"{num_tokens},{correctness},{triton_ms:.6f},"
                f"{helion_ms:.6f},{ratio:.3f}"
            )
        except Exception as e:
            print(
                f"{num_tokens},{correctness},{triton_ms:.6f},"
                f"ERROR,{type(e).__name__}: {e}"
            )


if __name__ == "__main__":
    main()
