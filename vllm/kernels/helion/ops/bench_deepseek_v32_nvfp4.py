#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone benchmark for DeepSeek V3.2 NVFP4 fused kernels.

This compares the Triton and Helion implementations used by the specialized
DeepSeek V3.2 NVFP4 vLLM path at serving-shaped token counts. It reports both
eager launch timing and CUDA graph replay timing for:

- fused_norm_rope with BF16 or FP8 MLA KV cache
- fused_q with BF16 or FP8 MQA output
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

import torch
import triton

from vllm.model_executor.specialized_models.deepseek_v3_2_nvfp4 import (
    helion_kernels,
)
from vllm.model_executor.specialized_models.deepseek_v3_2_nvfp4 import (
    kernels as triton_kernels,
)


def make_norm_rope_inputs(
    num_tokens: int,
    topk: int,
    mla_kv_cache_dtype: str,
) -> dict[str, object]:
    device = "cuda"
    dtype = torch.bfloat16
    block_size = 64
    num_blocks = triton.cdiv(num_tokens, block_size) + 1
    max_pos = max(4096, num_tokens + 16)
    mla_cache_dtype = torch.uint8 if mla_kv_cache_dtype != "auto" else dtype

    return {
        "positions": torch.arange(num_tokens, device=device, dtype=torch.int64),
        "q_c": torch.randn(num_tokens, 1536, device=device, dtype=dtype),
        "q_w": torch.randn(1536, device=device, dtype=dtype),
        "kv_c": torch.randn(num_tokens, 512, device=device, dtype=dtype),
        "kv_w": torch.randn(512, device=device, dtype=dtype),
        "k_pe": torch.randn(num_tokens, 64, device=device, dtype=dtype),
        "k_rope_cache": torch.randn(max_pos, 64, device=device, dtype=dtype),
        "index_k": torch.randn(num_tokens, 128, device=device, dtype=dtype),
        "index_w": torch.randn(128, device=device, dtype=dtype),
        "index_b": torch.randn(128, device=device, dtype=dtype),
        "index_rope_cache": torch.randn(max_pos, 128, device=device, dtype=dtype),
        "topk_buf": torch.empty(num_tokens, topk, device=device, dtype=torch.int32),
        "slot_mapping": torch.arange(num_tokens, device=device, dtype=torch.int64),
        "indexer_k_cache": torch.empty(
            num_blocks, block_size, 132, device=device, dtype=torch.uint8
        ),
        "mla_kv_cache": torch.empty(
            num_blocks, block_size, 576, device=device, dtype=mla_cache_dtype
        ),
        "mla_kv_cache_dtype": mla_kv_cache_dtype,
        "mla_k_scale": torch.ones(1, device=device, dtype=torch.float32),
    }


def make_fused_q_inputs(num_tokens: int) -> dict[str, object]:
    device = "cuda"
    dtype = torch.bfloat16
    max_pos = max(4096, num_tokens + 16)
    num_q_heads = 128
    num_index_q_heads = 256
    q_pe_dim = 64
    index_q_dim = 128
    ql_nope_dim = 512

    return {
        "positions": torch.arange(num_tokens, device=device, dtype=torch.int64),
        "q_pe": torch.randn(
            num_tokens, num_q_heads, q_pe_dim, device=device, dtype=dtype
        ),
        "q_pe_cache": torch.randn(max_pos, q_pe_dim, device=device, dtype=dtype),
        "index_q": torch.randn(
            num_tokens, num_index_q_heads, index_q_dim, device=device, dtype=dtype
        ),
        "index_q_cache": torch.randn(max_pos, index_q_dim, device=device, dtype=dtype),
        "ql_nope": torch.randn(
            num_tokens, num_q_heads, ql_nope_dim, device=device, dtype=dtype
        ),
        "q_scale": torch.ones(1, device=device, dtype=torch.float32),
        "index_weights": torch.randn(
            num_tokens, num_index_q_heads, device=device, dtype=dtype
        ),
        "index_weights_softmax_scale": index_q_dim**-0.5,
        "index_weights_head_scale": num_index_q_heads**-0.5,
    }


def clone_inputs(inputs: dict[str, object]) -> dict[str, object]:
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def run_norm_rope_triton(inputs: dict[str, object]) -> torch.Tensor:
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
        mla_kv_cache_dtype=inputs["mla_kv_cache_dtype"],
        mla_k_scale=inputs["mla_k_scale"],
    )


def run_norm_rope_helion(
    inputs: dict[str, object],
    mla_k_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    if mla_k_scale is None:
        mla_k_scale = inputs["mla_k_scale"]
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
        mla_kv_cache_dtype=inputs["mla_kv_cache_dtype"],
        mla_k_scale=mla_k_scale,
    )


def _mqa_q_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp8":
        return torch.float8_e4m3fn
    raise ValueError(f"Unsupported mqa_q dtype: {dtype_name}")


def run_fused_q_triton(
    inputs: dict[str, object],
    mqa_q_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    return triton_kernels.fused_q(
        inputs["positions"],
        inputs["q_pe"],
        inputs["q_pe_cache"],
        inputs["index_q"],
        inputs["index_q_cache"],
        inputs["ql_nope"],
        inputs["q_scale"],
        inputs["index_weights"],
        inputs["index_weights_softmax_scale"],
        inputs["index_weights_head_scale"],
        mqa_q_dtype,
    )


def run_fused_q_helion(
    inputs: dict[str, object],
    mqa_q_dtype: torch.dtype,
    q_scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    if q_scale is None:
        q_scale = (
            float(inputs["q_scale"].item())
            if mqa_q_dtype == torch.float8_e4m3fn
            else 1.0
        )
    return helion_kernels.fused_q_helion(
        inputs["positions"],
        inputs["q_pe"],
        inputs["q_pe_cache"],
        inputs["index_q"],
        inputs["index_q_cache"],
        inputs["ql_nope"],
        q_scale,
        inputs["index_weights"],
        inputs["index_weights_softmax_scale"],
        inputs["index_weights_head_scale"],
        mqa_q_dtype,
    )


def run_fused_q_helion_graph(
    inputs: dict[str, object],
    mqa_q_dtype: torch.dtype,
    q_scale: float,
) -> tuple[torch.Tensor, ...]:
    return helion_kernels.fused_q_helion(
        inputs["positions"],
        inputs["q_pe"],
        inputs["q_pe_cache"],
        inputs["index_q"],
        inputs["index_q_cache"],
        inputs["ql_nope"],
        q_scale,
        inputs["index_weights"],
        inputs["index_weights_softmax_scale"],
        inputs["index_weights_head_scale"],
        mqa_q_dtype,
    )


def bench(fn: Callable[[], object], warmup: int, repeat: int) -> float:
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


def bench_cuda_graph(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()


def _norm_cache_label(mla_kv_cache_dtype: str) -> str:
    return "fp8_mla" if mla_kv_cache_dtype != "auto" else "bf16_mla"


def check_norm_rope(
    num_tokens: int,
    topk: int,
    mla_kv_cache_dtype: str,
) -> str:
    base = make_norm_rope_inputs(num_tokens, topk, mla_kv_cache_dtype)
    triton_inputs = clone_inputs(base)
    helion_inputs = clone_inputs(base)
    q_t = run_norm_rope_triton(triton_inputs)
    q_h = run_norm_rope_helion(helion_inputs)
    torch.cuda.synchronize()

    mla_t = triton_inputs["mla_kv_cache"].view(-1, 576)[:num_tokens]
    mla_h = helion_inputs["mla_kv_cache"].view(-1, 576)[:num_tokens]
    if not torch.allclose(q_t, q_h, rtol=1e-2, atol=1e-2):
        return f"FAIL(q max_diff={max_diff(q_t, q_h):.6g})"
    if not torch.allclose(mla_t, mla_h, rtol=1e-2, atol=1e-2):
        return f"FAIL(mla max_diff={max_diff(mla_t, mla_h):.6g})"
    if not torch.equal(triton_inputs["topk_buf"], helion_inputs["topk_buf"]):
        return "FAIL(topk)"
    return "OK"


def check_fused_q(num_tokens: int, mqa_q_dtype: torch.dtype) -> str:
    base = make_fused_q_inputs(num_tokens)
    triton_out = run_fused_q_triton(clone_inputs(base), mqa_q_dtype)
    helion_out = run_fused_q_helion(clone_inputs(base), mqa_q_dtype)
    torch.cuda.synchronize()

    for name, lhs, rhs in zip(
        ["index_q_fp8", "index_weights", "mqa_q"], triton_out, helion_out
    ):
        if name == "index_q_fp8":
            ok = torch.equal(lhs, rhs)
        else:
            ok = torch.allclose(
                lhs.to(torch.float32),
                rhs.to(torch.float32),
                rtol=1e-2,
                atol=1e-2,
            )
        if not ok:
            return f"FAIL({name} max_diff={max_diff(lhs, rhs):.6g})"
    return "OK"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        default="8,1024",
        help="Comma-separated token counts to benchmark.",
    )
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--mqa-q-dtype",
        choices=("bf16", "fp8", "both"),
        default="bf16",
        help="MQA query output dtype to benchmark for fused_q.",
    )
    parser.add_argument(
        "--mla-kv-cache-dtype",
        choices=("bf16", "fp8", "both"),
        default="bf16",
        help="MLA KV cache dtype to benchmark for fused_norm_rope.",
    )
    args = parser.parse_args()

    print("gpu", torch.cuda.get_device_name())
    print("pid", os.getpid())
    print("tokens", args.tokens)
    print(
        "kernel,tokens,correctness,triton_ms,helion_ms,helion_over_triton,"
        "triton_graph_ms,helion_graph_ms,helion_graph_over_triton_graph"
    )

    norm_cache_dtypes = (
        ["auto", "fp8_e4m3"]
        if args.mla_kv_cache_dtype == "both"
        else ["fp8_e4m3" if args.mla_kv_cache_dtype == "fp8" else "auto"]
    )
    for norm_cache_dtype in norm_cache_dtypes:
        norm_label = _norm_cache_label(norm_cache_dtype)
        for token_s in args.tokens.split(","):
            num_tokens = int(token_s)
            correctness = check_norm_rope(num_tokens, args.topk, norm_cache_dtype)
            triton_inputs = make_norm_rope_inputs(
                num_tokens, args.topk, norm_cache_dtype
            )
            helion_inputs = make_norm_rope_inputs(
                num_tokens, args.topk, norm_cache_dtype
            )
            triton_ms = bench(
                lambda: run_norm_rope_triton(triton_inputs),
                args.warmup,
                args.repeat,
            )
            helion_ms = bench(
                lambda: run_norm_rope_helion(helion_inputs),
                args.warmup,
                args.repeat,
            )
            triton_graph_inputs = make_norm_rope_inputs(
                num_tokens, args.topk, norm_cache_dtype
            )
            helion_graph_inputs = make_norm_rope_inputs(
                num_tokens, args.topk, norm_cache_dtype
            )
            helion_graph_k_scale = (
                float(helion_graph_inputs["mla_k_scale"].item())
                if norm_cache_dtype != "auto"
                else 1.0
            )
            triton_graph_ms = bench_cuda_graph(
                lambda: run_norm_rope_triton(triton_graph_inputs),
                args.warmup,
                args.repeat,
            )
            helion_graph_ms = bench_cuda_graph(
                lambda: run_norm_rope_helion(
                    helion_graph_inputs, helion_graph_k_scale
                ),
                args.warmup,
                args.repeat,
            )
            print(
                f"fused_norm_rope_{norm_label},{num_tokens},{correctness},"
                f"{triton_ms:.6f},{helion_ms:.6f},{helion_ms / triton_ms:.3f},"
                f"{triton_graph_ms:.6f},{helion_graph_ms:.6f},"
                f"{helion_graph_ms / triton_graph_ms:.3f}"
            )

    dtype_names = ["bf16", "fp8"] if args.mqa_q_dtype == "both" else [args.mqa_q_dtype]
    for dtype_name in dtype_names:
        mqa_q_dtype = _mqa_q_dtype(dtype_name)
        for token_s in args.tokens.split(","):
            num_tokens = int(token_s)
            correctness = check_fused_q(num_tokens, mqa_q_dtype)
            triton_inputs = make_fused_q_inputs(num_tokens)
            helion_inputs = make_fused_q_inputs(num_tokens)
            triton_ms = bench(
                lambda: run_fused_q_triton(triton_inputs, mqa_q_dtype),
                args.warmup,
                args.repeat,
            )
            helion_ms = bench(
                lambda: run_fused_q_helion(helion_inputs, mqa_q_dtype),
                args.warmup,
                args.repeat,
            )
            triton_graph_inputs = make_fused_q_inputs(num_tokens)
            helion_graph_inputs = make_fused_q_inputs(num_tokens)
            helion_graph_q_scale = (
                float(helion_graph_inputs["q_scale"].item())
                if mqa_q_dtype == torch.float8_e4m3fn
                else 1.0
            )
            triton_graph_ms = bench_cuda_graph(
                lambda: run_fused_q_triton(triton_graph_inputs, mqa_q_dtype),
                args.warmup,
                args.repeat,
            )
            helion_graph_ms = bench_cuda_graph(
                lambda: run_fused_q_helion_graph(
                    helion_graph_inputs, mqa_q_dtype, helion_graph_q_scale
                ),
                args.warmup,
                args.repeat,
            )
            print(
                f"fused_q_{dtype_name}_mqa,{num_tokens},{correctness},"
                f"{triton_ms:.6f},{helion_ms:.6f},{helion_ms / triton_ms:.3f},"
                f"{triton_graph_ms:.6f},{helion_graph_ms:.6f},"
                f"{helion_graph_ms / triton_graph_ms:.3f}"
            )


if __name__ == "__main__":
    main()
