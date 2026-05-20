#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone benchmark for DeepSeek V3.2 NVFP4 fused kernels.

This compares the Triton and Helion implementations used by the specialized
DeepSeek V3.2 NVFP4 vLLM path at serving-shaped token counts. It reports both
eager launch timing and CUDA graph replay timing for:

- fused_norm_rope
- fused_q with BF16 MQA output
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


def make_norm_rope_inputs(num_tokens: int, topk: int) -> dict[str, object]:
    device = "cuda"
    dtype = torch.bfloat16
    block_size = 64
    num_blocks = triton.cdiv(num_tokens, block_size) + 1
    max_pos = max(4096, num_tokens + 16)

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
            num_blocks, block_size, 576, device=device, dtype=dtype
        ),
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
        mla_kv_cache_dtype="auto",
        mla_k_scale=inputs["mla_k_scale"],
    )


def run_norm_rope_helion(inputs: dict[str, object]) -> torch.Tensor:
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


def run_fused_q_triton(inputs: dict[str, object]) -> tuple[torch.Tensor, ...]:
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
        torch.bfloat16,
    )


def run_fused_q_helion(inputs: dict[str, object]) -> tuple[torch.Tensor, ...]:
    return helion_kernels.fused_q_helion(
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
        torch.bfloat16,
    )


def run_fused_q_helion_graph(inputs: dict[str, object]) -> tuple[torch.Tensor, ...]:
    return helion_kernels.fused_q_helion(
        inputs["positions"],
        inputs["q_pe"],
        inputs["q_pe_cache"],
        inputs["index_q"],
        inputs["index_q_cache"],
        inputs["ql_nope"],
        1.0,
        inputs["index_weights"],
        inputs["index_weights_softmax_scale"],
        inputs["index_weights_head_scale"],
        torch.bfloat16,
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


def check_norm_rope(num_tokens: int, topk: int) -> str:
    base = make_norm_rope_inputs(num_tokens, topk)
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


def check_fused_q(num_tokens: int) -> str:
    base = make_fused_q_inputs(num_tokens)
    triton_out = run_fused_q_triton(clone_inputs(base))
    helion_out = run_fused_q_helion(clone_inputs(base))
    torch.cuda.synchronize()

    for name, lhs, rhs in zip(
        ["index_q_fp8", "index_weights", "mqa_q"], triton_out, helion_out
    ):
        if name == "index_q_fp8":
            ok = torch.equal(lhs, rhs)
        else:
            ok = torch.allclose(lhs, rhs, rtol=1e-2, atol=1e-2)
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
    args = parser.parse_args()

    print("gpu", torch.cuda.get_device_name())
    print("pid", os.getpid())
    print("tokens", args.tokens)
    print(
        "kernel,tokens,correctness,triton_ms,helion_ms,helion_over_triton,"
        "triton_graph_ms,helion_graph_ms,helion_graph_over_triton_graph"
    )

    for token_s in args.tokens.split(","):
        num_tokens = int(token_s)
        correctness = check_norm_rope(num_tokens, args.topk)
        triton_inputs = make_norm_rope_inputs(num_tokens, args.topk)
        helion_inputs = make_norm_rope_inputs(num_tokens, args.topk)
        triton_ms = bench(
            lambda: run_norm_rope_triton(triton_inputs), args.warmup, args.repeat
        )
        helion_ms = bench(
            lambda: run_norm_rope_helion(helion_inputs), args.warmup, args.repeat
        )
        triton_graph_inputs = make_norm_rope_inputs(num_tokens, args.topk)
        helion_graph_inputs = make_norm_rope_inputs(num_tokens, args.topk)
        triton_graph_ms = bench_cuda_graph(
            lambda: run_norm_rope_triton(triton_graph_inputs),
            args.warmup,
            args.repeat,
        )
        helion_graph_ms = bench_cuda_graph(
            lambda: run_norm_rope_helion(helion_graph_inputs),
            args.warmup,
            args.repeat,
        )
        print(
            f"fused_norm_rope,{num_tokens},{correctness},"
            f"{triton_ms:.6f},{helion_ms:.6f},{helion_ms / triton_ms:.3f},"
            f"{triton_graph_ms:.6f},{helion_graph_ms:.6f},"
            f"{helion_graph_ms / triton_graph_ms:.3f}"
        )

    for token_s in args.tokens.split(","):
        num_tokens = int(token_s)
        correctness = check_fused_q(num_tokens)
        triton_inputs = make_fused_q_inputs(num_tokens)
        helion_inputs = make_fused_q_inputs(num_tokens)
        triton_ms = bench(
            lambda: run_fused_q_triton(triton_inputs), args.warmup, args.repeat
        )
        helion_ms = bench(
            lambda: run_fused_q_helion(helion_inputs), args.warmup, args.repeat
        )
        triton_graph_inputs = make_fused_q_inputs(num_tokens)
        helion_graph_inputs = make_fused_q_inputs(num_tokens)
        triton_graph_ms = bench_cuda_graph(
            lambda: run_fused_q_triton(triton_graph_inputs),
            args.warmup,
            args.repeat,
        )
        helion_graph_ms = bench_cuda_graph(
            lambda: run_fused_q_helion_graph(helion_graph_inputs),
            args.warmup,
            args.repeat,
        )
        print(
            f"fused_q_bf16_mqa,{num_tokens},{correctness},"
            f"{triton_ms:.6f},{helion_ms:.6f},{helion_ms / triton_ms:.3f},"
            f"{triton_graph_ms:.6f},{helion_graph_ms:.6f},"
            f"{helion_graph_ms / triton_graph_ms:.3f}"
        )


if __name__ == "__main__":
    main()
