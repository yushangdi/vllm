# DeepSeek V3.2 NVFP4 Fused Kernel Benchmark

This benchmark compares the Triton and Helion implementations used by the
specialized DeepSeek V3.2 NVFP4 path.

Script:

```bash
vllm/kernels/helion/ops/bench_deepseek_v32_nvfp4.py
```

## Local Environment

The recorded results below were taken on:

| Item | Value |
| --- | --- |
| Conda env | `pytorch-3.12` |
| Python | `3.12.13` |
| GPU | `NVIDIA B200` |
| GPU memory | `183359 MiB` per GPU |
| GPU compute capability | `10.0` |
| NVIDIA driver | `580.82.07` |
| System `nvcc` | `cuda_12.8.r12.8/compiler.35583870_0` |
| PyTorch | `2.13.0a0+git7c4c6c0` |
| PyTorch CUDA | `12.9` |
| Triton | `3.6.0` |
| Helion | `1.0.1.dev362+g99d62e001.d20260515` |
| FlashInfer Python | `0.6.7` |
| vLLM CLI | `0.19.1rc1.dev421+g8633f3059.d20260519` |
| Repo branch | `yushangdi-woosuk-ds-exp` |
| Repo commit | `8633f3059` |

## Command

Run from the repo root in the `pytorch-3.12` environment:

```bash
CUDA_VISIBLE_DEVICES=2 \
TORCH_NATIVE_SKIP_VERSION_CHECK=1 \
python vllm/kernels/helion/ops/bench_deepseek_v32_nvfp4.py \
  --tokens 8,1024 \
  --warmup 10 \
  --repeat 50 \
  --mla-kv-cache-dtype both \
  --mqa-q-dtype both
```

The token counts are serving-shaped:

- `8`: decode-like batch from the small serving benchmark.
- `1024`: prefill-like batch from the small serving benchmark.

The script prints CSV with:

```text
kernel,tokens,correctness,triton_ms,helion_ms,helion_over_triton,triton_graph_ms,helion_graph_ms,helion_graph_over_triton_graph
```

Lower `*_ms` is better. Ratios below `1.0` mean Helion is faster.

## Recorded Standalone Result

BF16 MQA result from `/tmp/ds32_serving_shapes_standalone.log`:

| Kernel | Tokens | Correctness | Triton eager ms | Helion eager ms | Eager H/T | Triton graph ms | Helion graph ms | Graph H/T |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fused_norm_rope` | 8 | OK | 0.034163 | 0.080600 | 2.359 | N/A | N/A | N/A |
| `fused_norm_rope` | 1024 | OK | 0.033638 | 0.079036 | 2.350 | N/A | N/A | N/A |
| `fused_q_bf16_mqa` | 8 | OK | 0.028989 | 0.093734 | 3.233 | N/A | N/A | N/A |
| `fused_q_bf16_mqa` | 1024 | OK | 0.410808 | 0.216173 | 0.526 | N/A | N/A | N/A |

FP8 MLA cache and FP8 MQA result from `/tmp/ds32_standalone_full_fp8_bench.log`:

```bash
CUDA_VISIBLE_DEVICES=2 \
PATH=/home/shangdiy/.conda/envs/pytorch-3.12/bin:$PATH \
TORCH_NATIVE_SKIP_VERSION_CHECK=1 \
/home/shangdiy/.conda/envs/pytorch-3.12/bin/python \
  vllm/kernels/helion/ops/bench_deepseek_v32_nvfp4.py \
  --tokens 8,1024 \
  --warmup 10 \
  --repeat 50 \
  --mla-kv-cache-dtype fp8 \
  --mqa-q-dtype fp8 \
  2>&1 | tee /tmp/ds32_standalone_full_fp8_bench.log
```

| Kernel | Tokens | Correctness | Triton eager ms | Helion eager ms | Eager H/T | Triton graph ms | Helion graph ms | Graph H/T |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fused_norm_rope_fp8_mla` | 8 | OK | 0.006256 | 2.429283 | 388.313 | 0.006221 | 0.004289 | 0.689 |
| `fused_norm_rope_fp8_mla` | 1024 | OK | 0.010316 | 2.464146 | 238.878 | 0.012372 | 0.010371 | 0.838 |
| `fused_q_fp8_mqa` | 8 | OK | 0.008751 | 2.443978 | 279.289 | 0.008267 | 0.008239 | 0.997 |
| `fused_q_fp8_mqa` | 1024 | OK | 1.116642 | 2.568621 | 2.300 | 0.938749 | 0.145550 | 0.155 |

Interpretation:

- Eager standalone timing favors Triton for the small launch-dominated shapes.
- Helion is faster under CUDA graph replay for the FP8 MLA cache norm/rope
  cases and for the large FP8 `fused_q_fp8_mqa` shape.
- CUDA graph replay is the more relevant mode for vLLM serving with graph
  capture enabled; use the current script output's `*_graph_ms` columns for
  graph replay comparisons.

## Full vLLM Serving Cross-Check

The clean full-serving comparison used:

- GPUs: `2,3,4,5`
- `--no-enable-prefix-caching`
- `--kv-cache-dtype bfloat16`
- `--moe-backend cutlass`
- `--compilation-config '{"max_cudagraph_capture_size": 256}'`
- workload: `128` prompts, `128` input tokens, `128` output tokens,
  `--request-rate inf`

Result:

| Path | Output tok/s | Mean TTFT ms | Mean TPOT ms | Prefix cache hit |
| --- | ---: | ---: | ---: | ---: |
| Triton/default | 2280.34 | 1332.20 | 45.82 | 0.0% |
| Helion autotuned | 2330.67 | 1263.43 | 45.14 | 0.0% |

Helion was `+2.2%` output tok/s in the clean capture-256 serving run.

## Full vLLM FP8 Serving Cross-Check

The FP8 serving comparison used the same workload and did not pass
`--kv-cache-dtype bfloat16`, so vLLM selected `kv_cache_dtype=fp8_e4m3`.

Result:

| Path | Output tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | Prefix cache hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Triton/default | 1643.69 | 3287.38 | 3767.34 | 48.52 | 0.0% |
| Helion patched | 1654.95 | 3309.91 | 3692.79 | 48.63 | 0.0% |

Helion patched was `+0.7%` output tok/s in the clean capture-256 FP8 serving
run. Startup/capture was still slower for Helion: graph capture was `36 s`
versus `26 s` for Triton/default.

## Notes

- The Helion path is selected in vLLM serving with `VLLM_DS32_USE_HELION=1`.
- The benchmark imports the specialized model kernels directly; it does not run
  the full vLLM scheduler, attention backend, MoE backend, sampling, or API
  server path.
- Full serving can differ from standalone kernel timing because the measured
  workload includes graph capture/replay behavior, attention, MoE, allreduce,
  sampling, scheduling, and request handling.
