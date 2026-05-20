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
  --repeat 50
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

Recorded command output from `/tmp/ds32_serving_shapes_standalone.log`:

| Kernel | Tokens | Correctness | Triton eager ms | Helion eager ms | Helion/Triton |
| --- | ---: | --- | ---: | ---: | ---: |
| `fused_norm_rope` | 8 | OK | 0.034163 | 0.080600 | 2.359 |
| `fused_norm_rope` | 1024 | OK | 0.033638 | 0.079036 | 2.350 |
| `fused_q_bf16_mqa` | 8 | OK | 0.028989 | 0.093734 | 3.233 |
| `fused_q_bf16_mqa` | 1024 | OK | 0.410808 | 0.216173 | 0.526 |

Interpretation:

- Eager standalone timing favors Triton for the small launch-dominated shapes.
- Helion is faster for the large `fused_q_bf16_mqa` shape.
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

## Notes

- The Helion path is selected in vLLM serving with `VLLM_DS32_USE_HELION=1`.
- The benchmark imports the specialized model kernels directly; it does not run
  the full vLLM scheduler, attention backend, MoE backend, sampling, or API
  server path.
- Full serving can differ from standalone kernel timing because the measured
  workload includes graph capture/replay behavior, attention, MoE, allreduce,
  sampling, scheduling, and request handling.
