# DS V3.2 Helion fused_norm_rope CUDA Graph Autotune

## Run

- Date: 2026-05-15
- GPU: NVIDIA B200
- Benchmark env: `pytorch-3.12`
- Tuning mode: one process per token count, CUDA graph replay benchmark inside
  Helion autotune.
- Effort: `HELION_AUTOTUNE_EFFORT=quick`
- Accuracy: Helion internal autotune accuracy check disabled because this kernel
  has side-effect cache writes; the benchmark still ran Triton-vs-Helion parity
  after each selected config.
- Log: `/tmp/ds32_helion_cudagraph_autotune_per_token.log`

Command pattern:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_NATIVE_SKIP_VERSION_CHECK=1 \
HELION_FORCE_AUTOTUNE=1 \
HELION_AUTOTUNE_EFFORT=quick \
HELION_AUTOTUNE_ACCURACY_CHECK=0 \
HELION_AUTOTUNE_IGNORE_ERRORS=1 \
HELION_AUTOTUNE_LOG_LEVEL=20 \
DS32_TOKENS=<tokens> \
conda run -n pytorch-3.12 python benchmarks/kernels/bench_ds32_fused_norm_rope_helion.py
```

## Perf

| tokens | correctness | Triton eager ms | Helion eager ms | Triton graph ms | Helion graph ms | Helion graph/Triton graph |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | OK | 0.035234 | 0.067976 | 0.004146 | 0.004151 | 1.001 |
| 2 | OK | 0.033433 | 0.067165 | 0.004151 | 0.004144 | 0.998 |
| 4 | OK | 0.034940 | 0.068998 | 0.004134 | 0.004141 | 1.002 |
| 8 | OK | 0.034202 | 0.066742 | 0.004130 | 0.004145 | 1.003 |
| 16 | OK | 0.033704 | 0.068136 | 0.004146 | 0.004139 | 0.998 |
| 32 | OK | 0.034619 | 0.067292 | 0.004144 | 0.004138 | 0.999 |
| 64 | OK | 0.033964 | 0.070029 | 0.004133 | 0.004130 | 0.999 |
| 128 | OK | 0.033857 | 0.068596 | 0.004139 | 0.004138 | 1.000 |
| 256 | OK | 0.035018 | 0.068960 | 0.006175 | 0.006164 | 0.998 |
| 512 | OK | 0.035459 | 0.069164 | 0.006185 | 0.006195 | 1.002 |
| 1024 | OK | 0.034037 | 0.065868 | 0.008228 | 0.006188 | 0.752 |

## Config Groups

The exact `@helion.kernel(config=...)` lines are in the log. The selected
configs group as follows:

| tokens | block_sizes | num_warps | num_stages | reduction_loops | indexing summary |
|---:|:---:|---:|---:|:---:|:---|
| 1, 2 | `[1]` | 1 | 4 | `[1024]` | mixed tensor_descriptor/pointer |
| 4, 8 | `[1]` | 1 | 1 | `[1024]` | mostly pointer, one tensor_descriptor |
| 16 | `[1]` | 1 | 5 | `[1024]` | mixed tensor_descriptor/pointer |
| 32, 64 | `[1]` | 1 | 5 | `[1024]` | mixed tensor_descriptor/pointer |
| 128 | `[1]` | 4 | 5 | `[128]` | mixed tensor_descriptor/pointer |
| 256 | `[1]` | 4 | 5 | `[128]` | mixed tensor_descriptor/pointer |
| 512 | `[1]` | 4 | 5 | `[512]` | mixed tensor_descriptor/pointer |
| 1024 | `[1]` | 1 | 1 | `[512]` | pointer |

## Notes

- CUDA graph replay removes the eager launch/enqueue gap. Under graph replay,
  all tuned configs are at parity with Triton except `T=1024`, where the best
  found config is faster.
- This was a quick-effort search, not full exhaustive tuning. Full graph-aware
  tuning should be run before landing hardcoded configs.
- To use these configs in production, we need a config picker keyed by
  `num_tokens` or separate bound kernels for token-count buckets. The current
  experimental kernel still leaves selection to Helion autotune/cache.
