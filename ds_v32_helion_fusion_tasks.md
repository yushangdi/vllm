# DeepSeek V3.2 Helion Fusion Task Plan

## Goal

Replicate the existing DeepSeek V3.2 vertical fusion path with Helion, then beat
the Triton implementation on decode and small mixed-batch latency.

## Tasks

1. Establish the Triton baseline
   - Benchmark the current specialized path for `nvidia/DeepSeek-V3.2-NVFP4`.
   - Capture per-kernel timings for `fused_norm_rope`, `fused_q`,
     `sparse_attn_indexer`, FlashMLA sparse attention, and W_UV up-projection.
   - Record decode shapes for `num_tokens` in `{1, 2, 4, 8, 16, 32, 64, 128,
     256, 512, 1024}` with MTP enabled and disabled.

2. Build kernel-level parity tests
   - Add direct correctness tests for Triton `fused_norm_rope` vs a reference.
   - Add direct correctness tests for Triton `fused_q` vs a reference.
   - Save representative DS V3.2 tensor shapes and tolerances for FP8 outputs,
     scales, cache writes, and top-k buffer initialization.

3. Add a Helion DS V3.2 ops module
   - Create a new Helion module under `vllm/kernels/helion/ops/`.
   - Mirror the public signatures of `fused_norm_rope` and `fused_q`.
   - Register the kernels with the existing vLLM Helion registration mechanism.

4. Implement Helion `fused_norm_rope`
   - Fuse Q RMSNorm, KV RMSNorm, KV RoPE, MLA cache write, index K LayerNorm,
     index K RoPE, index K FP8 quantization, indexer cache write, and top-k
     buffer initialization.
   - Avoid the current Triton fp32 scratch buffer and atomic read workaround for
     index K RoPE.
   - Validate cache writes for both BF16 and FP8 MLA KV cache modes.

5. Implement Helion `fused_q`
   - Fuse Q PE RoPE, W_UK_T-packed query quantization, index Q RoPE, index Q FP8
     quantization, and index weight scaling.
   - Evaluate whether one Helion kernel or two smaller Helion kernels is faster.
   - Keep output layouts identical to the current Triton path.

6. Add a runtime switch
   - Add a feature flag or kernel config option to select Helion vs Triton.
   - Keep Triton as the default fallback until Helion is faster and stable.
   - Ensure CUDA graph capture still works with the Helion path.

7. Tune Helion configs
   - Generate tuned configs for B200/GB200 first.
   - Add H100/H200 configs if the path is expected to run on Hopper.
   - Store configs using the existing `vllm/kernels/helion/configs/` pattern.

8. Benchmark full attention path
   - Compare Triton and Helion inside `monolithic_attn`.
   - Measure decode-only, prefill-only, and mixed prefill/decode batches.
   - Include MTP `num_speculative_tokens=1` because the current branch supports
     the DS V3.2 MTP path.

9. Optimize boundary overheads
   - Check whether Helion can remove extra temporary tensors from
     `fused_norm_rope` and `fused_q`.
   - Revisit top-k buffer initialization placement.
   - Avoid touching DeepGEMM logits, persistent top-k, and FlashMLA unless
     profiling proves they are the next bottleneck.

10. Validate end-to-end behavior
    - Run targeted kernel tests.
    - Run sparse MLA backend tests.
    - Run a small `nvidia/DeepSeek-V3.2-NVFP4` serving or offline generation test
      with the specialized model enabled.
    - Compare output correctness and latency against the Triton baseline.

## Initial Success Criteria

- Helion `fused_norm_rope` is faster than Triton for decode and small batches.
- Helion `fused_q` is at least parity, then faster after tuning.
- End-to-end DS V3.2 decode latency improves with no CUDA graph regression.
- Triton remains available as a fallback path.
