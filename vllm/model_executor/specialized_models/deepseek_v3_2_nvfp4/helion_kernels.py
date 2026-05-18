# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion kernels for the DeepSeek V3.2 specialized attention path.

This module intentionally is not imported from ``vllm.kernels.helion.ops`` yet.
The kernels here are an experimental parity target for the Triton kernels in
``kernels.py`` and should be wired behind an explicit runtime switch after
correctness tests and tuned configs are in place.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if has_helion():
    import helion
    import helion.language as hl


def _cuda_graph_autotune_benchmark(
    fns: list,
    *,
    repeat: int,
    desc: str | None = None,
) -> list[float]:
    timings: list[float] = []
    for fn in fns:
        try:
            for _ in range(5):
                fn()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()

            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end) / repeat)
        except Exception:
            timings.append(float("inf"))
    return timings


def _require_helion() -> None:
    if not has_helion():
        raise ImportError(
            "DeepSeek V3.2 Helion kernels require helion to be installed. "
            "Install it with: uv pip install 'vllm[helion]'"
        )


if has_helion():

    @helion.kernel(
        static_shapes=False,
        autotune_benchmark_fn=_cuda_graph_autotune_benchmark,
    )
    def _fused_norm_rope_helion_kernel(
        positions: torch.Tensor,
        # Q RMS norm
        q_c: torch.Tensor,
        q_rms_norm_w: torch.Tensor,
        q_rms_eps: float,
        # KV RMS norm
        kv_c: torch.Tensor,
        kv_rms_norm_w: torch.Tensor,
        kv_rms_eps: float,
        # KV RoPE
        k_pe: torch.Tensor,
        k_rope_cos_sin_cache: torch.Tensor,
        # Index K LayerNorm + RoPE
        index_k: torch.Tensor,
        index_k_layer_norm_w: torch.Tensor,
        index_k_layer_norm_bias: torch.Tensor,
        index_k_layer_norm_eps: float,
        index_k_rope_cos_sin_cache: torch.Tensor,
        # Top-k buffer
        topk_indices_buffer: torch.Tensor,
        # Cache routing
        slot_mapping: torch.Tensor,
        # Indexer K cache
        indexer_cache_fp8: torch.Tensor,
        indexer_cache_fp8_flat: torch.Tensor,
        indexer_cache_scale_flat: torch.Tensor,
        # MLA KV cache
        mla_cache: torch.Tensor,
        mla_cache_flat: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = positions.size(0)
        q_dim = hl.specialize(q_c.shape[1])
        kv_dim = hl.specialize(kv_c.shape[1])
        k_pe_dim = hl.specialize(k_pe.shape[1])
        k_pe_half = hl.specialize(k_pe_dim // 2)
        index_k_dim = hl.specialize(index_k.shape[1])
        index_k_half = hl.specialize(index_k_dim // 2)
        topk = hl.specialize(topk_indices_buffer.shape[1])
        indexer_cache_block_size = hl.specialize(indexer_cache_fp8.shape[1])
        indexer_cache_stride = hl.specialize(indexer_cache_fp8.shape[2])
        mla_cache_block_size = hl.specialize(mla_cache.shape[1])
        mla_block_stride = hl.specialize(mla_cache.stride(0))
        mla_entry_stride = hl.specialize(mla_cache.stride(1))

        q_c_out = torch.empty_like(q_c)

        for tile_t in hl.tile([num_tokens]):
            # Fill the sparse indexer output row up front. This mirrors the
            # current Triton kernel's pid==3 work and keeps stale indices from
            # leaking through padded tokens.
            topk_indices_buffer[tile_t, :topk] = -1

            slot_idx = slot_mapping[tile_t]

            # Q RMSNorm.
            q_vals = q_c[tile_t, :q_dim].to(torch.float32)
            q_w = q_rms_norm_w[:q_dim].to(torch.float32)
            q_mean_sq = torch.sum(q_vals * q_vals, dim=-1, keepdim=True) / q_dim
            q_rrms = torch.rsqrt(q_mean_sq + q_rms_eps)
            q_c_out[tile_t, :q_dim] = (q_vals * q_rrms * q_w).to(q_c_out.dtype)

            # KV RMSNorm.
            kv_vals = kv_c[tile_t, :kv_dim].to(torch.float32)
            kv_w = kv_rms_norm_w[:kv_dim].to(torch.float32)
            kv_mean_sq = torch.sum(kv_vals * kv_vals, dim=-1, keepdim=True) / kv_dim
            kv_rrms = torch.rsqrt(kv_mean_sq + kv_rms_eps)
            kv_normed = kv_vals * kv_rrms * kv_w

            # KV RoPE, interleaved layout.
            pos = positions[tile_t]
            kpe_offsets = hl.arange(0, k_pe_half)
            kv_cos = k_rope_cos_sin_cache[pos, kpe_offsets].to(torch.float32)
            kv_sin = k_rope_cos_sin_cache[pos, k_pe_half + kpe_offsets].to(
                torch.float32
            )
            kpe_even = k_pe[tile_t, kpe_offsets * 2].to(torch.float32)
            kpe_odd = k_pe[tile_t, kpe_offsets * 2 + 1].to(torch.float32)
            kpe_r1 = kpe_even * kv_cos - kpe_odd * kv_sin
            kpe_r2 = kpe_odd * kv_cos + kpe_even * kv_sin

            mla_block_idx = slot_idx // mla_cache_block_size
            mla_block_off = slot_idx - mla_block_idx * mla_cache_block_size
            mla_base = (
                mla_block_idx * mla_block_stride + mla_block_off * mla_entry_stride
            )
            kv_offsets = hl.arange(0, kv_dim)
            mla_cache_flat[mla_base[:, None] + kv_offsets[None, :]] = kv_normed.to(
                mla_cache_flat.dtype
            )
            mla_cache_flat[mla_base[:, None] + kv_dim + kpe_offsets[None, :] * 2] = (
                kpe_r1.to(mla_cache_flat.dtype)
            )
            mla_cache_flat[
                mla_base[:, None] + kv_dim + kpe_offsets[None, :] * 2 + 1
            ] = kpe_r2.to(mla_cache_flat.dtype)

            # Index K LayerNorm.
            idx_vals = index_k[tile_t, :index_k_dim].to(torch.float32)
            idx_masked = idx_vals
            idx_mean = torch.sum(idx_masked, dim=-1) / index_k_dim
            idx_diff = idx_vals - idx_mean[:, None]
            idx_var = torch.sum(idx_diff * idx_diff, dim=-1)
            idx_var = idx_var / index_k_dim
            idx_rstd = torch.rsqrt(idx_var + index_k_layer_norm_eps)

            # Index K RoPE, NeoX layout. Load the two halves directly from the
            # source tensor so Helion does not need to index an intermediate.
            idx_offsets = hl.arange(0, index_k_half)
            idx_lo = index_k[tile_t, idx_offsets].to(torch.float32)
            idx_hi = index_k[tile_t, index_k_half + idx_offsets].to(torch.float32)
            idx_w_lo = index_k_layer_norm_w[idx_offsets].to(torch.float32)
            idx_w_hi = index_k_layer_norm_w[index_k_half + idx_offsets].to(
                torch.float32
            )
            idx_b_lo = index_k_layer_norm_bias[idx_offsets].to(torch.float32)
            idx_b_hi = index_k_layer_norm_bias[index_k_half + idx_offsets].to(
                torch.float32
            )
            idx_normed_lo = (idx_lo - idx_mean[:, None]) * idx_rstd[:, None] * idx_w_lo
            idx_normed_hi = (idx_hi - idx_mean[:, None]) * idx_rstd[:, None] * idx_w_hi
            idx_normed_lo = idx_normed_lo + idx_b_lo
            idx_normed_hi = idx_normed_hi + idx_b_hi

            idx_cos = index_k_rope_cos_sin_cache[pos, idx_offsets].to(torch.float32)
            idx_sin = index_k_rope_cos_sin_cache[pos, index_k_half + idx_offsets].to(
                torch.float32
            )
            idx_roped_lo = idx_normed_lo * idx_cos - idx_normed_hi * idx_sin
            idx_roped_hi = idx_normed_hi * idx_cos + idx_normed_lo * idx_sin
            idx_abs = torch.maximum(torch.abs(idx_roped_lo), torch.abs(idx_roped_hi))
            amax = torch.amax(idx_abs, dim=-1)
            idx_scale = torch.clamp(amax, min=1e-4) / 448.0
            idx_scale = torch.exp2(torch.ceil(torch.log2(idx_scale)))

            idx_block_idx = slot_idx // indexer_cache_block_size
            idx_block_off = slot_idx - idx_block_idx * indexer_cache_block_size
            idx_block_start = (
                idx_block_idx * indexer_cache_block_size * indexer_cache_stride
            )
            idx_value_base = idx_block_start + idx_block_off * index_k_dim
            indexer_cache_fp8_flat[idx_value_base[:, None] + idx_offsets[None, :]] = (
                idx_roped_lo / idx_scale[:, None]
            ).to(indexer_cache_fp8_flat.dtype)
            indexer_cache_fp8_flat[
                idx_value_base[:, None] + index_k_half + idx_offsets[None, :]
            ] = (idx_roped_hi / idx_scale[:, None]).to(indexer_cache_fp8_flat.dtype)

            idx_scale_byte_off = (
                idx_block_start
                + indexer_cache_block_size * index_k_dim
                + idx_block_off * 4
            )
            indexer_cache_scale_flat[idx_scale_byte_off // 4] = idx_scale

        return q_c_out

    @helion.kernel(static_shapes=False)
    def _fused_q_helion_kernel(
        positions: torch.Tensor,
        q_pe: torch.Tensor,
        q_pe_cos_sin_cache: torch.Tensor,
        q_pe_cos_sin_cache_flat: torch.Tensor,
        index_q: torch.Tensor,
        index_q_cos_sin_cache: torch.Tensor,
        index_q_cos_sin_cache_flat: torch.Tensor,
        ql_nope: torch.Tensor,
        q_scale: torch.Tensor,
        index_weights: torch.Tensor,
        index_weights_softmax_scale: float,
        index_weights_head_scale: float,
        index_q_fp8: torch.Tensor,
        index_weights_out: torch.Tensor,
        mqa_q_fp8: torch.Tensor,
    ) -> None:
        num_tokens = positions.size(0)
        num_q_heads = hl.specialize(q_pe.shape[1])
        num_index_q_heads = hl.specialize(index_q.shape[1])
        q_pe_dim = hl.specialize(q_pe.shape[2])
        q_pe_half = hl.specialize(q_pe_dim // 2)
        q_pe_cache_stride = hl.specialize(q_pe_cos_sin_cache.stride(0))
        index_q_dim = hl.specialize(index_q.shape[2])
        index_q_half = hl.specialize(index_q_dim // 2)
        index_q_cache_stride = hl.specialize(index_q_cos_sin_cache.stride(0))
        ql_nope_dim = hl.specialize(ql_nope.shape[2])

        for pid, tile_t, tile_h in hl.grid([3, num_tokens, num_index_q_heads]):
            if pid == 0:
                if tile_h * 2 < num_q_heads:
                    qpe_offsets = hl.arange(0, q_pe_half)
                    pos = positions[tile_t]
                    qpe_base = pos * q_pe_cache_stride
                    qpe_cos = q_pe_cos_sin_cache_flat[qpe_base + qpe_offsets].to(
                        torch.float32
                    )
                    qpe_sin = q_pe_cos_sin_cache_flat[
                        qpe_base + q_pe_half + qpe_offsets
                    ].to(torch.float32)
                    qpe_scale = hl.load(q_scale, [0]).to(torch.float32)
                    for local_head in hl.static_range(2):
                        q_head = tile_h * 2 + local_head
                        if q_head < num_q_heads:
                            qpe_x1 = q_pe[tile_t, q_head, qpe_offsets * 2].to(
                                torch.float32
                            )
                            qpe_x2 = q_pe[tile_t, q_head, qpe_offsets * 2 + 1].to(
                                torch.float32
                            )
                            qpe_r1 = qpe_x1 * qpe_cos - qpe_x2 * qpe_sin
                            qpe_r2 = qpe_x2 * qpe_cos + qpe_x1 * qpe_sin
                            mqa_q_fp8[
                                tile_t, q_head, ql_nope_dim + qpe_offsets * 2
                            ] = (qpe_r1 / qpe_scale).to(mqa_q_fp8.dtype)
                            mqa_q_fp8[
                                tile_t, q_head, ql_nope_dim + qpe_offsets * 2 + 1
                            ] = (qpe_r2 / qpe_scale).to(mqa_q_fp8.dtype)
            elif pid == 1:
                if tile_h < num_index_q_heads:
                    index_offsets = hl.arange(0, index_q_half)
                    pos = positions[tile_t]
                    index_base = pos * index_q_cache_stride
                    index_cos = index_q_cos_sin_cache_flat[
                        index_base + index_offsets
                    ].to(torch.float32)
                    index_sin = index_q_cos_sin_cache_flat[
                        index_base + index_q_half + index_offsets
                    ].to(torch.float32)
                    index_x1 = index_q[tile_t, tile_h, index_offsets].to(
                        torch.float32
                    )
                    index_x2 = index_q[
                        tile_t, tile_h, index_q_half + index_offsets
                    ].to(torch.float32)
                    index_r1 = index_x1 * index_cos - index_x2 * index_sin
                    index_r2 = index_x2 * index_cos + index_x1 * index_sin
                    index_r1 = index_r1.to(index_q.dtype).to(torch.float32)
                    index_r2 = index_r2.to(index_q.dtype).to(torch.float32)

                    index_abs = torch.maximum(torch.abs(index_r1), torch.abs(index_r2))
                    index_amax = torch.amax(index_abs, dim=-1)
                    index_scale = torch.clamp(index_amax, min=1e-4) / 448.0
                    index_scale = torch.exp2(torch.ceil(torch.log2(index_scale)))
                    index_q_fp8[tile_t, tile_h, index_offsets] = (
                        index_r1 / index_scale
                    ).to(index_q_fp8.dtype)
                    index_q_fp8[tile_t, tile_h, index_q_half + index_offsets] = (
                        index_r2 / index_scale
                    ).to(index_q_fp8.dtype)

                    weights = index_weights[tile_t, tile_h].to(torch.float32)
                    weights = weights * index_scale
                    weights = weights * index_weights_softmax_scale
                    weights = weights * index_weights_head_scale
                    index_weights_out[tile_t, tile_h] = weights
            elif pid == 2:
                if tile_h * 2 < num_q_heads:
                    nope_offsets = hl.arange(0, ql_nope_dim)
                    nope_scale = hl.load(q_scale, [0]).to(torch.float32)
                    for local_head in hl.static_range(2):
                        q_head = tile_h * 2 + local_head
                        if q_head < num_q_heads:
                            nope_vals = ql_nope[tile_t, q_head, nope_offsets].to(
                                torch.float32
                            )
                            mqa_q_fp8[tile_t, q_head, nope_offsets] = (
                                nope_vals / nope_scale
                            ).to(mqa_q_fp8.dtype)

    @helion.kernel(static_shapes=False)
    def _fused_q_nope_pack_helion_kernel(
        ql_nope: torch.Tensor,
        q_scale: torch.Tensor,
        mqa_q_fp8: torch.Tensor,
    ) -> None:
        num_tokens = ql_nope.size(0)
        num_q_heads = hl.specialize(ql_nope.shape[1])
        ql_nope_dim = hl.specialize(ql_nope.shape[2])
        for tile_t, tile_h in hl.tile([num_tokens, num_q_heads]):
            offsets = hl.arange(0, ql_nope_dim)
            scale = hl.load(q_scale, [0]).to(torch.float32)
            vals = ql_nope[tile_t, tile_h, offsets].to(torch.float32)
            mqa_q_fp8[tile_t, tile_h, offsets] = (vals / scale).to(mqa_q_fp8.dtype)

    @helion.kernel(static_shapes=False)
    def _fused_q_pe_pack_helion_kernel(
        positions: torch.Tensor,
        q_pe: torch.Tensor,
        q_pe_cos_sin_cache: torch.Tensor,
        q_scale: torch.Tensor,
        mqa_q_fp8: torch.Tensor,
        ql_nope_dim: int,
    ) -> None:
        num_tokens = q_pe.size(0)
        num_q_heads = hl.specialize(q_pe.shape[1])
        q_pe_dim = hl.specialize(q_pe.shape[2])
        q_pe_half = hl.specialize(q_pe_dim // 2)
        for tile_t, tile_h in hl.tile([num_tokens, num_q_heads]):
            offsets = hl.arange(0, q_pe_half)
            scale = hl.load(q_scale, [0]).to(torch.float32)
            pos = positions[tile_t]
            cos = q_pe_cos_sin_cache[pos, offsets].to(torch.float32)
            sin = q_pe_cos_sin_cache[pos, q_pe_half + offsets].to(torch.float32)
            x1 = q_pe[tile_t, tile_h, offsets * 2].to(torch.float32)
            x2 = q_pe[tile_t, tile_h, offsets * 2 + 1].to(torch.float32)
            r1 = x1 * cos - x2 * sin
            r2 = x2 * cos + x1 * sin
            mqa_q_fp8[tile_t, tile_h, ql_nope_dim + offsets * 2] = (r1 / scale).to(
                mqa_q_fp8.dtype
            )
            mqa_q_fp8[tile_t, tile_h, ql_nope_dim + offsets * 2 + 1] = (r2 / scale).to(
                mqa_q_fp8.dtype
            )

    @helion.kernel(static_shapes=False)
    def _fused_index_q_helion_kernel(
        positions: torch.Tensor,
        index_q: torch.Tensor,
        index_q_cos_sin_cache: torch.Tensor,
        index_q_fp8: torch.Tensor,
        index_weights: torch.Tensor,
        index_weights_softmax_scale: float,
        index_weights_head_scale: float,
        index_weights_out: torch.Tensor,
    ) -> None:
        num_tokens = index_q.size(0)
        num_index_q_heads = hl.specialize(index_q.shape[1])
        index_q_dim = hl.specialize(index_q.shape[2])
        index_q_half = hl.specialize(index_q_dim // 2)
        for tile_t, tile_h in hl.tile([num_tokens, num_index_q_heads]):
            offsets = hl.arange(0, index_q_half)
            pos = positions[tile_t]
            cos = index_q_cos_sin_cache[pos, offsets].to(torch.float32)
            sin = index_q_cos_sin_cache[pos, index_q_half + offsets].to(torch.float32)
            x1 = index_q[tile_t, tile_h, offsets].to(torch.float32)
            x2 = index_q[tile_t, tile_h, index_q_half + offsets].to(torch.float32)
            r1 = x1 * cos - x2 * sin
            r2 = x2 * cos + x1 * sin
            r1 = r1.to(index_q.dtype).to(torch.float32)
            r2 = r2.to(index_q.dtype).to(torch.float32)

            abs_vals = torch.maximum(torch.abs(r1), torch.abs(r2))
            amax = torch.amax(abs_vals, dim=-1)
            scale = torch.clamp(amax, min=1e-4) / 448.0
            scale = torch.exp2(torch.ceil(torch.log2(scale)))
            index_q_fp8[tile_t, tile_h, offsets] = (r1 / scale[:, :, None]).to(
                index_q_fp8.dtype
            )
            index_q_fp8[tile_t, tile_h, index_q_half + offsets] = (
                r2 / scale[:, :, None]
            ).to(index_q_fp8.dtype)

            weights = index_weights[tile_t, tile_h].to(torch.float32)
            weights = weights * scale
            weights = weights * index_weights_softmax_scale
            weights = weights * index_weights_head_scale
            index_weights_out[tile_t, tile_h] = weights


_fused_norm_rope_helion_last_num_tokens: int | None = None
_fused_q_helion_last_num_tokens: int | None = None


def fused_norm_rope_helion(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    q_rms_norm_w: torch.Tensor,
    q_rms_eps: float,
    kv_c: torch.Tensor,
    kv_rms_norm_w: torch.Tensor,
    kv_rms_eps: float,
    k_pe: torch.Tensor,
    k_rope_cos_sin_cache: torch.Tensor,
    index_k: torch.Tensor,
    index_k_layer_norm_w: torch.Tensor,
    index_k_layer_norm_bias: torch.Tensor,
    index_k_layer_norm_eps: float,
    index_k_rope_cos_sin_cache: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    *,
    slot_mapping: torch.Tensor | None = None,
    indexer_k_cache: torch.Tensor | None = None,
    mla_kv_cache: torch.Tensor | None = None,
    mla_kv_cache_dtype: str = "auto",
    mla_k_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Helion version of ``kernels.fused_norm_rope``.

    The wrapper keeps the same public contract as the Triton function: it
    returns normalized ``q_c`` and mutates the top-k buffer plus the indexer/MLA
    KV caches when cache tensors are provided.
    """
    _require_helion()

    num_tokens = positions.shape[0]

    if slot_mapping is None:
        raise NotImplementedError(
            "fused_norm_rope_helion currently requires slot_mapping."
        )

    if indexer_k_cache is None:
        raise NotImplementedError(
            "fused_norm_rope_helion currently requires indexer_k_cache."
        )
    else:
        indexer_cache_fp8 = indexer_k_cache.view(current_platform.fp8_dtype())
        indexer_cache_scale_flat = indexer_k_cache.view(torch.float32).flatten()

    mla_cache_fp8 = mla_kv_cache_dtype != "auto"
    if mla_kv_cache is None or mla_cache_fp8:
        raise NotImplementedError(
            "fused_norm_rope_helion benchmark path currently supports only "
            "BF16 MLA KV cache."
        )

    # Helion currently reuses the first bound kernel even when the 1-D tile
    # extent changes, which leaves later tokens unwritten after a T=1 compile.
    # Reset only on shape changes so same-shape steady-state timing is not
    # affected.
    global _fused_norm_rope_helion_last_num_tokens
    if _fused_norm_rope_helion_last_num_tokens != num_tokens:
        _fused_norm_rope_helion_kernel.reset()
        _fused_norm_rope_helion_last_num_tokens = num_tokens

    return _fused_norm_rope_helion_kernel(
        positions,
        q_c,
        q_rms_norm_w,
        q_rms_eps,
        kv_c,
        kv_rms_norm_w,
        kv_rms_eps,
        k_pe,
        k_rope_cos_sin_cache,
        index_k,
        index_k_layer_norm_w,
        index_k_layer_norm_bias,
        index_k_layer_norm_eps,
        index_k_rope_cos_sin_cache,
        topk_indices_buffer,
        slot_mapping,
        indexer_cache_fp8,
        indexer_cache_fp8.flatten(),
        indexer_cache_scale_flat,
        mla_kv_cache,
        mla_kv_cache.flatten(),
    )


def fused_q_helion(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    q_pe_cos_sin_cache: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    ql_nope: torch.Tensor,
    q_scale: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helion version of ``kernels.fused_q``.

    This mirrors the Triton branch-on-pid structure in one Helion launch.
    """
    _require_helion()

    assert positions.ndim == 1
    assert q_pe.ndim == 3
    assert q_pe_cos_sin_cache.ndim == 2
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2
    assert ql_nope.ndim == 3
    assert ql_nope.shape[:2] == q_pe.shape[:2]

    mqa_q_fp8 = torch.empty(
        q_pe.shape[0],
        q_pe.shape[1],
        ql_nope.shape[2] + q_pe.shape[2],
        dtype=current_platform.fp8_dtype(),
        device=q_pe.device,
    )
    index_q_fp8 = torch.empty_like(index_q, dtype=current_platform.fp8_dtype())
    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    global _fused_q_helion_last_num_tokens
    if _fused_q_helion_last_num_tokens != positions.shape[0]:
        _fused_q_helion_kernel.reset()
        _fused_q_helion_last_num_tokens = positions.shape[0]

    _fused_q_helion_kernel(
        positions,
        q_pe,
        q_pe_cos_sin_cache,
        q_pe_cos_sin_cache.flatten(),
        index_q,
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.flatten(),
        ql_nope,
        q_scale,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_q_fp8,
        index_weights_out,
        mqa_q_fp8,
    )
    return index_q_fp8, index_weights_out, mqa_q_fp8
