# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Call-site routing for eager Helion quant ops.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)
_is_compiling = torch.compiler.is_compiling

_ROUTABLE_EAGER_OPS = frozenset(
    {
        "per_token_group_fp8_quant",
    }
)


@functools.cache
def _use_helion_kernels() -> bool:
    return envs.VLLM_USE_HELION_KERNELS


@functools.cache
def _eager_kernel(op_name: str) -> Callable:
    # The configured helion kernel, bypassing the torch custom-op dispatch
    # boundary + HelionKernelWrapper indirection (~4us/call of pure host
    # overhead). Safe here: route_quant only runs eager/capture (never under
    # torch.compile tracing, which takes the ``fn`` fallback), and the custom op
    # stays registered for the compiled FX-swap path. Cached: resolved once.
    from vllm.kernels.helion import get_kernel_by_name

    return get_kernel_by_name(op_name).eager_callable()


@functools.cache
def _checked_eager_kernel(op_name: str) -> Callable | None:
    if _helion_available(op_name):
        return _eager_kernel(op_name)
    return None


@functools.cache
def _helion_available(op_name: str) -> bool:
    # `import vllm.kernels.helion` transitively imports the third-party `helion`
    # package (see register.py). Since VLLM_USE_HELION_KERNELS defaults on, this
    # runs on every install; if `helion` is missing or broken, fall back to native
    # instead of crashing the forward.
    try:
        import vllm.kernels.helion  # noqa: F401  register ops
        from vllm.kernels.helion.register import _HOP_AVAILABLE

        ready = (not _HOP_AVAILABLE) and hasattr(torch.ops.vllm_helion, op_name)
    except Exception:
        ready = False
    if not ready:
        logger.warning_once(
            "VLLM_USE_HELION_KERNELS is set but Helion kernels are not available "
            "for '%s'; falling back to the native kernel. Install the `helion` "
            "package (pip install helion) to enable them.",
            op_name,
        )
    return ready


_PTG_OP_NAME = "per_token_group_fp8_quant"
_PTG_LAUNCH_CACHE: dict[tuple[Any, ...], Callable[..., None]] = {}
_PTG_FAST_LAUNCH_CACHE: dict[tuple[Any, ...], Callable[..., None]] = {}
_PTG_FAST_LAST: tuple[Any, ...] | None = None


def _ptg_dimensions(input: torch.Tensor) -> tuple[int, int] | None:
    if input.__class__ is not torch.Tensor:
        return None
    shape = input.shape
    if len(shape) != 2:
        return None
    return shape[0], shape[1]


def use_helion_per_token_group_fp8_quant() -> bool:
    return _use_helion_kernels() and not _is_compiling()


def _ptg_cache_key(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool,
    dummy_is_tma_aligned: bool,
    dimensions: tuple[int, int] | None = None,
) -> tuple[Any, ...] | None:
    if dimensions is None:
        dimensions = _ptg_dimensions(input)
        if dimensions is None:
            return None
    num_tokens, hidden_size = dimensions
    # Call-site invariant: fp8_utils.py already checked input/output shapes and
    # allocated output_s with the layout encoded by the two dummy flags. Keep
    # only guards that protect reuse of a cached aligned fused-launch closure.
    if group_size <= 0 or hidden_size % group_size != 0:
        return None
    groups_per_row = hidden_size // group_size
    if input.stride() != (hidden_size, 1) or output_q.stride() != (hidden_size, 1):
        return None
    # If an unusual view is unaligned, let Helion's own generic dispatch key
    # handle it instead of reusing a cached aligned launch closure.
    if (input.data_ptr() | output_q.data_ptr() | output_s.data_ptr()) & 15:
        return None
    if dummy_is_scale_transposed:
        tma_aligned_m = (
            ((num_tokens + 1023) // 1024) * 1024
            if dummy_is_tma_aligned
            else num_tokens
        )
        output_s_stride = (1, tma_aligned_m)
    else:
        output_s_stride = (groups_per_row, 1)
    return (
        num_tokens,
        hidden_size,
        input.dtype,
        input.device,
        output_q.dtype,
        output_s.dtype,
        output_s_stride,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        dummy_is_scale_transposed,
        dummy_is_tma_aligned,
    )


def _ptg_current_launches(
    kernel: Callable,
    args: tuple[object, ...],
) -> tuple[Callable[..., None], Callable[..., None] | None] | None:
    key_fn = getattr(kernel, "_fused_key_fn", None)
    recipes = getattr(kernel, "_fused_recipes", None)
    if key_fn is None or not recipes:
        return None
    try:
        launch = recipes.get(key_fn(args))
    except Exception:
        return None
    if launch is None:
        return None
    direct_launch = getattr(launch, "_helion_trusted_direct_launch", None)
    if direct_launch is None:
        direct_launch = getattr(launch, "_helion_direct_launch", None)
    if direct_launch is None:
        return None
    tensor_launch = getattr(launch, "_helion_trusted_tensor_direct_launch", None)
    if tensor_launch is None:
        tensor_launch = getattr(launch, "_helion_tensor_direct_launch", None)
    return direct_launch, tensor_launch


def _is_globals_changed(exc: Exception) -> bool:
    cls = exc.__class__
    return (
        cls.__name__ == "GlobalsChanged"
        and cls.__module__ == "helion.runtime._fused_launch"
    )


def route_per_token_group_fp8_quant(
    fn: Callable,
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
):
    """Fast route for vLLM's eager per-token-group fp8 quant call site."""
    if use_helion_per_token_group_fp8_quant():
        dimensions = _ptg_dimensions(input)
        if dimensions is not None:
            result = launch_per_token_group_fp8_quant(
                input,
                output_q,
                output_s,
                group_size,
                eps,
                fp8_min,
                fp8_max,
                scale_ue8m0,
                dummy_is_scale_transposed,
                dummy_is_tma_aligned,
                dimensions,
            )
            if result is not _PTG_FALLBACK:
                return result

    return fn(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        dummy_is_scale_transposed,
        dummy_is_tma_aligned,
    )


_PTG_FALLBACK = object()


def launch_per_token_group_fp8_quant(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool,
    dummy_is_tma_aligned: bool,
    dimensions: tuple[int, int],
):
    """Helion-only fast path for the 2D fp8_utils.py callsite."""
    global _PTG_FAST_LAST

    num_tokens, hidden_size = dimensions
    fast_cache_key: tuple[Any, ...] | None = None
    if (
        group_size > 0
        and hidden_size % group_size == 0
        and not (input.data_ptr() | output_q.data_ptr() | output_s.data_ptr()) & 15
    ):
        input_dtype = input.dtype
        input_device = input.device
        output_q_dtype = output_q.dtype
        output_s_dtype = output_s.dtype
        last = _PTG_FAST_LAST
        if (
            last is not None
            and num_tokens == last[0]
            and hidden_size == last[1]
            and input_dtype is last[2]
            and input_device == last[3]
            and output_q_dtype is last[4]
            and output_s_dtype is last[5]
            and group_size == last[6]
            and eps == last[7]
            and fp8_min == last[8]
            and fp8_max == last[9]
            and scale_ue8m0 == last[10]
            and dummy_is_scale_transposed == last[11]
            and dummy_is_tma_aligned == last[12]
        ):
            try:
                return last[13](input, output_q, output_s)
            except Exception as exc:
                if not _is_globals_changed(exc):
                    raise
                _PTG_FAST_LAST = None

        fast_cache_key = (
            num_tokens,
            hidden_size,
            input_dtype,
            input_device,
            output_q_dtype,
            output_s_dtype,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            dummy_is_scale_transposed,
            dummy_is_tma_aligned,
        )
    if fast_cache_key is not None:
        launch = _PTG_FAST_LAUNCH_CACHE.get(fast_cache_key)
        if launch is not None:
            try:
                result = launch(input, output_q, output_s)
                _PTG_FAST_LAST = (*fast_cache_key, launch)
                return result
            except Exception as exc:
                if not _is_globals_changed(exc):
                    raise
                _PTG_FAST_LAUNCH_CACHE.pop(fast_cache_key, None)
                if _PTG_FAST_LAST is not None and _PTG_FAST_LAST[13] is launch:
                    _PTG_FAST_LAST = None

    kernel = _checked_eager_kernel(_PTG_OP_NAME)
    if kernel is None:
        return _PTG_FALLBACK

    args = (
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        dummy_is_scale_transposed,
        dummy_is_tma_aligned,
    )
    cache_key = _ptg_cache_key(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        dummy_is_scale_transposed,
        dummy_is_tma_aligned,
        dimensions,
    )
    if cache_key is not None:
        launch = _PTG_LAUNCH_CACHE.get(cache_key)
        if launch is not None and getattr(kernel, "_fused_recipes", None):
            try:
                result = launch(
                    input,
                    output_q,
                    output_s,
                    group_size,
                    eps,
                    fp8_min,
                    fp8_max,
                    scale_ue8m0,
                    dummy_is_scale_transposed,
                    dummy_is_tma_aligned,
                )
                tensor_launch = getattr(
                    launch, "_helion_trusted_tensor_direct_launch", None
                )
                if tensor_launch is None:
                    tensor_launch = getattr(
                        launch, "_helion_tensor_direct_launch", None
                    )
                if fast_cache_key is not None and tensor_launch is not None:
                    _PTG_FAST_LAUNCH_CACHE[fast_cache_key] = tensor_launch
                    _PTG_FAST_LAST = (*fast_cache_key, tensor_launch)
                return result
            except Exception as exc:
                if not _is_globals_changed(exc):
                    raise
                _PTG_LAUNCH_CACHE.pop(cache_key, None)

    result = kernel(*args)
    if cache_key is not None:
        launches = _ptg_current_launches(kernel, args)
        if launches is not None:
            launch, tensor_launch = launches
            if len(_PTG_LAUNCH_CACHE) >= 128:
                _PTG_LAUNCH_CACHE.clear()
                _PTG_FAST_LAUNCH_CACHE.clear()
                _PTG_FAST_LAST = None
            _PTG_LAUNCH_CACHE[cache_key] = launch
            if fast_cache_key is not None and tensor_launch is not None:
                _PTG_FAST_LAUNCH_CACHE[fast_cache_key] = tensor_launch
                _PTG_FAST_LAST = (*fast_cache_key, tensor_launch)
    return result


def route_quant(op_name: str, fn: Callable, *args):
    """Dispatch a quant op to Helion when eager Helion routing is enabled.

    ``fn`` is the fallback, invoked as ``fn(*args)`` whenever Helion is not used:
    torch.compile tracing (``is_compiling()``), Helion unavailable/unsupported
    for ``op_name``, or the environment flag disabled. Passing the fallback in
    (rather than assuming ``torch.ops._C.<op_name>``) lets call sites route to
    any native implementation with the same signature as the Helion op.
    """
    if (
        op_name in _ROUTABLE_EAGER_OPS
        and _use_helion_kernels()
        and not _is_compiling()
        and (kernel := _checked_eager_kernel(op_name)) is not None
    ):
        return kernel(*args)
    return fn(*args)
