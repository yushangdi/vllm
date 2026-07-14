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
_PTG_LAUNCH_CACHE: dict[tuple[Any, ...], Callable[[tuple[object, ...]], None]] = {}
# Below this size the native vLLM op wins on H100 because launch overhead
# dominates. Larger prefill tensors still benefit from Helion's faster kernel.
_PTG_HELION_MIN_ELEMENTS = 5_000_000


def _ptg_dimensions(input: torch.Tensor) -> tuple[int, int] | None:
    if input.__class__ is not torch.Tensor:
        return None
    shape = input.shape
    if len(shape) != 2:
        return None
    return shape[0], shape[1]


def _ptg_use_helion(num_tokens: int, hidden_size: int) -> bool:
    return num_tokens * hidden_size >= _PTG_HELION_MIN_ELEMENTS


def use_helion_per_token_group_fp8_quant(input: torch.Tensor) -> bool:
    if not _use_helion_kernels() or _is_compiling():
        return False
    dimensions = _ptg_dimensions(input)
    return dimensions is not None and _ptg_use_helion(*dimensions)


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
    if (
        input.__class__ is not torch.Tensor
        or output_q.__class__ is not torch.Tensor
        or output_s.__class__ is not torch.Tensor
        or type(group_size) is not int
        or type(eps) is not float
        or type(fp8_min) is not float
        or type(fp8_max) is not float
        or type(scale_ue8m0) is not bool
        or type(dummy_is_scale_transposed) is not bool
        or type(dummy_is_tma_aligned) is not bool
    ):
        return None

    if dimensions is None:
        dimensions = _ptg_dimensions(input)
        if dimensions is None:
            return None
    num_tokens, hidden_size = dimensions
    if output_q.ndim != 2 or output_s.ndim != 2:
        return None
    if group_size <= 0:
        return None

    if hidden_size % group_size != 0:
        return None
    if output_q.shape != (num_tokens, hidden_size):
        return None
    if output_s.shape != (num_tokens, hidden_size // group_size):
        return None
    if input.device != output_q.device or input.device != output_s.device:
        return None
    if input.stride() != (hidden_size, 1) or output_q.stride() != (hidden_size, 1):
        return None
    # If an unusual view is unaligned, let Helion's own generic dispatch key
    # handle it instead of reusing a cached aligned launch closure.
    if (input.data_ptr() | output_q.data_ptr() | output_s.data_ptr()) & 15:
        return None
    if (
        getattr(input, "_dynamo_static_indices", None) is not None
        or getattr(output_q, "_dynamo_static_indices", None) is not None
        or getattr(output_s, "_dynamo_static_indices", None) is not None
    ):
        return None

    return (
        num_tokens,
        hidden_size,
        input.dtype,
        input.device,
        output_q.dtype,
        output_s.dtype,
        output_s.stride(),
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        dummy_is_scale_transposed,
        dummy_is_tma_aligned,
    )


def _ptg_current_launch(
    kernel: Callable,
    args: tuple[object, ...],
) -> Callable[[tuple[object, ...]], None] | None:
    key_fn = getattr(kernel, "_fused_key_fn", None)
    recipes = getattr(kernel, "_fused_recipes", None)
    if key_fn is None or not recipes:
        return None
    try:
        return recipes.get(key_fn(args))
    except Exception:
        return None


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
    if use_helion_per_token_group_fp8_quant(input):
        dimensions = _ptg_dimensions(input)
        assert dimensions is not None
        kernel = _checked_eager_kernel(_PTG_OP_NAME)
        if kernel is not None:
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
                        return launch(args)
                    except Exception as exc:
                        if not _is_globals_changed(exc):
                            raise
                        _PTG_LAUNCH_CACHE.pop(cache_key, None)

            result = kernel(*args)
            if cache_key is not None:
                launch = _ptg_current_launch(kernel, args)
                if launch is not None:
                    if len(_PTG_LAUNCH_CACHE) >= 128:
                        _PTG_LAUNCH_CACHE.clear()
                    _PTG_LAUNCH_CACHE[cache_key] = launch
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
