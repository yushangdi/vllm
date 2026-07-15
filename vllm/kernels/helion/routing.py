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
_PTG_CPP_LAUNCH_CACHE: dict[tuple[Any, ...], object] = {}
_PTG_FAST_LAST: tuple[Any, ...] | None = None
_PTG_CPP_LAST: tuple[Any, ...] | None = None


def _ptg_dtype_tag(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return 1
    if dtype is torch.float16:
        return 2
    if dtype is torch.float32:
        return 3
    if dtype is torch.float8_e4m3fn:
        return 4
    return 0


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


def _ptg_make_cpp_launcher(
    fast_cache_key: tuple[Any, ...],
    tensor_launch: Callable[..., None],
) -> object | None:
    try:
        from vllm.kernels.helion import cpp_launch

        namespace = tensor_launch.__globals__  # type: ignore[attr-defined]
        compiled = namespace["_compiled"]
        metadata = compiled.metadata
        signature = getattr(compiled.src, "signature", None)
        constants = getattr(compiled.src, "constants", None)
        signature_values = list(signature.values())
        if (
            signature_values[:9]
            != [
                "*bf16",
                "*fp32",
                "*fp8e4nv",
                "i32",
                "i32",
                "constexpr",
                "i32",
                "i32",
                "constexpr",
            ]
            or signature_values[11:]
            != [
                "fp32",
                "fp32",
                "u1",
                "fp32",
            ]
            or (signature_values[9], signature_values[10])
            not in {
                ("i32", "constexpr"),
                ("constexpr", "i32"),
            }
        ):
            return None
        output_s_stride0_const = signature_values[9] == "constexpr"
        scale_const_index = (9,) if output_s_stride0_const else (10,)
        scale_const_value = namespace["_c9" if output_s_stride0_const else "_c10"]
        if not isinstance(constants, dict) or (
            constants.get((5,)) != namespace["_c5"]
            or constants.get((8,)) != namespace["_c8"]
            or constants.get(scale_const_index) != scale_const_value
        ):
            return None
        if (
            int(getattr(metadata, "num_ctas", 1)) != 1
            or bool(getattr(metadata, "launch_cooperative_grid", False))
            or bool(getattr(metadata, "launch_pdl", False))
            or int(getattr(metadata, "global_scratch_size", 0)) != 0
            or int(getattr(metadata, "profile_scratch_size", 0)) != 0
        ):
            return None
        return cpp_launch.module().PtgFp8Launcher(
            int(compiled.function),
            int(fast_cache_key[0]),
            int(fast_cache_key[1]),
            _ptg_dtype_tag(fast_cache_key[2]),
            int(fast_cache_key[3].index or 0),
            _ptg_dtype_tag(fast_cache_key[4]),
            _ptg_dtype_tag(fast_cache_key[5]),
            int(fast_cache_key[6]),
            float(fast_cache_key[7]),
            float(fast_cache_key[8]),
            float(fast_cache_key[9]),
            bool(fast_cache_key[10]),
            bool(fast_cache_key[11]),
            bool(fast_cache_key[12]),
            output_s_stride0_const,
            int(namespace["_gx"]),
            int(namespace["_gy"]),
            int(namespace["_gz"]),
            int(metadata.num_warps),
            int(metadata.shared),
            int(namespace["_c3"]),
            int(namespace["_c4"]),
            int(namespace["_c5"]),
            int(namespace["_c6"]),
            int(namespace["_c7"]),
            int(namespace["_c8"]),
            int(namespace["_c9"]),
            int(namespace["_c10"]),
            float(namespace["_c11"]),
            float(namespace["_c12"]),
            bool(namespace["_c13"]),
            float(namespace["_c14"]),
        )
    except Exception:
        return None


def _ptg_cache_fast_launch(
    fast_cache_key: tuple[Any, ...],
    tensor_launch: Callable[..., None],
) -> None:
    global _PTG_CPP_LAST, _PTG_FAST_LAST

    _PTG_FAST_LAUNCH_CACHE[fast_cache_key] = tensor_launch
    _PTG_FAST_LAST = (*fast_cache_key, tensor_launch)
    cpp_launcher = _ptg_make_cpp_launcher(fast_cache_key, tensor_launch)
    if cpp_launcher is not None:
        _PTG_CPP_LAUNCH_CACHE[fast_cache_key] = cpp_launcher
        _PTG_CPP_LAST = (*fast_cache_key, cpp_launcher)


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
    if try_launch_per_token_group_fp8_quant(
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
    ):
        return None

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


def try_launch_per_token_group_fp8_quant(
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
) -> bool:
    """Helion-only fast path for the 2D fp8_utils.py callsite."""
    global _PTG_CPP_LAST, _PTG_FAST_LAST

    if not use_helion_per_token_group_fp8_quant() or not output_q.is_contiguous():
        return False
    cpp_last = _PTG_CPP_LAST
    if cpp_last is not None:
        try:
            if cpp_last[13].launch(
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
            ):
                return True
        except Exception:
            _PTG_CPP_LAST = None
            raise
    if dimensions is None:
        dimensions = _ptg_dimensions(input)
        if dimensions is None:
            return False
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
                last[13](input, output_q, output_s)
                return True
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
        cpp_launch = _PTG_CPP_LAUNCH_CACHE.get(fast_cache_key)
        if cpp_launch is not None:
            if cpp_launch.launch(
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
            ):
                _PTG_CPP_LAST = (*fast_cache_key, cpp_launch)
                return True
            _PTG_CPP_LAUNCH_CACHE.pop(fast_cache_key, None)
            if _PTG_CPP_LAST is not None and _PTG_CPP_LAST[13] is cpp_launch:
                _PTG_CPP_LAST = None
    if fast_cache_key is not None:
        launch = _PTG_FAST_LAUNCH_CACHE.get(fast_cache_key)
        if launch is not None:
            try:
                launch(input, output_q, output_s)
                _PTG_FAST_LAST = (*fast_cache_key, launch)
                return True
            except Exception as exc:
                if not _is_globals_changed(exc):
                    raise
                _PTG_FAST_LAUNCH_CACHE.pop(fast_cache_key, None)
                _PTG_CPP_LAUNCH_CACHE.pop(fast_cache_key, None)
                if _PTG_FAST_LAST is not None and _PTG_FAST_LAST[13] is launch:
                    _PTG_FAST_LAST = None
                if (
                    _PTG_CPP_LAST is not None
                    and _PTG_CPP_LAST[:13] == fast_cache_key
                ):
                    _PTG_CPP_LAST = None

    kernel = _checked_eager_kernel(_PTG_OP_NAME)
    if kernel is None:
        return False

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
                launch(
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
                    _ptg_cache_fast_launch(fast_cache_key, tensor_launch)
                return True
            except Exception as exc:
                if not _is_globals_changed(exc):
                    raise
                _PTG_LAUNCH_CACHE.pop(cache_key, None)

    kernel(*args)
    if cache_key is not None:
        launches = _ptg_current_launches(kernel, args)
        if launches is not None:
            launch, tensor_launch = launches
            if len(_PTG_LAUNCH_CACHE) >= 128:
                _PTG_LAUNCH_CACHE.clear()
                _PTG_FAST_LAUNCH_CACHE.clear()
                _PTG_CPP_LAUNCH_CACHE.clear()
                _PTG_FAST_LAST = None
                _PTG_CPP_LAST = None
            _PTG_LAUNCH_CACHE[cache_key] = launch
            if fast_cache_key is not None and tensor_launch is not None:
                _ptg_cache_fast_launch(fast_cache_key, tensor_launch)
    return True


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
