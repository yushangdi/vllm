# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Call-site routing for eager Helion quant ops.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ROUTABLE_EAGER_OPS = frozenset(
    {
        "per_token_group_fp8_quant",
    }
)
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


def route_quant(op_name: str, fn: Callable, *args):
    """Dispatch a quant op to Helion iff currently capturing a CUDA graph.

    ``fn`` is the fallback, invoked as ``fn(*args)`` whenever Helion is not used:
    plain eager, torch.compile tracing (``is_compiling()``), or Helion
    unavailable/unsupported for ``op_name``. Passing the fallback in (rather than
    assuming ``torch.ops._C.<op_name>``) lets call sites route to any native
    implementation with the same signature as the Helion op.
    """
    import vllm.envs as envs

    if (
        op_name in _ROUTABLE_EAGER_OPS
        and envs.VLLM_USE_HELION_KERNELS
        and not torch.compiler.is_compiling()
        and _helion_available(op_name)
    ):
        return _eager_kernel(op_name)(*args)
    return fn(*args)
