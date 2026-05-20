# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 NVFP4 Helion kernel registrations."""

# Importing this module executes the @register_kernel decorators in the
# specialized model module so the shared Helion autotune script can discover
# the DS32 kernels without importing the model layer.
import vllm.model_executor.specialized_models.deepseek_v3_2_nvfp4.helion_kernels  # noqa: F401
