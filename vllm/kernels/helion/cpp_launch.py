# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import functools
import os

from torch.utils.cpp_extension import load_inline

# Prototype C++ launch helpers for Helion eager fast paths.


_CPP_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <torch/extension.h>

#include <cstdint>
#include <sstream>

#define VLLM_HELION_CUDA_CHECK(EXPR)                                      \
  do {                                                                    \
    CUresult _status = (EXPR);                                            \
    if (_status != CUDA_SUCCESS) {                                        \
      const char* _name = nullptr;                                        \
      const char* _str = nullptr;                                         \
      cuGetErrorName(_status, &_name);                                    \
      cuGetErrorString(_status, &_str);                                   \
      std::ostringstream _oss;                                            \
      _oss << "CUDA driver error";                                        \
      if (_name) _oss << " " << _name;                                    \
      if (_str) _oss << ": " << _str;                                     \
      TORCH_CHECK(false, _oss.str());                                     \
    }                                                                     \
  } while (0)

static CUdeviceptr device_pointer(const at::Tensor& tensor) {
  CUdeviceptr dev_ptr = 0;
  VLLM_HELION_CUDA_CHECK(cuPointerGetAttribute(
      &dev_ptr,
      CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
      reinterpret_cast<CUdeviceptr>(tensor.data_ptr())));
  return dev_ptr;
}

static int scalar_type_tag(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::BFloat16:
      return 1;
    case c10::ScalarType::Half:
      return 2;
    case c10::ScalarType::Float:
      return 3;
    case c10::ScalarType::Float8_e4m3fn:
      return 4;
    default:
      return 0;
  }
}

class PtgFp8Launcher {
 public:
  PtgFp8Launcher(
      uint64_t function,
      int64_t num_tokens,
      int64_t hidden_size,
      int64_t input_dtype_tag,
      int64_t device_index,
      int64_t output_q_dtype_tag,
      int64_t output_s_dtype_tag,
      int64_t group_size,
      double eps,
      double fp8_min,
      double fp8_max,
      bool scale_ue8m0,
      bool scale_transposed,
      bool tma_aligned,
      bool output_s_stride0_const,
      int64_t grid_x,
      int64_t grid_y,
      int64_t grid_z,
      int64_t num_warps,
      int64_t shared_mem,
      int64_t c3,
      int64_t c4,
      int64_t c5,
      int64_t c6,
      int64_t c7,
      int64_t c8,
      int64_t c9,
      int64_t c10,
      double c11,
      double c12,
      bool c13,
      double c14)
      : function_(function),
        num_tokens_(num_tokens),
        hidden_size_(hidden_size),
        input_dtype_tag_(static_cast<int>(input_dtype_tag)),
        device_index_(device_index),
        output_q_dtype_tag_(static_cast<int>(output_q_dtype_tag)),
        output_s_dtype_tag_(static_cast<int>(output_s_dtype_tag)),
        group_size_(group_size),
        eps_(eps),
        fp8_min_(fp8_min),
        fp8_max_(fp8_max),
        scale_ue8m0_(scale_ue8m0),
        scale_transposed_(scale_transposed),
        tma_aligned_(tma_aligned),
        output_s_stride0_const_(output_s_stride0_const),
        grid_x_(grid_x),
        grid_y_(grid_y),
        grid_z_(grid_z),
        num_warps_(num_warps),
        shared_mem_(shared_mem),
        c3_(static_cast<int32_t>(c3)),
        c4_(static_cast<int32_t>(c4)),
        c5_(static_cast<int32_t>(c5)),
        c6_(static_cast<int32_t>(c6)),
        c7_(static_cast<int32_t>(c7)),
        c8_(static_cast<int32_t>(c8)),
        c9_(static_cast<int32_t>(c9)),
        c10_(static_cast<int32_t>(c10)),
        c11_(static_cast<float>(c11)),
        c12_(static_cast<float>(c12)),
        c13_(static_cast<uint32_t>(c13 ? 1 : 0)),
        c14_(static_cast<float>(c14)) {}

  bool launch(
      const at::Tensor& input,
      const at::Tensor& output_q,
      const at::Tensor& output_s,
      int64_t group_size,
      double eps,
      double fp8_min,
      double fp8_max,
      bool scale_ue8m0,
      bool scale_transposed,
      bool tma_aligned) const {
    if (!input.is_cuda() || !output_q.is_cuda() || !output_s.is_cuda()) {
      return false;
    }
    if (input.dim() != 2 || output_q.dim() != 2 || output_s.dim() != 2) {
      return false;
    }
    if (input.size(0) != num_tokens_ || input.size(1) != hidden_size_) {
      return false;
    }
    if (output_q.size(0) != num_tokens_ || output_q.size(1) != hidden_size_) {
      return false;
    }
    if (output_s.size(0) != num_tokens_ ||
        output_s.size(1) != hidden_size_ / group_size_) {
      return false;
    }
    if (scalar_type_tag(input.scalar_type()) != input_dtype_tag_ ||
        scalar_type_tag(output_q.scalar_type()) != output_q_dtype_tag_ ||
        scalar_type_tag(output_s.scalar_type()) != output_s_dtype_tag_) {
      return false;
    }
    if (input.get_device() != device_index_ ||
        output_q.get_device() != device_index_ ||
        output_s.get_device() != device_index_) {
      return false;
    }
    if (group_size != group_size_ || eps != eps_ || fp8_min != fp8_min_ ||
        fp8_max != fp8_max_ || scale_ue8m0 != scale_ue8m0_ ||
        scale_transposed != scale_transposed_ || tma_aligned != tma_aligned_) {
      return false;
    }
    if (input.stride(0) != hidden_size_ || input.stride(1) != 1 ||
        output_q.stride(0) != hidden_size_ || output_q.stride(1) != 1) {
      return false;
    }
    if (output_s.stride(0) != c9_ || output_s.stride(1) != c10_) {
      return false;
    }
    const uintptr_t ptr_bits =
        reinterpret_cast<uintptr_t>(input.data_ptr()) |
        reinterpret_cast<uintptr_t>(output_q.data_ptr()) |
        reinterpret_cast<uintptr_t>(output_s.data_ptr());
    if ((ptr_bits & 15) != 0) {
      return false;
    }

    c10::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_index_))
            .stream();

    CUdeviceptr input_ptr = device_pointer(input);
    CUdeviceptr output_s_ptr = device_pointer(output_s);
    CUdeviceptr output_q_ptr = device_pointer(output_q);

    CUdeviceptr global_scratch = 0;
    CUdeviceptr profile_scratch = 0;
    int32_t output_s_stride_arg = output_s_stride0_const_ ? c10_ : c9_;
    void* args[] = {
        &input_ptr,
        &output_s_ptr,
        &output_q_ptr,
        const_cast<int32_t*>(&c3_),
        const_cast<int32_t*>(&c4_),
        const_cast<int32_t*>(&c6_),
        const_cast<int32_t*>(&c7_),
        &output_s_stride_arg,
        const_cast<float*>(&c11_),
        const_cast<float*>(&c12_),
        const_cast<uint32_t*>(&c13_),
        const_cast<float*>(&c14_),
        &global_scratch,
        &profile_scratch,
    };

    CUlaunchConfig config = {};
    config.gridDimX = static_cast<unsigned int>(grid_x_);
    config.gridDimY = static_cast<unsigned int>(grid_y_);
    config.gridDimZ = static_cast<unsigned int>(grid_z_);
    config.blockDimX = static_cast<unsigned int>(32 * num_warps_);
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = static_cast<unsigned int>(shared_mem_);
    config.hStream = stream;
    config.attrs = nullptr;
    config.numAttrs = 0;

    CUresult launch_status = cuLaunchKernelEx(
        &config, reinterpret_cast<CUfunction>(function_), args, nullptr);
    return launch_status == CUDA_SUCCESS;
  }

 private:
  uint64_t function_;
  int64_t num_tokens_;
  int64_t hidden_size_;
  int input_dtype_tag_;
  int64_t device_index_;
  int output_q_dtype_tag_;
  int output_s_dtype_tag_;
  int64_t group_size_;
  double eps_;
  double fp8_min_;
  double fp8_max_;
  bool scale_ue8m0_;
  bool scale_transposed_;
  bool tma_aligned_;
  bool output_s_stride0_const_;
  int64_t grid_x_;
  int64_t grid_y_;
  int64_t grid_z_;
  int64_t num_warps_;
  int64_t shared_mem_;
  int32_t c3_;
  int32_t c4_;
  int32_t c5_;
  int32_t c6_;
  int32_t c7_;
  int32_t c8_;
  int32_t c9_;
  int32_t c10_;
  float c11_;
  float c12_;
  uint32_t c13_;
  float c14_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<PtgFp8Launcher>(m, "PtgFp8Launcher")
      .def(pybind11::init<
           uint64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           double,
           double,
           double,
           bool,
           bool,
           bool,
           bool,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           double,
           double,
           bool,
           double>())
      .def("launch", &PtgFp8Launcher::launch);
}
"""


@functools.cache
def module():
    cuda_home = os.environ.get("CUDA_HOME")
    extra_include_paths = [f"{cuda_home}/include"] if cuda_home else []
    return load_inline(
        name="vllm_helion_cpp_launch",
        cpp_sources=[_CPP_SRC],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=extra_include_paths,
        verbose=bool(os.environ.get("VLLM_HELION_CPP_LAUNCH_VERBOSE")),
    )
