/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_LAUNCH_CONFIG_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_LAUNCH_CONFIG_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <variant>

#include "absl/container/inlined_vector.h"

namespace mosaic {
namespace gpu {

// CUDA-style kernel Dim3.
struct Dim3 {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

// A tensor map the host code encoded for a launch: the arguments it passed to
// cuTensorMapEncodeTiled (or cuTensorMapEncodeIm2col), in the driver's order
// and enums, and where its 128-byte encoding sits among the launch's arguments
// (the host argument `arg`, at `offset`, found by its bytes: the launch's
// arguments are copies). A launch capture (capture.h) states each map by these
// arguments, as a simulator takes it, not by its opaque encoding.
struct TensorMapEncoding {
  std::byte encoded[128] = {};
  int32_t arg = -1;  // -1: in no host argument of the launch.

  int32_t offset = 0;
  uint32_t data_type = 0;
  uint32_t rank = 0;
  uint64_t global_address = 0;
  uint64_t dims[5] = {1, 1, 1, 1, 1};
  uint64_t strides[4] = {0, 0, 0, 0};  // In bytes, one fewer than `rank`.
  uint32_t box[5] = {1, 1, 1, 1, 1};
  uint32_t element_strides[5] = {1, 1, 1, 1, 1};
  uint32_t interleave = 0;
  uint32_t swizzle = 0;
  uint32_t l2_promotion = 0;
  uint32_t oob_fill = 0;
  // cuTensorMapEncodeIm2col's own, `box` unused.
  bool im2col = false;
  int32_t lower[3] = {0, 0, 0};
  int32_t upper[3] = {0, 0, 0};
  uint32_t channels_per_pixel = 0;
  uint32_t pixels_per_column = 0;
};

// Describes a kernel launch.
struct MosaicKernelSpec {
  struct DeviceArg {
    void* ptr = nullptr;
  };

  // A byval argument (e.g. a TMA descriptor bundle) whose payload occupies
  // `host_bytes[offset, offset + size)`.
  struct HostArg {
    int32_t offset = 0;
    int32_t size = 0;
  };

  using Arg = std::variant<DeviceArg, HostArg>;

  Dim3 grid;
  Dim3 block;
  Dim3 cluster = {0, 0, 0};  // {0, 0, 0} => no cluster attribute.
  uint32_t smem_bytes = 0;
  bool uses_pdl = false;

  absl::InlinedVector<Arg, 8> args;
  // Owns copied host-argument bytes (e.g. TMA descriptor bundles).
  // `HostArg::offset` above indexes into this vector.
  absl::InlinedVector<std::byte, 256> host_bytes;
  // The tensor maps encoded into the host arguments, each by its encoding's
  // arguments.
  absl::InlinedVector<TensorMapEncoding, 4> tensor_maps;

  bool IsHostArg(int32_t arg_index) const {
    return std::holds_alternative<HostArg>(args[arg_index]);
  }

  // Appends a device pointer argument.
  // Pre: `ptr` must be a valid device pointer.
  void AddDeviceArg(void* ptr) { args.push_back(DeviceArg{ptr}); }

  // Copies `size` bytes of byval payload into `host_bytes` and appends the
  // corresponding argument.
  // Pre: `size` must be positive.
  void AddHostArg(const void* data, int32_t size) {
    const size_t offset = host_bytes.size();
    host_bytes.resize(offset + size);
    std::memcpy(host_bytes.data() + offset, data, size);
    args.push_back(HostArg{static_cast<int32_t>(offset), size});
  }

  void Clear() {
    args.clear();
    host_bytes.clear();
    tensor_maps.clear();
  }

  // Pre-allocates space for arguments. Not mandatory to call this method.
  // However, it avoids reallocations when the arguments are added.
  void Reserve(int32_t num_args, int32_t host_bytes_size) {
    args.reserve(num_args);
    host_bytes.reserve(host_bytes_size);
  }

  // Returns a vector of kernel argument pointers, suitable for passing to
  // cuLaunchKernelEx.
  //
  // Borrows from `*this`: the result must not outlive this spec, and is
  // invalidated by any mutation of `args` or `host_bytes`.
  [[nodiscard]] absl::InlinedVector<void*, 8> kernel_params() {
    absl::InlinedVector<void*, 8> params;
    params.reserve(args.size());
    for (Arg& arg : args) {
      if (auto* host = std::get_if<HostArg>(&arg)) {
        // A byval argument is passed as the address of its payload.
        params.push_back(host_bytes.data() + host->offset);
      } else {
        // A device pointer is passed as the address of the pointer value.
        params.push_back(&std::get<DeviceArg>(arg).ptr);
      }
    }
    return params;
  }
};

}  // namespace gpu
}  // namespace mosaic

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_LAUNCH_CONFIG_H_
