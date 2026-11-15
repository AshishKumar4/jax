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

#ifndef JAXLIB_MOSAIC_GPU_CAPTURE_H_
#define JAXLIB_MOSAIC_GPU_CAPTURE_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "jaxlib/mosaic/gpu/launch_config.h"

namespace mosaic {
namespace gpu {

// A launch capture: with MOSAIC_GPU_CAPTURE_LAUNCHES=<dir> set, each Mosaic
// GPU launch writes <dir>/<pid>-<seq>-<kernel>/, a one-launch spec a simulator
// runs bit-exactly without a GPU (nvgpu-lean's NV/Launch/Spec.lean format):
//
//   spec.json            the kernel ("code": kernel.cubin, "function"), the
//                        grid, block, cluster and dynamic shared memory, every
//                        buffer the launch can address with its pre-launch
//                        contents, each tensor map by its encoding's
//                        arguments, the parameters by ordinal, and "expect":
//                        each operand's bytes after the launch (a workspace's,
//                        ws<i>, are the launch's own, held to none).
//   kernel.cubin         the binary the driver loaded.
//   <buffer>.bin         a buffer's bytes before the launch.
//   <buffer>.expected.bin  an operand's bytes after.
//
// Buffers are the custom call's operands (in<i>) and results (out<i>) and any
// other device allocation a parameter or tensor map points into (ws<i>:
// Mosaic's workspace, a collective's metadata), found by the driver's
// allocation range of the pointer. "mosaic" carries provenance a simulator
// may skip: the device, programmatic dependent launch, each operand's shape.
// Capturing synchronizes the stream before and after the launch.

// Whether launches are captured, and where (empty: not captured).
const std::string& CaptureDirectory();

struct CapturedBuffer {
  std::string name;  // in<i> or out<i>.
  void* data = nullptr;
  uint64_t bytes = 0;
  std::string dtype;  // XLA's name of its element type.
  std::vector<int64_t> dims;
};

// Captures the launch of `function` with `spec` on `stream`: writes its
// buffers' contents before calling `launch`, and after it the results.
absl::Status CaptureLaunch(std::string_view kernel_name,
                           std::string_view gpu_binary, CUstream stream,
                           const MosaicKernelSpec& spec,
                           const std::vector<CapturedBuffer>& operands,
                           absl::FunctionRef<absl::Status()> launch);

// Writes a launch's spec with no device: `spec` as the kernel's host function
// filled it over `operands`, each `data` a host buffer holding its contents
// before the launch, into <root>/<pid>-<seq>-<kernel>/, the directory it
// returns. No buffer outside the operands is captured (a pointer into none
// fails it), and no "expect": the launch never ran. `sm` the target it was
// compiled for (sm_90a, sm_100a).
absl::StatusOr<std::string> CaptureDeviceless(
    std::string_view kernel_name, std::string_view gpu_binary,
    std::string_view sm, const MosaicKernelSpec& spec,
    const std::vector<CapturedBuffer>& operands, std::string_view root);

}  // namespace gpu
}  // namespace mosaic

#endif  // JAXLIB_MOSAIC_GPU_CAPTURE_H_
