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

#include "jaxlib/mosaic/gpu/capture.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/cleanup/cleanup.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "jaxlib/mosaic/gpu/launch_config.h"

namespace mosaic {
namespace gpu {
namespace {

absl::Status Cuda(CUresult result, std::string_view what) {
  if (result == CUDA_SUCCESS) return absl::OkStatus();
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  return absl::InternalError(absl::StrCat("launch capture: ", what, " failed: ",
                                          name ? name : "unknown error"));
}

// A buffer of the capture: a device range the launch can address.
struct Buffer {
  std::string name;
  CUdeviceptr base = 0;
  uint64_t bytes = 0;
};

// The largest allocation a launch's workspace is captured whole from.
constexpr size_t kWorkspaceBytes = size_t{1} << 30;

// The buffer `address` falls in, finding the device allocation holding it
// where no buffer does (a workspace, ws<i>).
absl::StatusOr<const Buffer*> Holding(std::vector<Buffer>& buffers,
                                      uint64_t address) {
  for (const Buffer& buffer : buffers) {
    if (address >= buffer.base && address < buffer.base + buffer.bytes) {
      return &buffer;
    }
  }
  CUdeviceptr base = 0;
  size_t bytes = 0;
  const auto device_address = static_cast<CUdeviceptr>(address);
  if (absl::Status found =
          Cuda(cuMemGetAddressRange(&base, &bytes, device_address),
               absl::StrFormat("finding the allocation of 0x%x", address));
      !found.ok()) {
    return found;
  }
  // An operand of XLA's is a slice of its allocator's pool: a pointer in no
  // operand whose allocation is a pool names no buffer a spec can hold.
  if (bytes > kWorkspaceBytes) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "launch capture: 0x%x lies in no operand of the launch, and its "
        "allocation, %d bytes, is a pool rather than a buffer",
        address, bytes));
  }
  buffers.push_back({absl::StrCat("ws", buffers.size()), base, bytes});
  return &buffers.back();
}

absl::Status Write(const std::string& path, const void* data, size_t bytes) {
  std::error_code error;
  llvm::raw_fd_ostream out(path, error, llvm::sys::fs::OF_None);
  if (error) {
    return absl::InternalError(
        absl::StrCat("launch capture: writing ", path, ": ", error.message()));
  }
  out.write(static_cast<const char*>(data), bytes);
  return absl::OkStatus();
}

// Each buffer's bytes, written to <dir>/<name><suffix>.bin.
absl::Status Contents(const std::vector<Buffer>& buffers, CUstream stream,
                      const std::string& dir, std::string_view suffix) {
  if (absl::Status synced = Cuda(cuStreamSynchronize(stream), "a sync");
      !synced.ok()) {
    return synced;
  }
  for (const Buffer& buffer : buffers) {
    std::vector<std::byte> host(buffer.bytes);
    if (absl::Status read = Cuda(cuMemcpyDtoH(host.data(), buffer.base,
                                              buffer.bytes),
                                 absl::StrCat("reading ", buffer.name));
        !read.ok()) {
      return read;
    }
    if (absl::Status written =
            Write(absl::StrCat(dir, "/", buffer.name, suffix, ".bin"),
                  host.data(), host.size());
        !written.ok()) {
      return written;
    }
  }
  return absl::OkStatus();
}

std::string Hex(const std::byte* data, size_t bytes) {
  std::string hex;
  hex.reserve(2 * bytes);
  for (size_t i = 0; i < bytes; ++i) {
    absl::StrAppendFormat(&hex, "%02x", static_cast<unsigned>(data[i]));
  }
  return hex;
}

// A JSON string of `text`: quotes and backslashes escaped.
std::string Quoted(std::string_view text) {
  std::string quoted = "\"";
  for (char c : text) {
    if (c == '"' || c == '\\') quoted += '\\';
    quoted += c;
  }
  return quoted + "\"";
}

// The simulator's names of the driver's enums (NV/Launch/Spec.lean).
std::string_view DataType(uint32_t type) {
  switch (type) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8: return "u8";
    case CU_TENSOR_MAP_DATA_TYPE_UINT16: return "u16";
    case CU_TENSOR_MAP_DATA_TYPE_UINT32: return "u32";
    case CU_TENSOR_MAP_DATA_TYPE_INT32: return "s32";
    case CU_TENSOR_MAP_DATA_TYPE_UINT64: return "u64";
    case CU_TENSOR_MAP_DATA_TYPE_INT64: return "s64";
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16: return "f16";
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32: return "f32";
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64: return "f64";
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: return "bf16";
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ: return "f32_ftz";
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32: return "tf32";
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ: return "tf32_ftz";
    default: return "";
  }
}

constexpr std::string_view kInterleave[] = {"none", "16B", "32B"};
constexpr std::string_view kSwizzle[] = {
    "none", "32B", "64B", "128B", "128B_atom_32B", "128B_atom_32B_flip_8B",
    "128B_atom_64B"};
constexpr std::string_view kL2Promotion[] = {"none", "64B", "128B", "256B"};
constexpr std::string_view kOobFill[] = {"none", "nan_request_zero_fma"};

template <size_t N>
std::string Named(const std::string_view (&names)[N], uint32_t value) {
  return Quoted(value < N ? names[value] : "");
}

template <typename T>
std::string List(const T* values, size_t count) {
  return absl::StrCat("[", absl::StrJoin(values, values + count, ", "), "]");
}

// A tensor map of the spec, `name` its name, located in `buffers`.
absl::StatusOr<std::string> Map(const TensorMapEncoding& map,
                                std::string_view name,
                                std::vector<Buffer>& buffers) {
  if (DataType(map.data_type).empty()) {
    return absl::UnimplementedError(absl::StrCat(
        "launch capture: tensor map dtype ", map.data_type,
        " has no simulator name"));
  }
  absl::StatusOr<const Buffer*> held = Holding(buffers, map.global_address);
  if (!held.ok()) return held.status();
  std::string spec = absl::StrCat(
      "{ \"name\": ", Quoted(name), ", \"buffer\": ", Quoted((*held)->name),
      ", \"offset\": ", map.global_address - (*held)->base,
      ", \"dtype\": ", Quoted(DataType(map.data_type)),
      ", \"dims\": ", List(map.dims, map.rank),
      ", \"strides\": ", List(map.strides, map.rank - 1),
      ", \"elementStrides\": ", List(map.element_strides, map.rank),
      ", \"interleave\": ", Named(kInterleave, map.interleave),
      ", \"swizzle\": ", Named(kSwizzle, map.swizzle),
      ", \"l2Promotion\": ", Named(kL2Promotion, map.l2_promotion),
      ", \"oobFill\": ", Named(kOobFill, map.oob_fill));
  if (map.im2col) {
    absl::StrAppend(&spec, ", \"im2col\": { \"lower\": ",
                    List(map.lower, map.rank - 2), ", \"upper\": ",
                    List(map.upper, map.rank - 2), ", \"channelsPerPixel\": ",
                    map.channels_per_pixel, ", \"pixelsPerColumn\": ",
                    map.pixels_per_column, " }");
  } else {
    absl::StrAppend(&spec, ", \"box\": ", List(map.box, map.rank));
  }
  return spec + " }";
}

// The parameter `index` of `spec`: a device pointer into a buffer, or a byval
// payload of tensor maps (by name) and the bytes between them (hex).
absl::StatusOr<std::string> Param(const MosaicKernelSpec& spec, int32_t index,
                                  std::vector<Buffer>& buffers) {
  const MosaicKernelSpec::Arg& arg = spec.args[index];
  if (const auto* device = std::get_if<MosaicKernelSpec::DeviceArg>(&arg)) {
    uint64_t address = reinterpret_cast<uint64_t>(device->ptr);
    absl::StatusOr<const Buffer*> held = Holding(buffers, address);
    if (!held.ok()) return held.status();
    return absl::StrCat("{ \"ordinal\": ", index, ", \"ptr\": ",
                        Quoted((*held)->name), ", \"offset_in\": ",
                        address - (*held)->base, " }");
  }
  const auto& host = std::get<MosaicKernelSpec::HostArg>(arg);
  const std::byte* payload = spec.host_bytes.data() + host.offset;
  // A parameter that is one tensor map whole is the map's by name.
  for (size_t m = 0; m < spec.tensor_maps.size(); ++m) {
    const TensorMapEncoding& map = spec.tensor_maps[m];
    if (map.arg == index && map.offset == 0 &&
        host.size == static_cast<int32_t>(sizeof(map.encoded))) {
      return absl::StrCat("{ \"ordinal\": ", index, ", \"tensorMap\": \"map",
                          m, "\" }");
    }
  }
  std::vector<std::string> fields;
  int32_t at = 0;
  auto bytes_up_to = [&](int32_t end) {
    if (end > at) {
      fields.push_back(absl::StrCat("{ \"offset\": ", at, ", \"hex\": \"",
                                    Hex(payload + at, end - at), "\" }"));
    }
  };
  for (size_t m = 0; m < spec.tensor_maps.size(); ++m) {
    const TensorMapEncoding& map = spec.tensor_maps[m];
    if (map.arg != index) continue;
    bytes_up_to(map.offset);
    fields.push_back(absl::StrCat("{ \"offset\": ", map.offset,
                                  ", \"tensorMap\": \"map", m, "\" }"));
    at = map.offset + static_cast<int32_t>(sizeof(map.encoded));
  }
  bytes_up_to(host.size);
  return absl::StrCat("{ \"ordinal\": ", index, ", \"fields\": [ ",
                      absl::StrJoin(fields, ", "), " ], \"size\": ", host.size,
                      " }");
}

std::string Device() {
  CUdevice device = 0;
  char name[256] = {};
  int major = 0, minor = 0;
  if (cuCtxGetDevice(&device) == CUDA_SUCCESS) {
    cuDeviceGetName(name, sizeof(name), device);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                         device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                         device);
  }
  return absl::StrCat("\"device\": ", Quoted(name), ", \"sm\": \"sm_", major,
                      minor, "\"");
}

}  // namespace

const std::string& CaptureDirectory() {
  static const absl::NoDestructor<std::string> directory([] {
    const char* set = std::getenv("MOSAIC_GPU_CAPTURE_LAUNCHES");
    return std::string(set ? set : "");
  }());
  return *directory;
}

absl::Status CaptureLaunch(std::string_view kernel_name,
                           std::string_view gpu_binary, CUstream stream,
                           const MosaicKernelSpec& spec,
                           const std::vector<CapturedBuffer>& operands,
                           absl::FunctionRef<absl::Status()> launch) {
  // The stream's context current while the capture reads and writes memory.
  CUcontext context = nullptr;
  if (absl::Status got = Cuda(cuStreamGetCtx(stream, &context), "a context");
      !got.ok()) {
    return got;
  }
  if (absl::Status pushed = Cuda(cuCtxPushCurrent(context), "a context push");
      !pushed.ok()) {
    return pushed;
  }
  absl::Cleanup popped = [] { cuCtxPopCurrent(nullptr); };
  static std::atomic<int64_t> launches = 0;
  const std::string dir =
      absl::StrFormat("%s/%d-%06d-%s", CaptureDirectory(), getpid(),
                      launches.fetch_add(1), kernel_name);
  if (std::error_code error = llvm::sys::fs::create_directories(dir)) {
    return absl::InternalError(
        absl::StrCat("launch capture: creating ", dir, ": ", error.message()));
  }
  // Operands that alias (an output written in place) are one buffer.
  std::vector<Buffer> buffers;
  std::vector<std::string> shapes;
  for (const CapturedBuffer& operand : operands) {
    CUdeviceptr base = reinterpret_cast<CUdeviceptr>(operand.data);
    bool aliased = false;
    for (const Buffer& buffer : buffers) aliased |= buffer.base == base;
    if (!aliased && operand.bytes > 0) {
      buffers.push_back({operand.name, base, operand.bytes});
    }
    shapes.push_back(absl::StrCat(
        Quoted(operand.name), ": { \"dtype\": ", Quoted(operand.dtype),
        ", \"dims\": ", List(operand.dims.data(), operand.dims.size()), " }"));
  }
  // Every pointer the launch holds lands in a buffer before any is written.
  std::vector<std::string> maps, params;
  for (size_t m = 0; m < spec.tensor_maps.size(); ++m) {
    absl::StatusOr<std::string> map =
        Map(spec.tensor_maps[m], absl::StrCat("map", m), buffers);
    if (!map.ok()) return map.status();
    maps.push_back(*std::move(map));
  }
  for (int32_t i = 0; i < static_cast<int32_t>(spec.args.size()); ++i) {
    absl::StatusOr<std::string> param = Param(spec, i, buffers);
    if (!param.ok()) return param.status();
    params.push_back(*std::move(param));
  }
  std::vector<std::string> unplaced;
  for (size_t m = 0; m < spec.tensor_maps.size(); ++m) {
    if (spec.tensor_maps[m].arg < 0) {
      unplaced.push_back(absl::StrCat("\"map", m, "\""));
    }
  }

  const bool elf = gpu_binary.substr(0, 4) == "\x7f" "ELF";
  const std::string code = elf ? "kernel.cubin" : "kernel.ptx";
  if (absl::Status written = Write(absl::StrCat(dir, "/", code),
                                   gpu_binary.data(), gpu_binary.size());
      !written.ok()) {
    return written;
  }
  if (absl::Status before = Contents(buffers, stream, dir, ""); !before.ok()) {
    return before;
  }
  if (absl::Status launched = launch(); !launched.ok()) return launched;
  if (absl::Status after = Contents(buffers, stream, dir, ".expected");
      !after.ok()) {
    return after;
  }

  // Each buffer's bytes before the launch, and after it each operand's, which a
  // simulation is held to bit for bit (a workspace's are the launch's own).
  std::vector<std::string> declared, expected;
  for (const Buffer& buffer : buffers) {
    declared.push_back(absl::StrCat(
        "{ \"name\": ", Quoted(buffer.name), ", \"bytes\": ", buffer.bytes,
        ", \"init\": { \"gen\": \"file\", \"path\": \"", buffer.name,
        ".bin\" } }"));
    if (buffer.name.substr(0, 2) != "ws") {
      expected.push_back(absl::StrCat("{ \"buffer\": ", Quoted(buffer.name),
                                      ", \"offset\": 0, \"file\": \"",
                                      buffer.name, ".expected.bin\" }"));
    }
  }
  std::string json = absl::StrCat(
      "{ \"code\": ", Quoted(code), ", \"function\": ", Quoted(kernel_name),
      ",\n  \"grid\": [", spec.grid.x,
      ", ", spec.grid.y, ", ", spec.grid.z, "], \"block\": [", spec.block.x,
      ", ", spec.block.y, ", ", spec.block.z, "], \"cluster\": [",
      std::max<uint32_t>(spec.cluster.x, 1), ", ",
      std::max<uint32_t>(spec.cluster.y, 1), ", ",
      std::max<uint32_t>(spec.cluster.z, 1), "]");
  absl::StrAppend(
      &json, ", \"dynamicShared\": ", spec.smem_bytes,
      ",\n  \"buffers\": [\n    ",
      absl::StrJoin(declared, ",\n    "), " ],\n  \"tensorMaps\": [\n    ",
      absl::StrJoin(maps, ",\n    "), " ],\n  \"params\": [\n    ",
      absl::StrJoin(params, ",\n    "), " ],\n  \"expect\": [\n    ",
      absl::StrJoin(expected, ",\n    "), " ],\n  \"mosaic\": { ", Device(),
      ", \"usesPdl\": ", spec.uses_pdl ? "true" : "false",
      ", \"unplacedTensorMaps\": [", absl::StrJoin(unplaced, ", "),
      "], \"operands\": { ", absl::StrJoin(shapes, ", "), " } } }\n");
  return Write(absl::StrCat(dir, "/spec.json"), json.data(), json.size());
}

}  // namespace gpu
}  // namespace mosaic
