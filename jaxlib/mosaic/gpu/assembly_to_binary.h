/* Copyright 2025 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_GPU_ASSEMBLY_TO_BINARY_H_
#define JAXLIB_MOSAIC_GPU_ASSEMBLY_TO_BINARY_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/kernel_stats.h"

namespace mosaic {
namespace gpu {

// A note ptxas made of a loss of performance in a kernel's assembly: its code
// (`C7515`) and its message, as ptxas states them.
struct PtxasNote {
  std::string code;
  std::string message;
};

// The module attribute an assembled module's notes are kept in: an array of
// dictionaries, each its `code` and `message`.
inline constexpr char kAssemblerNotesAttr[] = "mosaic_gpu.assembler_notes";

// Each note of a loss of performance ptxas's report `log` of a module's
// assembly states, in order (`ptxas warning : (C7515) Potential Performance
// Loss: <message>`).
std::vector<PtxasNote> PtxasNotes(absl::string_view log);

// The module attribute an assembled module's spilled bytes are kept in
// (`LaunchSpilledBytes`), an integer, where they are known: absent where its
// launch's scope is not established.
inline constexpr char kSpilledBytesAttr[] = "mosaic_gpu.spilled_bytes";

// The bytes ptxas spilled assembling a binary of `entries` kernel entries: its
// functions' spill stores and loads (`stats`, StreamExecutor's, read from the
// spill warnings every compilation provider asks ptxas for, so a function it
// holds none of spilled nothing). Known where the binary has one entry, its
// every function then that entry or one it calls; unknown otherwise. A stack
// frame its functions' own locals take (libdevice's sinf range reduction) is
// none of them.
std::optional<int64_t> LaunchSpilledBytes(const ModuleStats& stats,
                                          int64_t entries);

// Registers a pass that converts `gpu.binary` ops wrapping PTX assembly into
// `gpu.binary` ops wrapping a CUBIN binary, each for the compute capability its
// object's NVVM target names, keeping what ptxas noted of a loss of
// performance in the module's `kAssemblerNotesAttr`.
void registerAssemblyToBinaryPass(
    const stream_executor::cuda::CompilationProvider* compilation_provider);

}  // namespace gpu
}  // namespace mosaic

#endif  // JAXLIB_MOSAIC_GPU_ASSEMBLY_TO_BINARY_H_
