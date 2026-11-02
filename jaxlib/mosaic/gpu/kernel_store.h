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

#ifndef JAXLIB_MOSAIC_GPU_KERNEL_STORE_H_
#define JAXLIB_MOSAIC_GPU_KERNEL_STORE_H_

#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace mosaic::gpu {

// What decides a Mosaic GPU kernel's compile: two compiles of equal inputs
// give the same cubin and host object, so their identity keys the in-process
// kernel cache, an executable's serialized kernels and the persistent store.
struct CompileInputs {
  // The kernel's serialized module.
  absl::string_view module;
  // The GPU target, its accelerated suffix included ("9.0a").
  absl::string_view gpu_target;
  // The PTX ISA version the module is lowered to.
  absl::string_view ptx_isa;
  // The compilation provider, by name: its build is the binary's own (see
  // `BuildIdentity`) where it compiles in process.
  absl::string_view provider;
  // The host target its launcher is built for: triple, CPU and features.
  absl::string_view host_target;
  // The bitcode files linked into the kernel (libdevice), by path; each is
  // read and keyed by its content.
  absl::Span<const std::string> bitcode;
  // The LLVM command-line options the compile sets.
  absl::Span<const std::string> llvm_options;
};

// The identity of a compile of `inputs`, a hex SHA-256, within one binary: the
// store keys an entry by it and the binary's own (`BuildIdentity`). A bitcode
// file that cannot be read is keyed by its path alone, marked unreadable, so
// it never matches a readable one.
std::string CompileIdentity(const CompileInputs& inputs);

// The SHA-256 of the binary holding this code, read once a process, the first
// time the store is used: what compiled a stored kernel, the statically linked
// PTX compiler among it.
const std::string& BuildIdentity();

// The persistent store's directory (MOSAIC_GPU_COMPILATION_CACHE_DIR), none
// where it is unset or empty.
std::optional<std::string> StoreDirectory();

// The entry stored under `identity` in `directory`, none where there is none
// or it cannot be read.
std::optional<std::string> ReadEntry(absl::string_view directory,
                                     absl::string_view identity);

// `entry` stored under `identity` in `directory`: written to a file of its
// own beside the destination, then renamed into place, so a reader sees a
// whole entry or none. A failure is logged, and the compile goes on uncached.
void WriteEntry(absl::string_view directory, absl::string_view identity,
                absl::string_view entry);

}  // namespace mosaic::gpu

#endif  // JAXLIB_MOSAIC_GPU_KERNEL_STORE_H_
