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

#include "jaxlib/mosaic/gpu/kernel_store.h"

#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>  // NOLINT
#include <thread>  // NOLINT

#include "absl/base/no_destructor.h"
#include "absl/log/log.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SHA256.h"

namespace mosaic::gpu {
namespace {

// The version of what an identity covers and how an entry is laid out: a
// change to either changes every identity.
constexpr absl::string_view kSchema = "mosaic-gpu-kernel-store-1";

std::string Hex(llvm::ArrayRef<uint8_t> digest) {
  return absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(digest.data()), digest.size()));
}

// One field of an identity: its length, then its bytes, so no two sequences of
// fields hash alike.
void Field(llvm::SHA256& hash, absl::string_view field) {
  hash.update(absl::StrCat(field.size(), ":"));
  hash.update(llvm::StringRef(field.data(), field.size()));
}

std::optional<std::string> ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return std::nullopt;
  }
  std::ostringstream content;
  content << in.rdbuf();
  if (in.bad()) {
    return std::nullopt;
  }
  return content.str();
}

// An entry's file: keyed by the identity under this binary, so another build's
// entries are never read.
std::string EntryPath(absl::string_view directory, absl::string_view identity) {
  llvm::SHA256 hash;
  Field(hash, kSchema);
  Field(hash, BuildIdentity());
  Field(hash, identity);
  return absl::StrCat(directory, "/", Hex(hash.final()), ".kernel");
}

}  // namespace

const std::string& BuildIdentity() {
  static const absl::NoDestructor<std::string> identity([] {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&BuildIdentity), &info) == 0 ||
        info.dli_fname == nullptr) {
      return std::string("unknown-binary");
    }
    std::optional<std::string> binary = ReadFile(info.dli_fname);
    if (!binary.has_value()) {
      return absl::StrCat("unreadable-binary:", info.dli_fname);
    }
    return Hex(llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t*>(binary->data()), binary->size())));
  }());
  return *identity;
}

std::string CompileIdentity(const CompileInputs& inputs) {
  llvm::SHA256 hash;
  Field(hash, kSchema);
  Field(hash, inputs.module);
  Field(hash, inputs.gpu_target);
  Field(hash, inputs.ptx_isa);
  Field(hash, inputs.provider);
  Field(hash, inputs.host_target);
  for (const std::string& path : inputs.bitcode) {
    std::optional<std::string> content = ReadFile(path);
    Field(hash, path);
    Field(hash, content.has_value()
                    ? Hex(llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
                          reinterpret_cast<const uint8_t*>(content->data()),
                          content->size())))
                    : "unreadable");
  }
  for (const std::string& option : inputs.llvm_options) {
    Field(hash, option);
  }
  return Hex(hash.final());
}

std::optional<std::string> StoreDirectory() {
  const char* directory = std::getenv("MOSAIC_GPU_COMPILATION_CACHE_DIR");
  if (directory == nullptr || *directory == '\0') {
    return std::nullopt;
  }
  return std::string(directory);
}

std::optional<std::string> ReadEntry(absl::string_view directory,
                                     absl::string_view identity) {
  return ReadFile(EntryPath(directory, identity));
}

void WriteEntry(absl::string_view directory, absl::string_view identity,
                absl::string_view entry) {
  if (std::error_code error = llvm::sys::fs::create_directories(
          llvm::StringRef(directory.data(), directory.size()))) {
    LOG(WARNING) << "Mosaic GPU: could not make the kernel store " << directory
                 << ": " << error.message() << "; the kernel is not stored";
    return;
  }
  std::string path = EntryPath(directory, identity);
  std::ostringstream thread;
  thread << std::this_thread::get_id();
  std::string partial = absl::StrCat(path, ".", getpid(), ".", thread.str(),
                                     ".partial");
  {
    std::ofstream out(partial, std::ios::binary | std::ios::trunc);
    out.write(entry.data(), entry.size());
    out.close();
    if (!out) {
      LOG(WARNING) << "Mosaic GPU: could not write the kernel store's entry "
                   << partial << "; the kernel is not stored";
      std::remove(partial.c_str());
      return;
    }
  }
  if (std::rename(partial.c_str(), path.c_str()) != 0) {
    LOG(WARNING) << "Mosaic GPU: could not move " << partial << " to " << path
                 << "; the kernel is not stored";
    std::remove(partial.c_str());
  }
}

}  // namespace mosaic::gpu
