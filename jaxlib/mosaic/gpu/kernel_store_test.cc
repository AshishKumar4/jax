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

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include <gtest/gtest.h>

namespace mosaic::gpu {
namespace {

std::string Scratch(const std::string& name) {
  const char* root = std::getenv("TEST_TMPDIR");
  return std::string(root == nullptr ? "/tmp" : root) + "/" + name;
}

void Write(const std::string& path, const std::string& content) {
  std::ofstream(path, std::ios::binary) << content;
}

// Each input that decides a compile changes its identity alone, and equal
// inputs give one identity: what keys a kernel in a process and in the store.
TEST(KernelStoreTest, EachInputChangesTheIdentity) {
  std::string bitcode = Scratch("libdevice.bc");
  Write(bitcode, "libdevice");
  std::vector<std::string> bitcodes{bitcode};
  std::vector<std::string> options{"-nvptx-prec-divf32=1"};
  CompileInputs inputs{.module = "module",
                       .gpu_target = "9.0a",
                       .ptx_isa = "ptx87",
                       .provider = "NvptxcompilerCompilationProvider",
                       .host_target = "x86_64|znver4|+avx512f",
                       .bitcode = bitcodes,
                       .llvm_options = options};
  std::string identity = CompileIdentity(inputs);
  EXPECT_EQ(identity, CompileIdentity(inputs));
  EXPECT_EQ(identity.size(), 64);

  std::vector<std::string> other_options{"-nvptx-prec-divf32=2"};
  std::vector<CompileInputs> changed(7, inputs);
  changed[0].module = "module2";
  changed[1].gpu_target = "8.0";
  changed[2].ptx_isa = "ptx86";
  changed[3].provider = "SubprocessCompilationProvider";
  changed[4].host_target = "aarch64|neoverse-v2|";
  changed[5].llvm_options = other_options;
  changed[6].module = "modul";
  changed[6].gpu_target = "e9.0a";
  for (const CompileInputs& each : changed) {
    EXPECT_NE(CompileIdentity(each), identity);
  }
  // A bitcode file's content, not its path, decides.
  Write(bitcode, "libdevice, rebuilt");
  EXPECT_NE(CompileIdentity(inputs), identity);
}

// An entry written is read whole, under its identity alone; none is read where
// none was written, and a directory is made where there is none.
TEST(KernelStoreTest, AnEntryIsReadWholeUnderItsIdentity) {
  std::string directory = Scratch("store/nested");
  EXPECT_EQ(ReadEntry(directory, "a"), std::nullopt);
  WriteEntry(directory, "a", std::string("kernel\0bytes", 12));
  EXPECT_EQ(ReadEntry(directory, "a"), std::string("kernel\0bytes", 12));
  EXPECT_EQ(ReadEntry(directory, "b"), std::nullopt);
  WriteEntry(directory, "a", "rewritten");
  EXPECT_EQ(ReadEntry(directory, "a"), "rewritten");
}

// Writers of one identity at once leave one whole entry, one of theirs.
TEST(KernelStoreTest, WritersAtOnceLeaveOneWholeEntry) {
  std::string directory = Scratch("store-concurrent");
  std::vector<std::string> entries;
  for (int index = 0; index < 8; ++index) {
    entries.push_back(std::string(1 << 20, static_cast<char>('a' + index)));
  }
  std::vector<std::thread> writers;
  for (const std::string& entry : entries) {
    writers.emplace_back([&directory, &entry] {
      WriteEntry(directory, "same", entry);
    });
  }
  for (std::thread& writer : writers) {
    writer.join();
  }
  std::optional<std::string> read = ReadEntry(directory, "same");
  ASSERT_TRUE(read.has_value());
  EXPECT_NE(std::find(entries.begin(), entries.end(), *read), entries.end());
}

}  // namespace
}  // namespace mosaic::gpu
