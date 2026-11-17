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

#include "jaxlib/mosaic/gpu/dump.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace {

#if !defined(_WIN32)
class SassDiscoveryTest : public testing::Test {
 protected:
  void SetUp() override {
    if (const char* value = getenv("MOSAIC_GPU_CUDA_ROOT")) cuda_root_ = value;
    if (const char* value = getenv("PATH")) path_ = value;
    std::string pattern = testing::TempDir() + "/mosaic-split-tools-XXXXXX";
    ASSERT_NE(mkdtemp(pattern.data()), nullptr);
    directory_ = pattern;
  }

  void TearDown() override {
    Restore("MOSAIC_GPU_CUDA_ROOT", cuda_root_);
    Restore("PATH", path_);
    std::filesystem::remove_all(directory_);
  }

  static void Restore(const char* name,
                      const std::optional<std::string>& value) {
    if (value) {
      setenv(name, value->c_str(), 1);
    } else {
      unsetenv(name);
    }
  }

  std::filesystem::path directory_;
  std::optional<std::string> cuda_root_;
  std::optional<std::string> path_;
};

TEST_F(SassDiscoveryTest, DisassemblerInSeparatePackageRemainsDiscoverable) {
  // Model a ptxas-only nvcc wheel beside a cu13 disassembler wheel. Running a
  // fake nvdisasm on a binary object exercises the native dumping consumer
  // without CUDA hardware, ptxas, or a valid cubin.
  auto nvcc = directory_ / "nvidia" / "cuda_nvcc";
  auto cu13 = directory_ / "nvidia" / "cu13";
  std::filesystem::create_directories(nvcc / "bin");
  std::filesystem::create_directories(cu13 / "bin");
  std::ofstream(nvcc / "bin" / "ptxas") << "assembler";
  auto disassembler = cu13 / "bin" / "nvdisasm";
  std::ofstream(disassembler) << "#!/bin/sh\nprintf 'disassembled\\n'\n";
  std::filesystem::permissions(disassembler, std::filesystem::perms::owner_all);
  ASSERT_EQ(setenv("PATH", (cu13 / "bin").c_str(), 1), 0);

  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::gpu::GPUDialect, mlir::NVVM::NVVMDialect>();
  auto location = mlir::UnknownLoc::get(&ctx);
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  mlir::OpBuilder builder(module->getBodyRegion());
  auto object = mlir::gpu::ObjectAttr::get(
      mlir::NVVM::NVVMTargetAttr::get(&ctx),
      mlir::gpu::CompilationTarget::Binary, builder.getStringAttr("ELF"));
  mlir::OperationState state(location, mlir::gpu::BinaryOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr("kernel"));
  state.addAttribute("objects", builder.getArrayAttr({object}));
  auto binary = mlir::cast<mlir::gpu::BinaryOp>(builder.create(state));

  // Control: the exact same fake tool is executable when it is in the root.
  ASSERT_EQ(setenv("MOSAIC_GPU_CUDA_ROOT", cu13.c_str(), 1), 0);
  mosaic::gpu::DumpSass(binary, directory_.string(), "control", false);
  ASSERT_TRUE(std::filesystem::exists(directory_ / "control.sass"));

  // JAX selects the package containing ptxas as its CUDA root. The separate
  // nvdisasm is still discoverable on PATH and should produce the same file.
  ASSERT_EQ(setenv("MOSAIC_GPU_CUDA_ROOT", nvcc.c_str(), 1), 0);
  mosaic::gpu::DumpSass(binary, directory_.string(), "split", false);
  EXPECT_TRUE(std::filesystem::exists(directory_ / "split.sass"));
}
#endif

TEST(DumpTest, GetOrSetDumpOptionsForModuleReturnsConsistentBasenameForModule) {
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(
      mlir::UnknownLoc::get(&ctx), /*name=*/"test_module");
  mosaic::gpu::DumpOptions opts1 =
      mosaic::gpu::GetOrSetDumpOptionsForModule(*module);
  mosaic::gpu::DumpOptions opts2 =
      mosaic::gpu::GetOrSetDumpOptionsForModule(*module);
  // The module basename should be consistent across calls for the same module.
  EXPECT_EQ(opts1.module_basename, opts2.module_basename);
}

TEST(
    DumpTest,
    GetOrSetDumpOptionsForModuleReturnsConsistentBasenameForDifferentModulesWithTheSameName) {  // NOLINT(whitespace/line_length)
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module1 = mlir::ModuleOp::create(
      mlir::UnknownLoc::get(&ctx), /*name=*/"test_module");
  mlir::OwningOpRef<mlir::ModuleOp> module2 = mlir::ModuleOp::create(
      mlir::UnknownLoc::get(&ctx), /*name=*/"test_module");
  mosaic::gpu::DumpOptions opts1 =
      mosaic::gpu::GetOrSetDumpOptionsForModule(*module1);
  mosaic::gpu::DumpOptions opts2 =
      mosaic::gpu::GetOrSetDumpOptionsForModule(*module2);
  // The module basename should be different for different modules, even though
  // they have the same name.
  EXPECT_NE(opts1.module_basename, opts2.module_basename);
}

}  // namespace
