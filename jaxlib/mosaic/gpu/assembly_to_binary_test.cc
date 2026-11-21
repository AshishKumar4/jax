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

#include "jaxlib/mosaic/gpu/assembly_to_binary.h"

#include <cstdint>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "jaxlib/mosaic/gpu/gpu_module_to_assembly.h"
#include "jaxlib/mosaic/gpu/target.h"
#include "xla/stream_executor/cuda/compilation_provider.h"

namespace mosaic::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::Field;

TEST(PtxasNotesTest, EachLossNoteOfAReportIsItsCodeAndMessageInOrder) {
  // ptxas's verbose report of one kernel: its resource lines, a note of no
  // loss, and two of a loss.
  constexpr char kLog[] =
      "ptxas info    : 0 bytes gmem\n"
      "ptxas info    : Compiling entry function 'kernel' for 'sm_90a'\n"
      "ptxas warning : (C7515) Potential Performance Loss: wgmma.mma_async "
      "instructions are serialized due to the presence of Extern calls in the "
      "function 'kernel'.\n"
      "ptxas info    : Used 168 registers, used 1 barriers\n"
      "ptxas warning : (C7507) Potential Performance Loss: setmaxnreg ignored; "
      "unable to determine register count at entry.  \n"
      "ptxas warning : Unresolved extern function 'vprintf'\n";
  EXPECT_THAT(
      PtxasNotes(kLog),
      ElementsAre(
          testing::AllOf(Field(&PtxasNote::code, "C7515"),
                Field(&PtxasNote::message,
                      "wgmma.mma_async instructions are serialized due to "
                      "the presence of Extern calls in the function "
                      "'kernel'.")),
          testing::AllOf(Field(&PtxasNote::code, "C7507"),
                Field(&PtxasNote::message,
                      "setmaxnreg ignored; unable to determine register "
                      "count at entry."))));
}

TEST(PtxasNotesTest, AReportNotingNoLossHasNone) {
  EXPECT_THAT(PtxasNotes("ptxas info    : Used 32 registers\n"),
              testing::IsEmpty());
}

// A PTX module of kernel entries, each holding `values` floats live at once
// under `max_registers` (none where zero): each value is multiplied by its
// mirror, loaded last, by volatile loads ptxas keeps in their order.
std::string PtxModule(int entries, int values, int max_registers) {
  std::string ptx = ".version 8.0\n.target sm_80\n.address_size 64\n";
  for (int entry = 0; entry < entries; ++entry) {
    absl::StrAppend(&ptx, ".visible .entry kernel", entry,
                    "(.param .u64 kernel", entry, "_out)\n");
    if (max_registers) {
      absl::StrAppend(&ptx, ".maxnreg ", max_registers, "\n");
    }
    absl::StrAppend(&ptx, "{\n.reg .b64 %rd<2>;\n.reg .f32 %f<", values + 2,
                    ">;\nld.param.u64 %rd1, [kernel", entry, "_out];\n");
    for (int i = 1; i <= values; ++i) {
      absl::StrAppend(&ptx, "ld.volatile.global.f32 %f", i, ", [%rd1+",
                      4 * i, "];\n");
    }
    int sum = values + 1;
    absl::StrAppend(&ptx, "mov.f32 %f", sum, ", 0f00000000;\n");
    for (int i = 1; i <= values / 2; ++i) {
      absl::StrAppend(&ptx, "fma.rn.f32 %f", sum, ", %f", i, ", %f",
                      values + 1 - i, ", %f", sum, ";\n");
    }
    absl::StrAppend(&ptx, "st.global.f32 [%rd1], %f", sum, ";\nret;\n}\n");
  }
  return ptx;
}

class AssemblyToBinaryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    absl::StatusOr<stream_executor::cuda::CompilationProvider*> provider =
        GetAssemblyToBinaryCompilationProvider();
    ASSERT_TRUE(provider.ok()) << provider.status();
    registerAssemblyToBinaryPass(*provider);
  }

  AssemblyToBinaryTest() {
    context_.loadDialect<mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                         mlir::NVVM::NVVMDialect>();
  }

  // A module of one binary wrapping `ptx`, whose object states
  // `kernel_entries`, assembled by the pass.
  mlir::OwningOpRef<mlir::ModuleOp> Assembled(
      const std::string& ptx, int64_t kernel_entries,
      std::optional<int64_t> held_spilled_bytes = std::nullopt) {
    mlir::OpBuilder b(&context_);
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::ModuleOp::create(b.getUnknownLoc());
    if (held_spilled_bytes.has_value()) {
      (*module)->setAttr(kSpilledBytesAttr,
                         b.getI64IntegerAttr(*held_spilled_bytes));
    }
    b.setInsertionPointToEnd(module->getBody());
    mlir::Attribute object = b.getAttr<mlir::gpu::ObjectAttr>(
        mlir::NVVM::NVVMTargetAttr::get(&context_, /*optLevel=*/3,
                                        "nvptx64-nvidia-cuda", "sm_80",
                                        "+ptx80"),
        mlir::gpu::CompilationTarget::Assembly, b.getStringAttr(ptx),
        b.getDictionaryAttr({b.getNamedAttr(
            kKernelEntriesProperty, b.getI64IntegerAttr(kernel_entries))}),
        /*kernels=*/nullptr);
    mlir::gpu::BinaryOp::create(b, b.getUnknownLoc(), "binary",
                                /*handler=*/nullptr, b.getArrayAttr({object}));
    mlir::PassManager pm(&context_);
    EXPECT_TRUE(mlir::succeeded(
        mlir::parsePassPipeline("mosaic-gpu-assembly-to-binary", pm)));
    EXPECT_TRUE(mlir::succeeded(pm.run(*module)));
    return module;
  }

  static std::optional<int64_t> SpilledBytes(mlir::ModuleOp module) {
    if (auto spilled =
            module->getAttrOfType<mlir::IntegerAttr>(kSpilledBytesAttr)) {
      return spilled.getInt();
    }
    return std::nullopt;
  }

  mlir::MLIRContext context_;
};

TEST_F(AssemblyToBinaryTest, AnEntrySpillingNothingStatesZeroBytes) {
  EXPECT_EQ(SpilledBytes(*Assembled(PtxModule(1, 4, 0), 1)), 0);
}

TEST_F(AssemblyToBinaryTest, AnEntryShortOfRegistersStatesTheBytesItSpilled) {
  // 32 floats live at once in 16 registers a thread.
  std::optional<int64_t> spilled =
      SpilledBytes(*Assembled(PtxModule(1, 64, 16), 1));
  ASSERT_TRUE(spilled.has_value());
  EXPECT_GT(*spilled, 0);
}

TEST_F(AssemblyToBinaryTest, ABinaryOfTwoEntriesStatesNoneWhatTheModuleHeld) {
  // Neither entry spills, so a zero the module held before would read as one
  // measured.
  EXPECT_EQ(SpilledBytes(*Assembled(PtxModule(2, 4, 0), 2,
                                    /*held_spilled_bytes=*/0)),
            std::nullopt);
}

}  // namespace
}  // namespace mosaic::gpu
