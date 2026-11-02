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

// This pass lowers existing PTX into a `gpu.binary` op using stream executor
// compilation providers. The stock MLIR pipeline uses `ptxas` in a subprocess
// to compile PTX by default. This does not work reliably in all environments,
// and stream executor's compilation providers are meant to remedy this problem.

#include "jaxlib/mosaic/gpu/assembly_to_binary.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/mosaic/gpu/dump.h"
#include "jaxlib/mosaic/pass_boilerplate.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/tsl/platform/statusor.h"

namespace mosaic {
namespace gpu {

namespace {

namespace se = ::stream_executor;

absl::StatusOr<std::string> SetPtxIsaVersionToHighestSupported(
    const se::cuda::CompilationProvider& compilation_provider,
    llvm::StringRef ptx_str) {
  ABSL_ASSIGN_OR_RETURN(int ptx_isa_latest_version,
                        compilation_provider.GetLatestPtxIsaVersion());

  int major_version = ptx_isa_latest_version / 10;
  int minor_version = ptx_isa_latest_version % 10;

  uint64_t start_pos = ptx_str.find(".version");
  uint64_t end_pos = ptx_str.find('\n', start_pos);

  if (start_pos == std::string::npos || end_pos == std::string::npos) {
    return absl::InternalError(
        "Failed to find PTX version during PTX to SASS lowering.");
  }

  return absl::StrCat(
      ptx_str.substr(0, start_pos).str(),
      absl::StrFormat(".version %d.%d", major_version, minor_version),
      ptx_str.substr(end_pos).str());
}

// The compute capability an NVVM target's chip names ("sm_90a", "sm_100a",
// "sm_86"), which a binary is assembled for: each binary's own, so one process
// assembles for several capabilities, in either order or at once.
absl::StatusOr<se::CudaComputeCapability> CapabilityOfChip(
    absl::string_view chip) {
  if (!absl::ConsumePrefix(&chip, "sm_")) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected an sm_ chip, got ", chip));
  }
  size_t digits = 0;
  while (digits < chip.size() && absl::ascii_isdigit(chip[digits])) {
    ++digits;
  }
  if (digits < 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected a chip of a major and a minor, got sm_", chip));
  }
  return se::CudaComputeCapability::FromString(
      absl::StrCat(chip.substr(0, digits - 1), ".",
                   chip.substr(digits - 1)));
}

class AssemblyToBinaryPass
    : public jaxlib::mlir::Pass<AssemblyToBinaryPass, mlir::ModuleOp> {
 public:
  using jaxlib::mlir::Pass<AssemblyToBinaryPass, mlir::ModuleOp>::Pass;

  explicit AssemblyToBinaryPass(
      const se::cuda::CompilationProvider* compilation_provider)
      : compilation_provider_(compilation_provider) {}

  static constexpr llvm::StringLiteral kArgumentName =
      "mosaic-gpu-assembly-to-binary";
  static constexpr llvm::StringLiteral kPassName = "GpuAssemblyToBinaryPass";

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder b(module);
    mlir::MLIRContext* ctx = module.getContext();
    DumpOptions dump_opts = GetOrSetDumpOptionsForModule(module);

    // The compilation log is always asked for: it and the PTX are kept with the
    // kernel (`kPtxAttr`, `kCompilationLogAttr`), which a kernel read from the
    // store replays where the dumps ask for them, with no compile.
    se::cuda::CompilationOptions compilation_options;
    compilation_options.dump_compilation_log = true;
    compilation_options.generate_line_info = true;
    std::string kept_ptx;
    std::string kept_log;

    mlir::WalkResult result = module.walk([&](mlir::gpu::BinaryOp binary) {
      if (binary.getObjects().size() != 1) {
        binary.emitOpError("Expected exactly one object in the binary.");
        return mlir::WalkResult::interrupt();
      }

      mlir::gpu::ObjectAttr object =
          mlir::cast<mlir::gpu::ObjectAttr>(*binary.getObjects().begin());
      if (object.getFormat() != mlir::gpu::CompilationTarget::Assembly) {
        binary.emitOpError("Expected an assembly object.");
        return mlir::WalkResult::interrupt();
      }

      llvm::StringRef ptx_str = object.getObject().getValue();

      auto target =
          mlir::dyn_cast<mlir::NVVM::NVVMTargetAttr>(object.getTarget());
      if (!target) {
        binary.emitOpError("Expected an NVVM target.");
        return mlir::WalkResult::interrupt();
      }
      absl::StatusOr<se::CudaComputeCapability> cc =
          CapabilityOfChip(target.getChip().str());
      if (!cc.ok()) {
        binary.emitOpError(cc.status().message());
        return mlir::WalkResult::interrupt();
      }

      auto dump_ptx_if_enabled = [&]() {
        if (dump_opts.ptx) {
          DumpToFileOrStdout(ptx_str, dump_opts.module_basename + ".ptx",
                             dump_opts.dump_path);
        }
      };

      absl::StatusOr<std::string> ptx_str_or =
          SetPtxIsaVersionToHighestSupported(*compilation_provider_, ptx_str);

      if (!ptx_str_or.ok()) {
        // Dump the PTX before emitting the error, for better diagnostics.
        dump_ptx_if_enabled();
        binary.emitOpError(ptx_str_or.status().message());
        return mlir::WalkResult::interrupt();
      }
      ptx_str = *ptx_str_or;
      dump_ptx_if_enabled();
      kept_ptx += ptx_str.str();

      absl::StatusOr<se::cuda::Assembly> sass_or =
          compilation_provider_->Compile(*cc, ptx_str, compilation_options);
      if (!sass_or.ok()) {
        binary.emitOpError(sass_or.status().message());
        return mlir::WalkResult::interrupt();
      }

      if (sass_or->compilation_log.has_value()) {
        kept_log += *sass_or->compilation_log;
      }
      if (dump_opts.ptxas) {
        if (!sass_or->compilation_log.has_value()) {
          binary.emitOpError("Expected a compilation log to be available.");
          return mlir::WalkResult::interrupt();
        }
        DumpToFileOrStdout(*sass_or->compilation_log,
                           dump_opts.module_basename + ".ptxas",
                           dump_opts.dump_path);
      }

      mlir::StringAttr sass = mlir::StringAttr::get(
          ctx, std::string(sass_or->cubin.begin(), sass_or->cubin.end()));
      b.setInsertionPointAfter(binary);

      mlir::gpu::ObjectAttr new_object = mlir::gpu::ObjectAttr::get(
          ctx, object.getTarget(), mlir::gpu::CompilationTarget::Binary, sass,
          object.getProperties(), object.getKernels());
      binary.setObjectsAttr(mlir::ArrayAttr::get(ctx, {new_object}));

      if (dump_opts.sass || dump_opts.sass_ctrl) {
        DumpSass(binary, dump_opts.dump_path, dump_opts.module_basename,
                 dump_opts.sass_ctrl);
      }

      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    module->setAttr(kPtxAttr, mlir::StringAttr::get(ctx, kept_ptx));
    module->setAttr(kCompilationLogAttr, mlir::StringAttr::get(ctx, kept_log));
  }

 private:
  const se::cuda::CompilationProvider* compilation_provider_;
};

}  // namespace

void registerAssemblyToBinaryPass(
    const se::cuda::CompilationProvider* compilation_provider) {
  ::mlir::registerPass([compilation_provider]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<AssemblyToBinaryPass>(compilation_provider);
  });
}

}  // namespace gpu
}  // namespace mosaic
