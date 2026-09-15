/* Copyright 2023 The JAX Authors.

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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "nanobind/nanobind.h"
#include "xla/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"

namespace nb = nanobind;

using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyAttribute;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMlirContext;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyOperation;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyOperationBase;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyType;

namespace jax {

struct PyFloat8EXMYType
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<
          PyFloat8EXMYType> {
  static constexpr const char* pyClassName = "Float8EXMYType";
  static bool isaFunction(MlirType t) { return mlirTpuIsAFloat8EXMYType(t); }
  using Base::Base;
  static void bindDerived(ClassTy& cls) {
    cls.def_static(
        "get",
        [](nb::object exmy_type_py, DefaultingPyMlirContext ctx) {
          MlirType exmy_type = {nullptr};
          if (!exmy_type_py.is_none()) {
            exmy_type = nb::cast<PyType&>(exmy_type_py).get();
          }
          return PyFloat8EXMYType(
              ctx.resolve().getRef(),
              mlirTpuFloat8EXMYTypeGet(ctx.resolve().get(), exmy_type));
        },
        nb::arg("exmy_type") = nb::none(), nb::arg("ctx") = nb::none());
    cls.def_prop_ro("underlying_type", [](PyFloat8EXMYType& self) {
      return mlirTpuFloat8EXMYTypeGetUnderlyingType(self.get());
    });
  }
};

NB_MODULE(_tpu_ext, m) {
  mlirTpuRegisterMosaicSerdePass();

  m.def(
      "register_dialect",
      [](PyMlirContext& context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__tpu__();
        mlirDialectHandleRegisterDialect(dialect, context.get());
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context.get());
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  m.def("private_has_communication", [](PyOperation& op) {
    bool has_communication;
    bool has_custom_barrier;
    mlirTPUAnalyzePotentialCommunication(op.get(), &has_communication,
                                         &has_custom_barrier);
    return std::make_pair(has_communication, has_custom_barrier);
  });

  m.def("private_unzip_debug_locations", [](PyOperationBase& op) {
    MlirOperation root_op = op.getOperation().get();
    MlirContext ctx = mlirOperationGetContext(root_op);
    MlirLocation unknown_loc = mlirLocationUnknownGet(ctx);

    struct WalkState {
      MlirLocation unknown_loc;
      llvm::DenseMap<const void*, int32_t> loc_ptr_to_idx;
      std::vector<std::string> locations;
      std::vector<int32_t> indices;
    } state;
    state.unknown_loc = unknown_loc;

    auto walk_fn = [](MlirOperation cur_op, void* user_data) -> MlirWalkResult {
      auto* s = static_cast<WalkState*>(user_data);
      MlirLocation loc = mlirOperationGetLocation(cur_op);
      auto [it, inserted] = s->loc_ptr_to_idx.try_emplace(
          loc.ptr, static_cast<int32_t>(s->locations.size()));
      if (inserted) {
        std::string loc_str;
        auto print_cb = [](MlirStringRef str, void* data) {
          static_cast<std::string*>(data)->append(str.data, str.length);
        };
        mlirLocationPrint(loc, print_cb, &loc_str);
        s->locations.push_back(std::move(loc_str));
      }
      s->indices.push_back(it->second);
      mlirOperationSetLocation(cur_op, s->unknown_loc);

      intptr_t num_regions = mlirOperationGetNumRegions(cur_op);
      for (intptr_t r = 0; r < num_regions; ++r) {
        MlirRegion region = mlirOperationGetRegion(cur_op, r);
        for (MlirBlock block = mlirRegionGetFirstBlock(region);
             !mlirBlockIsNull(block); block = mlirBlockGetNextInRegion(block)) {
          intptr_t num_args = mlirBlockGetNumArguments(block);
          for (intptr_t a = 0; a < num_args; ++a) {
            mlirBlockArgumentSetLocation(mlirBlockGetArgument(block, a),
                                         s->unknown_loc);
          }
        }
      }
      return MlirWalkResultAdvance;
    };

    mlirOperationWalk(root_op, walk_fn, &state, MlirWalkPreOrder);
    return std::make_pair(std::move(state.locations), std::move(state.indices));
  });

  // TODO(apaszke): All of those should be upstreamed to MLIR Python bindings.
  m.def("private_set_arg_attr", [](PyOperationBase& op, unsigned i,
                                   std::string name, PyAttribute& attr) {
    mlirFuncSetArgAttr(op.getOperation().get(), i,
                       mlirStringRefCreateFromCString(name.c_str()),
                       attr.get());
  });

  PyFloat8EXMYType::bind(m);
}

}  // namespace jax
