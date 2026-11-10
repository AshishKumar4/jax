# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deviceless launch captures: a Mosaic GPU kernel's launch as a simulator's spec, with no GPU.

`kernels` finds the Mosaic GPU kernels a lowered program calls, and `capture_deviceless` writes one's launch
over host buffers of its operands: compiled for a stated compute capability, its spec filled by the kernel's
own host function, its tensor maps recorded by their encoding's arguments (jaxlib's capture.h names the
format, `MOSAIC_GPU_CAPTURE_LAUNCHES`'s on a GPU, less the contents after the launch, which never ran).
"""

from __future__ import annotations

import ctypes
import dataclasses
from collections.abc import Sequence

import numpy as np

from jax._src import custom_call_targets
from jax._src import dtypes
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir



@dataclasses.dataclass(frozen=True)
class Kernel:
  """A Mosaic GPU kernel a lowered program calls: its module (`module`, the custom call's bytecode), its
  operands' and results' shapes and dtypes, in the order its launch reads its buffers, and whether it
  reads XLA's collective metadata (which no deviceless capture holds)."""

  module: bytes
  operands: tuple[tuple[tuple[int, ...], np.dtype], ...]
  results: tuple[tuple[tuple[int, ...], np.dtype], ...]
  collective: bool


def _dtype(element_type: ir.Type) -> np.dtype:
  """The dtype whose IR type `element_type` is (`mlir.dtype_to_ir_type`'s inverse; i1 a bool)."""
  for dtype, made in mlir._dtype_to_ir_type.items():
    if dtype != dtypes.float0 and made() == element_type:
      return dtype
  raise ValueError(f"no dtype has the IR type {element_type}")


def _shaped(value_type: ir.Type) -> tuple[tuple[int, ...], np.dtype]:
  ranked = ir.RankedTensorType(value_type)
  return tuple(ranked.shape), _dtype(ranked.element_type)


def kernels(module: ir.Module) -> list[Kernel]:
  """The Mosaic GPU kernels `module` (a lowered program's StableHLO, `Lowered.compiler_ir()`) calls, in the
  order its operations hold them."""
  found = []

  def walk(operation):
    for region in operation.regions:
      for block in region.blocks:
        for inner in block.operations:
          op = inner.operation
          if (op.name == "stablehlo.custom_call"
              and ir.StringAttr(op.attributes["call_target_name"]).value == custom_call_targets.MOSAIC_GPU):
            config = ir.DictAttr(op.attributes["mhlo.backend_config"])
            found.append(Kernel(
                module=ir.StringAttr(config["module"]).value_bytes,
                operands=tuple(_shaped(operand.type) for operand in op.operands),
                results=tuple(_shaped(result.type) for result in op.results),
                collective=("uses_xla_collective_metadata" in config
                            and ir.BoolAttr(config["uses_xla_collective_metadata"]).value)))
          walk(op)

  walk(module.operation)
  return found


def _library() -> ctypes.CDLL:
  try:
    try:
      import jax_plugins.xla_cuda13 as cuda_plugin  # pyrefly: ignore[missing-import]
    except ImportError:
      import jax_plugins.xla_cuda12 as cuda_plugin  # pyrefly: ignore[missing-import]
  except ImportError:
    return ctypes.CDLL(None)
  return ctypes.CDLL(cuda_plugin._get_library_path())


# XLA's names of the element types (the GPU capture's `primitive_util::LowercasePrimitiveTypeName`).
_XLA_NAMES = {"bool": "pred", "int4": "s4", "int8": "s8", "int16": "s16", "int32": "s32", "int64": "s64",
              "uint4": "u4", "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64",
              "float16": "f16", "bfloat16": "bf16", "float32": "f32", "float64": "f64",
              "float8_e4m3fn": "f8e4m3fn", "float8_e5m2": "f8e5m2", "float8_e8m0fnu": "f8e8m0fnu",
              "float4_e2m1fn": "f4e2m1fn"}


def _xla_name(dtype: np.dtype) -> str:
  return _XLA_NAMES.get(np.dtype(dtype).name, np.dtype(dtype).name)


def _buffers(kernel: Kernel, operands: Sequence[np.ndarray]) -> list[tuple[str, np.ndarray]]:
  """Each operand and zeroed result of `kernel` by its capture's name, in C order and of the shape and
  dtype the kernel reads. A sub-byte element type is refused: the kernel reads two elements a byte, a
  host array holds one."""
  buffers = []
  for index, (array, (shape, dtype)) in enumerate(zip(operands, kernel.operands, strict=True)):
    array = np.asarray(array, order="C")  # A scalar stays rank 0.
    if array.shape != shape or array.dtype != dtype:
      raise ValueError(f"operand {index} is {array.dtype}{list(array.shape)}, the kernel reads "
                       f"{dtype}{list(shape)}")
    buffers.append((f"in{index}", array))
  buffers += [(f"out{index}", np.zeros(shape, dtype)) for index, (shape, dtype) in enumerate(kernel.results)]
  for name, array in buffers:
    if array.dtype.itemsize == 1 and dtypes.itemsize_bits(array.dtype) < 8:
      raise ValueError(f"{name} is {array.dtype}, which the kernel reads packed two elements a byte; a "
                       "deviceless capture takes whole bytes")
  return buffers


def capture_deviceless(kernel: Kernel, compute_capability: str, operands: Sequence[np.ndarray],
                       directory: str) -> str:
  """Writes `kernel`'s launch over `operands` (host arrays of its operands' shapes and dtypes; its results
  start zeroed) as compiled for `compute_capability` ("9.0", "10.0"), with no GPU, into
  `directory`/<pid>-<seq>-<kernel>/, the directory it returns."""
  if kernel.collective:
    raise ValueError("a deviceless capture holds no collective metadata: the kernel reads XLA's")
  if len(operands) != len(kernel.operands):
    raise ValueError(f"the kernel reads {len(kernel.operands)} operands, not {len(operands)}")
  buffers = _buffers(kernel, operands)

  major, minor = (int(part) for part in compute_capability.split("."))
  count = len(buffers)
  data = (ctypes.c_void_p * count)(*(array.ctypes.data for _, array in buffers))
  sizes = (ctypes.c_uint64 * count)(*(array.nbytes for _, array in buffers))
  names = (ctypes.c_char_p * count)(*(name.encode() for name, _ in buffers))
  dtypes = (ctypes.c_char_p * count)(*(_xla_name(array.dtype).encode() for _, array in buffers))
  ranks = (ctypes.c_int * count)(*(array.ndim for _, array in buffers))
  flat = [dim for _, array in buffers for dim in array.shape]
  dims = (ctypes.c_int64 * max(1, len(flat)))(*flat)
  out = ctypes.create_string_buffer(4096)

  capture = _library().MosaicGpuCaptureDeviceless
  capture.restype = ctypes.c_int
  failed = capture(kernel.module, len(kernel.module), major, minor, count, data, sizes, names, dtypes, ranks,
                   dims, directory.encode(), out, len(out))
  written = out.value.decode()
  if failed:
    raise RuntimeError(f"deviceless capture failed: {written}")
  return written
