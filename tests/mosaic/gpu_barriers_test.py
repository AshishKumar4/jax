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
"""Mosaic GPU drops a collective warpgroup barrier repeating the one before it."""

from absl.testing import absltest
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.experimental.mosaic.gpu import utils


_MODULE = """
func.func @kernel(%ref: memref<128xf32, 3>, %x: f32, %i: index) {
  %c1 = arith.constant 1 : i32
  %c128 = arith.constant 128 : i32
  %tid = gpu.thread_id x
  %t = arith.index_cast %tid : index to i32
  %wg = arith.divui %t, %c128 : i32
  %id = arith.addi %wg, %c1 : i32
  nvvm.barrier id = %id number_of_threads = %c128 {mosaic_gpu.collective_barrier}
  memref.store %x, %ref[%i] : memref<128xf32, 3>
  nvvm.barrier id = %id number_of_threads = %c128 {mosaic_gpu.collective_barrier}
  %tid2 = gpu.thread_id x
  %t2 = arith.index_cast %tid2 : index to i32
  %wg2 = arith.divui %t2, %c128 : i32
  %id2 = arith.addi %wg2, %c1 : i32
  nvvm.barrier id = %id2 number_of_threads = %c128 {mosaic_gpu.collective_barrier}
  memref.store %x, %ref[%i] : memref<128xf32, 3>
  nvvm.barrier id = %id number_of_threads = %c128 {mosaic_gpu.collective_barrier}
  nvvm.barrier id = %id number_of_threads = %c128 {mosaic_gpu.collective_barrier, other = 1 : i32}
  nvvm.barrier id = %id number_of_threads = %c128
  nvvm.barrier id = %id number_of_threads = %c128
  nvvm.barrier id = %id number_of_threads = %c1 {mosaic_gpu.collective_barrier}
  %n8 = arith.index_cast %i : index to i8
  %a = arith.extui %n8 : i8 to i32
  %n16 = arith.index_cast %i : index to i16
  %b = arith.extui %n16 : i16 to i32
  nvvm.barrier id = %a number_of_threads = %c128 {mosaic_gpu.collective_barrier}
  nvvm.barrier id = %b number_of_threads = %c128 {mosaic_gpu.collective_barrier}
  return
}
"""


class RepeatedBarriersTest(jtu.JaxTestCase):

  def test_a_collective_barrier_repeating_one_with_no_memory_access_between_is_dropped(self):
    with ir.Context() as context, ir.Location.unknown():
      context.append_dialect_registry(mlir_interpreter.upstream_dialects)
      context.load_all_available_dialects()
      context.allow_unregistered_dialects = True
      module = ir.Module.parse(_MODULE)
      block = module.body.operations[0].regions[0].blocks[0]

      def others():
        return [str(op) for op in block.operations
                if op.operation.name != "nvvm.barrier"]

      before = others()
      mgpu_core._drop_repeated_barriers(module)
      barriers = [
          op.operation
          for op in block.operations
          if op.operation.name == "nvvm.barrier"
      ]
      marks = [b for b in barriers if utils.COLLECTIVE_BARRIER in b.attributes]
      # No other operation is merged or erased: the launch context still holds
      # values of the module, which its TMA descriptors' host initialization
      # reads (a CSE here freed them, and that initialization segfaulted).
      self.assertEqual(others(), before)
    # The barrier after a store and the one before the next, its index
    # recomputed alike, are one; a barrier after a store stays; one of the
    # same operands and other attributes stays; unmarked barriers, whose
    # participants a branch may split, stay even repeated; another thread count
    # is another barrier; ids alike but for a cast's width are two. No mark
    # outlives the pass.
    self.assertLen(barriers, 9)
    self.assertEmpty(marks)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
