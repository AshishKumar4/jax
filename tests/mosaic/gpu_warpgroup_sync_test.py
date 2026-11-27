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
"""A reader of a lowered program's Mosaic GPU kernels finds each warpgroup sync
as the typed operation Mosaic emits for one (`warpgroup_barrier`: an
`nvvm.barrier` of a warpgroup's threads), in the kernel's module read back into
its dialects (`deserialized`). Lowered for a compile-only H100, no GPU
needed."""

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.lib.mlir import ir
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import custom_call_targets
import jax.numpy as jnp


def _kernel_modules(lowered) -> list[ir.Module]:
  """Each Mosaic GPU kernel `lowered` calls, its module as its custom call
  holds it, read back into its operations' own dialects."""
  found = []

  def walk(operation):
    for region in operation.regions:
      for block in region.blocks:
        for inner in block.operations:
          op = inner.operation
          if (op.name == "stablehlo.custom_call"
              and ir.StringAttr(op.attributes["call_target_name"]).value
              == custom_call_targets.MOSAIC_GPU):
            config = ir.DictAttr(op.attributes["mhlo.backend_config"])
            found.append(
                mgpu.deserialized(ir.StringAttr(config["module"]).value_bytes))
          walk(op)

  walk(lowered.compiler_ir("stablehlo").operation)
  return found


def _warpgroup_syncs(module: ir.Module) -> int:
  """The `nvvm.barrier`s of `module` whose thread count is a constant
  warpgroup's."""
  syncs = 0

  def visit(op):
    nonlocal syncs
    if op.name == "nvvm.barrier":
      # Its operands by segment: barrierId, then numberOfThreads.
      segments = list(ir.DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
      count = op.operands[segments[0]].owner.operation
      if (segments[1] and count.name == "arith.constant"
          and ir.IntegerAttr(count.attributes["value"]).value
          == mgpu.utils.WARPGROUP_SIZE):
        syncs += 1
    return ir.WalkResult.ADVANCE

  module.operation.walk(visit)
  return syncs


def _lowered_for_h100(call, *operands):
  """`call` lowered for a compile-only H100: no GPU in the process."""
  device = jax.sharding.AbstractDevice(
      device_kind="GPU", platform="gpu", num_cores=132, compute_capability="9.0")
  with jax.sharding.use_abstract_mesh(
      jax.sharding.AbstractMesh((), (), abstract_device=device)):
    return jax.jit(call).trace(*operands).lower(lowering_platforms=("cuda",))


class WarpgroupSyncReaderTest(jtu.JaxTestCase):

  def _syncs(self, body) -> int:
    call = plgpu.kernel(
        body, out_type=jax.ShapeDtypeStruct((128,), jnp.float32))
    lowered = _lowered_for_h100(call, jnp.ones(128, jnp.float32))
    (module,) = _kernel_modules(lowered)
    return _warpgroup_syncs(module)

  def test_a_warp_maps_two_syncs_are_read_from_its_kernel(self):
    def plain(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    def mapped(x_ref, o_ref):
      # Under lane semantics a warp's region syncs its warpgroup before and
      # after it, so every warp sees the others' effects.
      @plgpu.warp_map
      def _(warp):
        del warp

      o_ref[...] = x_ref[...] + 1.0

    self.assertEqual(self._syncs(mapped) - self._syncs(plain), 2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
