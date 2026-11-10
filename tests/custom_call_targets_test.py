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
"""A lowering writes the custom-call target jax.extend.custom_call_targets names for it."""

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.lib.mlir import ir
from jax.extend import custom_call_targets
import jax.numpy as jnp


def _targets(lowered) -> set[str]:
  found = set()

  def walk(operation):
    for region in operation.regions:
      for block in region.blocks:
        for inner in block.operations:
          op = inner.operation
          if op.name == "stablehlo.custom_call":
            found.add(ir.StringAttr(op.attributes["call_target_name"]).value)
          walk(op)

  walk(lowered.compiler_ir("stablehlo").operation)
  return found


class CustomCallTargetsTest(jtu.JaxTestCase):

  def test_an_empty_array_lowers_to_the_named_allocation(self):
    lowered = jax.jit(lambda: jax.lax.empty((4,), jnp.float32)).lower()
    self.assertIn(custom_call_targets.ALLOCATE_BUFFER, _targets(lowered))

  def test_a_gpu_threefry_lowers_to_its_platforms_named_kernel(self):
    for platform in ("cuda", "rocm"):
      with self.subTest(platform):
        lowered = jax.jit(lambda: jax.random.bits(jax.random.key(0, impl="threefry2x32"), (8,))).trace().lower(
            lowering_platforms=(platform,))
        self.assertIn(custom_call_targets.THREEFRY2X32[platform], _targets(lowered))

  def test_exports_stable_list_names_the_targets_by_their_constants(self):
    from jax._src.export import _export
    for target in (custom_call_targets.ALLOCATE_BUFFER, custom_call_targets.MOSAIC_GPU,
                   custom_call_targets.TPU_CUSTOM_CALL, custom_call_targets.THREEFRY2X32["cuda"],
                   custom_call_targets.THREEFRY2X32["rocm"]):
      self.assertIn(target, _export._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
