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
"""A Pallas Mosaic GPU kernel's caller-stated metadata reaches its custom call."""

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.lib.mlir import ir
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import custom_call_targets
import jax.numpy as jnp


def _calls(lowered) -> list[ir.DictAttr]:
  found = []

  def walk(operation):
    for region in operation.regions:
      for block in region.blocks:
        for inner in block.operations:
          op = inner.operation
          if (op.name == "stablehlo.custom_call"
              and ir.StringAttr(op.attributes["call_target_name"]).value
              == custom_call_targets.MOSAIC_GPU):
            found.append(ir.DictAttr(op.attributes["mhlo.backend_config"]))
          walk(op)

  walk(lowered.compiler_ir("stablehlo").operation)
  return found


class KernelMetadataTest(jtu.JaxTestCase):

  def _lowered(self, **params):
    def body(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    call = plgpu.kernel(
        body, out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
        compiler_params=plgpu.CompilerParams(**params))
    return jax.jit(call).trace(jnp.ones(128, jnp.float32)).lower(
        lowering_platforms=("cuda",))

  def test_a_kernels_stated_metadata_is_its_calls_attribute(self):
    (config,) = _calls(self._lowered(metadata=b'{"launch": "a1b2"}'))
    self.assertEqual(ir.StringAttr(config["metadata"]).value_bytes,
                     b'{"launch": "a1b2"}')

  def test_a_kernel_stating_none_carries_none(self):
    (config,) = _calls(self._lowered())
    self.assertNotIn("metadata", config)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
