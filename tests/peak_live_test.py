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

import os

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp

jax.config.parse_flags_with_absl()


class PeakLiveTest(jtu.JaxTestCase):

  def test_a_compile_asked_reads_its_peaks_buffers_and_leaves_no_dump(self):
    # The live set is read from XLA's own buffer assignment of a real
    # compile: its buffers fit the temporary allocation the compile reports,
    # the products' f32 results among them, and the dump is gone once read.
    def step(a):
      b = jnp.tanh(a @ a)
      return (jnp.exp(b) @ b).sum()

    x = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    scratch = self.create_tempdir().full_path
    compiled = jax.jit(jax.grad(step)).trace(x).lower().compile(
        peak_live_in=scratch)
    live = compiled.peak_live()
    self.assertIsNotNone(live)
    self.assertTrue(live)
    self.assertLessEqual(sum(buffer.bytes for buffer in live),
                         compiled.memory_analysis().temp_size_in_bytes)
    self.assertIn(((256, 256), "f32"),
                  {(buffer.shape, buffer.dtype) for buffer in live})
    self.assertTrue(all(buffer.kind in ("parameter", "constant", "temporary")
                        and (buffer.parameter is None)
                        == (buffer.kind != "parameter") for buffer in live))
    self.assertEqual(os.listdir(scratch), [])

  def test_a_compile_not_asked_reads_none(self):
    compiled = jax.jit(jnp.sin).trace(jnp.ones(8)).lower().compile()
    self.assertIsNone(compiled.peak_live())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
