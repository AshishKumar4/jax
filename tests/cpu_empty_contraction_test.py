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

import subprocess
import sys

from absl.testing import absltest
from jax._src import test_util as jtu

# A dot of float32 constants over an empty contraction, [256, 0] by [0, 128],
# compiled for the CPU with XLA's algebraic simplifier off: constant folding
# evaluates it, and the evaluator's Eigen contraction divided by the empty
# depth, killing the process with SIGFPE. Its product is zeros.
_FOLDED = """
import jax
import jax.numpy as jnp
import numpy as np

options = {'xla_disable_hlo_passes': 'algsimp'}
got = jax.jit(lambda: jnp.zeros((256, 0), jnp.float32)
              @ jnp.zeros((0, 128), jnp.float32),
              compiler_options=options)()
np.testing.assert_array_equal(np.asarray(got), np.zeros((256, 128), np.float32))
print('zeros')
"""


class CpuEmptyContractionTest(jtu.JaxTestCase):

  def test_a_folded_dot_over_an_empty_contraction_is_zeros(self):
    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')
    # Its own process: the fault it guards against kills the process.
    out = subprocess.run([sys.executable, '-c', _FOLDED], capture_output=True,
                         text=True, check=False, timeout=600)
    self.assertEqual(out.returncode, 0, out.stderr[-2000:])
    self.assertIn('zeros', out.stdout)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
