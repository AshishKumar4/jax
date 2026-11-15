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

import json
import os
import subprocess
import sys

from absl.testing import absltest
from jax._src import test_util as jtu

# A bfloat16 cosine over (32, 128, 24): each operand over its L2 norm, then the
# sum of their product on the last axis, jitted. XLA:CPU's YNNPACK fusion of its
# reductions read past a broadcast operand's scalar (XNNPACK's matrix packer
# before 0e7357e2), so each process's result differed, NaNs among them.
_PROGRAM = """
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np


def normalized(x, eps):
    norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(x), axis=-1, keepdims=True), eps * eps))
    return x / jnp.maximum(norm, eps)


@jax.jit
def cosine(fields, options):
    return jnp.sum(normalized(fields, 1e-8) * normalized(options, 1e-8), axis=-1)


shape = (32, 128, 24)
z = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
fields = jnp.asarray(np.sin(z) * 2, jnp.bfloat16)
options = jnp.asarray(np.cos(z) * 2, jnp.bfloat16)
got = np.asarray(cosine(fields, options), np.float64)
a, b = (np.asarray(v, np.float64) for v in (fields, options))
truth = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
print(json.dumps({"jax": jax.__version__, "dtype": str(cosine(fields, options).dtype),
                  "digest": hashlib.sha256(got.tobytes()).hexdigest()[:16],
                  "nans": int(np.isnan(got).sum()),
                  "worst_from_float64": float(np.nanmax(np.abs(got - truth)))}))
"""


class CpuYnnReduceTest(jtu.JaxTestCase):

  @jtu.run_on_devices("cpu")
  def test_a_bf16_cosines_reductions_give_one_result_in_every_process(self):
    results = []
    for perturb in (None, "37", "41", "43"):
      env = dict(os.environ, JAX_PLATFORMS="cpu")
      if perturb is not None:
        env["MALLOC_PERTURB_"] = perturb  # a read past an allocation reads junk
      out = subprocess.run([sys.executable, "-c", _PROGRAM], env=env,
                           capture_output=True, text=True, check=True)
      results.append(json.loads(out.stdout.strip().splitlines()[-1]))
    self.assertLen({result["digest"] for result in results}, 1, results)
    self.assertEqual(results[0]["nans"], 0)
    self.assertLess(results[0]["worst_from_float64"], 0.01)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
