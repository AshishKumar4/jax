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

# Two bfloat16 programs whose reductions XLA:CPU fuses with YNNPACK beside a
# clamp scalar the fusion shares, broadcast at stride zero: YNNPACK's unary
# convert read the scalar as contiguous memory, past its end, so each process's
# result differed, NaNs among them. Each prints its result's digest, NaN count
# and largest error.
_SHARED_CLAMP = """
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def f(x):
    return (jnp.maximum(jnp.sum(x, axis=-1), 1.)
            + jnp.maximum(jnp.sum(-x, axis=-1), 1.))

x = jnp.asarray(np.sin(np.arange(8192, dtype=np.float32)).reshape(4096, 2) * 2,
                jnp.bfloat16)
got = np.asarray(f(x), np.float64)
a = np.asarray(x, np.float64)
sums = a.sum(-1)
truth = np.maximum(sums, 1.) + np.maximum(-sums, 1.)
print(json.dumps({'jax': jax.__version__,
                  'digest': hashlib.sha256(got.tobytes()).hexdigest()[:16],
                  'nans': int(np.isnan(got).sum()),
                  'worst': float(np.nanmax(np.abs(got - truth)))}))
"""

_COSINE = """
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


def _runs(program: str) -> list[dict]:
  results = []
  for perturb in (None, "37", "41", "43"):
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    if perturb is not None:
      env["MALLOC_PERTURB_"] = perturb  # a read past an allocation reads junk
    out = subprocess.run([sys.executable, "-c", program], env=env,
                         capture_output=True, text=True, check=True)
    results.append(json.loads(out.stdout.strip().splitlines()[-1]))
  return results


class CpuYnnReduceTest(jtu.JaxTestCase):

  @jtu.run_on_devices("cpu")
  def test_sums_beside_a_shared_clamp_give_one_result_in_every_process(self):
    results = _runs(_SHARED_CLAMP)
    self.assertLen({result["digest"] for result in results}, 1, results)
    self.assertEqual(results[0]["nans"], 0)
    self.assertLess(results[0]["worst"], 0.05)

  @jtu.run_on_devices("cpu")
  def test_a_bf16_cosine_gives_one_result_in_every_process(self):
    results = _runs(_COSINE)
    self.assertLen({result["digest"] for result in results}, 1, results)
    self.assertEqual(results[0]["nans"], 0)
    self.assertLess(results[0]["worst_from_float64"], 0.01)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
