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
"""A deviceless capture's spec is the GPU capture's of the same launch, but what the launch wrote."""

import glob
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap

from absl.testing import absltest
import jax
from jax._src import test_util as jtu

# The kernel, run where a GPU captures it (MOSAIC_GPU_CAPTURE_LAUNCHES, read once a process) and lowered
# here: a swizzled TMA load of a 64 x 64 half tile, stored back.
KERNEL = textwrap.dedent("""
    import jax.numpy as jnp
    import numpy as np
    import jax.experimental.mosaic.gpu as mgpu
    from jax._src.lib.mlir import ir
    from jax.experimental.mosaic.gpu.utils import c

    def kernel(ctx, src, dst, smem):
      tmp, barrier = smem
      ctx.async_copy(src_ref=src, dst_ref=tmp, swizzle=128, barrier=barrier)
      barrier.wait_parity(c(0, ir.IntegerType.get_signless(1)))
      ctx.async_copy(src_ref=tmp, dst_ref=dst, swizzle=128)
      ctx.await_async_copy(0)

    x = np.arange(64 * 64, dtype=np.float16).reshape(64, 64)
    launch = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, (x, mgpu.Barrier(1)))
""")


class DevicelessCaptureTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("TMA needs an sm90 GPU to capture against")

  def test_a_deviceless_spec_is_the_gpu_captures(self):
    from jax.experimental.mosaic.gpu import capture

    root = pathlib.Path(tempfile.mkdtemp())
    script = root / "run.py"
    script.write_text(KERNEL + "\nlaunch(x).block_until_ready()\n")
    subprocess.run([sys.executable, str(script)], check=True,
                   env={**os.environ, "MOSAIC_GPU_CAPTURE_LAUNCHES": str(root / "gpu")})
    (gpu,) = glob.glob(str(root / "gpu" / "*"))

    scope = {}
    exec(KERNEL, scope)
    (kernel,) = capture.kernels(jax.jit(scope["launch"]).lower(scope["x"]).compiler_ir("stablehlo"))
    cc = jax.devices()[0].compute_capability
    deviceless = capture.capture_deviceless(kernel, cc, [scope["x"]], str(root / "deviceless"))

    def read(directory):
      return json.loads((pathlib.Path(directory) / "spec.json").read_text())

    on_gpu, off = read(gpu), read(deviceless)
    for field in ("code", "function", "grid", "block", "cluster", "dynamicShared", "buffers", "tensorMaps",
                  "params"):
      self.assertEqual(off[field], on_gpu[field], field)
    self.assertNotIn("expect", off)
    self.assertEqual(off["mosaic"]["operands"], on_gpu["mosaic"]["operands"])
    for name in (off["code"], "in0.bin"):
      self.assertEqual((pathlib.Path(deviceless) / name).read_bytes(), (pathlib.Path(gpu) / name).read_bytes(),
                       name)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
