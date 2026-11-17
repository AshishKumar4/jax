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
"""A WGMMA's shared-operand descriptors are formed at its instructions: an
operand at one address across a loop's steps holds no more registers than one
whose address moves each step, where the compiler would otherwise hoist each of
its groups' descriptors out of the loop and hold them all across it. Compiled
for an H100 with no GPU (a deviceless capture's cubin)."""

import pathlib
import struct
import tempfile

from absl.testing import absltest
import jax
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np

# A cubin's `.nv.info` attribute holding a function's registers a thread.
_REGISTER_COUNT = 0x2F
_STEPS, _K, _N = 4, 512, 64


def _registers(cubin: bytes) -> int:
  """The registers a thread the kernel in `cubin` takes, as its `.nv.info`
  states them."""
  (sections_at,) = struct.unpack_from("<Q", cubin, 0x28)
  entry, count, names = struct.unpack_from("<HHH", cubin, 0x3A)
  headers = [struct.unpack_from("<IIQQQQIIQQ", cubin, sections_at + i * entry)
             for i in range(count)]
  names_at = headers[names][4]
  found = []
  for name, _, _, _, offset, size, *_ in headers:
    title = cubin[names_at + name:cubin.index(b"\0", names_at + name)]
    if not title.startswith(b".nv.info."):
      continue
    at = offset
    while at + 4 <= offset + size:
      form, attribute, length = struct.unpack_from("<BBH", cubin, at)
      if form == 0x04:  # A sized attribute: its symbol, then its value.
        if attribute == _REGISTER_COUNT:
          found.append(struct.unpack_from("<I", cubin, at + 8)[0])
        at += 4 + length
      else:
        at += 4
  (registers,) = found
  return registers


def _launch(moving: bool):
  """A kernel summing `_STEPS` products of a (64, `_K`) operand and a (`_K`,
  `_N`) one, both staged in shared memory once: the left one at one address
  every step, or, `moving`, at one of two slots by the step's parity."""
  transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))
  a_shape = (2, 64, _K) if moving else (64, _K)

  def body(a_gmem, b_gmem, o_gmem):
    def scoped(a_smem, b_smem, o_smem, barriers):
      plgpu.copy_gmem_to_smem(a_gmem, a_smem, barriers.at[0])
      plgpu.copy_gmem_to_smem(b_gmem, b_smem, barriers.at[1])
      plgpu.barrier_wait(barriers.at[0])
      plgpu.barrier_wait(barriers.at[1])

      def step(i, total):
        a = a_smem.at[lax.rem(i, 2)] if moving else a_smem

        def product(acc_ref):
          plgpu.wgmma(acc_ref, a, b_smem)
          return acc_ref[...]

        return total + pl.run_scoped(product, plgpu.ACC((64, _N), jnp.float32))

      start = plgpu.layout_cast(
          jnp.zeros((64, _N), jnp.float32), plgpu.Layout.WGMMA)
      o_smem[...] = lax.fori_loop(0, _STEPS, step, start)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(o_smem, o_gmem)
      plgpu.wait_smem_to_gmem(0)

    pl.run_scoped(
        scoped,
        plgpu.SMEM(a_shape, jnp.bfloat16, transforms=transforms),
        plgpu.SMEM((_K, _N), jnp.bfloat16, transforms=transforms),
        plgpu.SMEM((64, _N), jnp.float32),
        plgpu.Barrier(num_barriers=2),
    )

  call = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((64, _N), jnp.float32))
  operands = (np.zeros(a_shape, jnp.bfloat16), np.zeros((_K, _N), jnp.bfloat16))
  return call, operands


def _lowered_for_h100(call, *operands):
  """`call` lowered for a compile-only H100: no GPU in the process."""
  device = jax.sharding.AbstractDevice(
      device_kind="GPU", platform="gpu", num_cores=132, compute_capability="9.0")
  with jax.sharding.use_abstract_mesh(
      jax.sharding.AbstractMesh((), (), abstract_device=device)):
    return jax.jit(call).trace(*operands).lower(lowering_platforms=("cuda",))


class WgmmaDescriptorsTest(jtu.JaxTestCase):

  def _registers(self, moving: bool) -> int:
    from jax.experimental.mosaic.gpu import capture

    call, operands = _launch(moving)
    lowered = _lowered_for_h100(call, *operands)
    (kernel,) = capture.kernels(lowered.compiler_ir("stablehlo"))
    directory = capture.capture_deviceless(
        kernel, "9.0", list(operands), tempfile.mkdtemp())
    return _registers((pathlib.Path(directory) / "kernel.cubin").read_bytes())

  def test_an_operand_at_one_address_holds_no_more_registers_than_a_moving_one(self):
    self.assertLessEqual(self._registers(moving=False),
                         self._registers(moving=True))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
