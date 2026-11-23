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

"""Which nvJitLink library the CUDA plugin names, once XLA links with it."""

import os
import subprocess
import sys

from absl.testing import absltest

from jax._src import test_util as jtu


class CudaLibrariesTest(jtu.JaxTestCase):

  @jtu.run_on_devices("cuda")
  def test_nvjitlink_build_id_is_the_mapped_linkers_once_xla_links_with(self):
    # In a fresh process whose cold compile links PTX with nvJitLink (forced
    # on, no compilation cache), the stub's library is named: its build id in
    # hex, the same at every call, the build id of the file it mapped from;
    # with nvJitLink off, none of either.
    script = (
        "import jax, jax.numpy as jnp\n"
        "try:\n"
        "  from jax_plugins import xla_cuda13 as plugin\n"
        "except ImportError:\n"
        "  from jax_plugins import xla_cuda12 as plugin\n"
        "f = jax.jit(lambda x: jnp.sin(x) * 3 + 1)\n"
        "f(jnp.ones(7)).block_until_ready()\n"
        "print(plugin.nvjitlink_build_id(), plugin.nvjitlink_build_id(),\n"
        "      plugin.nvjitlink_library())\n")
    found = {}
    for mode in ("true", "false"):
      ran = subprocess.run(
          [sys.executable, "-c", script], capture_output=True, text=True,
          check=True, env={
              **os.environ, "JAX_ENABLE_COMPILATION_CACHE": "false",
              "XLA_FLAGS": f"--xla_gpu_enable_libnvjitlink={mode}"})
      found[mode] = ran.stdout.split()[-3:]
    self.assertEqual(found["false"], ["None", "None", "None"])
    build_id, again, linked = found["true"]
    self.assertEqual(build_id, again)
    self.assertRegex(build_id, r"^[0-9a-f]{16,}$")
    self.assertTrue(os.path.isabs(linked) and os.path.isfile(linked), linked)
    self.assertRegex(os.path.basename(linked),
                     r"^libnvJitLink\.so\.1[23](\.|$)")
    with open(linked, "rb") as library:
      self.assertIn(bytes.fromhex(build_id), library.read())


  @jtu.run_on_devices("cuda")
  def test_a_resolved_build_id_names_the_linker_before_the_first_compile(self):
    # In a fresh process, its backend initialized and nothing compiled, the
    # stub has loaded no library: resolving loads the one a compile would, its
    # build id the same at every call and the one the stub names once a compile
    # links PTX with it.
    script = (
        "import jax, jax.numpy as jnp\n"
        "try:\n"
        "  from jax_plugins import xla_cuda13 as plugin\n"
        "except ImportError:\n"
        "  from jax_plugins import xla_cuda12 as plugin\n"
        "jax.devices()\n"
        "before = plugin.nvjitlink_build_id()\n"
        "resolved = plugin.nvjitlink_build_id(resolve=True)\n"
        "again = plugin.nvjitlink_build_id(resolve=True)\n"
        "f = jax.jit(lambda x: jnp.sin(x) * 3 + 1)\n"
        "f(jnp.ones(7)).block_until_ready()\n"
        "print(before, resolved, again, plugin.nvjitlink_build_id())\n")
    ran = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        check=True, env={
            **os.environ, "JAX_ENABLE_COMPILATION_CACHE": "false",
            "XLA_FLAGS": "--xla_gpu_enable_libnvjitlink=true"})
    before, resolved, again, linked = ran.stdout.split()[-4:]
    self.assertEqual(before, "None")
    self.assertRegex(resolved, r"^[0-9a-f]{16,}$")
    self.assertEqual([again, linked], [resolved, resolved])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
