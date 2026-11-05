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

"""Which installed CUDA toolkit JAX takes as its CUDA root."""

import os
import pathlib
import sys
import tempfile

from absl.testing import absltest

from jax._src import lib


class CudaPathTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = pathlib.Path(self.enter_context(tempfile.TemporaryDirectory()))
    self.enter_context(absltest.mock.patch.dict(os.environ, {}, clear=False))
    os.environ.pop('CUDA_ROOT', None)
    os.environ.pop('PYTHON_RUNFILES', None)
    self.enter_context(absltest.mock.patch.object(sys, 'path', [str(self.root), *sys.path]))
    saved = {name: module for name, module in sys.modules.items()
             if name == 'nvidia' or name.startswith('nvidia.')}
    for name in saved:
      del sys.modules[name]
    self.addCleanup(sys.modules.update, saved)
    self.addCleanup(lambda: [sys.modules.pop(name) for name in list(sys.modules)
                             if name == 'nvidia' or name.startswith('nvidia.')])

  def installed(self, package: str, ptxas: bool, init: bool) -> pathlib.Path:
    """A wheel's directory under nvidia/, as pip lays one out: a namespace package (no __init__.py) or a regular
    one, with bin/ptxas or with another tool alone."""
    directory = self.root / 'nvidia' / package
    (directory / 'bin').mkdir(parents=True)
    (directory / 'bin' / ('ptxas' if ptxas else 'nvdisasm')).write_text('')
    if init:
      (directory / '__init__.py').write_text('')
    return directory

  def test_a_cu13_namespace_without_ptxas_yields_to_the_cuda_nvcc_wheel(self):
    # CUDA 13's nvdisasm wheel alone lays out nvidia/cu13 with no ptxas, beside CUDA 12's nvcc wheel.
    self.installed('cu13', ptxas=False, init=False)
    nvcc = self.installed('cuda_nvcc', ptxas=True, init=True)
    self.assertEqual(lib._cuda_path(), str(nvcc))

  def test_a_cu13_toolkit_with_ptxas_is_the_root(self):
    cu13 = self.installed('cu13', ptxas=True, init=False)
    self.installed('cuda_nvcc', ptxas=True, init=True)
    self.assertEqual(lib._cuda_path(), str(cu13))

  def test_no_toolkit_with_ptxas_is_no_root(self):
    self.installed('cu13', ptxas=False, init=False)
    self.assertIsNone(lib._cuda_path())


if __name__ == '__main__':
  absltest.main()
