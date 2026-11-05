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
"""Tests of the canonical encoding a Mosaic GPU lowering's key is made of.

The cases (`mosaic_gpu_lowering_cache_support.encoder_cases`) run in a process
that has the sealed site's jax, as sealed classes (enums, dataclasses, avals) are
only encodable where the installation is sealed.
"""

import os
import sys

from absl.testing import absltest

# The repository's pytest runs with --import-mode=importlib, which leaves this
# directory (and the support package in it) off sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mosaic_gpu_lowering_cache_support import harness  # pylint: disable=g-import-not-at-top,g-bad-import-order  # noqa: E402


class EncoderTest(harness.LoweringCacheTestCase):

  def test_the_canonicalizer_is_exact_and_strict(self):
    result = self.run_child(mode="encoder")["canonicalizer"]
    self.assertEqual(result["outcomes"].keys(), result["expected"].keys())
    for name, expected in result["expected"].items():
      with self.subTest(name):
        self.assertEqual(result["outcomes"][name], expected)


class ConstantNamesTest(absltest.TestCase):
  """Which names of a module's source are constants (an analysis of the source)."""

  def test_a_name_assigned_a_literal_at_module_level_is_a_constant(self):
    from jax._src.pallas.mosaic_gpu import lowering_cache_constants  # pylint: disable=g-import-not-at-top
    source = b"""
A = 1
B: int = -2
C = (1, "x", None)
D = 1 + 2
E = 1
E = compute()
F = 0
def bump():
  global F
  F += 1
G, H = 1, 2
import os
P = 1
from sys import path as P
Q = 1
del Q
R = 1
for R in range(3):
  pass
S = 1
S += 1
T = 1
try:
  pass
except ValueError as T:
  pass
U = 1
def U(): pass
V = 1
class V: pass
if flag:
  I = 3
else:
  I = 4
try:
  J = "j"
except ImportError:
  J = None
K = ...
L = [1, 2]
__all__ = ["A"]
class M:
  N = 1
"""
    names = lowering_cache_constants.constant_names(source, "<test>")
    # A computed value, one a name is also assigned elsewhere, one a function
    # assigns (the state of the process), a tuple target, an Ellipsis, a list,
    # a dunder and a class attribute are not.
    self.assertEqual(names, ("A", "B", "C", "I", "J"))

  def test_the_constants_of_the_lowerings_core_include_its_alignment(self):
    from jax._src.pallas.mosaic_gpu import core as gpu_core  # pylint: disable=g-import-not-at-top
    from jax._src.pallas.mosaic_gpu import lowering_cache_constants  # pylint: disable=g-import-not-at-top
    with open(gpu_core.__file__, "rb") as f:
      names = lowering_cache_constants.constant_names(f.read(), gpu_core.__file__)
    self.assertIn("SMEM_ALIGNMENT", names)


if __name__ == "__main__":
  absltest.main()
