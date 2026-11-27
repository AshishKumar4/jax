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


class JaxprEncodedOnceTest(harness.LoweringCacheTestCase):

  def test_a_jaxpr_met_before_is_read_and_keys_on_alike(self):
    # A kernel's jaxpr met again (a held trace a caller lowers in each program
    # calling it) takes its encoding without walking its equations, and the
    # key's later parts encode on from the same state either way.
    result = self.run_child(mode="jaxpr once")["jaxpr once"]
    self.assertGreater(result["walked first"], 0)
    self.assertEqual(result["walked again"], 0)
    self.assertTrue(result["same encoding"])
    self.assertTrue(result["same later part"])
    self.assertTrue(result["same primitives"])


class RulesTest(harness.LoweringCacheTestCase):

  def test_a_primitive_with_no_rule_is_refused_and_one_with_a_rule_reached(self):
    # A kernel's primitive is lowered by a Mosaic GPU rule or removed by a state
    # discharge rule; one with neither is not cached (`State.reached`).
    result = self.run_child(mode="rules")["rules"]
    self.assertEqual(result["a Mosaic GPU rule"], "reached")
    self.assertGreater(result["how many"], 0)
    self.assertEqual(result["a discharge rule alone"], ["reached"])
    self.assertEqual(result["no rule"], "refused")


if __name__ == "__main__":
  absltest.main()
