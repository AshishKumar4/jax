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


class UniverseTest(harness.LoweringCacheTestCase):

  def test_two_processes_that_changed_nothing_have_one_universe(self):
    # What the identity holds is what code defines, not what a process happened to
    # do: a variable a loop at module level leaves, which is one of a set that is
    # another in each process, is no part of it.
    a = self.run_child(mode="universe")["universe"]
    b = self.run_child(mode="universe")["universe"]
    self.assertGreater(len(a["modules"]), 300)
    different = [n for n in a["modules"] if a["modules"][n] != b["modules"].get(n)]
    self.assertEmpty(different)
    self.assertEqual(a["digest"], b["digest"])

  def test_a_function_rebound_is_another_universe(self):
    plain = self.run_child(mode="universe")["universe"]
    for tweak in ("bcast_to_wg", "util_split_list", "add_a_abstract_eval"):
      with self.subTest(tweak):
        changed = self.run_child(mode="universe", tweaks=[tweak])["universe"]
        self.assertNotEqual(changed["digest"], plain["digest"])
        different = [n for n in plain["modules"]
                     if plain["modules"][n] != changed["modules"].get(n)]
        self.assertNotEmpty(different)


if __name__ == "__main__":
  absltest.main()
