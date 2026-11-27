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


class SourceTest(harness.LoweringCacheTestCase):

  def test_a_file_whose_path_cannot_resolve_to_a_declared_one_is_not_resolved(self):
    # The seal answers whether a file is a declared package's by its path and,
    # where that misses, by its resolved one. A path whose trailing components,
    # no links, end no declared path is answered without resolving it; a path
    # through a link (to a directory or to the file), a declared file replaced
    # by a copy of its bytes and asked through another link, a path with a
    # component the system cannot read (missing, or through more links than
    # it follows) and a path of another type are resolved; every answer is
    # what resolving gives. A seal
    # declaring no package answers a path without reading the file system.
    # POSIX paths alone: elsewhere every miss is resolved.
    if os.name != "posix":
      self.skipTest("POSIX paths")
    package = self.tmp / "declaredpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "part.py").write_text("x = 1\n")
    path = os.pathsep.join(
        [str(self._site), str(harness.TESTS_PALLAS), str(self.tmp)])
    found = self.run_child(
        mode="sources", package="declaredpkg",
        env={"PYTHONPATH": path,
             "JAX_PALLAS_LOWERING_CACHE_SOURCES": "declaredpkg"})["sources"]
    answers = found["answers"]
    self.assertIsNotNone(answers["declared"])
    for name in ("declared through a linked directory",
                 "declared through a link to it",
                 "through more links than the system follows",
                 "a string of a class of its own",
                 "replaced, through another link"):
      self.assertEqual(answers[name], answers["declared"], name)
    for name in ("no declared package's", "named as a declared file elsewhere",
                 "missing", "of a seal declaring no package"):
      self.assertIsNone(answers[name], name)
    self.assertEqual(found["resolved"], [
        "a string of a class of its own", "declared through a link to it",
        "declared through a linked directory", "missing",
        "replaced, through another link",
        "through more links than the system follows"])
    self.assertNotIn("of a seal declaring no package", found["read"])


class DeclaredTest(harness.LoweringCacheTestCase):

  def test_what_an_inline_body_runs_is_checked_not_its_wrapper(self):
    # A function, and what a partial, a jitted function or a callable object
    # runs: none of a declared package here, so none is admitted.
    found = self.run_child(mode="declared")["declared"]
    self.assertEqual(found, dict.fromkeys(found, False))
    self.assertLen(found, 5)

  def test_only_a_declared_function_or_an_exact_partial_of_one_is_admitted(self):
    # A package the process declares: its function and an exact partial of it
    # run that function alone; a partial subclass overriding `__call__`, a
    # callable object and a jitted function may run code no key holds.
    package = self.tmp / "declaredpkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import functools\n"
        "def body(ctx, ref):\n  del ctx, ref\n"
        "class Calling(functools.partial):\n"
        "  def __call__(self, *args):\n    return None\n"
        "class Body:\n  def __call__(self, ctx, ref):\n    del ctx, ref\n")
    path = os.pathsep.join(
        [str(self._site), str(harness.TESTS_PALLAS), str(self.tmp)])
    found = self.run_child(
        mode="declared", package="declaredpkg",
        env={"PYTHONPATH": path,
             "JAX_PALLAS_LOWERING_CACHE_SOURCES": "declaredpkg"})["declared"]
    self.assertEqual(found, {
        "a declared function": True,
        "a partial of a declared function": True,
        "a partial subclass of a declared function": False,
        "a declared callable object": False,
        "a jitted declared function": False,
    })


if __name__ == "__main__":
  absltest.main()
