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
"""An audit of what real Mosaic GPU lowerings read, against what the cache covers.

A cached lowering is right if everything it reads is in its key, sealed, or cannot
change what it makes. The audit
(`mosaic_gpu_lowering_cache_support.state_audit`) lowers real kernels, traces the
code they run and finds every mutable global that code loads. A global that the
cache neither covers nor accepts, for a reason written down, fails it: this test
is what keeps the list of what the cache knows of as long as the lowering's.
"""

import os
import sys

from absl.testing import absltest

# The repository's pytest runs with --import-mode=importlib, which leaves this
# directory (and the support package in it) off sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mosaic_gpu_lowering_cache_support import harness  # pylint: disable=g-import-not-at-top,g-bad-import-order  # noqa: E402


class AuditTest(harness.LoweringCacheTestCase):

  def test_every_mutable_global_a_lowering_reads_is_covered_or_accepted(self):
    result = self.run_child(mode="audit", cache=False)["audit"]
    self.assertEqual(
        result["lowered"],
        ["attention", "blackwell matmul", "hopper matmul", "simple", "warps"])
    print(f"\naudited {result['executed_code_objects']} code objects: "
          f"{result['covered']} globals covered, {len(result['accepted'])} accepted")
    for name, reason in result["accepted"].items():
      print(f"  accepted {name}: {reason}")
    self.assertEmpty(
        result["uncovered"],
        "mutable globals that real lowerings read and the cache neither covers "
        f"nor accepts: {result['uncovered']}")


if __name__ == "__main__":
  absltest.main()
