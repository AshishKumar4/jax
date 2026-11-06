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
"""Frozen and mutable dataclasses of the sealed site's jax (sealed code).

Added to the sealed site as ``jax/_src/lowering_cache_test_classes.py``, which is
what the canonicalizer cases need classes of sealed code for.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class WithIgnored:
  value: int
  ignored: int = dataclasses.field(default=0, compare=False)


@dataclasses.dataclass
class Mutable:
  value: int
