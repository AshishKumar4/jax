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
"""A module of the sealed site that registers rules when it is imported.

Added to the sealed site as ``jax/_src/lowering_cache_test_registrant.py``. As a
library's module a process imports late would, it registers a state discharge
rule for a primitive of its own, ``registered_p``, which no Mosaic GPU rule
lowers and no kernel of the cases holds. ``other_discharge_rule`` is another
sealed rule a case puts in its place.
"""

from jax._src import core
from jax._src.state import discharge

registered_p = core.Primitive("lowering_cache_test_registered")


@discharge.register_discharge_rule(registered_p)
def _registered_discharge_rule(in_avals, out_avals, *args, **params):
  del in_avals, out_avals, params
  return (None,) * len(args), args[:1]


def other_discharge_rule(in_avals, out_avals, *args, **params):
  del in_avals, out_avals, params
  return (None,) * len(args), args[-1:]
