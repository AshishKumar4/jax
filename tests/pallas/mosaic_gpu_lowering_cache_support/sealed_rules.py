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
"""Primitives and their Mosaic GPU lowering rules, of the sealed site's jax.

Added to the sealed site as ``jax/_src/lowering_cache_test_rules.py``, beside the
other sealed code. ``touch_p`` is a stand-in: it lowers to its operand, and before
it does it runs a hook the process sets, which the cases use to change a registry
from inside the lowering. ``add_a_p`` and ``add_b_p`` are both named "add", as
``lax.add_p`` is, and their lane rules are the lane add and the lane subtract rule:
primitives of one name, whose rules (both sealed code) differ. ``add_c_p`` and
``add_d_p`` are sums whose rules call a helper (`_combine`, whose keyword default
is the operation; `combine_closure`, whose closure holds it) that no registry
holds: what a helper is, is what the rule does.
"""

import operator

from jax import lax
from jax._src import core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
import jax.experimental.mosaic.gpu as mgpu

touch_p = core.Primitive("lowering_cache_test_touch")
touch_p.def_abstract_eval(lambda x, *, target: x)

HOOKS = {}


def touch(x, target):
  return touch_p.bind(x, target=target)


def _touch_lowering_rule(ctx, x, *, target):
  del ctx
  hook = HOOKS.get(target)
  if hook is not None:
    hook()
  return x


lowering.register_lowering_rule(
    touch_p, mgpu.LoweringSemantics.Warpgroup)(_touch_lowering_rule)

add_a_p = core.Primitive("add")
add_a_p.def_abstract_eval(lambda x, y: x)
add_b_p = core.Primitive("add")
add_b_p.def_abstract_eval(lambda x, y: x)


def add_a(x, y):
  return add_a_p.bind(x, y)


def add_b(x, y):
  return add_b_p.bind(x, y)


add_c_p = core.Primitive("add_c")
add_c_p.def_abstract_eval(lambda x, y: x)
add_d_p = core.Primitive("add_d")
add_d_p.def_abstract_eval(lambda x, y: x)


def add_c(x, y):
  return add_c_p.bind(x, y)


def add_d(x, y):
  return add_d_p.bind(x, y)


def _combine(x, y, *, op=operator.add):
  return op(x, y)


def _make_combine(op):
  def combine(x, y):
    return op(x, y)
  return combine


combine_closure = _make_combine(operator.add)


def _add_c_rule(ctx, x, y):
  return lowering._binary_op_lowering_rule(
      ctx, x, y, impl=lambda a, b: _combine(a, b))


def _add_d_rule(ctx, x, y):
  return lowering._binary_op_lowering_rule(
      ctx, x, y, impl=lambda a, b: combine_closure(a, b))


_lane_rules = lowering.mosaic_lowering_rules[gpu_core.LANExWG_SEMANTICS]
_lane_rules[add_a_p] = _lane_rules[lax.add_p]
_lane_rules[add_b_p] = _lane_rules[lax.sub_p]
_lane_rules[add_c_p] = _add_c_rule
_lane_rules[add_d_p] = _add_d_rule
