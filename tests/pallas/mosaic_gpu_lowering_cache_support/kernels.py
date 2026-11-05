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
"""The kernels the cache tests lower.

Each is a function of (size, power, params, scale) that returns the kernel. The
child process copies this file into the case's work directory, with lines added
above it, so that a kernel moved in its file can be told from one that changed.
`scale` is a literal of the jaxpr, `power` a param of its integer_pow, `size` an
aval's shape and `params` its compiler params (or None).
"""

import collections
import dataclasses
import importlib

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax._src import core as jax_core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
import numpy as np


def plain(size, power, params, scale):
  kwargs = {} if params is None else {"compiler_params": params}

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((size,), jnp.float32),
      num_threads=1,
      thread_name="wg",
      **kwargs,
  )
  def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] ** power * scale + 1.0

  return kernel


def smem(size, power, params, scale):
  """A kernel that holds shared memory: `size` float32s of it."""
  del params

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((size,), jnp.float32),
      scratch_types=[plgpu.SMEM((size,), jnp.float32)],
      num_threads=1,
      thread_name="wg",
  )
  def kernel(x_ref, o_ref, smem):
    smem[...] = x_ref[...] ** power * scale + 1.0
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(smem, o_ref)
    plgpu.wait_smem_to_gmem(0)

  return kernel


def warp(size, power, params, scale):
  """A kernel whose warps each have a part: warp_map names its axis with an
  object of its own, made afresh by every process."""
  del size, power, params

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((4, 2, 64), jnp.float32),
      scratch_types=(plgpu.SMEM((2, 4, 64), jnp.float32),),
  )
  def kernel(in_ref, out_ref, smem):
    smem[...] = in_ref[...] * scale

    @plgpu.warp_map
    def _(warp_id):
      @pl.when(warp_id == 0)
      def _():
        plgpu.copy_smem_to_gmem(smem.transpose((1, 0, 2)), out_ref)
        plgpu.wait_smem_to_gmem(0)

  return kernel


def lane(size, power, params, scale):
  """A kernel of the lane semantics: its sum is the lane add rule's."""
  del power, params

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((size,), jnp.float32),
      num_threads=1,
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(
          lowering_semantics=plgpu.LoweringSemantics.Lane),
  )
  def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] + scale

  return kernel


@dataclasses.dataclass(frozen=True)
class OwnLayout(gpu_core.SomeLayout):
  """A layout of a class this file (no sealed file) defines."""
  tag: int = 0

  def to_mgpu(self, *args, **kwargs):
    return plgpu.Layout.WGMMA.to_mgpu()


def own_layout(size, power, params, scale):
  """A kernel with a param of a class no sealed file defines."""
  del power, params

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((64, size), jnp.float32),
      num_threads=1, thread_name="wg")
  def kernel(x_ref, o_ref):
    o_ref[...] = plgpu.layout_cast(x_ref[...], OwnLayout()) * scale

  return kernel


@dataclasses.dataclass(frozen=True)
class ForgedLayout(gpu_core.SomeLayout):
  """A layout whose class says it is of a sealed module (see `forged_layout`)."""

  def to_mgpu(self, *args, **kwargs):
    if _FORGED_VARIANT[0] == "b":
      raise ValueError("the layout of variant b")
    return plgpu.Layout.WGMMA.to_mgpu()


_FORGED_VARIANT = ["a"]


def forged_layout(size, power, params, scale, *, forge, variant):
  """A kernel with a layout that says it is of a sealed module.

  `forge` is how: "module" only claims it (`__module__`), "inject" also puts the
  class there, so that looking it up by its name finds it; "numpy", "stdlib" and
  "extension" claim a module of NumPy, the standard library (`collections`) and
  a compiled one (`jaxlib._jax`) and put it there. `variant` is what the class
  does: a layout ("a"), or an error ("b"), which no field says.
  """
  del power, params
  namespaces = {
      "module": (gpu_core.__name__, None),
      "inject": (gpu_core.__name__, gpu_core),
      "numpy": ("numpy", np),
      "stdlib": ("collections", collections),
      "extension": ("jaxlib._jax", importlib.import_module("jaxlib._jax")),
  }
  module_name, namespace = namespaces[forge]
  ForgedLayout.__module__ = module_name
  if namespace is not None:
    namespace.ForgedLayout = ForgedLayout
  _FORGED_VARIANT[0] = variant

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((64, size), jnp.float32),
      num_threads=1, thread_name="wg")
  def kernel(x_ref, o_ref):
    o_ref[...] = plgpu.layout_cast(x_ref[...], ForgedLayout()) * scale

  return kernel


def touch(size, power, params, scale):
  """A kernel that runs the stand-in rule, which runs the process's hook."""
  del power, params
  from jax._src import lowering_cache_test_rules as rules  # pylint: disable=g-import-not-at-top

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((size,), jnp.float32),
      num_threads=1, thread_name="wg")
  def kernel(x_ref, o_ref):
    o_ref[...] = rules.touch(x_ref[...] * scale, "registry")

  return kernel


def compare(size, power, params, scale):
  """A kernel with a float comparison: the dialect's cmpf rule."""
  del power, params

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((size,), jnp.float32),
      num_threads=1, thread_name="wg")
  def kernel(x_ref, o_ref):
    v = x_ref[...]
    o_ref[...] = jnp.where(v > scale, v, v * 2.0)

  return kernel


def wgmma_layout(size, power, params, scale):
  """A kernel that casts to the WGMMA layout, a constant of the mosaic package."""
  del power, params

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((64, size), jnp.float32),
      num_threads=1, thread_name="wg")
  def kernel(x_ref, o_ref):
    o_ref[...] = plgpu.layout_cast(x_ref[...], plgpu.Layout.WGMMA) * scale

  return kernel


def _lane_add(add):
  """The lane kernel `add(x_ref, scale)`, its operands as they are (a weak scalar)."""

  def factory(size, power, params, scale):
    del power, params

    @plgpu.kernel(
        out_type=jax.ShapeDtypeStruct((size,), jnp.float32),
        num_threads=1,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Lane),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = add(x_ref[...], scale)

    return kernel

  return factory


def _sealed_rules():
  # Imported for what it registers too: the lane rules of its primitives.
  return importlib.import_module("jax._src.lowering_cache_test_rules")


def alias_a(size, power, params, scale, *, register_unsealed=False):
  """The sum a primitive named "add" of the sealed site makes, whose lane rule
  is the lane add rule. `register_unsealed` registers `alias_unsealed`'s
  primitive too, so that the registries are those of that kernel's process."""
  if register_unsealed:
    _register_unsealed_add()
  return _lane_add(_sealed_rules().add_a)(size, power, params, scale)


def alias_c(size, power, params, scale):
  """A sum by a helper with a keyword default (`_combine`'s operation)."""
  return _lane_add(_sealed_rules().add_c)(size, power, params, scale)


def alias_d(size, power, params, scale):
  """A sum by a helper that closes over its operation (`combine_closure`)."""
  return _lane_add(_sealed_rules().add_d)(size, power, params, scale)


def alias_b(size, power, params, scale):
  """The same, by another primitive of the sealed site named "add", whose lane
  rule is the lane subtract rule: its jaxpr is `alias_a`'s but for the primitive."""
  return _lane_add(_sealed_rules().add_b)(size, power, params, scale)


# A primitive named "add" that no sealed module holds, given the lane subtract rule.
_unsealed_add_p = jax_core.Primitive("add")
_unsealed_add_p.def_abstract_eval(lambda x, y: x)


def _register_unsealed_add():
  rules = lowering.mosaic_lowering_rules[gpu_core.LANExWG_SEMANTICS]
  rules[_unsealed_add_p] = rules[lax.sub_p]


def alias_unsealed(size, power, params, scale):
  """The same, by a primitive named "add" that no sealed module holds, whose
  lane rule is the lane subtract rule."""
  _register_unsealed_add()
  _sealed_rules()  # So the registries are `alias_a`'s.
  return _lane_add(lambda x, y: _unsealed_add_p.bind(x, y))(
      size, power, params, scale)
