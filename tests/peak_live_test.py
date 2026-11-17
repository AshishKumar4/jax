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

from typing import Any

from absl.testing import absltest
import jax
from jax import ad_checkpoint
from jax._src import hlo_reader
from jax._src import peak_live
from jax._src import test_util as jtu
from jax._src.lib import xla_client as xc
import jax.experimental.topologies as topologies
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

jax.config.parse_flags_with_absl()


def _step(x, w):
  return jnp.tanh(x @ w) @ w


class PeakLiveTest(jtu.JaxTestCase):

  def _names_of_lowered_results(self, fun, opcode, shapes):
    lowered = jax.jit(fun).lower(
        *[jax.ShapeDtypeStruct(shape, jnp.float32) for shape in shapes])
    serialized = lowered.compiler_ir("hlo").as_serialized_hlo_module_proto()
    module = hlo_reader.message_class("xla.HloModuleProto").FromString(serialized)
    instruction, = [instruction
                    for computation in module.computations
                    for instruction in computation.instructions
                    if instruction.opcode == opcode]
    # Give each array result a logical buffer at the peak. This isolates name
    # transport from backend scheduling and allocation choices.
    assignment = hlo_reader.message_class("xla.BufferAssignmentProto")()
    allocation = assignment.buffer_allocations.add(index=0, maybe_live_out=True)
    peak = assignment.peak_buffers.add(index=allocation.index)
    indices = ([(i,) for i in range(len(instruction.shape.tuple_shapes))]
               if instruction.shape.tuple_shapes else [()])
    for number, index in enumerate(indices):
      buffer = assignment.logical_buffers.add(id=number, size=4)
      buffer.defined_at.instruction_id = instruction.id
      buffer.defined_at.shape_index.extend(index)
      peak.logical_buffer_ids.append(number)
    return [buffer.names for buffer in peak_live.live(
        assignment.SerializeToString(), serialized)]

  def test_checkpoint_name_with_comma_is_one_name(self):
    def f(x):
      return ad_checkpoint.checkpoint_name(jnp.sin(x), "left,right")

    self.assertEqual(self._names_of_lowered_results(f, "sine", [(8,)]),
                     [("left,right",)])

  def test_checkpoint_name_without_comma_is_one_name(self):
    def f(x):
      return ad_checkpoint.checkpoint_name(jnp.sin(x), "left")

    self.assertEqual(self._names_of_lowered_results(f, "sine", [(8,)]),
                     [("left",)])

  def test_checkpoint_names_belong_to_individual_sort_results(self):
    def f(keys, values):
      keys, values = jax.lax.sort_key_val(keys, values)
      return (ad_checkpoint.checkpoint_name(keys, "keys"),
              ad_checkpoint.checkpoint_name(values, "values"))

    self.assertEqual(self._names_of_lowered_results(f, "sort", [(8,), (8,)]),
                     [("keys",), ("values",)])

  def test_checkpoint_names_on_separate_result_operations_stay_separate(self):
    def f(keys, values):
      keys, values = jax.lax.sort_key_val(keys, values)
      return (ad_checkpoint.checkpoint_name(jnp.sin(keys), "keys"),
              ad_checkpoint.checkpoint_name(jnp.cos(values), "values"))

    self.assertEqual(self._names_of_lowered_results(f, "sine", [(8,), (8,)]),
                     [("keys",)])
    self.assertEqual(self._names_of_lowered_results(f, "cosine", [(8,), (8,)]),
                     [("values",)])

  @jtu.run_on_devices("cpu", "cuda")
  def test_a_compiles_peak_holds_its_parameters_and_the_temporaries_it_reads_at_once(self):
    # tanh(x @ w) @ w: at the second product's peak, x's and w's buffers and
    # the first product's tanh, one temporary of a 256 x 256 f32 tile.
    tile = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    compiled = jax.jit(_step).lower(tile, tile).compile()
    live = compiled.peak_live()
    self.assertIsNotNone(live)
    parameters = [buffer for buffer in live if buffer.kind == "parameter"]
    self.assertLen(parameters, 2)
    for buffer in parameters:
      self.assertEqual((buffer.shape, buffer.dtype, buffer.bytes),
                       ((256, 256), "f32", 256 * 256 * 4))
    temporaries = [buffer for buffer in live if buffer.kind == "temporary"]
    self.assertTrue(any(
        buffer.shape == (256, 256) and buffer.dtype == "f32"
        and "tanh" in buffer.op_name for buffer in temporaries), live)
    self.assertLessEqual(sum(buffer.bytes for buffer in temporaries),
                         compiled.memory_analysis().temp_size_in_bytes)

  @jtu.run_on_devices("cpu", "cuda")
  def test_a_residual_a_policy_saves_is_named_by_its_checkpoint_name(self):
    # sin(tanh(x @ w))'s gradient in w under a policy saving the residual
    # named "hidden" alone: the backward reads the saved tanh beside the
    # cotangent it forms from it, the rows' widest buffers, so at its peak the
    # residual is live, its buffer named as jax.ad_checkpoint.checkpoint_name
    # named it; each parameter's buffer states its entry parameter's number.
    def step(x, w):
      hidden = ad_checkpoint.checkpoint_name(jnp.tanh(x @ w), "hidden")
      return jnp.sum(jnp.sin(hidden))

    saved = jax.checkpoint(
        step, policy=jax.checkpoint_policies.save_only_these_names("hidden"))
    rows = jax.ShapeDtypeStruct((2048, 256), jnp.float32)
    weight = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    live = jax.jit(jax.grad(saved, argnums=1)).lower(rows, weight).compile().peak_live()
    self.assertIsNotNone(live)
    named = [buffer for buffer in live if buffer.names]
    self.assertTrue(named, live)
    self.assertEqual({buffer.names for buffer in named}, {("hidden",)})
    self.assertEqual(
        sorted(buffer.parameter for buffer in live if buffer.kind == "parameter"),
        [0, 1])

  @jtu.run_on_devices("cuda")
  def test_a_probe_on_the_devices_twin_states_the_devices_own_compile(self):
    # The probe: a compile for a compile-only twin of the attached device,
    # its target config the device's own (its memory size, cores, clocks and
    # library versions, as XLA reads them), so jax compiles it through the
    # device's client (`_get_cross_compile_backend`), whose executor autotunes
    # as the device's own compile does, stopping after buffer assignment. It
    # states the memory and the buffers live at its peak as the device's own
    # whole compile of the program does.
    devices = jax.devices()[:1]
    topology = topologies.get_topology_desc(
        platform="cuda",
        target_config=xc.get_topology_for_devices(devices).target_config,
        topology="1x1x1")
    twin = jax.sharding.NamedSharding(
        topologies.make_mesh(topology, (1,), ("x",)), P())
    probe = jax.jit(_step).lower(
        *[jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=twin)] * 2
    ).compile({"xla_gpu_experimental_early_exit":
               "EARLY_EXIT_POINT_AFTER_BUFFER_ASSIGNMENT"})
    attached = jax.sharding.NamedSharding(
        jax.sharding.Mesh(devices, ("x",)), P())
    whole = jax.jit(_step).lower(
        *[jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=attached)] * 2
    ).compile()
    self.assertEqual(probe.memory_analysis().temp_size_in_bytes,
                     whole.memory_analysis().temp_size_in_bytes)
    self.assertIsNotNone(whole.peak_live())
    self.assertEqual(probe.peak_live(), whole.peak_live())

def _descended(text: str) -> dict[str, Any]:
  """The provenance of each instruction of HLO module ``text``, by name."""
  module = hlo_reader.message_class("xla.HloModuleProto").FromString(
      hlo_reader.module_from_text(text).as_serialized_hlo_module_proto())
  found = peak_live.provenance(module)
  return {instruction.name: found[instruction.id]
          for computation in module.computations
          for instruction in computation.instructions
          if instruction.id in found}


def _of(*pairs: tuple[int, int]) -> frozenset[tuple[int, int]]:
  return frozenset(pairs)


class DimensionProvenanceTest(jtu.JaxTestCase):

  def test_custom_call_root_alias_does_not_require_output_dims(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        x = f32[8,16] parameter(0)
        ROOT called = f32[8,16] custom-call(x), custom_call_target="kernel",
            output_to_operand_aliasing={{}: (0, {})}
      }""")
    self.assertEqual(descended["called"], (_of((0, 0)), _of((0, 1))))

  def test_custom_call_root_alias_overrides_unknown_output_dims(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        x = f32[8,16] parameter(0)
        ROOT called = f32[8,16] custom-call(x), custom_call_target="kernel",
            output_to_operand_aliasing={{}: (0, {})},
            frontend_attributes={output_dims="[null]"}
      }""")
    self.assertEqual(descended["called"], (_of((0, 0)), _of((0, 1))))

  def test_custom_call_nested_alias_resolves_both_shape_indices(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        x = f32[8,16] parameter(0)
        arg = (f32[8,16]) tuple(x)
        ROOT called = ((f32[8,16])) custom-call(arg),
            custom_call_target="kernel",
            output_to_operand_aliasing={{0,0}: (0, {0})},
            frontend_attributes={output_dims="[null]"}
      }""")
    self.assertEqual(descended["called"], [[(_of((0, 0)), _of((0, 1)))]])

  def test_custom_call_alias_resolves_operand_shape_index(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        x = f32[8,16] parameter(0)
        arg = (f32[8,16]) tuple(x)
        ROOT called = (f32[8,16]) custom-call(arg),
            custom_call_target="kernel",
            output_to_operand_aliasing={{0}: (0, {0})},
            frontend_attributes={output_dims="[null]"}
      }""")
    self.assertEqual(descended["called"], [(_of((0, 0)), _of((0, 1)))])

  def test_custom_call_tuple_alias_does_not_require_output_dims(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        x = f32[8,16] parameter(0)
        ROOT called = (f32[8,16]) custom-call(x),
            custom_call_target="kernel",
            output_to_operand_aliasing={{0}: (0, {})}
      }""")
    self.assertEqual(descended["called"], [(_of((0, 0)), _of((0, 1)))])

  def test_triangular_solve_free_dimension_does_not_descend_from_matrix(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        a = f32[8,8] parameter(0)
        b = f32[8,3] parameter(1)
        ROOT solved = f32[8,3] triangular-solve(a, b), left_side=true,
            lower=true, unit_diagonal=false, transpose_a=NO_TRANSPOSE
      }""")
    self.assertEqual(descended["solved"][1], _of((1, 1)))

  def test_a_dot_keeps_its_batch_and_free_dimensions_and_drops_the_contracted(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        x = f32[4,8,16] parameter(0)
        w = f32[4,16,32] parameter(1)
        ROOT d = f32[4,8,32] dot(x, w), lhs_batch_dims={0}, rhs_batch_dims={0},
            lhs_contracting_dims={2}, rhs_contracting_dims={1}
      }""")
    self.assertEqual(descended["d"], (_of((0, 0), (1, 0)), _of((0, 1)),
                                      _of((1, 2))))

  def test_a_reshape_merges_and_splits_and_a_size_one_dimension_descends_from_none(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        x = f32[6,4,5] parameter(0)
        merged = f32[24,5] reshape(x)
        ROOT split = f32[2,3,1,20] reshape(merged)
      }""")
    self.assertEqual(descended["merged"], (_of((0, 0), (0, 1)), _of((0, 2))))
    self.assertEqual(descended["split"],
                     (_of((0, 0), (0, 1)), _of((0, 0), (0, 1)), _of(),
                      _of((0, 0), (0, 1), (0, 2))))

  def test_broadcast_transpose_and_reduce_place_and_drop_dimensions(self):
    descended = _descended("""
      HloModule m
      add {
        a = f32[] parameter(0)
        b = f32[] parameter(1)
        ROOT s = f32[] add(a, b)
      }
      ENTRY e {
        x = f32[8,16] parameter(0)
        wide = f32[3,8,16] broadcast(x), dimensions={1,2}
        t = f32[16,3,8] transpose(wide), dimensions={2,0,1}
        zero = f32[] constant(0)
        ROOT r = f32[16,8] reduce(t, zero), dimensions={1}, to_apply=add
      }""")
    self.assertEqual(descended["wide"], (_of(), _of((0, 0)), _of((0, 1))))
    self.assertEqual(descended["t"], (_of((0, 1)), _of(), _of((0, 0))))
    self.assertEqual(descended["r"], (_of((0, 1)), _of((0, 0))))

  def test_a_weight_gradient_descends_from_no_row_of_the_batch_it_reads(self):
    # The gradient of sum((x @ w) ** 2) by w: x's rows contracted away.
    descended = _descended("""
      HloModule m
      ENTRY e {
        x = f32[64,512] parameter(0)
        w = f32[512,256] parameter(1)
        y = f32[64,256] dot(x, w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
        two = f32[] constant(2)
        twos = f32[64,256] broadcast(two), dimensions={}
        g = f32[64,256] multiply(y, twos)
        ROOT dw = f32[512,256] dot(x, g), lhs_contracting_dims={0}, rhs_contracting_dims={0}
      }""")
    self.assertEqual(descended["y"], (_of((0, 0)), _of((1, 1))))
    self.assertEqual(descended["dw"], (_of((0, 1)), _of((1, 1))))

  def test_a_while_loop_carries_what_its_body_adds_to_a_fixed_point(self):
    # The body adds the transpose of its square carry into it, so after a pass
    # each dimension descends from both of the carry's axes.
    descended = _descended("""
      HloModule m
      body {
        p = (s32[], f32[8,8]) parameter(0)
        i = s32[] get-tuple-element(p), index=0
        c = f32[8,8] get-tuple-element(p), index=1
        t = f32[8,8] transpose(c), dimensions={1,0}
        n = f32[8,8] add(c, t)
        one = s32[] constant(1)
        j = s32[] add(i, one)
        ROOT out = (s32[], f32[8,8]) tuple(j, n)
      }
      cond {
        p = (s32[], f32[8,8]) parameter(0)
        i = s32[] get-tuple-element(p), index=0
        k = s32[] constant(4)
        ROOT less = pred[] compare(i, k), direction=LT
      }
      ENTRY e {
        x = f32[8,8] parameter(0)
        zero = s32[] constant(0)
        init = (s32[], f32[8,8]) tuple(zero, x)
        loop = (s32[], f32[8,8]) while(init), condition=cond, body=body
        ROOT result = f32[8,8] get-tuple-element(loop), index=1
      }""")
    both = _of((0, 0), (0, 1))
    self.assertEqual(descended["result"], (both, both))
    self.assertEqual(descended["n"], (both, both))

  def test_a_fusion_reads_its_operands_through_its_body(self):
    descended = _descended("""
      HloModule m
      fused {
        a = f32[8,4] parameter(0)
        ROOT t = f32[4,8] transpose(a), dimensions={1,0}
      }
      ENTRY e {
        x = f32[8,4] parameter(0)
        ROOT f = f32[4,8] fusion(x), kind=kLoop, calls=fused
      }""")
    self.assertEqual(descended["f"], (_of((0, 1)), _of((0, 0))))

  def test_a_gather_takes_its_batch_from_its_indices_and_its_offsets_from_its_operand(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        table = f32[100,32] parameter(0)
        ids = s32[16,1] parameter(1)
        ROOT rows = f32[16,32] gather(table, ids), offset_dims={1},
            collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
            slice_sizes={1,32}
      }""")
    self.assertEqual(descended["rows"], (_of((1, 0)), _of((0, 1))))

  def test_a_custom_call_states_its_dimensions_or_they_are_unknown(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        x = f32[8,16] parameter(0)
        stated = f32[16,8] custom-call(x), custom_call_target="kernel",
            frontend_attributes={output_dims="[[[[0,1]],[[0,0]]]]"}
        ROOT unstated = f32[8,16] custom-call(stated), custom_call_target="other"
      }""")
    self.assertEqual(descended["stated"], (_of((0, 1)), _of((0, 0))))
    self.assertEqual(descended["unstated"], (None, None))

  def test_a_custom_calls_aliased_output_is_its_operands_and_a_null_output_unknown(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        x = f32[8,16] parameter(0)
        acc = f32[4] parameter(1)
        called = (f32[16,8], f32[4], f32[32], f32[2]) custom-call(x, acc),
            custom_call_target="kernel",
            output_to_operand_aliasing={{1}: (1, {})},
            frontend_attributes={output_dims="[[[[0,1]],[[0,0]]], null]"}
        ROOT unstated = f32[8,16] custom-call(x), custom_call_target="kernel",
            frontend_attributes={output_dims="[null]"}
      }""")
    self.assertEqual(descended["called"],
                     [(_of((0, 1)), _of((0, 0))), (_of((1, 0)),), (None,),
                      (None,)])
    self.assertEqual(descended["unstated"], (None, None))

  def test_iota_and_constants_descend_from_none(self):
    descended = _descended("""
      HloModule m
      ENTRY e {
        x = f32[8] parameter(0)
        i = f32[8] iota(), iota_dimension=0
        ROOT s = f32[8] add(x, i)
      }""")
    self.assertEqual(descended["i"], (_of(),))
    self.assertEqual(descended["s"], (_of((0, 0)),))

  @jtu.run_on_devices("cpu")
  def test_dimensions_name_the_calls_arguments_past_the_ones_it_drops(self):
    # The first argument unused, so pruned from the module: x is the call's
    # argument 1 though it is the module's parameter 0.
    def f(unused, x):
      del unused
      return jnp.tanh(x).sum(axis=1)

    rows = jax.ShapeDtypeStruct((24, 40), jnp.float32)
    live = jax.jit(f).lower(rows, rows).compile().peak_live()
    self.assertIsNotNone(live)
    parameter, = (buffer for buffer in live if buffer.kind == "parameter")
    self.assertEqual(parameter.dims, (_of((1, 0)), _of((1, 1))))
    self.assertTrue(any(buffer.dims == (_of((1, 0)),) for buffer in live
                        if buffer.kind != "parameter"), live)

  @jtu.run_on_devices("cpu")
  def test_an_ordered_effects_token_is_no_argument(self):
    # An ordered callback's token comes first among the module's parameters:
    # x is still the call's argument 0.
    def f(x):
      jax.debug.callback(lambda: None, ordered=True)
      return jnp.tanh(x).sum(axis=1)

    rows = jax.ShapeDtypeStruct((24, 40), jnp.float32)
    live = jax.jit(f).lower(rows).compile().peak_live()
    self.assertIsNotNone(live)
    self.assertIn((_of((0, 0)), _of((0, 1))),
                  [buffer.dims for buffer in live if buffer.kind == "parameter"])
    self.assertTrue(any(buffer.dims == (_of((0, 0)),) for buffer in live
                        if buffer.kind != "parameter"), live)

  @jtu.run_on_devices("cpu")
  def test_each_live_buffer_states_which_dimensions_hold_the_batchs_rows(self):
    # A step over 64 rows of 512 features: every live dimension of 64 descends
    # from the rows of x (parameter 0, axis 0) and no other does, the weight
    # gradients among them reading x but holding none of its rows.
    def loss(ws, x):
      for w in ws:
        x = jnp.tanh(x @ w)
      return jnp.sum(x * x)

    w = jax.ShapeDtypeStruct((512, 512), jnp.float32)
    x = jax.ShapeDtypeStruct((64, 512), jnp.float32)
    live = jax.jit(jax.grad(loss)).lower([w, w], x).compile().peak_live()
    self.assertIsNotNone(live)
    rows = (2, 0)  # x, after the two weights among the call's arguments.
    for buffer in live:
      self.assertLen(buffer.dims, len(buffer.shape))
      for size, dimension in zip(buffer.shape, buffer.dims):
        self.assertIsNotNone(dimension, buffer)
        self.assertEqual(rows in dimension, size == 64, buffer)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
