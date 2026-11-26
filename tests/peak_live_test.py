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

import collections
from typing import Any

from absl.testing import absltest
import jax
from jax import ad_checkpoint
from jax._src import hlo_reader
from jax._src import peak_live
from jax._src import test_util as jtu
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
import jax.experimental.topologies as topologies
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

jax.config.parse_flags_with_absl()


def _step(x, w):
  return jnp.tanh(x @ w) @ w


class PeakLiveTest(jtu.JaxTestCase):

  @jtu.run_on_devices("cpu", "cuda")
  def test_a_compiles_peak_holds_its_parameters_and_the_temporaries_it_reads_at_once(self):
    # tanh(x @ w) @ w: at the second product's peak, x's and w's buffers and
    # the first product's tanh, one temporary of a 256 x 256 f32 tile, its rows
    # x's and its columns w's (whichever fusion computes it: XLA:GPU's GEMM
    # fusion takes the tanh into the product's epilogue).
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
        and buffer.dims == (frozenset({(0, 0)}), frozenset({(1, 1)}))
        for buffer in temporaries), live)
    self.assertLessEqual(sum(buffer.bytes for buffer in temporaries),
                         compiled.memory_analysis().temp_size_in_bytes)

  @jtu.run_on_devices("cuda")
  def test_a_compiles_intervals_hold_its_peak_buffers_at_each_allocations_peak(
      self):
    # At the first position a trace's charged intervals hold their most bytes,
    # those covering it are the buffers peak_live names of it, as XLA's own
    # count of its peak reads the same simulation (XLA:GPU's, which keeps its
    # traces; XLA:CPU's assignment states none).
    def step(x, w):
      for _ in range(3):
        x = jnp.tanh(x @ w)
      return x.sum(), x

    tile = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    compiled = jax.jit(jax.grad(step, has_aux=True)).lower(tile, tile).compile()
    intervals, live = compiled.live_intervals(), compiled.peak_live()
    self.assertTrue(intervals)
    peaked = []
    for trace in {each.trace for each in intervals}:
      within = [each for each in intervals if each.trace == trace]
      at = max(range(max(each.end for each in within)),
               key=lambda at, within=within: _counted(within, at))
      peaked += [each for each in within
                 if each.charged and each.start <= at < each.end]
    self.assertEqual(_temporaries(peaked), _temporaries(live))

  def test_an_interval_shares_memory_only_with_a_buffer_still_live(self):
    # A heap simulation's events: an output in place over a live operand
    # shares its memory; one placed where a freed buffer was is new memory
    # again (XLA's FREE then SHARE_WITH); a heap's pages each hold their own
    # buffers of one trace; an allocation simulated per computation has a
    # trace each, positions from 0 in each; a share of a share, freed or not,
    # lies in the first's memory.
    hlo = """
HloModule m, entry_computation_layout={(f32[4]{0})->f32[4]{0}}

ENTRY e {
  p = f32[4]{0} parameter(0)
  a = f32[4]{0} negate(p), frontend_attributes={checkpoint_name="first"}
  b = f32[4]{0} exponential(a)
  c = f32[4]{0} sine(b)
  d = f32[4]{0} cosine(c)
  e = f32[4]{0} tanh(d)
  ROOT r = f32[4]{0} add(e, c)
}
"""
    module = hlo_reader.module_from_text(hlo).as_serialized_hlo_module_proto()
    [entry] = hlo_reader.message_class("xla.HloModuleProto").FromString(
        module).computations
    defined = {instruction.name: instruction.id
               for instruction in entry.instructions}
    assignment = hlo_reader.message_class("xla.BufferAssignmentProto")()
    ids = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    for name, buffer in ids.items():
      logical = assignment.logical_buffers.add(id=buffer, size=16)
      logical.defined_at.instruction_id = defined[name]
    for index, names in enumerate(("ab", "cde")):
      allocation = assignment.buffer_allocations.add(index=index, size=32)
      for name in names:
        allocation.assigned.add(logical_buffer_id=ids[name], size=16)
    alloc, free, share = 0, 1, 2
    events = [(alloc, "a", None), (alloc, "c", None), (free, "a", None),
              (share, "b", "a"), (share, "d", "c"), (free, "c", None),
              (free, "d", None), (share, "e", "d"), (free, "e", None),
              (free, "b", None)]
    for allocation, steps in ((0, events), (1, events),
                              (0, [(alloc, "a", None), (free, "a", None)])):
      trace = assignment.heap_simulator_traces.add(
          buffer_allocation_index=allocation,
          whole_module_simulation=steps is events)
      for kind, name, shared in steps:
        trace.events.add(kind=kind, buffer_id=ids[name],
                         share_with_canonical_id=ids[shared] if shared else 0)
    found = peak_live.intervals(assignment.SerializeToString(), module)
    self.assertEqual(
        sorted((each.instruction, each.allocation, each.trace, each.start,
                each.end, each.storage) for each in found),
        [("a", 0, 0, 0, 2, 1), ("a", 0, 2, 0, 1, 1), ("b", 0, 0, 3, 9, 1),
         ("c", 1, 1, 1, 5, 3), ("d", 1, 1, 4, 6, 3), ("e", 1, 1, 7, 8, 3)])
    self.assertEqual({each.names for each in found if each.instruction == "a"},
                     {("first",)})
    self.assertEqual({(each.trace, each.whole) for each in found},
                     {(0, True), (1, True), (2, False)})
    page = [each for each in found if each.trace == 0]
    # b in a's freed memory holds it anew; d in c's live memory holds none.
    self.assertEqual([_held(page, at) for at in range(10)],
                     [16, 16, 0, 16, 16, 16, 16, 16, 16, 0])
    other = [each for each in found if each.trace == 1]
    self.assertEqual([_held(other, at) for at in range(10)],
                     [0, 16, 16, 16, 16, 16, 0, 16, 0, 0])
    # XLA's count charges d nothing, so once c is freed it counts none of the
    # memory d holds; e, sharing d with no other instance of d live, it
    # charges.
    self.assertEqual({each.instruction for each in found if not each.charged},
                     {"d"})
    self.assertEqual([_counted(other, at) for at in range(10)],
                     [0, 16, 16, 16, 16, 0, 0, 16, 0, 0])

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
  def test_a_scans_stacked_residual_is_named_by_what_its_body_writes(self):
    # A scan saves each step's named residual into a stacked output, a loop
    # state element whose buffer is the zeros a broadcast makes (Dew row
    # 13304): that buffer, live at the gradient's peak, is named by the
    # residual the body writes into it.
    def step(xs, w):
      def body(carry, x):
        hidden = ad_checkpoint.checkpoint_name(jnp.tanh(x @ w), "hidden")
        return carry + jnp.sum(jnp.sin(hidden)), None
      saved = jax.checkpoint(
          body, policy=jax.checkpoint_policies.save_only_these_names("hidden"))
      return jax.lax.scan(saved, 0.0, xs)[0]

    rows = jax.ShapeDtypeStruct((16, 512, 256), jnp.float32)
    weight = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    compiled = jax.jit(jax.grad(step, argnums=1)).lower(rows, weight).compile()
    live = compiled.peak_live()
    self.assertIsNotNone(live)
    stacked = [buffer for buffer in live if buffer.shape == (16, 512, 256)]
    self.assertTrue(stacked, live)
    self.assertIn(("hidden",), {buffer.names for buffer in stacked}, stacked)

  def test_a_loops_state_is_named_only_by_a_scans_whole_stack(self):
    # A scan's stack, which the loop's steps fill a row each, forwards or
    # backwards, is named by the row it writes; a populated matrix whose one
    # row each step overwrites is not, nor a filled one written at one row,
    # nor a square turned by a bitcast to another layout and a copy back
    # before each write, nor one written at the counter passed through any op
    # (a bitcast to a signed type of its width may take it negative), nor a
    # stack whose loop runs fewer steps than its rows.
    self.assertEqual(_loop_names(4), {"zeros": ("hidden",),
                                      "ones": ("late",), "populated": (),
                                      "filled": (), "spread": (),
                                      "blank": ()})
    self.assertEqual(_loop_names(2), {"zeros": (), "ones": (),
                                      "populated": (), "filled": (),
                                      "spread": (), "blank": ()})

  @jtu.run_on_devices("cpu", "cuda")
  def test_a_residual_named_through_reshapes_names_the_buffer_they_bitcast(self):
    # The saved residual is named after two reshapes, which XLA may fold into a
    # bitcast of the tanh's buffer, the reshapes that carried the name gone (as
    # it did a GPU gemm's converted output, Dew row 12898): the tanh's own op is
    # named through them, so the buffer live at the gradient's peak is named,
    # once, however XLA lays the reshapes out.
    def step(x, w):
      hidden = jnp.tanh(x @ w).reshape(2048, 16, 16).reshape(2048 * 16, 16)
      hidden = ad_checkpoint.checkpoint_name(hidden, "hidden")
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

  def test_a_name_reaches_through_reshapes_to_their_producer_and_no_further(self):
    # Each reshape on the way and the op making their operand carry the name,
    # once however often it is given; an argument reshaped has no producer, so
    # its reshape alone does; an op the producer reads is never named; nor is a
    # producer of several results, whose unnamed sibling would read it.
    def names(f, *args):
      found = []

      def walk(operation):
        for region in operation.regions:
          for block in region.blocks:
            for inner in block.operations:
              op = inner.operation
              if "mhlo.frontend_attributes" in op.attributes:
                attributes = ir.DictAttr(op.attributes["mhlo.frontend_attributes"])
                if "checkpoint_name" in attributes:
                  found.append((op.name, ir.StringAttr(attributes["checkpoint_name"]).value))
              walk(op)

      walk(jax.jit(f).lower(*args).compiler_ir("stablehlo").operation)
      return found

    x = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    hidden = lambda x: ad_checkpoint.checkpoint_name(ad_checkpoint.checkpoint_name(
        jnp.tanh(jnp.exp(x)).reshape(2, 64).reshape(128), "hidden"), "hidden")
    self.assertEqual(names(hidden, x), [("stablehlo.tanh", "hidden"),
                                        ("stablehlo.reshape", "hidden"),
                                        ("stablehlo.reshape", "hidden")])
    flat = lambda x: ad_checkpoint.checkpoint_name(x.reshape(128), "flat")
    self.assertEqual(names(flat, x), [("stablehlo.reshape", "flat")])

    def sorted_keys(x):
      keys, values = jax.lax.sort((x, jnp.exp(x)), num_keys=1)
      return ad_checkpoint.checkpoint_name(keys.reshape(128), "keys"), values

    self.assertEqual(names(sorted_keys, x), [("stablehlo.reshape", "keys")])

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

_LOOP = """
HloModule m, entry_computation_layout={(f32[], f32[4,2]{1,0})->f32[4,2]{1,0}}

forwards {
  stack = f32[4,2]{1,0} parameter(0)
  row = f32[2]{0} parameter(1)
  laid = f32[1,2]{1,0} bitcast(row)
  at = s32[] parameter(2)
  zero = s32[] constant(0)
  ROOT put = f32[4,2]{1,0} dynamic-update-slice(stack, laid, at, zero)
}

moved {
  stack = f32[4,4]{1,0} parameter(0)
  other = f32[4,4]{0,1} bitcast(stack)
  back = f32[4,4]{1,0} copy(other)
  row = f32[1,4]{1,0} parameter(1)
  at = s32[] parameter(2)
  zero = s32[] constant(0)
  ROOT put = f32[4,4]{1,0} dynamic-update-slice(back, row, at, zero)
}

backwards {
  stack = f32[4,2]{1,0} parameter(0)
  row = f32[1,2]{1,0} parameter(1)
  i = s32[] parameter(2)
  last = s32[] constant(3)
  at = s32[] subtract(last, i)
  zero = s32[] constant(0)
  ROOT put = f32[4,2]{1,0} dynamic-update-slice(stack, row, at, zero)
}

body {
  state = (f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,4]{1,0}, f32[4,2]{1,0}, s32[]) parameter(0)
  stack = f32[4,2]{1,0} get-tuple-element(state), index=0
  back = f32[4,2]{1,0} get-tuple-element(state), index=1
  matrix = f32[4,2]{1,0} get-tuple-element(state), index=2
  fixed = f32[4,2]{1,0} get-tuple-element(state), index=3
  square = f32[4,4]{1,0} get-tuple-element(state), index=4
  cast = f32[4,2]{1,0} get-tuple-element(state), index=5
  i = s32[] get-tuple-element(state), index=6
  hidden = f32[2]{0} constant({1, 2}), frontend_attributes={checkpoint_name="hidden"}
  written = f32[4,2]{1,0} fusion(stack, hidden, i), kind=kLoop, calls=forwards
  late = f32[1,2]{1,0} constant({{3, 4}}), frontend_attributes={checkpoint_name="late"}
  reversed = f32[4,2]{1,0} fusion(back, late, i), kind=kLoop, calls=backwards
  row = f32[1,2]{1,0} constant({{5, 6}}), frontend_attributes={checkpoint_name="row"}
  zero = s32[] constant(0)
  over = f32[4,2]{1,0} dynamic-update-slice(matrix, row, zero, zero)
  pinned = f32[4,2]{1,0} dynamic-update-slice(fixed, row, zero, zero)
  line = f32[1,4]{1,0} constant({{7, 8, 9, 10}}), frontend_attributes={checkpoint_name="line"}
  turned = f32[4,4]{1,0} fusion(square, line, i), kind=kLoop, calls=moved
  counted = s32[] bitcast(i)
  through = f32[4,2]{1,0} dynamic-update-slice(cast, row, counted, zero)
  one = s32[] constant(1)
  next = s32[] add(i, one)
  ROOT out = (f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,4]{1,0}, f32[4,2]{1,0}, s32[]) tuple(written, reversed, over, pinned, turned, through, next)
}

condition {
  state = (f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,4]{1,0}, f32[4,2]{1,0}, s32[]) parameter(0)
  i = s32[] get-tuple-element(state), index=6
  bound = s32[] constant(STEPS)
  ROOT less = pred[] compare(i, bound), direction=LT
}

ENTRY e {
  a = f32[] parameter(0)
  populated = f32[4,2]{1,0} parameter(1)
  zeros = f32[4,2]{1,0} broadcast(a), dimensions={}
  unit = f32[] constant(1)
  ones = f32[4,2]{1,0} broadcast(unit), dimensions={}
  filled = f32[4,2]{1,0} broadcast(a), dimensions={}
  spread = f32[4,4]{1,0} broadcast(a), dimensions={}
  blank = f32[4,2]{1,0} broadcast(a), dimensions={}
  start = s32[] constant(0)
  init = (f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,4]{1,0}, f32[4,2]{1,0}, s32[]) tuple(zeros, ones, populated, filled, spread, blank, start)
  loop = (f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,2]{1,0}, f32[4,4]{1,0}, f32[4,2]{1,0}, s32[]) while(init), condition=condition, body=body, backend_config={"known_trip_count":{"n":"STEPS"},"known_init_step":{"init":"0","step":"1"},"known_induction_variable":{"tuple_index":"6"}}
  ROOT x = f32[4,2]{1,0} get-tuple-element(loop), index=0
}
"""


def _loop_names(steps: int) -> dict[str, tuple[str, ...]]:
  """The names `peak_live.live` gives each loop state element's buffer of
  ``_LOOP`` run for ``steps`` steps, by the instruction defining it, each
  buffer at the loop's output too and all live at one peak."""
  module = hlo_reader.module_from_text(
      _LOOP.replace("STEPS", str(steps))).as_serialized_hlo_module_proto()
  defined = {instruction.name: instruction
             for computation in hlo_reader.message_class(
                 "xla.HloModuleProto").FromString(module).computations
             for instruction in computation.instructions
             if computation.name == "e"}
  assignment = hlo_reader.message_class("xla.BufferAssignmentProto")()
  assignment.buffer_allocations.add(index=0, size=128)
  peak = assignment.peak_buffers.add(index=0)
  for element, name in enumerate(
      ("zeros", "ones", "populated", "filled", "spread", "blank")):
    buffer = assignment.logical_buffers.add(id=element, size=32)
    buffer.defined_at.instruction_id = defined[name].id
    alias = assignment.buffer_aliases.add(source_buffer_id=element)
    alias.location.instruction_id = defined["loop"].id
    alias.location.shape_index.append(element)
    peak.logical_buffer_ids.append(element)
  live = peak_live.live(assignment.SerializeToString(), module)
  return {buffer.instruction: buffer.names for buffer in live}


def _held(intervals, at: int) -> int:
  """The bytes ``intervals``, of one trace, hold at position ``at``: each
  storage some interval covering it lies in, once, at its members' most."""
  storages: dict[int, int] = {}
  for each in intervals:
    if each.start <= at < each.end:
      storages[each.storage] = max(storages.get(each.storage, 0), each.bytes)
  return sum(storages.values())


def _counted(intervals, at: int) -> int:
  """The bytes XLA's count of a peak charges ``intervals``, of one trace, at
  position ``at``."""
  return sum(each.bytes for each in intervals
             if each.charged and each.start <= at < each.end)


def _temporaries(buffers) -> collections.Counter:
  """``buffers``' temporaries by their instruction and size."""
  return collections.Counter((buffer.instruction, buffer.bytes)
                             for buffer in buffers
                             if buffer.kind == "temporary")


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

  def test_each_elementwise_opcode_keeps_its_operands_dimensions(self):
    # By HLO's own opcode names (a softmax's "exponential", not "exp"): a name
    # the reader does not know leaves every value computed from it unknown.
    unary = ("exponential", "exponential-minus-one", "log-plus-one", "cosine",
             "sine", "tanh", "logistic", "rsqrt", "sqrt", "negate")
    lines = "\n".join(f"        unary{i} = f32[4,8] {opcode}(x)"
                      for i, opcode in enumerate(unary))
    descended = _descended(f"""
      HloModule m
      ENTRY e {{
        x = f32[4,8] parameter(0)
{lines}
        held = (f32[4,8]) tuple(unary0)
        barrier = (f32[4,8]) opt-barrier(held)
        b = f32[4,8] get-tuple-element(barrier), index=0
        ROOT out = f32[4,8] add(b, unary{len(unary) - 1})
      }}""")
    for i in range(len(unary)):
      self.assertEqual(descended[f"unary{i}"], (_of((0, 0)), _of((0, 1))), unary[i])
    self.assertEqual(descended["out"], (_of((0, 0)), _of((0, 1))))

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
