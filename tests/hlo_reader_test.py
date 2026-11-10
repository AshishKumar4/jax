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
"""jax.extend.xla.instructions reads a module's instructions through XLA's own schema."""

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax.extend import custom_call_targets
from jax.extend import xla as jex_xla
import jax.numpy as jnp


def _step(x):
  y = jax.lax.fori_loop(0, 3, lambda i, y: jnp.sin(y) * 2, x)
  return y, jax.lax.empty((4,), jnp.float32)


class HloReaderTest(jtu.JaxTestCase):

  def test_a_loop_calls_its_body_and_condition_and_a_custom_call_states_its_target(self):
    lowered = jax.jit(_step).trace(jnp.ones(8)).lower(lowering_platforms=("cuda",))
    found = list(jex_xla.instructions(lowered.compiler_ir("hlo").as_serialized_hlo_module_proto()))
    computations = {instruction.computation for instruction in found}
    (loop,) = [instruction for instruction in found if instruction.opcode == "while"]
    self.assertLen(loop.called, 2)
    self.assertLessEqual(set(loop.called), computations)
    self.assertIsNone(loop.custom_call_target)
    self.assertEqual([i.custom_call_target for i in found if i.opcode == "custom-call"],
                     [custom_call_targets.ALLOCATE_BUFFER])
    self.assertEqual(loop.proto.name, loop.name)
    # The loop is in the entry; its body and condition are not, and each has one root.
    self.assertTrue(loop.entry)
    for called in loop.called:
      inside = [i for i in found if i.computation == called]
      self.assertFalse(any(i.entry for i in inside))
      self.assertEqual(sum(i.root for i in inside), 1)
    # An instruction's operands are instructions of its own computation, the loop's its carried tuple.
    by_name = {(i.computation, i.name): i for i in found}
    for instruction in found:
      for operand in instruction.operands:
        self.assertIn((instruction.computation, operand), by_name)
    self.assertLen(loop.operands, 1)

  def test_a_compiled_module_reads_as_its_serialization_does(self):
    compiled = jax.jit(_step).lower(jnp.ones(8)).compile()
    (module,) = compiled.runtime_executable().hlo_modules()
    from_module = list(jex_xla.instructions(module))
    from_bytes = list(jex_xla.instructions(module.as_serialized_hlo_module_proto()))
    self.assertEqual([i[:8] for i in from_module], [i[:8] for i in from_bytes])
    text = module.to_string()
    for instruction in from_module:
      self.assertIn(instruction.name, text)
      for called in instruction.called:
        self.assertIn(called, text)
    # Its text, parsed back, reads as the module does (XLA's parser names what it printed).
    from_text = list(jex_xla.instructions(jex_xla.module_from_text(text)))
    self.assertEqual([(i.computation, i.name, i.opcode, i.called, i.root, i.entry) for i in from_text],
                     [(i.computation, i.name, i.opcode, i.called, i.root, i.entry) for i in from_module])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
