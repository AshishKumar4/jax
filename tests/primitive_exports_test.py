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

"""The primitives an analysis of a jaxpr dispatches on are exported."""


from absl.testing import absltest

import jax
import jax.extend as jex
from jax._src import core
from jax._src import test_util as jtu
from jax._src.lax import lax as lax_internal

jax.config.parse_flags_with_absl()

# The parameters a marked primitive binds with that its operands do not imply.
class PrimitiveExportsTest(jtu.JaxTestCase):

  def test_the_primitives_an_analysis_dispatches_on_are_exported(self):
    from jax._src import callback, custom_batching, debugging, pjit, shard_map
    from jax._src.lax import parallel
    from jax._src.lax.control_flow import conditionals
    from jax._src.pallas import primitives as pallas_primitives
    from jax._src.random import prng
    import jax.experimental.pallas as pl

    exported = jex.core.primitives
    for name, primitive in {
        "custom_vmap_p": custom_batching.custom_vmap_p, "shard_map_p": shard_map.shard_map_p,
        "random_wrap_p": prng.random_wrap_p, "random_unwrap_p": prng.random_unwrap_p,
        "ragged_dot_general_p": lax_internal.ragged_dot_general_p,
        "pure_callback_p": callback.pure_callback_p, "io_callback_p": callback.io_callback_p,
        "debug_callback_p": debugging.debug_callback_p,
        "platform_index_p": conditionals.platform_index_p, "reshard_p": pjit.reshard_p,
        "psum_invariant_p": parallel.psum_invariant_p, "reduce_scatter_p": parallel.reduce_scatter_p,
        "all_gather_invariant_p": parallel.all_gather_invariant_p, "pbroadcast_p": parallel.pbroadcast_p,
        "pvary_p": core.pvary_p}.items():
      with self.subTest(name):
        self.assertIs(getattr(exported, name), primitive)
    self.assertIs(pl.program_id_p, pallas_primitives.program_id_p)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
