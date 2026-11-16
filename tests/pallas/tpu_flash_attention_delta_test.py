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

"""The TPU flash attention backward's D, each query's sum of p * dp, run in TPU interpret mode."""

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu import flash_attention
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

_BLOCKS = flash_attention.BlockSizes(
    block_q=128, block_k_major=128, block_k=128, block_b=1,
    block_q_major_dkv=128, block_k_major_dkv=128, block_k_dkv=128,
    block_q_dkv=128, block_k_major_dq=128, block_k_dq=128, block_q_dq=128)


class FlashAttentionDeltaTest(jtu.JaxTestCase):

  def test_a_documents_first_query_has_no_query_gradient(self):
    # Three documents packed into 384 positions, causal: each document's first
    # query sees one key, its weight exactly 1, so its dq is exactly zero.
    # D formed as rowsum(o * do) left o's rounding in dp - D (about 1e-6).
    lengths = (100, 200, 84)
    ids = np.repeat(np.arange(1, 4, dtype=np.int32), lengths)[None]
    starts = np.cumsum((0,) + lengths[:-1])
    segment_ids = flash_attention.SegmentIds(q=jnp.asarray(ids), kv=jnp.asarray(ids))
    q, k, v, do = (
        jax.random.normal(key, (1, 2, ids.shape[1], 128), jnp.float32)
        for key in jax.random.split(jax.random.key(0), 4))

    def attention(q, k, v):
      return flash_attention.flash_attention(
          q, k, v, segment_ids=segment_ids, causal=True, sm_scale=0.125,
          block_sizes=_BLOCKS)

    with pltpu.force_tpu_interpret_mode():
      dq, dk, dv = jax.vjp(attention, q, k, v)[1](do)

    def reference(q, k, v):
      return flash_attention.mha_reference_no_custom_vjp(
          q, k, v, segment_ids=segment_ids, causal=True, sm_scale=0.125)

    with jax.enable_x64(True):
      wide = [x.astype(jnp.float64) for x in (q, k, v, do)]
      expected = jax.vjp(reference, *wide[:3])[1](wide[3])

    np.testing.assert_array_equal(np.asarray(dq)[:, :, starts], 0.0)
    for got, want in zip((dq, dk, dv), expected):
      want = np.asarray(want)
      np.testing.assert_allclose(
          np.asarray(got, np.float64), want, rtol=0,
          atol=1e-5 * np.abs(want).max())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
