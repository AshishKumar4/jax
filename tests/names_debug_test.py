from absl.testing import absltest
import jax
from jax import ad_checkpoint
import jax.numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class NamesDebugTest(jtu.JaxTestCase):

  def test_debug(self):
    def step(x, w):
      hidden = ad_checkpoint.checkpoint_name(jnp.tanh(x @ w), "hidden")
      return jnp.sum(jnp.sin(hidden @ w))

    saved = jax.checkpoint(
        step, policy=jax.checkpoint_policies.save_only_these_names("hidden"))
    tile = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    traced = jax.jit(jax.grad(saved, argnums=1)).trace(tile, tile)
    print("JAXPR_HAS_NAME", "name" in str(traced.jaxpr))
    print(str(traced.jaxpr)[:3000])
    lowered = traced.lower()
    text = lowered.as_text()
    print("STABLEHLO_HAS", "checkpoint_name" in text)
    for line in text.splitlines():
      if "checkpoint_name" in line or "tanh" in line:
        print("SHLO", line.strip()[:300])
    compiled = lowered.compile()
    for line in compiled.as_text().splitlines():
      if "checkpoint_name" in line or "tanh" in line:
        print("HLO", line.strip()[:300])
    for b in compiled.peak_live():
      print("LIVE", b)
    self.fail("debug")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
