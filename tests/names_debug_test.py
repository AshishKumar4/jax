from absl.testing import absltest
import jax
from jax import ad_checkpoint
import jax.numpy as jnp
from jax._src import ad_checkpoint as internal
from jax._src.interpreters import mlir
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class NamesDebugTest(jtu.JaxTestCase):

  def test_debug(self):
    print("FILE", internal.__file__)
    print("RULE", mlir._lowerings.get(internal.name_p))
    print("PLATFORM_RULES", {k: v.get(internal.name_p) for k, v in mlir._platform_specific_lowerings.items() if internal.name_p in v})
    f = jax.jit(lambda x: ad_checkpoint.checkpoint_name(jnp.tanh(x), "a") * 2)
    lowered = f.lower(jax.ShapeDtypeStruct((8,), jnp.float32))
    print("SIMPLE", lowered.as_text())
    self.fail("debug")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
