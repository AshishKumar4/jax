"""Temporary runtime instrumentation for red/green causal verification."""
import inspect


def _replace(module, name, old, new):
  function = getattr(module, name)
  source = inspect.getsource(function)
  source = source[source.index("def "):]
  assert old in source, (name, old)
  rewritten = source.replace(old, new)
  scope = {}
  exec(compile(rewritten, function.__code__.co_filename, "exec"),
       function.__globals__, scope)
  function.__code__ = scope[name].__code__


def pytest_sessionstart(session):
  del session
  from jax import lax
  from jax._src import shard_map
  from jax._src.interpreters import batching
  from jax._src.lax import lax as lax_impl
  from jax._src.pallas.mosaic_gpu import lowering, lowering_cache_seal
  from jax._src.pallas.triton import lowering as triton_lowering

  _replace(batching, "batch_subtrace_2",
           "return out_vals, list(out_dims)",
           "return out_vals, _BatchDims(out_dims, ["
           "getattr(x, 'staged_custom_jvp', None) for x in outs])")
  _replace(shard_map, "_shard_map_batch",
           "return out_vals.map2(out_dims, make_tracer)",
           "return out_vals.map3(out_dims, "
           "batching._staged_custom_jvps(out_dims), "
           "lambda v, d, name: make_tracer(v, d, staged_custom_jvp=name))")
  _replace(lax_impl, "_complex_dot_by_parts",
           "return dot_general(x, y,", "return _dot_general_bound(x, y,")
  _replace(triton_lowering, "_unit_slice",
           "  if (\n      len(sliced) != 1",
           "  if not sliced and (strides is None or all(s == 1 for s in strides)):\n"
           "    return x\n"
           "  if (\n      len(sliced) != 1")

  def bounded_accuracy(accuracy):
    return lax.Tolerance(atol=accuracy.atol,
                         rtol=max(accuracy.rtol, accuracy.atol),
                         ulps=accuracy.ulps)

  lowering._audit_bounded_accuracy = bounded_accuracy
  for name in ("_tanh_lowering_rule", "_logistic"):
    _replace(lowering, name, "_meets(accuracy,",
             "_meets(_audit_bounded_accuracy(accuracy),")
  _replace(lowering_cache_seal, "_sources",
           '        with open(path, "rb") as f:\n'
           '          digest = hashlib.sha256(f.read()).digest()',
           '        try:\n'
           '          with open(path, "rb") as f:\n'
           '            digest = hashlib.sha256(f.read()).digest()\n'
           '        except OSError as e:\n'
           '          return {}, {}, f"source package {package!r} cannot be read: {e}"')
