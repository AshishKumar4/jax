# Dew's patch stack on JAX

This fork carries the fixes [Dew](https://github.com/AshishKumar4/dew) needs from JAX before a release
has them, one branch a release: `vX.Y.Z-cache-key-pool-fingerprints`, the release `jax-vX.Y.Z` with
the fixes on top, versioned `X.Y.Z.postN`. Each fix is also sent upstream, and is dropped from the stack
once a release holds it.

| Fix | Issue | Pull request |
|---|---|---|
| A Pallas Triton kernel's cache key leaves out its source locations | [#41502](https://github.com/jax-ml/jax/issues/41502) | [#41777](https://github.com/jax-ml/jax/pull/41777) |

Each stack's wheel is published to Dew's package index, beside the release's own:

```
pip install dewml --extra-index-url https://pypi.dewml.dev/simple
uv pip install dewml --extra-index-url https://pypi.dewml.dev/simple
```

`.github/workflows/patch-stack.yml` looks for a new JAX release every six hours, rebases the newest
stack onto it with `dew/patch_stack.py`, runs the stack's tests, pushes the branch and
publishes its wheel; a conflict, a red test, or a release that holds every fix opens an issue here.
This branch, `dew`, holds only that; `main` follows upstream.

## How long Pallas Triton runs

JAX deprecates the Pallas Triton backend: since jax 0.11.2 (jax-ml/jax@adccdaa8f), every Pallas call lowered
to Triton warns `jax-pallas-triton`, "deprecated and will be removed in a future JAX version", pointing to
the Mosaic GPU backend. No release has removed it yet: jax and jaxlib 0.11.2, 0.12.0 and 0.12.0.1 all run it,
and upstream `main` at jax-ml/jax@1e3fa0800 still holds both its lowering (`jax/_src/pallas/triton`) and
jaxlib's Triton bindings (`jaxlib/triton`, `jaxlib/gpu_triton.py`). So the newest release that runs Pallas
Triton is, for now, the newest release: 0.12.0.1. A rebase onto a release that drops either one updates this
line with that release, the last before it, and the commit that removed it.
