# Dew's patch stack on JAX

This fork carries the fixes [Dew](https://github.com/AshishKumar4/dew) needs from JAX before a release
has them, one branch a release: `vX.Y.Z-cache-key-pool-fingerprints`, the release `jax-vX.Y.Z` with
the fixes on top, versioned `X.Y.Z.postN`. Each fix is also sent upstream, and is dropped from the stack
once a release holds it.

| Fix | Issue | Pull request |
|---|---|---|
| A computation that spans processes keys the persistent compilation cache by every process's topology fingerprint, and only across processes with the same accelerators | [#40940](https://github.com/jax-ml/jax/issues/40940) | [#41776](https://github.com/jax-ml/jax/pull/41776) |
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
