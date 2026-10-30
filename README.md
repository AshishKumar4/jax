# Dew's patch stack on JAX

This fork carries the fixes [Dew](https://github.com/AshishKumar4/dew) needs from JAX before a release
has them, one branch a release: `vX.Y.Z-cache-key-pool-fingerprints`, the release `jax-vX.Y.Z` with
the fixes on top, versioned `X.Y.Z.postN`. Each fix is also sent upstream, and is dropped from the stack
once a release holds it.

Each family of fixes the stack carries, at 0.12.0.1.post36 (`gs/post36`, b0824fe52), with what upstream holds
(jax-ml/jax `main` at b2fbcb706, 2026-10-29) and when it leaves the stack. Dew's ledger row, where one names it,
in brackets.

| Family (commits) | Upstream | Leaves the stack when |
|---|---|---|
| Persistent cache key: a CPU program keyed by the host's CPU (5a2d20258) | no | a release keys CPU executables by host CPU, or Dew stops sharing a cache across CPU classes |
| Persistent cache key: jax's own build named (29a36bd84) | no | Dew stops publishing `X.postN` builds over one jaxlib `X`, or upstream keys jax's version |
| Persistent cache key: a Pallas Triton or Mosaic GPU kernel's debug info stripped, through one helper `_strip_kernel_debug_info` (5a58685ac, 6259bb46c, merged in dfb9611a7, c63aed3a1) [1691; [#41502](https://github.com/jax-ml/jax/issues/41502), [#41777](https://github.com/jax-ml/jax/pull/41777)] | TPU kernels only | a release strips GPU kernels' debug info in `_canonicalize_ir` |
| A platform-dependent conditional keeping one branch lowers it inline (31cd9c409, 950416715) [1732] | 31cd9c409 is jax-ml/jax@f9cf4a609 | 31cd9c409 at the next rebase; 950416715 once upstream's lowering passes `test_platform_dependent_kept_branch_threads_ordered_effects` |
| x64: a wide dtype a trace holds is kept under transforms (6bb44f4f7, e3e2a5cfd) [1688] | no | a release keeps it in `dot_general`, `conv_general_dilated` and `convert_element_type` outside `enable_x64` |
| Transfer guard on a CPU array's host read (1fbd2dd81) | no (jaxlib unchecked) | jaxlib applies the device-to-host guard to zero-copy CPU reads |
| Pallas Triton: 16-bit divide and `erf_inv` through float32 (d074c9a85) | no | upstream rounds f16 and bf16 division once |
| Pallas Triton: `custom_vjp_call` lowered as its primal (35b362f2e) | no | upstream registers `custom_vjp_call_p` for Triton |
| Mosaic GPU: `ceil`, `floor`, `add_any`, `integer_pow` of any integer exponent (3b54f2bd7, aec02c1e3, c1c68b1b9) | no | upstream lowers them |
| Mosaic GPU layout inference solves unrelated unknowns apart (9231c658f) | no | upstream does, or Dew's sm80 grouped GEMM stops exhausting its fuel |
| Mosaic GPU: `AbstractDevice.compute_capability`; a compile-only target's shared memory limit (82211f353, 9771c862c) | no | upstream holds both, or Dew stops compiling for a device it is not on |
| Mosaic GPU MMA: warp split per op (`m_warps`), accumulator-to-LHS relayout (14158a67e, 08ce533c2, b15744c1d) | no | upstream holds both, or Dew's Ampere GEMM and sm80 P@V stop using them |
| Mosaic GPU: each warpgroup's own reduction scratch (5878c8ccf) | no | upstream's reductions do |
| Pallas Mosaic GPU: a unit row or column of a tiled array (876c57aa3 and its fixes) [1428] | no | upstream resolves `constraints.py:304`'s TODO |
| Mosaic GPU WGMMA: shared-operand descriptors formed at their instructions (5226f132d) | no | a release forms them at use, or Dew's D128 attention backward shows no loss without it |
| Mosaic GPU TMA: load coordinates ptxas cannot fold (4de45c521) [2254]; im2col loads build theirs the same way, unfixed, as Dew makes none | no; jax-ml/jax@ec456bfe3 renames the condition the fix sits under (`untransformed_gmem_ref is src_ref`), and the fix applies there unchanged | the ptxas Dew ships no longer miscompiles a literal-0 TMA coordinate (2254's repro, patch removed) |
| Mosaic GPU CP_ASYNC: a squeezed dimension between sliced ones under a tile transform (4f7868935) [2078] | no | upstream squeezes before the tile transform |
| Pallas Mosaic GPU `emit_pipeline`: ring slots carried (84547ea19) | no | a release carries them |
| Pallas Mosaic GPU interpreter: a thread map's constants and inputs moved once (2e522efea) [1850] | no | upstream's `thread_map.py` does |
| Pallas Mosaic GPU interpreter: its own callbacks take the host's NumPy arrays, no `device_put` a call (6c25cdff3); the public `io_callback` unchanged | no | upstream's interpreter skips the placement |
| Persistent cache: an entry written aside and renamed into place, so a reader beside its writer sees it whole or not at all (7c6ce00e8) | no | upstream's `LRUCache.put` writes atomically |
| `JAX_EXPLAIN_CACHE_MISSES` reads a trace keyed by keyword, as Pallas Triton's reductions trace their bodies (360a29dc6) [2302] | no | upstream's explainer reads keyword keys |

Fixes the stack carried and dropped, net zero at post34: the multi-process topology fingerprint exchange
[1731, 1689] (dropped in f30bea7b8), the `linear_call` patches [1690] (d302b6449), Splash's
`delta_from_weights` [1692] (2dc0045ee), `emit_pipeline`'s thread axis [1748] (42fd2d85c) and the
interpreter's per-device barrier check [1923] (85a47e9e9).

The tests `patch-stack.yml` runs before it publishes are the stack's own (`dew/patch_stack.py tests`: every
`tests/*_test.py` the stack changes, and `tests/api_test.py`) on a GitHub runner, which has no GPU: there the
Mosaic GPU families' tests skip. Their gate is Dew's: a pin moves only on a whole-suite verdict and the GPU
checks `docs/installation.md` names under "Bumping JAX", and Dew's daily GPU tier runs every test a GPU can take
on an A100 and an H100.

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
