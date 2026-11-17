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
"""The custom-call targets JAX's own lowerings write, each named once.

A lowering writes its target from here and a reader (the compilation cache's debug-info stripping, export's
stability list, an analysis of a compiled program) compares against the same name, so the two cannot drift.
Light by design: the core reads these without importing the kernels' backends.
"""

from typing import Final

MOSAIC_GPU: Final = "mosaic_gpu_v2"
"""A Mosaic GPU kernel (jax.experimental.mosaic.gpu's `as_gpu_kernel`, Pallas's Mosaic GPU backend)."""

TRITON: Final = "__gpu$xla.gpu.triton"
"""A Pallas Triton kernel (jax._src.pallas.triton)."""

TPU_CUSTOM_CALL: Final = "tpu_custom_call"
"""A Pallas TPU (Mosaic TPU) kernel (jax._src.tpu_custom_call)."""

ALLOCATE_BUFFER: Final = "AllocateBuffer"
"""An uninitialized buffer (`lax.empty`)."""

THREEFRY2X32: Final = {"cuda": "cu_threefry2x32_ffi", "rocm": "hip_threefry2x32_ffi",
                       "oneapi": "oneapi_threefry2x32_ffi"}
"""The threefry2x32 hash's FFI kernel by GPU platform (jax._src.random.threefry2x32)."""
