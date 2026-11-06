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
# ==============================================================================
"""The ambient answers a Mosaic GPU lowering reads, held fixed for a cache.

A Mosaic GPU lowering asks the process what GPU it targets, how much shared
memory a block may use, and what the device topology and NVSHMEM look like. A
lowering cache keys those answers, so every read inside the lowering it caches
must give the answer it keyed. ``using`` sets them for the duration of the
lowering, and each reader (``utils._infer_arch``, ``core._max_smem_bytes``,
``core.is_single_process_multi_device_topology``, ``core.is_nvshmem_available``,
``core._is_known_multihost_mesh``, and ``core._SMEM_SIZE_BOUND``) consults
``current()`` before it reads the process.

While they are set the lowering is a cache's: it carries no source locations
(``current() is not None``).
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class LoweringInputs:
  # ``utils._infer_arch()``.
  arch: tuple[int, int]
  # ``core._max_smem_bytes()``.
  max_smem_bytes: int
  # ``core.is_single_process_multi_device_topology()``.
  single_process_multi_device_topology: bool
  # ``core.is_nvshmem_available()``.
  nvshmem_available: bool
  # The mesh ``core._is_known_multihost_mesh`` was asked about, and its answer.
  # A call about another mesh (or none) is not this mesh's answer.
  mesh: Any
  known_multihost_mesh: bool
  # ``core._SMEM_SIZE_BOUND``.
  smem_size_bound: int | None


_current: contextvars.ContextVar[LoweringInputs | None] = contextvars.ContextVar(
    "mosaic_gpu_lowering_inputs", default=None
)


def current() -> LoweringInputs | None:
  return _current.get()


@contextlib.contextmanager
def using(inputs: LoweringInputs):
  token = _current.set(inputs)
  try:
    yield
  finally:
    _current.reset(token)
