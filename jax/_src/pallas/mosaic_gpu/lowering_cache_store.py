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

"""The persistent store of lowerings: a versioned envelope in an ``LRUCache``.

An entry holds a lowering's module (as bytecode, without its dump basename) and
the plain data of the rest of its result, under the key it was made for. Its
envelope is a magic line, a version and the size of the header, a JSON header and
the module. The header holds the result's fields and the sha256 of the whole
payload (the fields as canonical JSON, a newline and the module), so an entry
altered anywhere, valid bytecode included, is not the one that was stored; and an
entry that is whole but is not of the schema is not one either. Either is removed
to be replaced.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import struct
import threading
from typing import Any

from jax._src import lru_cache
import numpy as np

logger = logging.getLogger(__name__)

# The directory under the cache directory flag that holds the entries.
_SUBDIRECTORY = "mosaic_gpu_lowering"

_ENVELOPE_MAGIC = b"jax-mosaic-gpu-lowering-cache\n"
_ENVELOPE_VERSION = 2
_ENVELOPE_PREFIX = struct.Struct("<II")


class InvalidEntry(ValueError):
  """Bytes that are not the current entry of a key."""


@dataclasses.dataclass(frozen=True)
class Entry:
  module: bytes
  grid: tuple[int, ...]
  block: tuple[int, ...]
  # Each a list of [shape, physical dtype name].
  new_out_shapes: list[Any]
  gmem_scratch_shapes: list[Any]
  seconds: float


def _payload(fields: dict[str, Any], module: bytes) -> bytes:
  canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
  return canonical.encode() + b"\n" + module


def pack(key: str, entry: Entry) -> bytes:
  fields = {
      "key": key, "grid": list(entry.grid), "block": list(entry.block),
      "new_out_shapes": entry.new_out_shapes,
      "gmem_scratch_shapes": entry.gmem_scratch_shapes,
      "seconds": entry.seconds, "module_size": len(entry.module),
  }
  header = {
      **fields,
      "sha256": hashlib.sha256(_payload(fields, entry.module)).hexdigest(),
  }
  blob = json.dumps(header, sort_keys=True).encode()
  return (
      _ENVELOPE_MAGIC + _ENVELOPE_PREFIX.pack(_ENVELOPE_VERSION, len(blob))
      + blob + entry.module
  )


def _dims(value: Any, *, length: int | None = None, minimum: int) -> list[int]:
  if (
      not isinstance(value, list)
      or (length is not None and len(value) != length)
      or not all(type(d) is int and d >= minimum for d in value)
  ):
    raise InvalidEntry(f"{value!r} is not a list of integers of the schema")
  return value


def _shapes(value: Any) -> list[Any]:
  if not isinstance(value, list):
    raise InvalidEntry(f"{value!r} is not a list of shapes")
  for item in value:
    if not isinstance(item, list) or len(item) != 2:
      raise InvalidEntry(f"{item!r} is not a shape and a dtype")
    shape, dtype = item
    _dims(shape, minimum=0)
    try:
      if not isinstance(dtype, str):
        raise TypeError(dtype)
      np.dtype(dtype)
    except TypeError as e:
      raise InvalidEntry(f"{dtype!r} is not a dtype") from e
  return value


def unpack(key: str, data: bytes) -> Entry:
  """The entry `data` holds for `key`.

  Raises InvalidEntry (a ValueError), or a KeyError for a field it lacks, for
  bytes that are not the current entry of `key`: of another envelope version,
  another key, altered or damaged in any way, or not of the schema.
  """
  prefix_end = len(_ENVELOPE_MAGIC) + _ENVELOPE_PREFIX.size
  if len(data) < prefix_end or not data.startswith(_ENVELOPE_MAGIC):
    raise InvalidEntry("not an entry")
  version, header_size = _ENVELOPE_PREFIX.unpack(
      data[len(_ENVELOPE_MAGIC):prefix_end]
  )
  if version != _ENVELOPE_VERSION:
    raise InvalidEntry(f"an entry of the envelope version {version}")
  header = json.loads(data[prefix_end:prefix_end + header_size])
  module = data[prefix_end + header_size:]
  if not isinstance(header, dict):
    raise InvalidEntry("a header that is not a JSON object")
  fields = {k: v for k, v in header.items() if k != "sha256"}
  digest = header["sha256"]
  if (
      not isinstance(digest, str)
      or hashlib.sha256(_payload(fields, module)).hexdigest() != digest
  ):
    raise InvalidEntry("an entry that is not what was stored (its checksum)")
  if fields["key"] != key:
    raise InvalidEntry("an entry of another key")
  if fields["module_size"] != len(module):
    raise InvalidEntry("a module of another size")
  seconds = fields["seconds"]
  if (
      isinstance(seconds, bool) or not isinstance(seconds, (int, float))
      or not math.isfinite(seconds) or seconds < 0
  ):
    raise InvalidEntry(f"{seconds!r} is not a duration")
  return Entry(
      module=module,
      grid=tuple(_dims(fields["grid"], length=3, minimum=1)),
      block=tuple(_dims(fields["block"], length=3, minimum=1)),
      new_out_shapes=_shapes(fields["new_out_shapes"]),
      gmem_scratch_shapes=_shapes(fields["gmem_scratch_shapes"]),
      seconds=float(seconds),
  )


class Store:
  """The entries under one cache directory."""

  def __init__(self, directory: str):
    self._cache = lru_cache.LRUCache(
        os.path.join(directory, _SUBDIRECTORY), max_size=-1
    )

  def get(self, key: str) -> Entry | None:
    """The entry stored for `key`; one that is not usable is removed, and is none."""
    try:
      data = self._cache.get(key)
    except OSError as e:
      logger.warning("the Mosaic GPU lowering cache cannot be read: %s", e)
      return None
    if data is None:
      return None
    try:
      return unpack(key, data)
    except (ValueError, KeyError) as e:
      logger.warning("a Mosaic GPU lowering cache entry is not usable: %r", e)
      self.discard(key)
      return None

  def put(self, key: str, entry: Entry) -> None:
    self._cache.put(key, pack(key, entry))

  def discard(self, key: str) -> None:
    """Removes an entry that cannot be used: a put never replaces one."""
    try:
      self._cache.delete(key)
    except OSError as e:
      logger.warning("the Mosaic GPU lowering cache cannot remove an entry: %s", e)


_stores: dict[str, Store] = {}
_unusable: set[str] = set()
_lock = threading.Lock()


def open_store(directory: str) -> Store | None:
  """The store under `directory`; none (said once) if it cannot be made there."""
  with _lock:
    store = _stores.get(directory)
    if store is None:
      try:
        store = _stores[directory] = Store(directory)
      except OSError as e:
        if directory not in _unusable:
          _unusable.add(directory)
          logger.warning(
              "the Mosaic GPU lowering cache cannot use %s: %s", directory, e)
        return None
    return store
