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
"""A compiled Mosaic GPU kernel's record read through its own schema (the CUDA plugin's
`mosaic.gpu.MosaicGpuKernelProto`, jax._src.mosaic_gpu_protos): what a serialized executable keeps of each
kernel (its binary, its function's name, the dynamic shared memory its launch asks for, its cluster, its
caller's metadata)."""

from __future__ import annotations

import functools
from typing import Any


@functools.cache
def _record_class() -> Any:
  from google.protobuf import descriptor_pb2
  from google.protobuf import descriptor_pool
  from google.protobuf import message_factory
  from jax._src import mosaic_gpu_protos

  pool = descriptor_pool.DescriptorPool()
  for file in descriptor_pb2.FileDescriptorSet.FromString(
      mosaic_gpu_protos.FILE_DESCRIPTOR_SET
  ).file:
    pool.Add(file)
  return message_factory.GetMessageClass(
      pool.FindMessageTypeByName("mosaic.gpu.MosaicGpuKernelProto")
  )


def kernel_record(serialized: bytes) -> Any | None:
  """`serialized` read as a Mosaic GPU kernel's record (a `MosaicGpuKernelProto`), None where it is no
  such record: it does not parse as one, or holds a field the schema does not name, or names no kernel
  (no binary, no function)."""
  from google.protobuf.message import DecodeError
  from google.protobuf.unknown_fields import UnknownFieldSet

  record = _record_class()()
  try:
    record.ParseFromString(serialized)
  except DecodeError:
    return None
  if len(UnknownFieldSet(record)) or not record.gpu_binary or not record.kernel_name:
    return None
  return record
