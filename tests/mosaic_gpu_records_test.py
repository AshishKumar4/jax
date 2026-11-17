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

import importlib.util
import os

from absl.testing import absltest
from google.protobuf import descriptor_pb2

from jax._src import mosaic_gpu_protos
from jax._src import test_util as jtu
from jax.extend import mosaic_gpu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _generator():
  spec = importlib.util.spec_from_file_location(
      "generate_xla_protos",
      os.path.join(ROOT, "build", "generate_xla_protos.py"),
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class MosaicGpuRecordsTest(jtu.JaxTestCase):

  def test_the_shipped_schema_describes_the_plugins_proto(self):
    # The descriptor jax ships is protoc's description of the .proto the CUDA
    # plugin is built from, so the reader and the writer state one schema.
    generator = _generator()
    shipped = descriptor_pb2.FileDescriptorSet.FromString(
        mosaic_gpu_protos.FILE_DESCRIPTOR_SET
    )
    described = descriptor_pb2.FileDescriptorSet.FromString(
        generator.described(ROOT, [generator.MOSAIC_GPU_PROTO])
    )
    self.assertEqual(shipped, described)

  def test_a_kernel_record_reads_every_field_the_writer_sets(self):
    # A record as the plugin writes one, every field the reader names set.
    from jax._src.mosaic_gpu_records import _record_class

    record = _record_class()(
        version=4, gpu_binary=b"\x7fELF", kernel_name="kernel",
        smem_bytes=1024, cluster_size=2, caller_metadata=b"launch",
    )
    read = mosaic_gpu.kernel_record(record.SerializeToString())
    self.assertEqual(
        (read.version, read.gpu_binary, read.kernel_name, read.smem_bytes,
         read.cluster_size, read.caller_metadata),
        (4, b"\x7fELF", "kernel", 1024, 2, b"launch"),
    )

  def test_bytes_that_are_no_kernel_record_read_as_none(self):
    # Not a proto; a proto with a field the schema does not name; a record
    # naming no kernel.
    for serialized in (b"\xff\xff\xff", b"\xa8\x06\x01", b"\x08\x04"):
      self.assertIsNone(mosaic_gpu.kernel_record(serialized))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
