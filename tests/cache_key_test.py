# Copyright 2023 The JAX Authors.
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

import hashlib
import os
import re
import sys
import io
import types
import unittest
from typing import cast as type_cast

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pl_triton
from jax._src import cache_key
from jax._src import compiler
from jax._src import config
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.interpreters import mlir
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec as P
from jax._src.sharding_impls import NamedSharding
from jax._src.custom_partitioning import custom_partitioning


config.parse_flags_with_absl()


class CacheKeyTest(jtu.JaxTestCase):

  def test_serialized_compile_options(self):
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    hash1 = self.get_hashed_value(
        cache_key._hash_serialized_compile_options, compile_options
    )
    debug_options = compile_options.executable_build_options.debug_options
    debug_options.xla_force_host_platform_device_count = 2
    debug_options.xla_dump_to = "foo"
    debug_options.xla_dump_hlo_module_re = "bar"
    debug_options.xla_dump_hlo_pass_re = "baz"
    debug_options.xla_dump_hlo_as_text = True
    debug_options.xla_dump_hlo_as_proto = True
    debug_options.xla_dump_hlo_as_dot = True
    debug_options.xla_dump_hlo_as_url = True
    debug_options.xla_dump_hlo_as_html = True
    debug_options.xla_dump_fusion_visualization = True
    debug_options.xla_dump_hlo_snapshots = True
    debug_options.xla_dump_max_hlo_modules = True
    debug_options.xla_dump_module_metadata = True
    debug_options.xla_dump_compress_protos = True
    debug_options.xla_dump_hlo_as_long_text = True
    debug_options.xla_dump_disable_metadata = True
    debug_options.xla_dump_hlo_pipeline_re = "xyzzy"
    debug_options.xla_gpu_experimental_autotune_cache_mode = 2
    hash2 = self.get_hashed_value(
        cache_key._hash_serialized_compile_options, compile_options
    )
    self.assertEqual(hash1, hash2)

  @jtu.skip_on_devices("cpu")
  def test_hash_accelerator_devices(self):
    devices = np.array([[jax.local_devices()[0]]])

    dev_hash1 = self.get_hashed_value(cache_key._hash_devices, devices)
    dev_hash2 = self.get_hashed_value(cache_key._hash_devices, devices)
    self.assertEqual(dev_hash1, dev_hash2)

    acc_hash1 = self.get_hashed_value(
        cache_key._hash_accelerator_config, devices)
    acc_hash2 = self.get_hashed_value(
        cache_key._hash_accelerator_config, devices)
    self.assertEqual(acc_hash1, acc_hash2)

  def test_hash_platform(self):
    hash1 = self.get_hashed_value(
        cache_key._hash_platform, xla_bridge.get_backend()
    )
    hash2 = self.get_hashed_value(
        cache_key._hash_platform, xla_bridge.get_backend()
    )
    self.assertEqual(hash1, hash2)
    if xla_bridge.get_backend().platform != "cpu":
      cpu_backend = xla_bridge.get_backend("cpu")
      hash3 = self.get_hashed_value(cache_key._hash_platform, cpu_backend)
      self.assertNotEqual(hash1, hash3)

  @jtu.thread_unsafe_test()  # patches cache_key.host_cpu for the process
  def test_a_cpu_program_s_key_names_its_host_s_cpu(self):
    # XLA:CPU compiles for the features of the host that compiles, so a CPU
    # executable from a host of another CPU is not this host's to read.
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    cpu = xla_bridge.get_backend("cpu")
    devices = np.array([[cpu.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    original_host_cpu = cache_key.host_cpu
    keys = []
    try:
      for described in ("vendor_id: AuthenticAMD\ncpu family: 25\nmodel: 17\n"
                        "flags: avx2 avx512f",
                        "vendor_id: AuthenticAMD\ncpu family: 25\nmodel: 1\n"
                        "flags: avx2"):
        cache_key.host_cpu = lambda described=described: described
        keys.append(cache_key.get(computation, devices, compile_options, cpu))
    finally:
      cache_key.host_cpu = original_host_cpu
    self.assertNotEqual(keys[0], keys[1])
    if os.path.exists("/proc/cpuinfo"):
      self.assertRegex(cache_key.host_cpu(), "flags: |Features: ")

  @jtu.thread_unsafe_test()  # patches cache_key.jax_version for the process
  def test_a_program_s_key_names_its_jax_build(self):
    # A patch stack's releases over one jaxlib are builds of their own: one's
    # cache entries are not another's to read.
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    original_jax_version = cache_key.jax_version
    keys = []
    try:
      for build in ("0.12.0.1.post13", "0.12.0.1.post14"):
        cache_key.jax_version = lambda build=build: build
        keys.append(cache_key.get(computation, devices, compile_options, backend))
    finally:
      cache_key.jax_version = original_jax_version
    self.assertNotEqual(keys[0], keys[1])
    self.assertEqual(cache_key.jax_version(), jax.__version__)

  def test_an_accelerator_program_s_key_leaves_the_host_cpu_out(self):
    for platform_name in ("gpu", "tpu"):
      with_cpu, without = hashlib.sha256(), hashlib.sha256()
      cache_key._hash_host_cpu(
          with_cpu, types.SimpleNamespace(platform=platform_name))
      self.assertEqual(with_cpu.digest(), without.digest())

  def test_hash_string(self):
    hash1 = self.get_hashed_value(cache_key._hash_string, "foo")
    hash2 = self.get_hashed_value(cache_key._hash_string, "bar")
    hash3 = self.get_hashed_value(cache_key._hash_string, "bar")
    self.assertEqual(hash2, hash3)
    self.assertNotEqual(hash1, hash2)

  def test_same_key(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    self.assertEqual(
        cache_key.get(computation, devices, compile_options, backend),
        cache_key.get(computation, devices, compile_options, backend),
    )

  def test_different_key(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options_not_filled = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    compile_options_filled = self.filled_compile_options()
    backend = xla_bridge.get_backend()
    self.assertNotEqual(
        cache_key.get(
            computation, devices, compile_options_not_filled, backend
        ),
        cache_key.get(computation, devices, compile_options_filled, backend),
    )

  @jtu.thread_unsafe_test()
  def test_custom_hook(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    original_custom_hook = cache_key.custom_hook
    cache_key.custom_hook = lambda: "hook1"
    key1 = cache_key.get(computation, devices, compile_options, backend)
    cache_key.custom_hook = lambda: "hook2"
    key2 = cache_key.get(computation, devices, compile_options, backend)
    cache_key.custom_hook = original_custom_hook
    self.assertNotEqual(key1, key2)

  def test_different_computations(self):
    computation1 = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    computation2 = jax.jit(lambda x, y: x * y).lower(2, 2).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    self.assertNotEqual(
        cache_key.get(computation1, devices, compile_options, backend),
        cache_key.get(computation2, devices, compile_options, backend),
    )

  # TODO(phawkins): this test flakes if test concurrency is enabled.
  @jtu.thread_unsafe_test()
  @jtu.ignore_warning(category=DeprecationWarning,
                      message='`with mesh:` context manager')
  def test_custom_partitioning_ptr_removal(self):
    def _partition(mesh, arg_shapes, result_shape):
      arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
      result_shardings = NamedSharding(mesh, arg_shapes[0].sharding.spec)
      return mesh, jax.numpy.add, result_shardings, arg_shardings

    def _infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      return NamedSharding(mesh, arg_shapes[0].sharding.spec)

    @custom_partitioning
    def _cp_add(x, y):
      return jax.numpy.add(x, y)

    _cp_add.def_partition(
      infer_sharding_from_operands=_infer_sharding_from_operands,
      partition=_partition,
      sharding_rule='..., ... -> ...')

    devices = np.asarray(jax.devices())
    with Mesh(devices, ('x',)) as m:
      computation = jax.jit(
        _cp_add,
        in_shardings=(NamedSharding(m, P('x')),
                      NamedSharding(m, P('x'))),
                      out_shardings=NamedSharding(m, P('x'))
      ).lower(
        jax.ShapeDtypeStruct([1024], dtype=jax.numpy.float32),
        jax.ShapeDtypeStruct([1024], dtype=jax.numpy.float32),
      ).compiler_ir()
      pattern = (
          r'stablehlo\.custom_call @CustomSPMDPartitioning\('
          r'(.*?)\) \{'
          r'(.*?backend_config\s*=\s*"([^"]*)".*?)'
          r'\}'
      )
      with computation.context:
        updated_module = cache_key._remove_custom_partitioning_callbacks(
            type_cast(ir.Module, computation.operation.clone()),
        )
        bcs = [
            match[2]
            for match in re.findall(pattern, str(updated_module), re.DOTALL)
        ]
        for bc in bcs:
          self.assertEqual(bc, "REMOVED")

      compile_options = compiler.get_compile_options(
          num_replicas=1, num_partitions=1
      )
      backend = xla_bridge.get_backend()
      hash_without_callback_ptrs = cache_key.get(
          computation,
          devices,
          compile_options,
          backend,
          ignore_custom_partitioning=True,
      )
      expected_hash = cache_key.get(
          updated_module, devices, compile_options, backend
      )
      self.assertEqual(expected_hash, hash_without_callback_ptrs)

  def test_different_device_assignment(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options_1 = compiler.get_compile_options(
        num_replicas=1, num_partitions=1, device_assignment=np.array([[0]])
    )
    compile_options_2 = compiler.get_compile_options(
        num_replicas=1, num_partitions=1, device_assignment=np.array([[1]])
    )
    backend = xla_bridge.get_backend()
    hash_1 = cache_key.get(computation, devices, compile_options_1, backend)
    hash_2 = cache_key.get(computation, devices, compile_options_2, backend)
    if backend.platform == "gpu":
      self.assertEqual(hash_1, hash_2)
    else:
      self.assertNotEqual(hash_1, hash_2)

  @parameterized.parameters([False, True])
  @jtu.thread_unsafe_test()  # env vars are not thread-safe
  def test_identical_computations_different_metadata(self, include_metadata):
    f = lambda x, y: lax.mul(lax.add(x, y), 2)
    g = lambda x, y: lax.mul(lax.add(x, y), 2)
    assert id(f) != id(g)
    computation1 = jax.jit(f).lower(1, 1).compiler_ir()
    computation2 = jax.jit(g).lower(2, 3).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    with config.compilation_cache_include_metadata_in_key(include_metadata):
      key1 = cache_key.get(computation1, devices, compile_options, backend)
      key2 = cache_key.get(computation2, devices, compile_options, backend)
    self.assertEqual(include_metadata, key1 != key2)

  @parameterized.parameters([False, True])
  def test_identical_pallas_kernels_different_metadata(self, include_metadata):
    if jtu.test_device_matches(["tpu"]):
      compiler_params = None
    elif jtu.test_device_matches(["cuda"]):
      compiler_params = pl_triton.CompilerParams()
    else:
      self.skipTest("Pallas lowers its kernels for TPU and CUDA")

    # Same kernel body at different source lines; same __name__ to avoid kernel_name diff
    def make_caller_a():
      def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0
      return jax.jit(lambda x: pl.pallas_call(
          kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          compiler_params=compiler_params)(x))

    def make_caller_b():
      def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0
      return jax.jit(lambda x: pl.pallas_call(
          kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          compiler_params=compiler_params)(x))

    x = np.ones((128,), dtype=np.float32)
    computation1 = make_caller_a().lower(x).compiler_ir()
    computation2 = make_caller_b().lower(x).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    with config.compilation_cache_include_metadata_in_key(include_metadata):
      key1 = cache_key.get(computation1, devices, compile_options, backend)
      key2 = cache_key.get(computation2, devices, compile_options, backend)
    self.assertEqual(include_metadata, key1 != key2)

  @parameterized.parameters(("mosaic_gpu_v2", "module", "kernel_hash"),
                            ("__gpu$xla.gpu.triton", "ir", None))
  def test_identical_gpu_kernels_from_two_checkouts_key_alike(
      self, target, field, digest):
    # A Mosaic GPU or Pallas Triton kernel's module is serialized into its
    # custom call with its source locations, and a Mosaic GPU kernel's
    # kernel_hash is that serialization's digest: the same kernel compiled
    # from two checkouts must key alike.
    def program(path):
      with mlir.make_ir_context() as ctx, ir.Location.unknown():
        kernel = ir.Module.parse(
            "module {\n"
            "  func.func @kernel() {\n"
            f'    return loc("{path}/src/kernels.py":3:5)\n'
            "  }\n"
            "}\n")
        out = io.BytesIO()
        kernel.operation.write_bytecode(out, desired_version=0)
        body = out.getvalue()
        module = ir.Module.parse(
            "module {\n"
            "  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {\n"
            f"    %0 = stablehlo.custom_call @{target}(%arg0) "
            "{backend_config = \"\"} : (tensor<f32>) -> tensor<f32>\n"
            "    return %0 : tensor<f32>\n"
            "  }\n"
            "}\n")
        call = module.body.operations[0].regions[0].blocks[0].operations[0]
        config = {field: ir.StringAttr.get(body)}
        if digest is not None:
          config[digest] = ir.StringAttr.get(hashlib.sha256(body).digest())
        call.attributes["mhlo.backend_config"] = ir.DictAttr.get(config)
        return cache_key._canonicalize_ir(module, False)

    self.assertEqual(program("/home/ubuntu/s/trees/caea5dc4fa46-1"),
                     program("/home/ubuntu/s/trees/b7bd6a6dbee7-1"))

  @jtu.thread_unsafe_test()  # env vars are not thread-safe
  def test_xla_flags(self):
    if jtu.is_device_tpu(version=4):
      raise unittest.SkipTest("TODO(b/240151176)")

    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()

    orig_xla_flags = os.getenv("XLA_FLAGS")
    orig_argv = sys.argv
    try:
      os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
      key1 = cache_key.get(computation, devices, compile_options, backend)
      os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=1"
      key2 = cache_key.get(computation, devices, compile_options, backend)
      self.assertNotEqual(key1, key2)

      os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
      key3 = cache_key.get(computation, devices, compile_options, backend)
      self.assertEqual(key1, key3)

      # Test flag in _xla_flags_to_exclude_from_cache_key
      os.environ["XLA_FLAGS"] = (
          "--xla_gpu_autotune_level=0 --xla_force_host_platform_device_count=8"
      )
      key4 = cache_key.get(computation, devices, compile_options, backend)
      self.assertEqual(key1, key4)

      # Test flags given on command line
      del os.environ["XLA_FLAGS"]
      sys.argv.append("--xla_gpu_autotune_level=0")
      key5 = cache_key.get(computation, devices, compile_options, backend)
      self.assertEqual(key1, key5)
      sys.argv.append("--xla_force_host_platform_device_count=8")
      self.assertEqual(key1, key5)

    finally:
      if orig_xla_flags is not None:
        os.environ["XLA_FLAGS"] = orig_xla_flags
      elif os.getenv("XLA_FLAGS") is not None:
        del os.environ["XLA_FLAGS"]
      sys.argv = orig_argv

  @jtu.thread_unsafe_test()  # env vars are not thread-safe
  def test_libtpu_init_args(self):
    if jtu.is_device_tpu(version=4):
      raise unittest.SkipTest("TODO(b/240151176)")

    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()

    orig_libtpu_init_args = os.getenv("LIBTPU_INIT_ARGS")
    orig_argv = sys.argv
    try:
      os.environ["LIBTPU_INIT_ARGS"] = (
          "--xla_spmd_threshold_for_windowed_einsum_mib=0"
      )
      key1 = cache_key.get(computation, devices, compile_options, backend)
      os.environ["LIBTPU_INIT_ARGS"] = (
          "--xla_spmd_threshold_for_windowed_einsum_mib=1"
      )
      key2 = cache_key.get(computation, devices, compile_options, backend)
      self.assertNotEqual(key1, key2)

    finally:
      if orig_libtpu_init_args is not None:
        os.environ["LIBTPU_INIT_ARGS"] = orig_libtpu_init_args
      elif os.getenv("LIBTPU_INIT_ARGS") is not None:
        del os.environ["LIBTPU_INIT_ARGS"]
      sys.argv = orig_argv

  def filled_compile_options(self):
    compile_options = xla_client.CompileOptions()
    compile_options.num_replicas = 1
    compile_options.num_partitions = 1
    shape = xla_client.Shape.array_shape(np.dtype(np.float32), [2])
    shape_array = [shape, shape]
    compile_options.argument_layouts = shape_array
    compile_options.executable_build_options.result_layout = shape

    device_assignment = xla_client.DeviceAssignment.create(
        np.arange(4).reshape(2, 2)
    )
    compile_options.device_assignment = device_assignment
    compile_options.executable_build_options.device_assignment = (
        device_assignment
    )
    compile_options.executable_build_options.fdo_profile = b"test_profile"
    return compile_options

  def get_hashed_value(
      self, hash_function, hash_function_input1, hash_function_input2=None):
    hash_obj = hashlib.sha256()
    if hash_function_input2 is not None:
      hash_function(hash_obj, hash_function_input1, hash_function_input2)
    else:
      hash_function(hash_obj, hash_function_input1)
    return hash_obj.digest().hex()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
