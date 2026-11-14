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
"""Tests of the Mosaic GPU lowering cache (``jax_pallas_lowering_cache_dir``).

They run on the CPU, lowering a Mosaic GPU kernel for a compile-only GPU (a
mesh whose abstract device states its compute capability). Each lowering runs in
a fresh process against a temporary cache directory, as the cache is meant to be
used: by one process after another. See `mosaic_gpu_lowering_cache_support` for
the process (`child`), the kernels it lowers (`kernels`) and the sealed site the
cases that need a sealed installation run in (`harness`). The canonicalizer's
own cases are in `mosaic_gpu_lowering_cache_encoder_test`.
"""

import os
import shutil

import sys

from absl.testing import absltest

# The repository's pytest runs with --import-mode=importlib, which leaves this
# directory (and the support package in it) off sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mosaic_gpu_lowering_cache_support import harness  # pylint: disable=g-import-not-at-top,g-bad-import-order  # noqa: E402


class LoweringCacheTest(harness.LoweringCacheTestCase):

  # The kernel the cases lower: a function of `kernels`.
  kernel = "plain"

  def spawn_child(self, **kwargs):
    kwargs.setdefault("kernel", self.kernel)
    return super().spawn_child(**kwargs)

  # -- a miss, then a hit

  def test_a_second_process_hits_what_the_first_missed(self):
    first = self.run_child(report_seal=True)
    second = self.lower()
    first_run, = first["runs"]
    self.assertEqual(first_run["events"], ["miss"], first)
    self.assertEqual(second["events"], ["hit"])
    self.assertLen(second["seconds_saved"], 1)
    self.assertGreater(second["seconds_saved"][0], 0)
    self.assertEqual(first_run["normalized"], second["normalized"])
    self.assertEqual(first_run["text_sha"], second["text_sha"])
    self.assertTrue(first_run["normalized"]["profiler_spec_is_none"])
    # Hit and miss alike carry no source locations.
    self.assertFalse(first_run["has_file_locations"])
    self.assertFalse(second["has_file_locations"])
    seal = first["seal"]
    self.assertTrue(seal["sealed"], seal["reason"])
    print(
        f"\nsealing the installation: {seal['files']} files, "
        f"{seal['bytes'] / 1e6:.0f} MB hashed in {seal['seconds']:.3f} s")

  def test_the_flag_is_the_environment_variable(self):
    flag = {"jax_pallas_lowering_cache_dir": str(self.cache_dir)}
    first = self.lower(cache=False, config=flag)
    second = self.lower()
    self.assertEqual(first["events"], ["miss"], first)
    self.assertEqual(second["events"], ["hit"], second)
    self.assertEqual(first["normalized"], second["normalized"])

  def test_a_verified_hit_is_lowered_afresh_and_compared(self):
    first = self.lower()
    verified = self.lower(env={"JAX_PALLAS_LOWERING_CACHE_VERIFY": "1"})
    # The control is no part of the key, so it still hits; and it checked.
    self.assertCountEqual(verified["events"], ["verified", "hit"])
    self.assertEqual(first["normalized"], verified["normalized"])
    self.assertEqual(first["text_sha"], verified["text_sha"])

  def test_a_difference_a_verified_hit_finds_raises(self):
    # An entry that is whole (its checksum is of what it holds now) and holds
    # the module of another kernel: only a fresh lowering says it is not this one's.
    self.lower()
    (entry,) = self.entries()
    other = self.lower(scale=3.0)
    self.assertEqual(other["events"], ["miss"])
    (other_entry,) = (e for e in self.entries() if e != entry)
    _, _, other_module = harness.split_entry(other_entry.read_bytes())
    harness.rewrite_entry(entry, module=other_module)
    unverified = self.lower()
    self.assertEqual(unverified["events"], ["hit"])
    run = self.lower(env={"JAX_PALLAS_LOWERING_CACHE_VERIFY": "1"})
    self.assertEqual(run["error"], "LoweringCacheMismatch", run)
    self.assertIn("differs from a fresh one", run["message"])

  # -- a run that verifies must verify a hit

  def test_a_verifying_run_that_verified_no_hit_exits_with_an_error(self):
    verify = {"JAX_PALLAS_LOWERING_CACHE_VERIFY": "1"}
    # A cold cache: every lowering is a miss.
    cold = self.run_child(env=verify)
    self.assertEqual(cold["runs"][0]["events"], ["miss"], cold)
    self.assertNotEqual(cold["_returncode"], 0, cold)
    self.assertIn("no hit was verified", cold["_stderr"])
    # Everything bypassed: an installation no RECORD lists, and a lowering
    # that profiles.
    unsealed = self.run_child(env=verify, site="overlay")
    self.assertEqual(unsealed["runs"][0]["events"], ["unsealed"], unsealed)
    self.assertNotEqual(unsealed["_returncode"], 0, unsealed)
    profile = (self.tmp / "profile")
    profile.mkdir()
    uncacheable = self.run_child(
        env=verify, compiler_params=dict(profile_space=8, profile_dir=str(profile)))
    self.assertEqual(uncacheable["runs"][0]["events"], ["uncacheable"], uncacheable)
    self.assertNotEqual(uncacheable["_returncode"], 0, uncacheable)
    # The same kernel again, with the cache the first run filled: a hit, which
    # it verifies.
    warm = self.run_child(env=verify)
    self.assertCountEqual(warm["runs"][0]["events"], ["verified", "hit"], warm)
    self.assertEqual(warm["_returncode"], 0, warm)

  def test_a_run_that_does_not_verify_is_not_asked_to(self):
    # No flag, or the flag with no cache to verify: nothing to check.
    cold = self.run_child()
    self.assertEqual(cold["runs"][0]["events"], ["miss"], cold)
    self.assertEqual(cold["_returncode"], 0, cold)
    off = self.run_child(cache=False, env={"JAX_PALLAS_LOWERING_CACHE_VERIFY": "1"})
    self.assertEqual(off["runs"][0]["events"], [], off)
    self.assertEqual(off["_returncode"], 0, off)

  def test_without_the_flag_a_lowering_is_as_it_was(self):
    result = self.run_child(cache=False, direct=True, site="overlay")
    runs = result["direct"]
    self.assertIsNone(result["flag"])
    for mode in ("wrapper", "direct"):
      self.assertEqual(runs[mode]["events"], [], mode)
    self.assertEqual(runs["wrapper"]["normalized"], runs["direct"]["normalized"])
    self.assertEqual(runs["wrapper"]["text_sha"], runs["direct"]["text_sha"])
    # Its module keeps the source locations the cache leaves out.
    self.assertTrue(runs["wrapper"]["has_file_locations"])
    self.assertFalse(self.cache_dir.exists())

  # -- what makes a miss

  def test_the_callers_metadata_is_no_input_of_the_key_and_a_hit_carries_its_own(self):
    # A caller's statement of its launch (`CompilerParams.metadata`) rides the
    # call, not the module: kernels differing in it alone share one lowering,
    # each call carrying its own caller's.
    first = self.lower(compiler_params={"metadata": b"launch-a"})
    self.assertEqual(first["events"], ["miss"])
    second = self.lower(compiler_params={"metadata": b"launch-b"})
    self.assertEqual(second["events"], ["hit"], second)
    self.assertEqual((first["caller_metadata"], second["caller_metadata"]),
                     ("launch-a", "launch-b"))
    self.assertLen(self.entries(), 1)

  def test_each_input_of_the_key_is_one(self):
    base = self.lower()
    self.assertEqual(base["events"], ["miss"])
    changes = {
        "a param value": dict(power=3),
        "a literal value": dict(scale=3.0),
        "an aval's shape": dict(size=512),
        "a compiler param": dict(compiler_params={"approx_math": True}),
        "the stated GPU": dict(capability=harness.SM80),
        "a jax.config flag": dict(config={"jax_numpy_rank_promotion": "warn"}),
        "the shared memory bound": dict(smem_bound=200 * 1024),
        "an environment variable": dict(env={"MOSAIC_GPU_TEST_INPUT": "1"}),
    }
    for name, change in changes.items():
      with self.subTest(name):
        run = self.lower(**change)
        self.assertEqual(run["events"], ["miss"], run)
    # Each of them took its own entry, and the first is still there.
    self.assertLen(self.entries(), 1 + len(changes))
    again = self.lower()
    self.assertEqual(again["events"], ["hit"])
    self.assertEqual(again["normalized"], base["normalized"])

  def test_a_declared_source_package_names_its_code_by_its_tree(self):
    # A kernel holding an inline_mgpu body of a package of the process's own.
    package = self.tmp / "userpkg"
    package.mkdir()
    (package / "__init__.py").write_text("def body(ctx, ref):\n  del ctx, ref\n")
    (package / "other.py").write_text("VALUE = 1\n")
    path = os.pathsep.join([str(self._site), str(harness.TESTS_PALLAS), str(self.tmp)])
    kernel = dict(kernel="declared_source", kernel_args={"package": "userpkg"})
    declared = {"PYTHONPATH": path, "JAX_PALLAS_LOWERING_CACHE_SOURCES": "userpkg"}
    # Undeclared, the body has no identity: not cached.
    undeclared = self.lower(env={"PYTHONPATH": path}, **kernel)
    self.assertNotIn("error", undeclared, undeclared.get("message"))
    self.assertEqual(undeclared["events"], ["uncacheable"])
    # Declared, it is named with the package's tree: kept, and read again.
    self.assertEqual(self.lower(env=declared, **kernel)["events"], ["miss"])
    self.assertEqual(self.lower(env=declared, **kernel)["events"], ["hit"])
    # Any file of the package changed, another tree: lowered afresh.
    (package / "other.py").write_text("VALUE = 2\n")
    self.assertEqual(self.lower(env=declared, **kernel)["events"], ["miss"])
    # A kernel holding none of its code is read by declaring and undeclared processes alike.
    self.assertEqual(self.lower(env=declared)["events"], ["miss"])
    self.assertEqual(self.lower()["events"], ["hit"])
    # A declared package that no import finds unseals the process: the cache is bypassed.
    missing = dict(declared, JAX_PALLAS_LOWERING_CACHE_SOURCES="no_such_package")
    self.assertEqual(self.lower(env=missing)["events"], ["unsealed"])

  def test_what_no_lowering_reads_is_no_input_of_the_key(self):
    # A process's own settings: its compilation cache, its client's
    # allocations, XLA's dumps and a profiler's injection. Processes differing
    # in them alone share their lowerings.
    base = self.lower()
    self.assertEqual(base["events"], ["miss"])
    flags = os.environ.get("XLA_FLAGS", "")  # the children's own, kept
    unread = {
        "the client's share of memory": dict(env={
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}),
        "the compilation cache's directory": dict(env={
            "JAX_COMPILATION_CACHE_DIR": str(self.tmp / "elsewhere")}),
        "what the compilation cache keeps": dict(config={
            "jax_persistent_cache_min_compile_time_secs": 7.0}),
        "XLA's autotuning cache": dict(env={"XLA_FLAGS": (
            f"{flags} --xla_gpu_per_fusion_autotune_cache_dir=/tmp/picks")}),
        "XLA's dumps": dict(env={"XLA_FLAGS": (
            f"{flags} --xla_dump_to=/tmp/dumps --xla_dump_hlo_as_text"
            " --xla_dump_hlo_module_re=.*step.*")}),
        "a profiler's injection": dict(env={
            "CUDA_INJECTION64_PATH": "/opt/nsys/libToolsInjection64.so",
            "NVTX_INJECTION64_PATH": "/opt/nsys/libToolsInjection64.so"}),
    }
    for name, change in unread.items():
      with self.subTest(name):
        self.assertEqual(self.lower(**change)["events"], ["hit"])
    # Another XLA flag beside the cache's directory is still read.
    read = self.lower(env={"XLA_FLAGS": (
        f"{flags} --xla_gpu_per_fusion_autotune_cache_dir=/tmp/picks"
        " --xla_gpu_enable_command_buffer=")})
    self.assertEqual(read["events"], ["miss"])
    self.assertLen(self.entries(), 2)

  def test_what_the_process_answers_of_nvshmem_is_an_input_of_the_key(self):
    # Where the answer is all that differs: the environment of the two NVSHMEM
    # runs is one, and only a file's existence is another.
    self.assertEqual(self.lower()["events"], ["miss"])
    bc = self.tmp / "nvshmem.bc"
    absent = self.lower(nvshmem_bc=str(bc))
    bc.write_bytes(b"")
    present = self.lower(nvshmem_bc=str(bc))
    self.assertEqual(absent["events"], ["miss"], absent)
    self.assertEqual(present["events"], ["miss"], present)
    self.assertLen(self.entries(), 3)
    self.assertEqual(self.lower(nvshmem_bc=str(bc))["events"], ["hit"])

  def test_a_source_line_moved_is_still_a_hit(self):
    first = self.lower()
    moved = self.lower(pad=7)
    self.assertEqual(first["events"], ["miss"])
    self.assertEqual(moved["events"], ["hit"])
    self.assertEqual(first["normalized"], moved["normalized"])
    self.assertEqual(first["text_sha"], moved["text_sha"])
    self.assertFalse(moved["has_file_locations"])

  def test_a_hit_over_the_shared_memory_bound_raises_as_a_fresh_lowering_does(self):
    # The bound is part of the key, so no entry made without it answers a
    # lowering with it, and that one raises as it does with no cache.
    self.kernel = "smem"
    first = self.lower()
    self.assertEqual(first["events"], ["miss"])
    bound = dict(smem_bound=512)
    cached = self.lower(**bound)
    fresh = self.lower(cache=False, **bound)
    self.assertEqual(fresh["error"], "ValueError", fresh)
    self.assertIn("exceeds available shared memory", fresh["message"])
    self.assertEqual(cached["error"], fresh["error"])
    self.assertEqual(cached["message"], fresh["message"])
    self.assertEqual(cached["events"], [])
    self.assertLen(self.entries(), 1)
    self.assertEqual(self.lower()["events"], ["hit"])

  # -- what is not sealed

  def test_a_replaced_or_added_lowering_rule_is_unsealed(self):
    registries = {
        "mosaic_lowering_rules": "rule",
        "a rule registered later for a sealed primitive": "register",
        "dialect_lowering._lowerings": "dialect",
        "layout_inference._constraint_system_derivation_rules": "layout",
        "state.discharge._discharge_rules": "discharge",
    }
    for name, mutate in registries.items():
      with self.subTest(name):
        self.cache_dir = self.tmp / f"cache-{mutate}"
        clean, replaced = self.run_child(mutate=mutate)["runs"]
        self.assertEqual(clean["events"], ["miss"])
        self.assertEqual(replaced["events"], ["unsealed"], replaced)
        self.assertNotIn("error", replaced)

  def test_an_installation_that_differs_from_its_record_is_unsealed(self):
    site = self.tmp / "site"
    shutil.copytree(self._site, site)
    victim = site / "jax" / "_src" / "pallas" / "mosaic_gpu" / "lowering.py"
    stat = victim.stat()
    data = victim.read_bytes()
    edited = data.replace(b"Copyright 2024", b"Copyright 2025", 1)
    self.assertNotEqual(edited, data)
    self.assertEqual(len(edited), len(data))
    victim.write_bytes(edited)
    os.utime(victim, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = victim.stat()
    self.assertEqual((after.st_size, after.st_mtime_ns),
                     (stat.st_size, stat.st_mtime_ns))
    control = self.lower(site="sealed")
    self.assertEqual(control["events"], ["miss"])
    result = self.run_child(site=site, report_seal=True)
    run, = result["runs"]
    self.assertEqual(run["events"], ["unsealed"], result)
    self.assertFalse(result["seal"]["sealed"])
    self.assertIn("sha256", result["seal"]["reason"])
    self.assertIn("lowering.py", result["seal"]["reason"])

  def test_an_installation_with_no_record_is_unsealed(self):
    site = self.tmp / "site"
    harness.make_site(site, with_record=False)
    result = self.run_child(site=site, report_seal=True)
    run, = result["runs"]
    self.assertEqual(run["events"], ["unsealed"], result)
    self.assertIn("RECORD", result["seal"]["reason"])

  def test_jax_that_no_record_lists_is_unsealed(self):
    # The jax under test, imported from its source tree: no installed file.
    result = self.run_child(site="overlay", report_seal=True)
    run, = result["runs"]
    self.assertEqual(run["events"], ["unsealed"], result)
    self.assertFalse(result["seal"]["sealed"])

  # -- the code a rule is: its definition, not its name

  def assert_lowers_as_fresh(self, cached, fresh):
    """A lowering through the cache is the one without it: its module as text
    (the cache's modules carry no locations), or its error."""
    if "error" in fresh:
      self.assertEqual(
          (cached.get("error"), cached.get("message")),
          (fresh["error"], fresh["message"]))
    else:
      self.assertNotIn("error", cached, cached)
      self.assertEqual(
          cached["normalized"]["asm"], fresh["normalized"]["asm"])

  def test_a_rule_swapped_for_one_that_is_named_alike_is_a_miss(self):
    # The lane add rule and the lane subtract rule are lambdas of one file with
    # one name, `<lambda>`: a process that has the one in the other's place is
    # not the process that cached an add.
    self.kernel = "lane"
    plain = self.lower()
    self.assertEqual(plain["events"], ["miss"], plain)
    self.assertEqual(self.lower()["events"], ["hit"])
    swapped = self.lower(tweaks=["lane_add_sub"])
    fresh = self.lower(cache=False, tweaks=["lane_add_sub"])
    self.assertEqual(swapped["events"], ["miss"], swapped)
    self.assert_lowers_as_fresh(swapped, fresh)
    self.assertNotEqual(swapped["normalized"]["asm"], plain["normalized"]["asm"])
    # Each took its own entry, and each is a hit for its own process.
    self.assertLen(self.entries(), 2)
    self.assertEqual(self.lower(tweaks=["lane_add_sub"])["events"], ["hit"])
    self.assertEqual(self.lower()["events"], ["hit"])

  # -- the dispatch configuration is part of the key, whole

  def test_an_op_removed_from_the_warp_semantics_is_a_miss(self):
    self.kernel = "warp"
    plain = self.lower(shape=[2, 4, 64])
    self.assertEqual(plain["events"], ["miss"], plain)
    tweaks = ["warp_without_async_store"]
    cached = self.lower(shape=[2, 4, 64], tweaks=tweaks)
    fresh = self.lower(cache=False, shape=[2, 4, 64], tweaks=tweaks)
    self.assertEqual(fresh["error"], "NotImplementedError", fresh)
    self.assertIn("does not support warp semantics", fresh["message"])
    self.assertNotIn("hit", cached["events"], cached)
    self.assert_lowers_as_fresh(cached, fresh)

  def test_a_dialect_or_layout_rule_swapped_for_another_is_a_miss(self):
    for tweak in ("dialect_addf_subf", "layout_constant_broadcast"):
      with self.subTest(tweak):
        self.cache_dir = self.tmp / f"cache-{tweak}"
        plain = self.lower()
        self.assertEqual(plain["events"], ["miss"], plain)
        cached = self.lower(tweaks=[tweak])
        fresh = self.lower(cache=False, tweaks=[tweak])
        self.assertNotIn("hit", cached["events"], cached)
        self.assert_lowers_as_fresh(cached, fresh)

  # -- a rule that registers another while its kernel lowers

  def test_a_registry_that_changes_during_a_lowering_is_never_stored(self):
    # A rule of the sealed site's jax registers an entry in a registry while its
    # kernel is lowered (as a rule that imports a module does): the registries
    # are not what the key was made of, so the result is for this call only.
    self.kernel = "touch"
    control = self.lower()
    self.assertEqual(control["events"], ["miss"], control)
    self.assertLen(self.entries(), 1)
    for name in ("rules", "estimators", "call_primitives", "discharge", "dialect",
                 "warp", "layout"):
      with self.subTest(name):
        self.cache_dir = self.tmp / f"cache-{name}"
        run = self.lower(touch={"registry": name})
        self.assertNotIn("error", run, run)
        self.assertEqual(run["events"], ["unsealed"], run)
        self.assertEmpty(self.entries())

  # -- what can and what cannot be keyed

  def test_a_kernel_with_a_warp_map_is_cached(self):
    # Its axis name is an object no other process has: the same kernel all the
    # same.
    self.kernel = "warp"
    first = self.lower(shape=[2, 4, 64])
    second = self.lower(shape=[2, 4, 64])
    self.assertEqual(first["events"], ["miss"], first)
    self.assertEqual(second["events"], ["hit"], second)
    self.assertEqual(first["normalized"], second["normalized"])
    self.assertEqual(first["text_sha"], second["text_sha"])
    changed = self.lower(shape=[2, 4, 64], scale=3.0)
    self.assertEqual(changed["events"], ["miss"], changed)

  def test_a_param_class_outside_the_sealed_files_is_uncacheable(self):
    # A layout_cast to a frozen dataclass a file outside the sealed ones defines,
    # whose to_mgpu the lowering calls.
    self.kernel = "own_layout"
    for _ in range(2):
      run = self.lower(size=128, rows=64)
      self.assertEqual(run["events"], ["uncacheable"], run)
      self.assertNotIn("error", run)
    self.assertEmpty(self.entries())

  def test_a_param_class_of_an_installed_package_jax_does_not_require_is_uncacheable(self):
    # The same layout_cast to a class a user's package defines, installed as a
    # wheel installs it: its to_mgpu may read what no key holds, so it is no more
    # cacheable than one of a source file.
    self.kernel = "installed_layout"
    for _ in range(2):
      run = self.lower(size=128, rows=64)
      self.assertEqual(run["events"], ["uncacheable"], run)
      self.assertNotIn("error", run)
    self.assertEmpty(self.entries())

  def test_a_profiling_lowering_is_uncacheable(self):
    params = dict(profile_space=8, profile_dir=str(self.tmp / "profile"))
    (self.tmp / "profile").mkdir()
    for _ in range(2):
      run = self.lower(compiler_params=params)
      self.assertEqual(run["events"], ["uncacheable"], run)
    self.assertEmpty(self.entries())

  def test_a_lowering_dumping_its_passes_is_uncacheable(self):
    passes = self.tmp / "passes"
    passes.mkdir()
    for dumps in ({"MOSAIC_GPU_DUMP_MLIR_PASSES": "1"},
                  {"MOSAIC_GPU_DUMP_CONSTRAINT_SYSTEM": "1"},
                  {}):  # MOSAIC_GPU_DUMP_TO alone dumps every kind.
      with self.subTest(str(dumps)):
        run = self.lower(env={**dumps, "MOSAIC_GPU_DUMP_TO": str(passes)})
        self.assertEqual(run["events"], ["uncacheable"], run)
    self.assertEmpty(self.entries())

  def test_a_hit_dumps_the_resource_estimate_a_dumping_miss_dumps(self):
    dumping = {"MOSAIC_GPU_DUMP_RESOURCES": "1"}
    plain = self.lower()  # The entry made by a process that dumps nothing.
    hit = self.run_child(env=dumping)
    self.cache_dir = self.tmp / "another_cache"
    self.cache_dir.mkdir()
    miss = self.run_child(env=dumping)
    (hit_run,), (miss_run,) = hit["runs"], miss["runs"]
    self.assertEqual([plain["events"], hit_run["events"], miss_run["events"]],
                     [["miss"], ["hit"], ["miss"]])
    self.assertEqual(hit_run["normalized"], miss_run["normalized"])
    self.assertTrue(miss["_printed"])
    self.assertEqual(hit["_printed"], miss["_printed"])

  def test_default_compiler_params_are_cacheable(self):
    # None, as `plgpu.kernel` is called without compiler_params, and the
    # default CompilerParams passed.
    default = self.lower()
    explicit = self.lower(compiler_params={})
    self.assertEqual(default["events"], ["miss"])
    self.assertEqual(explicit["events"], ["hit"])

  # -- the store

  def test_a_corrupt_entry_is_a_miss(self):
    first = self.lower()
    (entry,) = self.entries()
    data = entry.read_bytes()
    _, header, _ = harness.split_entry(data)
    module_start = len(data) - header["module_size"]
    damages = {
        "empty": b"",
        "not an entry": b"not an entry",
        "truncated": data[:40],
        "a module that is not bytecode": (
            data[:module_start] + b"garbage!" + data[module_start + 8:]),
    }
    for name, damage in damages.items():
      with self.subTest(name):
        entry.write_bytes(damage)
        run = self.lower()
        self.assertEqual(run["events"], ["miss"], run)
        self.assertEqual(run["normalized"], first["normalized"])
        # The damaged entry was replaced: the next lowering hits.
        self.assertEqual(self.lower()["events"], ["hit"])

  def test_an_envelope_of_another_version_is_a_miss(self):
    first = self.lower()
    (entry,) = self.entries()
    data = entry.read_bytes()
    magic = data[:data.index(b"\n") + 1]
    version, header, module = harness.split_entry(data)
    entry.write_bytes(harness.join_entry(magic, version + 1, header, module))
    run = self.lower()
    self.assertEqual(run["events"], ["miss"], run)
    self.assertEqual(run["normalized"], first["normalized"])
    self.assertEqual(self.lower()["events"], ["hit"])

  def test_an_entry_of_another_key_is_a_miss(self):
    self.lower()
    (entry,) = self.entries()
    data = entry.read_bytes()
    magic = data[:data.index(b"\n") + 1]
    version, header, module = harness.split_entry(data)
    header["key"] = "0" * 64
    entry.write_bytes(harness.join_entry(magic, version, header, module))
    self.assertEqual(self.lower()["events"], ["miss"])

  # -- an entry that is whole is the one that was stored

  def test_an_entry_altered_but_still_valid_is_a_miss(self):
    first = self.lower()
    (entry,) = self.entries()
    other = self.lower(scale=3.0)
    self.assertEqual(other["events"], ["miss"])
    (other_entry,) = (e for e in self.entries() if e != entry)
    _, _, original = harness.split_entry(entry.read_bytes())
    _, _, altered = harness.split_entry(other_entry.read_bytes())
    # The module of the kernel with another constant in it: as long, as valid
    # as bytecode, and not what was stored. Only a checksum says so.
    self.assertEqual(len(altered), len(original))
    self.assertNotEqual(altered, original)
    harness.rewrite_entry(entry, module=altered, digest=False)
    run = self.lower()
    self.assertEqual(run["events"], ["miss"], run)
    self.assertEqual(run["normalized"], first["normalized"])
    # The altered entry was replaced: the next lowering hits what was lowered.
    again = self.lower()
    self.assertEqual(again["events"], ["hit"])
    self.assertEqual(again["normalized"], first["normalized"])

  def test_an_entry_with_altered_metadata_is_a_miss(self):
    first = self.lower()
    altered = {
        "an output's shape": {"new_out_shapes": [[[512], "float32"]]},
        "the grid": {"grid": [2, 1, 1]},
        "the block": {"block": [256, 1, 1]},
        "the seconds saved": {"seconds": 1e9},
    }
    for name, header in altered.items():
      with self.subTest(name):
        if not self.entries():
          self.lower()
        (entry,) = self.entries()
        harness.rewrite_entry(entry, header=header, digest=False)
        run = self.lower()
        self.assertEqual(run["events"], ["miss"], run)
        self.assertEqual(run["normalized"], first["normalized"])

  def test_an_entry_that_is_whole_but_is_not_an_entry_is_a_miss(self):
    # The checksum is of what the entry holds, so none of these is damage it
    # can see: what the entry holds is not an entry of the schema.
    first = self.lower()
    broken = {
        "a grid that is not three integers": {"grid": "abc"},
        "a block of the wrong length": {"block": [128, 1]},
        "a shape with a negative extent": {"new_out_shapes": [[[-1], "float32"]]},
        "a dtype that is not one": {"new_out_shapes": [[[256], "not-a-dtype"]]},
        "an output that is not a pair": {"new_out_shapes": [7]},
        "scratch that is not a list": {"gmem_scratch_shapes": None},
        "seconds that are not a number": {"seconds": "slow"},
    }
    for name, header in broken.items():
      with self.subTest(name):
        if not self.entries():
          self.lower()
        (entry,) = self.entries()
        harness.rewrite_entry(entry, header=header)
        run = self.lower()
        self.assertEqual(run["events"], ["miss"], run)
        self.assertEqual(run["normalized"], first["normalized"])
    with self.subTest("a module that is not bytecode"):
      if not self.entries():
        self.lower()
      (entry,) = self.entries()
      harness.rewrite_entry(entry, module=b"this is not a Mosaic GPU module" * 20)
      run = self.lower()
      self.assertEqual(run["events"], ["miss"], run)
      self.assertEqual(run["normalized"], first["normalized"])

  # -- what the cache expects to fail, and what it does not

  def test_a_store_that_cannot_be_made_is_a_miss_that_is_not_kept(self):
    blocker = self.tmp / "blocker"
    blocker.write_text("a file, where a directory would be")
    self.cache_dir = blocker / "cache"
    run = self.lower()
    self.assertNotIn("error", run, run)
    self.assertEqual(run["events"], ["miss"], run)

  def test_a_store_that_cannot_be_written_is_a_miss_that_is_not_kept(self):
    if os.geteuid() == 0:
      self.skipTest("root writes where it is told not to")
    self.assertEqual(self.lower()["events"], ["miss"])
    store = self.cache_dir / "mosaic_gpu_lowering"
    store.chmod(0o555)
    self.addCleanup(store.chmod, 0o755)
    run = self.lower(scale=3.0)
    self.assertNotIn("error", run, run)
    self.assertEqual(run["events"], ["miss"], run)
    self.assertLen(self.entries(), 1)

  def test_a_gpu_mosaic_does_not_support_is_lowered_as_it_is_without_the_cache(self):
    # A ROCm target: the reading of the GPU the cache makes raises what the
    # lowering raises, and the lowering is made as it is without a cache.
    cached = self.lower(capability="gfx90a")
    fresh = self.lower(cache=False, capability="gfx90a")
    self.assertEqual(fresh["error"], "ValueError", fresh)
    self.assertIn("ROCm", fresh["message"])
    self.assertEqual(
        (cached["error"], cached["message"]), (fresh["error"], fresh["message"]))
    self.assertEqual(cached["events"], ["uncacheable"], cached)
    self.assertEmpty(self.entries())

  def test_a_bug_of_the_cache_is_not_a_bypass(self):
    # What the cache does not expect (a bug in its own code) is an error of the
    # lowering, not a lowering made afresh that says nothing of it.
    for bug in ("encoder", "inputs", "seal"):
      with self.subTest(bug):
        self.cache_dir = self.tmp / f"cache-{bug}"
        run = self.lower(bug=bug)
        self.assertEqual(run["error"], "AttributeError", run)
        self.assertIn(f"a bug of the {bug}", run["message"])
        self.assertEmpty(self.entries())
    with self.subTest("a failure to seal is not remembered"):
      runs = self.run_child(bug="seal", repeat=2)["runs"]
      for run in runs:
        self.assertEqual(run["error"], "AttributeError", run)

  # -- a kernel that reads what the topology is

  def test_a_multi_device_kernel_reads_the_snapshot_and_lowers_as_it_does_without_it(self):
    devices = {"XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    verify = {**devices, "JAX_PALLAS_LOWERING_CACHE_VERIFY": "1"}
    for mesh in ("abstract2", "concrete2"):
      with self.subTest(mesh):
        self.cache_dir = self.tmp / f"cache-{mesh}"
        fresh = self.lower(cache=False, mesh=mesh, env=devices)
        miss = self.lower(mesh=mesh, env=devices)
        hit = self.lower(mesh=mesh, env=verify)
        for name, run in (("fresh", fresh), ("miss", miss), ("hit", hit)):
          self.assertNotIn("error", run, (name, run))
        self.assertEqual(miss["events"], ["miss"], miss)
        self.assertCountEqual(hit["events"], ["verified", "hit"], hit)
        # Hit and miss are one lowering, and its emitted call is the same.
        self.assertEqual(miss["normalized"], hit["normalized"])
        self.assertEqual(miss["text_sha"], hit["text_sha"])
        self.assert_lowers_as_fresh(miss, fresh)
        # Without the cache the lowering reads the topology of the process;
        # within it, every such read is answered from the snapshot.
        self.assertIn("device_count", fresh["reads_outside"])
        self.assertEqual(miss["reads_in_snapshot"], [], miss)
        self.assertEqual(hit["reads_in_snapshot"], [], hit)

  # -- processes that lower the same kernel at once

  def test_processes_that_lower_together_leave_one_entry_that_hits(self):
    processes = [self.spawn_child() for _ in range(3)]
    runs = [self.collect_child(p)["runs"][0] for p in processes]
    for run in runs:
      self.assertIn(run["events"], (["miss"], ["hit"]), run)
      self.assertEqual(run["normalized"], runs[0]["normalized"])
    self.assertLen(self.entries(), 1)
    self.assertEmpty(list((self.cache_dir / "mosaic_gpu_lowering").glob("*.tmp")))
    after = self.lower()
    self.assertEqual(after["events"], ["hit"])
    self.assertEqual(after["normalized"], runs[0]["normalized"])

  # -- a primitive is the object a sealed module holds, not its name

  def test_two_primitives_named_alike_with_sealed_rules_are_keyed_apart(self):
    # Two primitives of the sealed site, both named "add": the lane rule of the
    # first is the lane add rule, that of the second the lane subtract rule. The
    # two kernels' jaxprs are one but for which primitive their sum is, and the
    # processes' registries are one.
    self.kernel = "alias_a"
    added = self.lower()
    self.assertEqual(added["events"], ["miss"], added)
    self.kernel = "alias_b"
    subtracted = self.lower()
    fresh = self.lower(cache=False)
    self.assertEqual(subtracted["events"], ["miss"], subtracted)
    self.assert_lowers_as_fresh(subtracted, fresh)
    self.assertNotEqual(
        subtracted["normalized"]["asm"], added["normalized"]["asm"])
    again = self.lower()
    self.assertEqual(again["events"], ["hit"], again)
    self.assert_lowers_as_fresh(again, fresh)
    self.assertLen(self.entries(), 2)

  def test_a_primitive_no_sealed_module_holds_is_uncacheable(self):
    # Named "add", registered with a sealed rule, held by no sealed module: it is
    # no primitive a sealed installation has, whatever its name and rule. (The
    # first process registers it too: the registries are one.)
    self.kernel = "alias_a"
    self.assertEqual(
        self.lower(kernel_args={"register_unsealed": True})["events"], ["miss"])
    self.kernel = "alias_unsealed"
    fresh = self.lower(cache=False)
    for _ in range(2):
      alias = self.lower()
      self.assertEqual(alias["events"], ["uncacheable"], alias)
      self.assert_lowers_as_fresh(alias, fresh)
    self.assertLen(self.entries(), 1)

  def test_a_rule_registered_for_a_primitive_of_its_own_bypasses_only_its_kernels(self):
    # No kernel of a cacheable lowering has the primitive, so the rule (which is
    # not sealed code) is nothing they read.
    runs = self.run_child(tweaks=["foreign_primitive_rule"], repeat=2)["runs"]
    self.assertEqual([r["events"] for r in runs], [["miss"], ["hit"]], runs)

  # -- a module loaded after the process was sealed is checked as the others were

  def test_a_module_loaded_after_sealing_is_checked_against_its_record(self):
    # (A module as RECORD lists it is not a reason: what it registers when it is
    # imported may be another key, a miss, and that is all it can be.)
    cases = {
        "unlisted": (["unsealed"],), "edited": (["unsealed"],),
        "listed": (["hit"], ["miss"]),
    }
    for how, expected in cases.items():
      with self.subTest(how):
        self.cache_dir = self.tmp / f"cache-{how}"
        site = self.tmp / f"site-{how}"
        shutil.copytree(self._site, site)
        first, second = self.run_child(site=site, late_module=how)["runs"]
        self.assertEqual(first["events"], ["miss"], first)
        self.assertIn(second["events"], expected, second)

  # -- a constant of the lowering's code is read, not assumed

  def test_a_constant_of_the_lowering_that_is_replaced_is_part_of_the_key(self):
    # `SMEM_ALIGNMENT` is assigned a literal by the sealed source of the lowering:
    # a process that gives it a value of its own (one the key can hold) is another
    # key, before the first lookup and after the first lowering.
    self.kernel = "smem"
    stock = self.lower()
    self.assertEqual(stock["events"], ["miss"], stock)
    run = self.lower(tweaks=["smem_alignment:2048"])
    fresh = self.lower(cache=False, tweaks=["smem_alignment:2048"])
    self.assertEqual(run["events"], ["miss"], run)
    self.assert_lowers_as_fresh(run, fresh)
    self.assertNotEqual(run["normalized"]["module"], stock["normalized"]["module"])
    again = self.lower(tweaks=["smem_alignment:2048"])
    self.assertEqual(again["events"], ["hit"], again)
    self.cache_dir = self.tmp / "cache-after"
    warm, replaced = self.run_child(after=["smem_alignment:2048"])["runs"]
    self.assertEqual(warm["events"], ["miss"], warm)
    self.assertEqual(replaced["events"], ["miss"], replaced)

  def test_a_constant_replaced_by_what_a_key_cannot_hold_is_unsealed(self):
    # An object, a NumPy scalar and an instance of a subclass of int: none is the
    # literal kind of value the source assigns the constant, so the lowering
    # bypasses the cache, before the first lookup or after the first lowering.
    self.kernel = "smem"
    for tweak in ("smem_alignment_object", "smem_alignment_numpy:2048",
                  "smem_alignment_subclass:2048"):
      with self.subTest(f"{tweak} before the first lookup"):
        self.cache_dir = self.tmp / f"cache-before-{tweak}"
        run = self.lower(tweaks=[tweak])
        fresh = self.lower(cache=False, tweaks=[tweak])
        self.assertEqual(run["events"], ["unsealed"], run)
        self.assert_lowers_as_fresh(run, fresh)
        self.assertEmpty(self.entries())
      with self.subTest(f"{tweak} after the first lowering"):
        self.cache_dir = self.tmp / f"cache-after-{tweak}"
        warm, replaced = self.run_child(after=[tweak])["runs"]
        fresh = self.lower(cache=False, tweaks=[tweak])
        self.assertEqual(warm["events"], ["miss"], warm)
        self.assertEqual(replaced["events"], ["unsealed"], replaced)
        self.assert_lowers_as_fresh(replaced, fresh)

  # -- what a lowering reads of the process

  def test_a_lowering_reads_the_inputs_it_is_given(self):
    from jax.experimental.mosaic.gpu import core as mgpu_core
    from jax.experimental.mosaic.gpu import lowering_inputs
    from jax.experimental.mosaic.gpu import utils as mgpu_utils

    mesh = object()
    process = (mgpu_utils._infer_arch(), mgpu_core._max_smem_bytes(),
               mgpu_core.is_single_process_multi_device_topology(),
               mgpu_core.is_nvshmem_available(), mgpu_core._SMEM_SIZE_BOUND)
    inputs = lowering_inputs.LoweringInputs(
        arch=(8, 6), max_smem_bytes=12345,
        single_process_multi_device_topology=not process[2],
        nvshmem_available=not process[3], mesh=mesh, known_multihost_mesh=True,
        smem_size_bound=777)
    self.assertIsNone(lowering_inputs.current())
    with lowering_inputs.using(inputs):
      self.assertIs(lowering_inputs.current(), inputs)
      self.assertEqual(mgpu_utils._infer_arch(), (8, 6))
      self.assertEqual(mgpu_core._max_smem_bytes(), 12345)
      self.assertEqual(
          mgpu_core.is_single_process_multi_device_topology(), not process[2])
      self.assertEqual(mgpu_core.is_nvshmem_available(), not process[3])
      # The answer for the mesh it was asked about, and no other.
      self.assertTrue(mgpu_core._is_known_multihost_mesh(mesh))
      self.assertFalse(mgpu_core._is_known_multihost_mesh(None))
    self.assertIsNone(lowering_inputs.current())
    self.assertEqual(
        (mgpu_utils._infer_arch(), mgpu_core._max_smem_bytes(),
         mgpu_core.is_single_process_multi_device_topology(),
         mgpu_core.is_nvshmem_available(), mgpu_core._SMEM_SIZE_BOUND), process)



if __name__ == "__main__":
  absltest.main()
