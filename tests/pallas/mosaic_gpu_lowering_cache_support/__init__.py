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
"""Support for the tests of the Mosaic GPU lowering cache.

``harness`` drives the tests (a sealed site, a process per lowering, the cache's
entries), ``child`` is the process, ``kernels`` the kernels it lowers,
``encoder_cases`` the canonicalizer's, and ``sealed_classes`` and ``sealed_rules``
are modules that are added to the sealed site's jax, so that they are sealed code.
"""
