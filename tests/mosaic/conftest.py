# Copyright 2026 The JAX Authors. All Rights Reserved.
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
"""Scratch: a fork jax over a suffixed jaxlib (0.12.0.1 against 0.12.0.1.postN)
runs the layout inference tests their version check skips."""
import jax
import jax.version

jax.version._version = jax.lib.__version__
