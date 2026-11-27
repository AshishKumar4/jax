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

"""The custom-call targets JAX's own lowerings write, for analyses of a lowered or compiled program."""

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.custom_call_targets import (
    ALLOCATE_BUFFER as ALLOCATE_BUFFER,
    BUFFER_CALLBACK as BUFFER_CALLBACK,
    MOSAIC_GPU as MOSAIC_GPU,
    PYTHON_CALLBACK as PYTHON_CALLBACK,
    THREEFRY2X32 as THREEFRY2X32,
    TPU_CUSTOM_CALL as TPU_CUSTOM_CALL,
    TRITON as TRITON,
)
