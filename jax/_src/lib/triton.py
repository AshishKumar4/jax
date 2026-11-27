# Copyright 2024 The JAX Authors.
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

import sys

from jaxlib.mlir.dialects._ods_common import _cext
from jaxlib.triton import dialect  # noqa: F401

# MLIR finds the Python classes of a dialect's operations by importing
# `<prefix>.<namespace>` under each dialect search prefix, and remembers only a
# module it found. The Triton dialect's module is `jaxlib.triton.dialect`, no
# `<prefix>.tt`, so each lookup of a `tt` operation's class searched every
# prefix's directories again: in a kernel's lowering, most of its time. Named
# `tt` under a search prefix of its own, the first lookup finds it and MLIR
# keeps it.
sys.modules[f"{__name__}.tt"] = dialect
_cext.globals.append_dialect_search_prefix(__name__)
