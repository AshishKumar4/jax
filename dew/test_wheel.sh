#!/bin/bash
# A package's tests run against its installed wheel, never its source tree: the test paths (and the checkout's
# root conftest.py and pytest.ini, where it has them) copied out of the checkout into a directory of their own,
# run there by an interpreter that puts no working directory on its path (-P), after each module the wheel
# installs is checked to import from the environment. A source-tree run imports the checkout first (python -m
# pytest puts the working directory ahead of site-packages), so a module the wheel leaves out, or packages
# wrongly, passed it (audit of 2026-11-01).
#
#   test_wheel.sh PYTHON WHEEL CHECKOUT TEST...    PYTHON: the environment's interpreter, the wheel installed;
#                                                 PYTEST_ARGS: pytest's own arguments (-n auto, say)
set -euo pipefail
# The interpreter absolute but its own link kept: a virtual environment's python is a link to its base
# interpreter, and only the link's path finds the environment (its pyvenv.cfg).
python="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" wheel=$(realpath "$2") checkout=$(realpath "$3")
shift 3
apart=$(mktemp -d)
trap 'rm -rf "$apart"' EXIT
for path in "$@" conftest.py pytest.ini; do
  if [ -e "$checkout/$path" ]; then
    mkdir -p "$apart/$(dirname "$path")"
    cp -r "$checkout/$path" "$apart/$path"
  fi
done
# The wheel's top-level modules, from its RECORD (the dist-info and data directories aside).
modules=$("$python" -c '
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
packages = {name.split("/")[0] for name in names if "/" in name}
modules = {name.removesuffix(".py") for name in names if "/" not in name and name.endswith(".py")}
print(" ".join(sorted(modules | {top for top in packages if not top.endswith((".dist-info", ".data"))})))
' "$wheel")
cd "$apart"
site=$("$python" -P -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
for module in $modules; do
  "$python" -P -c "
import importlib, os, sys
found = importlib.import_module('$module')
where = os.path.realpath(getattr(found, '__file__', '') or next(iter(found.__path__)))
sys.exit(0 if where.startswith(os.path.realpath('$site')) else f'$module imports from {where}, not the wheel')
"
done
read -ra arguments <<< "${PYTEST_ARGS:-}"
"$python" -P -m pytest -q -p no:cacheprovider "${arguments[@]}" "$@"
