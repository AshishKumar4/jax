#!/bin/bash
# A project taken back off Dew's package index (https://pypi.dewml.dev/simple), the undoing of its first
# publication: its root first rewritten without it (so no root lists a project whose pages are gone), then both
# names of its page deleted, so an installer that lists the index first finds the project on PyPI alone again
# (uv's first-index rule passes to the next index where this one answers 404), then its manifest, so publishing it
# again is again a new project's (NEW_PROJECT, ledger 3116). Its files stay under packages/, which no page links.
#
#   unlist.sh SITE PROJECT     R2, BUCKET and the AWS_* variables name the storage
set -euo pipefail
site=$1 package=$2
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
. "$here/index_lib.sh"
projects=$(published_projects) || exit 1
grep -qx "$package" <<< "$projects" || echo "unlist: the root does not list $package; its pages are deleted anyway"
python3 "$here/patch_stack.py" root "$site" --projects $(grep -vx "$package" <<< "$projects" || true)
put_page simple/
for key in "simple/$package/" "simple/$package/index.html" "packages/$package/manifest.json"; do
  aws s3api delete-object --endpoint-url "$R2" --bucket "$BUCKET" --key "$key" > /dev/null
done
echo "unlist: $package is off the index; its root lists $(sed -n 's|.*<a href="\([^/"]*\)/">.*|\1|p' \
  "$site/simple/index.html" | tr '\n' ' ')"
