#!/bin/bash
# One project's wheels indexed on Dew's package index (https://pypi.dewml.dev/simple), the files it lists already
# kept as they are: its manifest read (absent only before its first publication, which NEW_PROJECT must name, as
# the index then serves it in place of PyPI's to an installer listing the index first: ledger 3116), the pages
# written by patch_stack.py index, the wheels uploaded, the manifest and pages last.
#
#   index.sh SITE WHEEL...     every WHEEL one project's; R2, BUCKET and the AWS_* variables name the storage;
#                              YANK the files it lists to mark yanked (PEP 592), REASON why; PACKAGE the project
#                              where no WHEEL names it (a run that only yanks)
set -euo pipefail
site=$1
shift
here=$(cd "$(dirname "$0")" && pwd)
# A project's name as PEP 503 serves it, from a wheel's file name (PEP 427: its - written _).
project() { basename "$1" | cut -d- -f1 | tr '_.A-Z' '--a-z'; }
if [ $# -gt 0 ]; then package=$(project "$1"); else package=${PACKAGE:-}; fi
[ -n "$package" ] && { [ $# -gt 0 ] || [ -n "${YANK:-}" ]; } || { echo "::error::no wheel to index"; exit 1; }
for wheel in "$@"; do
  [ "$(project "$wheel")" = "$package" ] || { echo "::error::$wheel is not $package's"; exit 1; }
done
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
if aws s3api head-object --endpoint-url "$R2" --bucket "$BUCKET" --key "packages/$package/manifest.json" \
    > /dev/null 2> "$work/head.txt"; then
  aws s3 cp --endpoint-url "$R2" "s3://$BUCKET/packages/$package/manifest.json" "$work/manifest.json"
elif grep -q "(404)" "$work/head.txt"; then
  [ "${NEW_PROJECT:-}" = "$package" ] || {
    echo "::error::$package is new to the index, which would serve it in place of PyPI's to every installer that lists the index first (Dew's lock among them); publish it with new_project=$package once Dew's pyproject maps $package to this index explicitly"
    exit 1
  }
  echo '{}' > "$work/manifest.json"
else
  cat "$work/head.txt"
  exit 1
fi
projects=$(aws s3api list-objects-v2 --endpoint-url "$R2" --bucket "$BUCKET" --prefix packages/ \
             --delimiter / --query 'CommonPrefixes[].Prefix' --output text \
           | tr '\t' '\n' | sed -n 's|^packages/\(.*\)/$|\1|p')
read -ra names <<< "$(printf '%s' "${YANK:-}" | tr '\r\n' '  ')"
yanks=()
if [ ${#names[@]} -gt 0 ]; then yanks=(--yank "${names[@]}" "--reason=${REASON:-}"); fi
python3 "$here/patch_stack.py" index "$work/manifest.json" "$site" "$@" "${yanks[@]}" --package "$package" \
  --projects $projects > "$work/new.txt"
cat "$work/new.txt"
put() { aws s3api put-object --endpoint-url "$R2" --bucket "$BUCKET" --key "$1" --body "$2" --content-type "$3" \
          ${4:+--cache-control "$4"} > /dev/null; }
while read -r name; do
  put "packages/$package/$name" "$site/packages/$package/$name" application/octet-stream
  put "packages/$package/$name.metadata" "$site/packages/$package/$name.metadata" text/plain
done < "$work/new.txt"
put "packages/$package/manifest.json" "$work/manifest.json" application/json
# Each page under both names installers ask for (simple/, simple/PROJECT/), revalidated at every request
# (no-cache: R2 answers a conditional request with 304), so no installer resolves from a page older than the index.
for page in simple/ "simple/$package/"; do
  put "$page" "$site/${page}index.html" text/html no-cache
  put "${page}index.html" "$site/${page}index.html" text/html no-cache
done
