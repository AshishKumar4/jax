# Sourced by index.sh and unlist.sh: Dew's package index's storage (R2, BUCKET and the AWS_* variables name it),
# read and written through a scratch directory `work` of the caller's.

# The index's projects, as its published root lists them: a project is listed once its pages are, never for objects
# under packages/ alone, which a publish that failed part-way through its uploads leaves (review, 2026-11-15:
# jaxlib's first publish). No root yet, none; any other failure to read it ends the caller.
published_projects() {
  if aws s3 cp --endpoint-url "$R2" "s3://$BUCKET/simple/index.html" "$work/root.html" \
      > /dev/null 2> "$work/root.txt"; then
    sed -n 's|.*<a href="\([^/"]*\)/">.*|\1|p' "$work/root.html"
  elif ! grep -q "(404)" "$work/root.txt"; then
    cat "$work/root.txt" >&2
    exit 1
  fi
}

# put KEY FILE CONTENT-TYPE [CACHE-CONTROL]
put() {
  aws s3api put-object --endpoint-url "$R2" --bucket "$BUCKET" --key "$1" --body "$2" --content-type "$3" \
    ${4:+--cache-control "$4"} > /dev/null
}

# A page under both names installers ask for (PAGE/ and PAGE/index.html), from the site's PAGE/index.html,
# revalidated at every request (no-cache: R2 answers a conditional request with 304), so no installer resolves
# from a page older than the index.
put_page() {
  put "$1" "$site/${1}index.html" text/html no-cache
  put "${1}index.html" "$site/${1}index.html" text/html no-cache
}
