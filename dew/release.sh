#!/usr/bin/env bash
# A post release of the stack, composed and published in one run: the newest published release branch, the
# reviewed and green commits it gains in their order, the version bumped, both branches pushed at once and
# patch-stack.yml asked to publish, its own run (by the token its name carries) watched to its end. Any step
# that fails ends the run there: one before the push leaves the fork as it was, so no push or publish follows
# a broken step (Main, 2026-10-30: a typed chain pushed a half-composed post40 and began its publish); one
# after it leaves both branches at the composed release, unpublished, and says how to publish it again.
#
#   dew/release.sh CHECKOUT BASE NEW MESSAGE COMMIT...
#
# CHECKOUT is a clone of the fork with `origin` its remote; BASE the published release it grows from
# (0.12.0.1.post39), NEW the release it makes (0.12.0.1.post40), MESSAGE the release commit's message, and each
# COMMIT a commit or a range (`A..B`) cherry-picked in order. The stack branch must be at BASE's release branch,
# and NEW's must not exist yet.
set -euo pipefail

checkout=$1 base=$2 new=$3 message=$4
shift 4
if (( $# == 0 )); then
  echo "release.sh: name the commits $new gains" >&2
  exit 2
fi
release=${base%.post*}
stack=v$release-cache-key-pool-fingerprints

# The branches as origin holds them now, wherever the clone last saw them.
git -C "$checkout" fetch -q origin "+refs/heads/release/$base:refs/remotes/origin/release/$base" \
  "+refs/heads/$stack:refs/remotes/origin/$stack"
published=$(git -C "$checkout" rev-parse "origin/release/$base")
if [[ $(git -C "$checkout" rev-parse "origin/$stack") != "$published" ]]; then
  echo "release.sh: $stack is not at release/$base ($published)" >&2
  exit 1
fi
# ls-remote says 2 where the ref is absent; anything else is a ref there or a remote not read.
status=0
git -C "$checkout" ls-remote --exit-code origin "refs/heads/release/$new" > /dev/null || status=$?
if (( status != 2 )); then
  echo "release.sh: release/$new exists already, or origin could not be read ($status)" >&2
  exit 1
fi

work=$(mktemp -d)
trap 'git -C "$checkout" worktree remove --force "$work" 2> /dev/null || rm -rf "$work"' EXIT
git -C "$checkout" worktree add -q --detach "$work" "$published"
for commit in "$@"; do
  git -C "$work" cherry-pick "$commit" > /dev/null
done
# The versions as literal strings: a dot in one matches only a dot.
sed -i "s/^_release_version: str | None = \"${base//./\\.}\"\$/_release_version: str | None = \"$new\"/" \
  "$work/jax/version.py"
grep -Fqx "_release_version: str | None = \"$new\"" "$work/jax/version.py"
git -C "$work" commit -q -am "$message"
composed=$(git -C "$work" rev-parse HEAD)

# Both branches move together, or neither: the stack branch only from where it was found, the release branch
# only where none is.
git -C "$work" push -q --atomic "--force-with-lease=refs/heads/$stack:$published" \
  "--force-with-lease=refs/heads/release/$new:" origin \
  "$composed:refs/heads/release/$new" "$composed:refs/heads/$stack"

# From here a failure leaves both branches at the composed release, unpublished: its run's log says why, and
# `gh workflow run patch-stack.yml -R AshishKumar4/jax --ref dew -f publish=RELEASE` publishes it again.
token=release-$new-$(date +%s)-$$
gh workflow run patch-stack.yml -R AshishKumar4/jax --ref dew -f "publish=$release" -f "token=$token"
run=
for _ in $(seq 60); do
  run=$(gh run list -R AshishKumar4/jax --workflow patch-stack.yml -L 20 --json databaseId,displayTitle \
    --jq ".[] | select(.displayTitle == \"patch stack $token\") | .databaseId")
  [[ -n $run ]] && break
  sleep 5
done
if [[ -z $run ]]; then
  echo "release.sh: no run of patch-stack.yml named $token in 5 minutes; release/$new and $stack are at" \
    "$composed, unpublished" >&2
  exit 1
fi
gh run watch "$run" -R AshishKumar4/jax --exit-status > /dev/null
echo "$new published from $composed (run $run)"
