"""Bazel's credential helper for Dew's remote cache: `bazel_credentials.py TOKEN_FILE get`.

Bazel runs it for each request to the cache's host (`--credential_helper=HOST=LAUNCHER`, the launcher passing the
token file's path and Bazel's own arguments): it reads Bazel's request on stdin and prints the request's headers as
one JSON object on stdout (https://github.com/EngFlow/credential-helper-spec, "get"). The token is read from its file
at each request and appears in no option, command line or message: a header given as a Bazel option was once split
at its space and printed as a target in a public log (Dew ledger 6204). A file that holds no single token (empty,
several words, a quote or backslash) is refused, nothing printed but why, and Bazel reports the helper failed.
"""

import json
import sys
from pathlib import Path


def headers(token_file: str) -> dict:
    """The headers of a request to the cache: its bearer token, read from `token_file` (its line ending, LF or
    CRLF, dropped)."""
    token = Path(token_file).read_text(encoding="utf-8").strip()
    if not token or any(character.isspace() or character in "\"\\" for character in token):
        raise SystemExit(f"{token_file}: no single token (a token is one word of no quote or backslash)")
    return {"headers": {"Authorization": [f"Bearer {token}"]}}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit("usage: bazel_credentials.py TOKEN_FILE get")
    sys.stdin.read()
    print(json.dumps(headers(argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
