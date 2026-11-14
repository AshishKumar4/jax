"""dew/bazel_credentials.py: Bazel's credential helper for Dew's remote cache, the token read from its file."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import bazel_credentials

HELPER = Path(__file__).with_name("bazel_credentials.py")


@pytest.mark.parametrize("written", ["t0ken-abc", "t0ken-abc\n", "t0ken-abc\r\n"], ids=["bare", "lf", "crlf"])
def test_bazel_reads_the_tokens_header_as_json_whatever_its_files_line_ending(tmp_path: Path, written: str):
    """Run as Bazel runs it (its request on stdin, `get` after the file), it prints one JSON object holding the
    bearer header of the file's token, its line ending dropped."""
    token = tmp_path / "bazel.token"
    token.write_bytes(written.encode())
    done = subprocess.run([sys.executable, str(HELPER), str(token), "get"], input='{"uri": "https://cache/ac/x"}',
                          capture_output=True, text=True, check=True)
    assert json.loads(done.stdout) == {"headers": {"Authorization": ["Bearer t0ken-abc"]}}


@pytest.mark.parametrize("written", ["", "two words", 'a"quote', "a\\backslash"])
def test_a_file_holding_no_single_token_is_refused_and_prints_nothing_of_it(tmp_path: Path, written: str):
    """A file that holds no single token makes the helper fail, its stdout empty and its message naming the file,
    never the file's contents."""
    token = tmp_path / "bazel.token"
    token.write_text(written)
    done = subprocess.run([sys.executable, str(HELPER), str(token), "get"], input="{}", capture_output=True, text=True)
    assert done.returncode != 0 and done.stdout == ""
    assert str(token) in done.stderr and (not written or written not in done.stderr)
    with pytest.raises(SystemExit):
        bazel_credentials.headers(str(token))
