"""REQ-011: no private or business names in the public repository.

The terms are not in the repository (that would publish them): CI reads them from the
PRIVATE_TERMS secret, comma-separated; the test is skipped where the variable is unset.
"""
import os
import re
import subprocess

import pytest

from bench import paths

TERMS = [t.strip() for t in os.environ.get("PRIVATE_TERMS", "").split(",") if t.strip()]


@pytest.mark.skipif(not TERMS, reason="PRIVATE_TERMS is not set")
def test_no_private_terms_in_tracked_files():
    files = subprocess.run(["git", "-C", str(paths.ROOT), "ls-files"], capture_output=True, text=True,
                           check=True).stdout.split()
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, TERMS)) + r")\b", re.IGNORECASE)
    hits = []
    for name in files:
        try:
            text = (paths.ROOT / name).read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        hits += [f"{name}:{n}" for n, line in enumerate(text.splitlines(), 1) if pattern.search(line)]
    assert not hits, f"private terms found at {hits[:20]}"
