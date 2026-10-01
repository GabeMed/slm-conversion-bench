"""What every execution records about the code that produced it, and nothing about the machine or a
credential."""
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from bench import paths


def git_state(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or paths.ROOT

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def scrub(text: str) -> str:
    """Remove the origin from free text that reaches a record (SPEC S1: anonymize with respect to
    origin): the repository path and the home directory become placeholders."""
    return text.replace(str(paths.ROOT), "<repo>").replace(str(Path.home()), "~")


SECRET_MIN_LENGTH = 8  # a key is longer; shorter values are placeholders ("x", "1", "EMPTY"), and
# replacing every "1" of a record would corrupt it
_KEY_LIKE = re.compile(r"\b(sk|hf)[-_][A-Za-z0-9_\-*.]{6,}")


def redact(text: str, config: Dict[str, Any]) -> str:
    """What reaches a record never carries a credential (provider error text included, whatever the
    record: a committed manifest, calls.jsonl, a preflight report): the values of every
    credential variable of the configuration (API keys and `headers_env` values, C2's
    `credential_envs`), and anything shaped like a key (provider error bodies sometimes echo a
    masked one)."""
    from bench.contracts.config import credential_envs
    values = {os.environ.get(name) for name in credential_envs(config)}
    for value in sorted((v for v in values if v and len(v) >= SECRET_MIN_LENGTH), key=len, reverse=True):
        text = text.replace(value, "<redacted>")  # the longest first: one value may contain another
    return _KEY_LIKE.sub("<redacted>", text)
