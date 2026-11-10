"""What every execution records about the code that produced it, and nothing about the machine or a
credential."""
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench import paths


DIRTY_PATHS_MAX = 100  # the entries a record lists; `dirty_paths_total` counts them all


def dirty_entries(porcelain_z: str) -> List[Dict[str, str]]:
    """`git status --porcelain -z` as [{"status": XY, "path": path}], sorted by path. A rename or copy is
    one entry, its new path, with the old one as `from`."""
    fields, entries = porcelain_z.split("\0"), []
    i = 0
    while i < len(fields):
        field = fields[i]
        i += 1
        if len(field) < 4:
            continue
        entry = {"status": field[:2], "path": field[3:]}
        if field[0] in "RC" or field[1] in "RC":  # -z: the original path is the next field
            entry["from"] = fields[i] if i < len(fields) else ""
            i += 1
        entries.append(entry)
    return sorted(entries, key=lambda e: (e["path"], e["status"]))


def git_state(root: Optional[Path] = None) -> Dict[str, Any]:
    """The commit, whether the tree differs from it (`dirty`, as always), and how: the first
    DIRTY_PATHS_MAX of `git status --porcelain` entries by path (`dirty_paths`, relative to the repository,
    untracked files included) and their number (`dirty_paths_total`)."""
    root = root or paths.ROOT

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout
    entries = dirty_entries(git("status", "--porcelain", "-z"))
    return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(entries),
            "dirty_paths": entries[:DIRTY_PATHS_MAX], "dirty_paths_total": len(entries)}


def scrub(text: str) -> str:
    """Remove the origin from free text that reaches a record (SPEC S1: anonymize with respect to
    origin): the repository path and the home directory become placeholders."""
    return text.replace(str(paths.ROOT), "<repo>").replace(str(Path.home()), "~")


SECRET_MIN_LENGTH = 8  # a key is longer; shorter values are placeholders ("x", "1", "EMPTY"), and
# replacing every "1" of a record would corrupt it
_KEY_LIKE = re.compile(r"\b(sk|hf)[-_][A-Za-z0-9_\-*.]{6,}")


def redact(text: str, config: Dict[str, Any]) -> str:
    """No record the harness writes carries a credential (a run's manifest, written whole through this, a
    C1 line's error, a harness error, a load test's and a preflight's report; CHESS's own history files under
    runs/<run>/chess/ are vendor code and keep its text: local and never published as they are): the values of every
    credential variable of the configuration (API keys and `headers_env` values, C2's
    `credential_envs`), and anything shaped like a key (provider error bodies sometimes echo a
    masked one)."""
    from bench.contracts.config import credential_envs
    values = {os.environ.get(name) for name in credential_envs(config) + ["ANTHROPIC_WORKSPACE_ID"]}
    for value in sorted((v for v in values if v and len(v) >= SECRET_MIN_LENGTH), key=len, reverse=True):
        text = text.replace(value, "<redacted>")  # the longest first: one value may contain another
    return _KEY_LIKE.sub("<redacted>", text)
