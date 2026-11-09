"""Apply a config patch (configs/runpod.patch.yaml) to config.yaml: exact text replacements, so every comment
of the registered configuration is kept, then a check of what the patched configuration says (`expect`) and
that it still loads (bench.contracts.config.load_config). Prints the unified diff and the configuration's
sha256 before and after; writes only with --write.

    python scripts/apply_config_patch.py configs/runpod.patch.yaml [--config config.yaml] [--write]
"""
import argparse
import difflib
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class PatchError(RuntimeError):
    pass


def _indent(text: str, n: int) -> str:
    return "".join((" " * n + line) if line.strip() else line for line in text.splitlines(keepends=True))


def patched_text(original: str, patch: Dict[str, Any]) -> str:
    text = original
    for i, change in enumerate(patch.get("replace") or []):
        old, new = _indent(change["old"], change.get("indent", 0)), _indent(change["new"], change.get("indent", 0))
        count = text.count(old)
        if count != 1:
            raise PatchError(f"replacement {i + 1}: its text occurs {count} times in the configuration, not once "
                             f"(already applied, or the configuration changed): {old.splitlines()[0].strip()!r}")
        text = text.replace(old, new)
    return text


def lookup(config: Any, dotted: str) -> Any:
    """A dotted path; `[name]` selects the item of a list whose `name` it is (roles.slm_candidates[qwen3-8b])."""
    node = config
    for key, name in re.findall(r"([^.\[\]]+)|\[([^\]]+)\]", dotted):
        if name:
            found = [item for item in node if isinstance(item, dict) and item.get("name") == name] \
                if isinstance(node, list) else []
            if len(found) != 1:
                raise PatchError(f"{dotted}: no single list item named {name!r}")
            node = found[0]
            continue
        segment = key
        if isinstance(node, dict) and segment in node:
            node = node[segment]
        else:
            raise PatchError(f"{dotted}: no {segment!r}")
    return node


def check(text: str, patch: Dict[str, Any], base_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    """The patched configuration loaded as bench loads it, and the `expect` paths it does not satisfy."""
    from bench.contracts.config import load_config

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", dir=base_dir, delete=False) as f:
        f.write(text)
    try:
        config = load_config(f.name)
    finally:
        Path(f.name).unlink()
    wrong = []
    for dotted, expected in (patch.get("expect") or {}).items():
        try:
            found = lookup(config, dotted)
        except PatchError as e:
            wrong.append(str(e))
            continue
        if found != expected:
            wrong.append(f"{dotted}: {found!r}, expected {expected!r}")
    return config, wrong


def apply(patch_path: Path, config_path: Path, write: bool = False) -> Dict[str, Any]:
    from bench.contracts.config import config_sha256, load_config

    patch = yaml.safe_load(patch_path.read_text())
    original = config_path.read_text()
    text = patched_text(original, patch)
    config, wrong = check(text, patch, config_path.resolve().parent)
    if wrong:
        raise PatchError("the patched configuration is not what the patch expects: " + "; ".join(wrong))
    diff = "".join(difflib.unified_diff(original.splitlines(keepends=True), text.splitlines(keepends=True),
                                        f"a/{config_path.name}", f"b/{config_path.name}"))
    before = config_sha256(load_config(config_path))
    if write:
        config_path.write_text(text)
    return {"diff": diff, "sha256_before": before, "sha256_after": config_sha256(config), "written": write}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("patch")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--write", action="store_true", help="write the patched configuration (default: only show it)")
    args = parser.parse_args(argv)
    try:
        result = apply(Path(args.patch), Path(args.config), args.write)
    except PatchError as e:
        print(f"apply_config_patch: {e}", file=sys.stderr)
        return 2
    print(result["diff"], end="")
    print(f"config sha256 {result['sha256_before']} -> {result['sha256_after']} "
          f"({'written' if result['written'] else 'not written: --write applies it'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
