"""`bench prereg` (SPEC 6.1): before the test split is touched, register the hash of the SPEC, of the
configuration, of the splits (the calibration and test ids), of the data manifest, and the rule
that computes Δ.

It writes the two files the test barrier reads (`bench/barrier.py` documents the format):
`prereg/manifest.json`, and `prereg/HASH`, one line with the sha256 of the manifest's bytes; and it
commits them on a clean tree. It never pushes: publishing is the author's act, and the barrier
opens only once `origin/main` has the registration. `commit` is informational (design §5): each
test execution records its own.
"""
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from bench import barrier, paths
from bench.contracts.config import config_sha256, load_config
from bench.data import DataError, pilot_sample
from bench.judge import j4

CODE_ROOT = Path(__file__).resolve().parent.parent


class PreregError(DataError):
    pass


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise PreregError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_ok(root: Path, *args: str) -> bool:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True).returncode == 0


def _config_chain(config_file: Path) -> List[Path]:
    """The configuration and every file it `extends`, followed as bench/contracts/config.py does."""
    chain = []
    path: Optional[Path] = config_file
    while path is not None and path not in chain:
        chain.append(path)
        if not path.is_file():
            break
        parent = (yaml.safe_load(path.read_text()) or {}).get("extends")
        path = (path.parent / parent).resolve() if parent else None
    return chain


def _check_committed(root: Path, chain: List[Path]) -> None:
    """Every file of the chain must be what the registration commit holds."""
    for path in chain:
        if not path.is_relative_to(root):
            raise PreregError(f"{path.name} is outside the repository: the commit cannot hold the configuration")
        rel = path.relative_to(root).as_posix()
        if not _git_ok(root, "ls-files", "--error-unmatch", "--", rel):
            raise PreregError(f"{rel} is not tracked: the commit would not hold the configuration")
        if _git_ok(root, "check-ignore", "-q", "--no-index", "--", rel):
            raise PreregError(f"{rel} is ignored by git: a configuration file must not be")
        if not _git_ok(root, "diff", "--quiet", "HEAD", "--", rel):
            raise PreregError(f"{rel} has uncommitted changes")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def delta_rule(config: Dict[str, Any], calib: List[str]) -> Dict[str, Any]:
    """The rule J4 applies (bench/judge/j4.py), in words and parameters, the pilot it takes d from,
    and the code that applies it."""
    implementation = Path(j4.__file__).resolve()
    size, seed = config["stats"]["pilot_size"], config["seeds"]["calib_split"]
    return {
        "formula": "delta = (z_0.95 + z_0.80) * sqrt(d / n)",
        "z": {"one_sided_confidence": j4.CONFIDENCE, "power": j4.POWER},
        "d": "paired discordance, the fraction of questions exactly one of the two arms gets right, "
             "measured on the pilot",
        "pilot": {"rule": "the stats.pilot_size calibration ids with the smallest sha256('<seeds.calib_split>:pilot:<id>')",
                  "size": size, "ids": pilot_sample(calib, size, seed)},
        "n": "the number of paired test questions (498)",
        "cap_pp": config["thresholds"]["delta_cap_pp"],
        "above_cap": "not testable with this n: reported as descriptive only",
        "noninferior_if": "the one-sided 95% paired-bootstrap lower bound of EX_A - EX_B is above -delta",
        "bootstrap": {"unit": "question", "n_boot": config["stats"]["n_boot"], "seed": config["seeds"]["bootstrap"],
                      "quantile": "inverted CDF"},
        "calibration_margin": "delta / 2 when choosing engines on the calibration split (B5)",
        "implementation": {"path": implementation.relative_to(CODE_ROOT).as_posix(), "sha256": _sha256(implementation)},
    }


def unset(node: Any, where: str = "") -> List[str]:
    """The configuration values still null: registering them fixes them as undecided."""
    if node is None:
        return [where]
    items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else []
    return [path for key, value in items for path in unset(value, f"{where}.{key}" if where else str(key))]


def register(config_path: str, root: Optional[Path] = None, replace: bool = False) -> Dict[str, Any]:
    """Write and commit the pre-registration of `config_path` (relative to `root` unless absolute).
    Registering the same inputs again changes nothing; registering different ones over an
    existing registration needs `replace`, because test runs made under it stop being scorable."""
    root = Path(root or paths.ROOT).resolve()
    if _git(root, "status", "--porcelain"):
        raise PreregError("the working tree has uncommitted changes: commit them, so the registration "
                          "hashes what the repository holds")
    config_file = (root / config_path).resolve()
    if not config_file.is_relative_to(root):
        raise PreregError(f"the configuration must be inside the repository, so the commit holds it: {config_path}")
    _check_committed(root, _config_chain(config_file))
    config = load_config(config_file)
    for key in ("n_boot", "pilot_size"):
        if not isinstance((config.get("stats") or {}).get(key), int):
            raise PreregError(f"the configuration has no stats.{key}: the delta rule needs it")
    calib = json.loads((root / barrier.REGISTERED["splits_sha256"]).read_text()).get("calib") or []
    manifest = {"config_path": config_file.relative_to(root).as_posix(), "config_sha256": config_sha256(config),
                **{key: _sha256(root / rel) for key, rel in barrier.REGISTERED.items()},
                "commit": _git(root, "rev-parse", "HEAD"), "delta_rule": delta_rule(config, calib)}
    manifest_path, hash_path = root / barrier.PREREG_MANIFEST, root / barrier.PREREG_HASH
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if {k: v for k, v in existing.items() if k != "commit"} == {k: v for k, v in manifest.items() if k != "commit"}:
            return {"hash": hash_path.read_text().strip(), "commit": _git(root, "rev-parse", "HEAD"), "new": False,
                    "unset": unset(config)}
        if not replace:
            raise PreregError(f"a different pre-registration is in force ({hash_path.read_text().strip()}); "
                              f"pass --replace to register anew (test runs made under it can no longer be scored)")
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_bytes(raw)
    hash_path.write_text(digest + "\n")
    files = (barrier.PREREG_MANIFEST, barrier.PREREG_HASH)
    _git(root, "add", "--", *files)
    _git(root, "commit", "-q", "-m", f"Pre-registration {digest}\n\nSPEC.md, {manifest['config_path']}, the splits, "
         f"the data manifest and the delta rule, before the test split is touched. Push to publish.", "--", *files)
    return {"hash": digest, "commit": _git(root, "rev-parse", "HEAD"), "new": True, "unset": unset(config)}
