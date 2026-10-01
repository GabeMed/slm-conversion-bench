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

# The analysis code the registration hashes, so no reading of the results changes without changing
# prereg/HASH: every tracked file under these paths, which is every file whose code can change a score,
# a judgment or a verdict: the contracts too (agreement, the C1 reader, the facts, the configuration)
# and the paths they resolve. The contracts include the router, so a routing change after the
# registration needs a new one too.
ANALYSIS_CODE = ("bench/judge", "bench/evaluate.py", "bench/data.py", "bench/report.py", "bench/contracts",
                 "bench/paths.py")
REQUIRED_ANALYSIS = ("bench/judge/j4.py", "bench/evaluate.py", "bench/data.py")


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
    """Every file of the chain must be what the registration commit holds: inside the repository,
    tracked, not matched by .gitignore; a modified one is already refused by the clean-tree check."""
    for path in chain:
        if not path.is_relative_to(root):
            raise PreregError(f"{path.name} is outside the repository: the commit cannot hold the configuration")
        rel = path.relative_to(root).as_posix()
        if not _git_ok(root, "ls-files", "--error-unmatch", "--", rel):
            raise PreregError(f"{rel} is not tracked: the commit would not hold the configuration")
        if _git_ok(root, "check-ignore", "-q", "--no-index", "--", rel):
            raise PreregError(f"{rel} is ignored by git: a configuration file must not be")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analysis_code(root: Path) -> Dict[str, str]:
    """The sha256 of every tracked file of the analysis code, by path. Python code under it that git does
    not track would run here while no commit holds it: refused (bytecode, .DS_Store and the like are not)."""
    def untracked(*flags: str) -> List[str]:
        return sorted(rel for rel in _git(root, "ls-files", "--others", *flags, "--", *ANALYSIS_CODE).splitlines()
                      if rel.endswith(".py"))
    ignored = untracked("--ignored", "--exclude-standard")
    if ignored:
        raise PreregError(f"analysis code ignored by git: {', '.join(ignored)}")
    uncommitted = untracked("--exclude-standard")
    if uncommitted:
        raise PreregError(f"analysis code not committed: {', '.join(uncommitted)}")
    files = sorted(_git(root, "ls-files", "--", *ANALYSIS_CODE).splitlines())
    missing = [rel for rel in REQUIRED_ANALYSIS if rel not in files]
    if missing:
        raise PreregError(f"the analysis code to register is not tracked: {', '.join(missing)}")
    return {rel: _sha256(root / rel) for rel in files}


def delta_rule(config: Dict[str, Any], calib: List[str]) -> Dict[str, Any]:
    """The rule J4 applies (bench/judge/j4.py), in words and parameters, and the pilot it takes d from."""
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
                      "quantile": "inverted CDF: ci_low the ceil(n_boot/20)-th smallest resample, ci_high the "
                                  "ceil(n_boot/20)-th largest, from the same resamples"},
        "calibration_margin": "delta / 2 when choosing engines on the calibration split (B5)",
    }


def unset(node: Any, where: str = "") -> List[str]:
    """The configuration values still null: registering them fixes them as undecided."""
    if node is None:
        return [where]
    items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else []
    return [path for key, value in items for path in unset(value, f"{where}.{key}" if where else str(key))]


def check_registered_analysis_code(root: Path) -> None:
    """A test result is produced only by the analysis code that was registered (prereg/manifest.json),
    file by file; a registration that recorded none matches no code. For whatever scores or reads the
    test: bench eval, and the report."""
    manifest = barrier.read_registered(root)
    if manifest is None:
        raise PreregError(f"{barrier.PREREG_MANIFEST} is not a JSON object: no analysis code is registered")
    registered = manifest.get("analysis_code") or {}
    now = analysis_code(root)
    changed = sorted(rel for rel in set(registered) | set(now) if registered.get(rel) != now.get(rel))
    if changed:
        raise PreregError(f"the analysis code differs from the pre-registered one: {', '.join(changed[:10])}"
                          + (f" and {len(changed) - 10} more" if len(changed) > 10 else ""))


def register(config_path: str, root: Optional[Path] = None, replace: bool = False) -> Dict[str, Any]:
    """Write and commit the pre-registration of `config_path` (relative to `root` unless absolute).
    Registering the same inputs again changes nothing; registering different ones over an
    existing registration needs `replace`, because test runs made under it stop being scorable."""
    root = Path(root or paths.ROOT).resolve()
    if _git(root, "status", "--porcelain"):
        raise PreregError("the working tree has uncommitted changes: commit them, so the registration "
                          "hashes what the repository holds")
    config_file = (root / config_path).resolve()
    _check_committed(root, _config_chain(config_file))
    config = load_config(config_file)
    for key in ("n_boot", "pilot_size"):
        if not isinstance((config.get("stats") or {}).get(key), int):
            raise PreregError(f"the configuration has no stats.{key}: the delta rule needs it")
    inputs = barrier.REGISTERED
    calib = json.loads((root / inputs["splits_sha256"]).read_text()).get("calib") or []
    manifest = {"config_path": config_file.relative_to(root).as_posix(), "config_sha256": config_sha256(config),
                **{key: _sha256(root / rel) for key, rel in inputs.items()},
                "commit": _git(root, "rev-parse", "HEAD"), "delta_rule": delta_rule(config, calib),
                "analysis_code": analysis_code(root)}
    manifest_path, hash_path = root / barrier.PREREG_MANIFEST, root / barrier.PREREG_HASH
    if manifest_path.exists() or hash_path.exists():
        existing = barrier.read_registered(root)  # both halves, as the barrier reads them
        intact = existing is not None and hash_path.is_file() and \
            hash_path.read_bytes().strip() == _sha256(manifest_path).encode()
        if not intact:
            if not replace:
                raise PreregError(f"the registration in prereg/ is broken ({barrier.PREREG_MANIFEST} is not a JSON object, "
                                  f"or {barrier.PREREG_HASH} is not its sha256): pass --replace to register anew")
        elif {k: v for k, v in existing.items() if k != "commit"} == {k: v for k, v in manifest.items() if k != "commit"}:
            return {"hash": hash_path.read_bytes().strip().decode(), "commit": _git(root, "rev-parse", "HEAD"),
                    "new": False, "unset": unset(config)}
        elif not replace:
            raise PreregError(f"a different pre-registration is in force ({hash_path.read_bytes().strip().decode()}); "
                              f"pass --replace to register anew (test runs made under it can no longer be scored)")
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_bytes(raw)
    hash_path.write_text(digest + "\n")
    files = (barrier.PREREG_MANIFEST, barrier.PREREG_HASH)
    _git(root, "add", "--", *files)
    _git(root, "commit", "-q", "-m", f"Pre-registration {digest}\n\nSPEC.md, {manifest['config_path']}, the splits, "
         f"the data manifest, the delta rule and the analysis code, before the test split is touched. Push to publish.", "--", *files)
    return {"hash": digest, "commit": _git(root, "rev-parse", "HEAD"), "new": True, "unset": unset(config)}
