"""`bench data`: download the pinned inputs, check their sha256, and write the hashes and splits.

Facts produced (committed): `data/MANIFEST.json` (the hash of every input and of every database
file) and `data/splits.json` (train, calib, test, excluded). The inputs themselves stay out of git.
"""
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

from bench import barrier, paths

SPLIT_NAMES = ("train", "calib", "test")
BIRD_DEV_SIZE = 1534
MINI_DEV_SIZE = 500


class DataError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(pin: dict, dest: Path) -> Path:
    """Download `pin["url"]` to `dest` unless present; either way the file must match the pin."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(pin["url"]) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.rename(dest)
    actual = sha256_file(dest)
    if actual != pin["sha256"]:
        raise DataError(f"{dest}: sha256 {actual} differs from the pin {pin['sha256']}")
    return dest


def _raw_path(name: str, pin: dict) -> Path:
    return paths.RAW / f"{name}{Path(pin['url']).suffix}"


def extract_databases(config: dict, zip_path: Path) -> Path:
    """Unpack BIRD's nested zip (dev.zip -> <root>/dev_databases.zip) once."""
    root = paths.bird_root(config)
    databases = root / "dev_databases"
    if databases.is_dir():
        return databases
    root.mkdir(parents=True, exist_ok=True)
    inner_name = f"{config['data']['bird_dev_databases']['root']}/dev_databases.zip"
    inner = paths.RAW / "dev_databases.zip"
    with zipfile.ZipFile(zip_path) as outer, outer.open(inner_name) as src, open(inner, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with zipfile.ZipFile(inner) as z:
        z.extractall(root, members=[m for m in z.namelist() if not m.startswith("__MACOSX")])
    inner.unlink()
    return databases


def hash_databases(databases: Path) -> Dict[str, dict]:
    hashes = {}
    for db_dir in sorted(p for p in databases.iterdir() if p.is_dir()):
        descriptions = db_dir / "database_description"
        hashes[db_dir.name] = {
            "sqlite": sha256_file(db_dir / f"{db_dir.name}.sqlite"),
            "descriptions": {p.name: sha256_file(p) for p in sorted(descriptions.glob("*.csv"))},
        }
    return hashes


def _normalize(item: dict) -> dict:
    """BIRD ids are int and Plat-SQL ids are str; evidence may be null. One shape for both."""
    return {**item, "question_id": str(item["question_id"]), "evidence": item.get("evidence") or ""}


def _read_pinned(config: dict, name: str) -> List[dict]:
    """A pinned JSON input, checked against its sha256 on every read (the gold comes from here)."""
    pin = config["data"][name]
    path = _raw_path(name, pin)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != pin["sha256"]:
        raise DataError(f"{path}: sha256 {actual} differs from the pin {pin['sha256']}")
    return [_normalize(item) for item in json.loads(raw)]


def _dev_questions(config: dict) -> List[dict]:
    """BIRD dev, which also holds the test ids: never read directly outside this module."""
    return _read_pinned(config, "bird_dev_questions")


def _test_questions(config: dict) -> List[dict]:
    """The raw test file. Only `bench data` reads it directly (for the ids of the splits); every
    execution goes through `questions_for`, which applies the test barrier."""
    return _read_pinned(config, "plat_sql_test")


def calib_sample(pool: List[str], size: int, seed: int) -> List[str]:
    """The calibration ids: the `size` ids of the pool with the smallest sha256("<seed>:<id>").
    Independent of the Python version, unlike `random.sample`."""
    ranked = sorted(pool, key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())
    return sorted(ranked[:size], key=int)


def build_splits(config: dict, dev: List[dict], test: List[dict]) -> dict:
    dev_ids = [q["question_id"] for q in dev]
    test_ids = [q["question_id"] for q in test]
    excluded = list(config["splits"]["excluded"])
    if len(set(dev_ids)) != BIRD_DEV_SIZE or len(dev_ids) != BIRD_DEV_SIZE:
        raise DataError(f"BIRD dev has {len(dev_ids)} items ({len(set(dev_ids))} unique), expected {BIRD_DEV_SIZE}")
    if len(set(test_ids)) != len(test_ids):
        raise DataError("duplicate ids in the test set")
    if set(test_ids) & set(excluded):
        raise DataError("an excluded id is in the test set")
    mini_dev = set(test_ids) | set(excluded)
    if len(mini_dev) != MINI_DEV_SIZE or not mini_dev <= set(dev_ids):
        raise DataError("test ids plus exclusions are not the 500 Mini-Dev ids inside BIRD dev")
    dev_db = {q["question_id"]: q["db_id"] for q in dev}
    mismatched = [q["question_id"] for q in test if dev_db[q["question_id"]] != q["db_id"]]
    if mismatched:
        raise DataError(f"db_id differs between BIRD dev and the test set for ids {mismatched[:10]}")
    pool = sorted(set(dev_ids) - mini_dev, key=int)
    calib = calib_sample(pool, config["splits"]["calib_size"], config["seeds"]["calib_split"])
    calib_set = set(calib)
    train = [q for q in pool if q not in calib_set]
    return {
        "seed": config["seeds"]["calib_split"],
        "train": train,
        "calib": calib,
        "test": sorted(test_ids, key=int),
        "excluded": sorted(excluded, key=int),
    }


def check_splits_unchanged(splits: dict) -> None:
    """The splits are a fact: rebuilding must give the committed file, or `bench data` stops."""
    if paths.SPLITS.exists() and json.loads(paths.SPLITS.read_text()) != splits:
        raise DataError("the rebuilt splits differ from the committed data/splits.json; "
                        "change them only on purpose, by deleting the file")


def run(config: dict) -> dict:
    """Fetch, check and pin every input; write data/MANIFEST.json and data/splits.json."""
    inputs = {}
    for name, pin in config["data"].items():
        fetch(pin, _raw_path(name, pin))
        inputs[name] = {"url": pin["url"], "sha256": pin["sha256"]}
    databases = extract_databases(config, _raw_path("bird_dev_databases", config["data"]["bird_dev_databases"]))
    db_hashes = hash_databases(databases)
    if paths.DATA_MANIFEST.exists():
        recorded = json.loads(paths.DATA_MANIFEST.read_text())["databases"]
        if recorded != db_hashes:
            changed = sorted(db for db in set(recorded) | set(db_hashes) if recorded.get(db) != db_hashes.get(db))
            raise DataError(f"database files differ from data/MANIFEST.json: {changed} "
                            f"(delete {paths.bird_root(config)} and run `bench data` again)")
    splits = build_splits(config, _dev_questions(config), _test_questions(config))
    check_splits_unchanged(splits)
    manifest = {"inputs": inputs, "databases": db_hashes}
    paths.DATA_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    paths.SPLITS.write_text(json.dumps(splits, indent=1) + "\n")
    return {"databases": len(db_hashes), **{k: len(splits[k]) for k in SPLIT_NAMES}}


def load_splits() -> dict:
    return json.loads(paths.SPLITS.read_text())


def questions_for(config: dict, split: str) -> Dict[str, dict]:
    """The questions of one split, by id. train and calib come from BIRD dev; test from Plat-SQL."""
    if split not in SPLIT_NAMES:
        raise DataError(f"unknown split {split!r}")
    barrier.ensure_split_allowed(split, config)
    ids = set(load_splits()[split])
    source = _test_questions(config) if split == "test" else _dev_questions(config)
    return {q["question_id"]: q for q in source if q["question_id"] in ids}


def check_database(config: dict, db_id: str) -> None:
    """The database file must be the one data/MANIFEST.json recorded."""
    expected = json.loads(paths.DATA_MANIFEST.read_text())["databases"][db_id]["sqlite"]
    actual = sha256_file(paths.sqlite_path(config, db_id))
    if actual != expected:
        raise DataError(f"{db_id}.sqlite sha256 {actual} differs from data/MANIFEST.json")
