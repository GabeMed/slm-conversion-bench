"""`bench data`: download the pinned inputs, check their sha256, and write the hashes and splits.

Facts produced (committed): `data/MANIFEST.json` (the hash of every input and of every database
file; `gold_check`, the golds that fail with an execution error on those databases, which
reproduces on any machine and is guarded; `gold_observed`, what the recording machine saw, never
compared) and `data/splits.json` (train, calib, test, excluded). The inputs themselves stay out of git. The test questions carry the difficulty of
the Mini-Dev, which Arcwise-Plat-SQL does not have.
"""
import hashlib
import json
import shutil
import sqlite3
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

from bench import barrier, paths

SPLIT_NAMES = ("train", "calib", "test")
BIRD_DEV_SIZE = 1534
MINI_DEV_SIZE = 500
DIFFICULTIES = ("simple", "moderate", "challenging")


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
    """The raw test file, with the Mini-Dev difficulty. Only `bench data` reads it directly (for the
    ids of the splits and the gold check); every execution goes through `questions_for`, which
    applies the test barrier."""
    return attach_difficulty(_read_pinned(config, "plat_sql_test"), _read_pinned(config, "mini_dev"))


def attach_difficulty(test: List[dict], mini_dev: List[dict]) -> List[dict]:
    """Each test question with the difficulty the Mini-Dev gives its id (BIRD dev's own label
    differs for some ids, so the source is pinned)."""
    difficulty = {q["question_id"]: q["difficulty"] for q in mini_dev}
    missing = [q["question_id"] for q in test if q["question_id"] not in difficulty]
    if missing:
        raise DataError(f"test ids with no Mini-Dev difficulty: {missing[:10]}")
    return [{**q, "difficulty": difficulty[q["question_id"]]} for q in test]


def check_mini_dev(test: List[dict], excluded: List[str], mini_dev: List[dict]) -> None:
    """The test ids plus the exclusions are exactly the Mini-Dev ids (`build_splits` checks there
    are 500 of them), on the same databases."""
    ids = [q["question_id"] for q in mini_dev]
    if len(set(ids)) != len(ids):
        raise DataError("duplicate ids in the Mini-Dev")
    expected = {q["question_id"] for q in test} | set(excluded)
    if set(ids) != expected:
        raise DataError(f"test ids plus exclusions differ from the Mini-Dev ids: only in the Mini-Dev "
                        f"{sorted(set(ids) - expected)[:10]}, only in test or exclusions {sorted(expected - set(ids))[:10]}")
    mini_db = {q["question_id"]: q["db_id"] for q in mini_dev}
    mismatched = [q["question_id"] for q in test if mini_db[q["question_id"]] != q["db_id"]]
    if mismatched:
        raise DataError(f"db_id differs between the Mini-Dev and the test set for ids {mismatched[:10]}")
    unknown = sorted({q["difficulty"] for q in mini_dev} - set(DIFFICULTIES))
    if unknown:
        raise DataError(f"unknown difficulty in the Mini-Dev: {unknown}")


def calib_sample(pool: List[str], size: int, seed: int) -> List[str]:
    """The calibration ids: the `size` ids of the pool with the smallest sha256("<seed>:<id>").
    Independent of the Python version, unlike `random.sample`."""
    ranked = sorted(pool, key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())
    return sorted(ranked[:size], key=int)


def pilot_sample(calib: List[str], size: int, seed: int) -> List[str]:
    """The pilot (SPEC 6.4: the calibration questions that measure d before the test): the `size`
    calibration ids with the smallest sha256("<seed>:pilot:<id>"), a draw of its own."""
    if not 0 < size <= len(calib):
        raise DataError(f"a pilot of {size} ids needs that many calibration ids, there are {len(calib)}")
    ranked = sorted(calib, key=lambda q: hashlib.sha256(f"{seed}:pilot:{q}".encode()).hexdigest())
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
    from bench.evaluate import check_golds, fixed_date  # evaluate imports this module
    dev, test = _dev_questions(config), _test_questions(config)
    splits = build_splits(config, dev, test)
    check_mini_dev(test, config["splits"]["excluded"], _read_pinned(config, "mini_dev"))
    check_splits_unchanged(splits)
    by_id = {q["question_id"]: q for q in dev}
    questions = {"train": [by_id[q] for q in splits["train"]], "calib": [by_id[q] for q in splits["calib"]],
                 "test": sorted(test, key=lambda q: int(q["question_id"]))}
    check = check_golds(questions, lambda db_id: paths.sqlite_path(config, db_id),
                        config["eval"]["timeout_s"], fixed_date(config))
    registered, observed = check["registered"], check["observed"]
    recorded = json.loads(paths.DATA_MANIFEST.read_text()) if paths.DATA_MANIFEST.exists() else {}
    if "gold_check" in recorded:
        if _gold_identity(recorded["gold_check"]) != _gold_identity(registered):
            then = (recorded.get("gold_observed") or {}).get("sqlite_version")
            raise DataError(f"the gold check differs from data/MANIFEST.json: (fixed date, golds checked, golds failing "
                            f"with an execution error) {_gold_identity(recorded['gold_check'])} recorded, "
                            f"{_gold_identity(registered)} now; SQLite {then} then, {sqlite3.sqlite_version} now. "
                            f"data/MANIFEST.json is left as recorded. If the SQLite here lacks what the golds use, "
                            f"use another; if the data changed on purpose, remove gold_check and gold_observed to "
                            f"record them anew: the pre-registration hashes this file")
        # the same fact: keep the file byte for byte (an error's wording and this machine's observations may differ)
        registered, observed = recorded["gold_check"], recorded.get("gold_observed", observed)
    manifest = {"inputs": inputs, "databases": db_hashes, "gold_check": registered, "gold_observed": observed}
    paths.DATA_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    paths.SPLITS.write_text(json.dumps(splits, indent=1) + "\n")
    return {"databases": len(db_hashes), **{k: len(splits[k]) for k in SPLIT_NAMES},
            "gold_errors": len(registered["errors"]),  # this machine's observations, as it ran:
            "gold_timeouts": len(check["observed"]["timeouts"]), "gold_empty": len(check["observed"]["empty"])}


def _gold_identity(gold_check: dict) -> tuple:
    """What the gold check registers: the date, the counts, and which golds fail with an error."""
    errors = gold_check.get("errors")
    return (gold_check.get("fixed_date"), gold_check.get("checked"),
            None if errors is None else sorted((e["split"], e["question_id"]) for e in errors))


def load_splits() -> dict:
    return json.loads(paths.SPLITS.read_text())


def pilot_ids(config: dict) -> List[str]:
    """The pilot questions (design §6.2: their only accessor). PROVISIONAL, until Front D's stratified
    pilot lands: today's `pilot_sample` over the calib split, with stats.pilot_size and seeds.calib_split."""
    return pilot_sample(load_splits()["calib"], config["stats"]["pilot_size"], config["seeds"]["calib_split"])


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
