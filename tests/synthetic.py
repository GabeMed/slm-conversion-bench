"""A synthetic repository shaped like the real one: one tiny BIRD-style database with column
descriptions, pinned question files, splits and a data manifest, under a temporary root that
`bench.paths` is pointed at. No download, no server."""
import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import yaml

from bench import paths
from bench.contracts.config import load_config

GOLD = "SELECT count(*) FROM gas_t WHERE country = 'CZE' AND segment = 'Premium'"
DESCRIPTION_HEADER = "original_column_name,column_name,column_description,data_format,value_description\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_db(db_dir: Path) -> Path:
    """dev_databases/tiny/: two tables (only gas_t answers the questions) and their descriptions."""
    (db_dir / "database_description").mkdir(parents=True)
    path = db_dir / "tiny.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE gas_t (id INTEGER PRIMARY KEY, country TEXT, segment TEXT)")
    connection.execute("CREATE TABLE other_u (id INTEGER PRIMARY KEY, note TEXT)")
    connection.executemany("INSERT INTO gas_t VALUES (?, ?, ?)",
                           [(1, "CZE", "Premium"), (2, "CZE", "Value"), (3, "SVK", "Premium")])
    connection.execute("INSERT INTO other_u VALUES (1, 'unrelated')")
    connection.commit()
    connection.close()
    (db_dir / "database_description" / "gas_t.csv").write_text(
        DESCRIPTION_HEADER + "id,id,station id,integer,\ncountry,country,country code,text,\nsegment,segment,price segment,text,\n")
    (db_dir / "database_description" / "other_u.csv").write_text(
        DESCRIPTION_HEADER + "id,id,note id,integer,\nnote,note,a note,text,\n")
    return path


def make_repo(tmp_path: Path, monkeypatch, base_config: str = "configs/smoke-local.yaml"):
    """Returns (root, config path, config). Train ids 1-3 and test id 9, all on database `tiny`."""
    for name, value in {"ROOT": tmp_path, "DATA": tmp_path / "data", "RAW": tmp_path / "data" / "raw",
                        "SPLITS": tmp_path / "data" / "splits.json", "RUNS": tmp_path / "runs",
                        "DATA_MANIFEST": tmp_path / "data" / "MANIFEST.json"}.items():
        monkeypatch.setattr(paths, name, value)
    config = copy.deepcopy(load_config(Path(__file__).resolve().parent.parent / base_config))
    db_path = make_db(paths.bird_root(config) / "dev_databases" / "tiny")
    paths.RAW.mkdir(parents=True)
    dev = [{"question_id": q, "db_id": "tiny", "question": "How many gas stations in CZE are Premium?",
            "evidence": None, "difficulty": "simple", "SQL": GOLD} for q in (1, 2, 3)]
    test = [{"question_id": "9", "db_id": "tiny", "question": "?", "evidence": "", "SQL": "SELECT 9"}]
    mini_dev = [{"question_id": 9, "db_id": "tiny", "difficulty": "simple"}]  # the test ids' difficulty
    for name, items in (("bird_dev_questions", dev), ("plat_sql_test", test), ("mini_dev", mini_dev)):
        path = paths.RAW / f"{name}.json"
        path.write_text(json.dumps(items))
        config["data"][name]["sha256"] = sha256(path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    paths.SPLITS.write_text(json.dumps({"train": ["1", "2", "3"], "calib": [], "test": ["9"], "excluded": []}))
    paths.DATA_MANIFEST.write_text(json.dumps({"inputs": {}, "databases": {
        "tiny": {"sqlite": sha256(db_path), "descriptions": {}}}}))
    return tmp_path, config_path, load_config(config_path)
