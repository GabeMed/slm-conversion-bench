"""Acceptance (design §3.3, F3): AIPerf v0.13.0 runs against the local llama.cpp server with a handful of
payloads replayed from a run's calls, at low concurrency, and `runs/loadtest-*/` holds AIPerf's export as
it comes out plus the manifest. Skipped where AIPerf is not installed or the smoke server is not up
(scripts/serve-local.sh)."""
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import pytest

from bench import cli, paths
from synthetic import make_repo
from test_train_fixtures import TINY, save

LLAMACPP = "http://127.0.0.1:8080/v1"
CALLS = json.loads((Path(__file__).parent / "fixtures_calls.json").read_text())


def _llamacpp_serves(model):
    try:
        with urllib.request.urlopen(f"{LLAMACPP}/models", timeout=3) as response:
            return model in {m["id"] for m in json.loads(response.read())["data"]}
    except OSError:
        return False


def test_aiperf_replays_a_runs_calls_against_llamacpp(tmp_path, monkeypatch):
    """Also: the API key AIPerf is given appears in none of its artifacts (llama.cpp ignores it)."""
    if not (Path(sys.executable).parent / "aiperf").exists() and not shutil.which("aiperf"):
        pytest.skip("AIPerf is not installed (env/train)")
    _, config_path, config = make_repo(tmp_path, monkeypatch)
    if not _llamacpp_serves(config["roles"]["production_llm"]["model"]):
        pytest.skip("the llama.cpp smoke server is not up")
    config["roles"]["production_llm"]["params"]["max_tokens"] = 16
    config["roles"]["production_llm"]["endpoint"]["api_key_env"] = "F3_TEST_KEY"
    monkeypatch.setenv("F3_TEST_KEY", "sk-f3-secret-never-written")
    config["loadtest"].update({"concurrency": [2], "request_count": 4, "warmup_request_count": 1})
    save(config, config_path)
    source = paths.RUNS / "agent-B0-train-fixture"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({"run_id": source.name, "split": "train", "status": "done"}))
    (source / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CALLS))

    assert cli.main(["loadtest", "--config", str(config_path), "--engine", "production_llm", "--source", source.name,
                     "--on", "local", "--tokenizer", TINY["repo"], "--tokenizer-revision", TINY["revision"]]) == 0
    run_dir = next(paths.RUNS.glob("loadtest-production_llm-c2-*"))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "done" and manifest["aiperf_returncode"] == 0
    assert (manifest["concurrency"], manifest["source_calls"], manifest["where"]) == (2, 2, "local")
    assert manifest["payloads_repeat"] is True  # two source calls cannot fill 4 requests without repeating
    assert manifest["served_model"]["id"] == config["roles"]["production_llm"]["model"]
    assert manifest["endpoint_kind"] == "llamacpp" and manifest["gpu"] is None
    assert isinstance(manifest["observed"]["prompt_cache_read_pct"], (int, float))  # from llama.cpp's own usage
    export = json.loads((run_dir / "profile_export_aiperf.json").read_text())
    assert export["aiperf_version"].startswith("0.13.0")
    assert export["request_count"]["avg"] == 4 and export["was_cancelled"] is False
    assert export["request_latency"]["p95"] > 0 and export["request_throughput"]["avg"] > 0
    assert not export.get("error_summary")
    leaked = [p for p in run_dir.rglob("*") if p.is_file() and b"sk-f3-secret-never-written" in p.read_bytes()]
    assert not leaked
