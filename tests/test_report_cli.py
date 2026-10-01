"""F4's block of the `bench` command: each subcommand is wired to its function, a judgment's
refusal is exit code 2 with its reason, and a library bug is not swallowed as one."""
import pytest

from bench import cli
from fixtures.fake import call, repo, write_run


def loadtest(run_id, concurrency, p95_ms, rps):
    write_run(run_id, {"type": "loadtest", "engine": "slm:qwen3-8b", "gpu": "L4", "concurrency": concurrency,
                       "prefix_cache": True, "sweep_id": "sweep-1"},
              files={"profile_export_aiperf.json": {"request_latency": {"unit": "ms", "p95": p95_ms},
                                                    "request_throughput": {"unit": "requests/sec", "avg": rps}}})
    return run_id


def test_a_judgment_runs_and_prints_its_result(tmp_path, monkeypatch, capsys):
    config_path, _ = repo(tmp_path, monkeypatch, {"cost": {"p95_slo_cap_ms": 1000},
                                                  "modal": {"gpu_prices": {"as_of": "2026-09-30", "usd_per_s": {"L4": 0.001}}}})
    runs = [loadtest("lt-1", 1, 300, 2.0), loadtest("lt-8", 8, 800, 9.0)]
    write_run("agent-B0-calib", {"type": "agent", "arm": "B0", "split": "calib"},
              [call("agent-B0-calib", "1", "select_tables", latency_ms=500)])
    judge = ["judge", "j8", "--loadtest", runs[0], "--loadtest", runs[1], "--config", str(config_path)]
    with pytest.raises(SystemExit):  # the SLO rule needs the pilot's B0 run: there is no J8 without --slo-from
        cli.main(judge)
    capsys.readouterr()
    assert cli.main(judge + ["--slo-from", "agent-B0-calib"]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed.endswith("result.json") and "/judgments/J8/" in printed
    from bench.judge.base import read_result
    assert read_result(printed, "J8")["result"]["slo"]["slo_ms"] == 500.0  # the pilot's p95, stricter than the cap


def test_a_refusal_is_exit_code_2_with_its_reason(tmp_path, monkeypatch, capsys):
    config_path, _ = repo(tmp_path, monkeypatch)
    assert cli.main(["judge", "j3", "--run", "nowhere", "--config", str(config_path)]) == 2
    assert "no execution nowhere" in capsys.readouterr().err
    assert cli.main(["judge", "j6", "--zeroshot", "not-a-pair", "--teacher-eval", "e", "--config", str(config_path)]) == 2
    assert "expected RUN=EVAL" in capsys.readouterr().err


def test_every_f4_command_is_wired(tmp_path, monkeypatch):
    config_path, _ = repo(tmp_path, monkeypatch)
    seen = []
    for module, name in (("bench.curate", "run_curate"), ("bench.curate", "write_datasets"), ("bench.embed", "run_embed"),
                         ("bench.report", "run")):
        monkeypatch.setattr(__import__(module, fromlist=[name]), name, lambda *a, _n=name, **k: seen.append(_n) or "ok")
    for argv in (["curate", "--source", "r"], ["datasets", "--curated", "c", "--j5", "j"], ["embed", "--source", "r"],
                 ["report", "--plan", "p.yaml"]):
        assert cli.main(argv + ["--config", str(config_path)]) == 0
    assert seen == ["run_curate", "write_datasets", "run_embed", "run"]


def test_a_library_bug_is_not_reported_as_a_refusal(tmp_path, monkeypatch):
    config_path, _ = repo(tmp_path, monkeypatch)
    from bench.judge import j8
    monkeypatch.setattr(j8, "run", lambda *a, **k: {}["bug"])
    with pytest.raises(KeyError):
        cli.main(["judge", "j8", "--loadtest", "x", "--slo-from", "y", "--config", str(config_path)])
