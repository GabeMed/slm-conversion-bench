"""`bench verify`: every stored judgment the report or an arm's fact reads is recomputed from its
`reads` with today's code and configuration. An untouched world is clean; a result that was edited,
fabricated, computed with another configuration or code, or left without what it read is listed,
and the command fails."""
import copy
import json

import pytest
import yaml

from bench import cli, data, paths, verify
from bench.contracts import facts
from bench.judge import j2, j3, j5, j6, j7, j8
from bench.judge.base import read_result, relative, write_result
from fixtures.fake import call, repo, usage, write_run
from fixtures.world import GOLD_SITES, gold_correct, per_call_eval, replay, teacher, trained_on

QUESTIONS = [str(q) for q in range(1000, 1200)]
PRICES = {"as_of": "2026-09-30", "table": {
    "teacher-model": {"input_per_mtok": 2.0, "cached_input_per_mtok": 0.5, "output_per_mtok": 8.0, "batch_discount": 0.5},
    "engine-model": {"input_per_mtok": 0.5, "cached_input_per_mtok": 0.05, "output_per_mtok": 1.5, "batch_discount": 0.5}}}


def loadtest(run_id, concurrency, p95_ms, rps):
    write_run(run_id, {"type": "loadtest", "engine": "slm:qwen3-8b", "gpu": "L4", "concurrency": concurrency,
                       "prefix_cache": True, "sweep_id": "sweep-1"},
              files={j8.EXPORT: {"request_latency": {"unit": "ms", "p95": p95_ms},
                                 "request_throughput": {"unit": "requests/sec", "avg": rps}}})
    return run_id


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The judgments of a small benchmark, each written by its own `run`: S4's choice (J6), the load
    test (J8), the allocation built on both (J7), an SLM arm's cost built on J8 (J3) and its format
    (J2), and a per-call comparison (J2); and the report plan that names them."""
    config_path, config = repo(tmp_path, monkeypatch, {
        "prices": PRICES, "roles": {"production_llm": {"model": "teacher-model"}},
        "allocation": {"min_calls": 5}, "stats": {"n_boot": 100}, "cost": {"p95_slo_cap_ms": 1000},
        "modal": {"gpu_prices": {"as_of": "2026-09-30", "usd_per_s": {"L4": 0.8 / 3600}}}})
    # what J7 reads beside its arguments: the size of the test split and the registered pilot
    monkeypatch.setattr(data, "load_splits", lambda: {"test": [str(q) for q in range(500)]})
    monkeypatch.setattr(j7, "registered_pilot", lambda config: QUESTIONS[:50])

    t = teacher("agent-B0-calib", "calib", QUESTIONS)  # every call takes 100 ms: the SLO is 100 ms, under the cap
    write_run("agent-B0-calib", {"type": "agent", "arm": "B0", "split": "calib", "question_ids": QUESTIONS}, t)
    teacher_ok = gold_correct("t", 0.9)
    per_call_eval("eval-t", "agent-B0-calib", t, teacher_ok)
    zeroshots = {}
    for name, quality in (("qwen3-8b", 0.9), ("granite-4.2-8b", 0.5)):
        calls = replay(t, f"zeroshot-{name}", f"slm:{name}", quality)
        write_run(f"zeroshot-{name}", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": f"slm:{name}",
                                       "split": "calib"}, calls)
        zeroshots[f"zeroshot-{name}"] = per_call_eval(f"eval-zeroshot-{name}", f"zeroshot-{name}", calls,
                                                      lambda c, n=name, q=quality: teacher_ok(c) and gold_correct(n, q)(c))
    j6_path, choice = j6.run(zeroshots, "eval-t", config)
    j8_path = j8.run([loadtest("lt-4", 4, 60, 5.0), loadtest("lt-8", 8, 80, 12.0)], config, slo_from="agent-B0-calib")

    embedding = {"model": "fake-embedder", "revision": "0" * 40, "max_seq_length": 64, "truncation": "tail", "text": "prompt"}
    centroids = facts.write_fact("J5", "centroids", {"embedding": embedding, "clusters": {"c0": [1.0, 0.0], "c1": [0.0, 1.0]}})
    adapters = facts.write_fact("S5", "adapters", {
        "slm": "qwen3-8b", "choice": choice.parent.name, "centroids": centroids.parent.name, **trained_on(config),
        "adapters": {c: {"served_name": f"qwen3-8b-{c}", "sha256": c[1] * 64} for c in ("c0", "c1")}})
    b4 = replay(t, "replay-b4", "slm:qwen3-8b+lora:x", 1.0, cluster_of=lambda c: "c0" if c["call_site"] in GOLD_SITES else "c1")
    write_run("replay-b4", {"type": "replay", "source_run_id": "agent-B0-calib", "arm": "B4", "split": "calib",
                            "facts": {"choice": choice.parent.name, "centroids": centroids.parent.name,
                                      "adapters": adapters.parent.name}}, b4)
    per_call_eval("eval-b4", "replay-b4", b4, lambda c: teacher_ok(c) and gold_correct("slm", 0.5)(c))
    cheap = replay(t, "replay-cheap", "cheap_alt", 0.9)
    write_run("replay-cheap", {"type": "replay", "source_run_id": "agent-B0-calib", "engine": "cheap_alt", "split": "calib"}, cheap)
    per_call_eval("eval-cheap", "replay-cheap", cheap, teacher_ok)
    j7_path, allocation = j7.run(str(centroids), str(adapters), {"cheap_alt": ("replay-cheap", "eval-cheap"),
                                                                "slm": ("replay-b4", "eval-b4")},
                                 "eval-t", str(j8_path), str(j6_path), config)

    slm_calls = [call("agent-B3", q, "select_tables", parsed={}, role="slm", engine="slm:qwen3-8b", model="qwen3-8b",
                      use=usage(1000, 0, 10)) for q in ("1", "2")]
    write_run("agent-B3", {"type": "agent", "arm": "B3", "split": "calib", "question_ids": ["1", "2"]}, slm_calls)
    write_run("eval-B3", {"type": "eval", "source_run_id": "agent-B3", "status": None},
              files={"results.jsonl": [{"question_id": "1", "correct": True}, {"question_id": "2", "correct": False}]})
    results = {"j6": j6_path, "j7": j7_path, "j8": j8_path, "cost": j3.run("agent-B3", "eval-B3", str(j8_path), config),
               "format": j2.run("agent-B3"), "per_call": j2.run(None, "replay-cheap", "eval-cheap", "eval-t")}
    plan = {"split": "calib", "arms": {"B3": {"eval": "eval-B3", "cost": relative(results["cost"])}},
            "format": {"B3": relative(results["format"])}, "per_call": relative(results["per_call"]),
            "j6": relative(j6_path), "j7": relative(j7_path), "j8": relative(j8_path)}
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan))
    return {"config": config, "config_path": config_path, "plan": plan, "plan_path": plan_path, "results": results,
            "choice": choice, "allocation": allocation}


def stored(root=None):
    return sorted(str(p) for p in (paths.ROOT / "judgments").rglob("*.json"))


def test_an_untouched_world_verifies_clean_and_writes_nothing(world, capsys):
    before = stored()
    verified, divergences = verify.verify(world["config"], world["plan"])
    assert divergences == []
    assert sorted(verified) == sorted(relative(path) for path in world["results"].values())  # J8 once, though read three times
    assert stored() == before  # every recomputation landed on the stored bytes
    assert cli.main(["verify", "--plan", str(world["plan_path"]), "--config", str(world["config_path"])]) == 0
    printed = capsys.readouterr()
    assert printed.out == "" and "6 judgment(s) recomputed to the stored bytes, 0 divergence(s)" in printed.err


def test_a_result_edited_in_place_is_listed_and_the_command_fails(world, capsys):
    path = world["results"]["j8"]
    raw = path.read_bytes()
    assert b'"slo_ms":100.0' in raw
    path.write_bytes(raw.replace(b'"slo_ms":100.0', b'"slo_ms":999.0'))
    verified, divergences = verify.verify(world["config"], world["plan"])
    assert any(line.startswith(f"{relative(path)}: ") and "content sha256 is not its directory name" in line for line in divergences)
    # what was built on the edited result cannot be recomputed on it either: the J3 cost and the J7 allocation
    assert sorted(verified) == sorted(relative(world["results"][key]) for key in ("j6", "format", "per_call"))
    assert cli.main(["verify", "--plan", str(world["plan_path"]), "--config", str(world["config_path"])]) == 1
    printed = capsys.readouterr()
    assert relative(path) in printed.out and "3 divergence(s)" in printed.err


def test_a_fabricated_result_does_not_come_out_of_its_own_reads(world):
    genuine = read_result(world["results"]["j8"], "J8")
    cheaper = {**genuine["result"], "cost_per_request": {u: c / 2 for u, c in genuine["result"]["cost_per_request"].items()}}
    forged = write_result("J8", genuine["reads"], cheaper)  # a well-formed result: its sha256 names its directory
    verified, divergences = verify.verify(world["config"], {**world["plan"], "j8": relative(forged)})
    assert divergences == [f"{relative(forged)}: recomputed as {relative(world['results']['j8'])} "
                           f"(read {', '.join(j8.CONFIG_KEYS)} of the configuration)"]
    assert relative(world["results"]["j8"]) in verified  # the genuine one, which the J3 cost and J7 built on


def test_a_result_of_another_configuration_or_code_is_listed(world, monkeypatch):
    other = copy.deepcopy(world["config"])
    other["serving"]["memory_gib"] = 64  # a key J8 read: the container costs more
    before = stored()
    verified, divergences = verify.verify(other, world["plan"])
    assert len(divergences) == 1 and divergences[0].startswith(f"{relative(world['results']['j8'])}: recomputed as judgments/J8/")
    assert "serving.memory_gib" in divergences[0]
    assert len(stored()) == len(before) + 1  # the recomputed result, written beside the stored one to compare
    assert len(verified) == 5  # the others read the stored J8, which is still the bytes they were built on
    monkeypatch.setattr(j2, "truncation", lambda calls: {})  # the code changed since the result was stored
    _, divergences = verify.verify(world["config"], world["plan"])
    assert [line.split(": ")[0] for line in divergences] == [relative(world["results"]["format"])]


def test_the_judgments_behind_the_arms_facts_are_verified(world):
    config = copy.deepcopy(world["config"])
    config["arms"]["B3"] = {"choice": relative(world["choice"])}
    config["arms"]["B5"]["allocation"] = relative(world["allocation"])
    only_j8 = {"split": "calib", "j8": world["plan"]["j8"]}
    verified, divergences = verify.verify(config, only_j8)  # the plan does not name J6 or J7: the facts lead to them
    assert divergences == [] and sorted(verified) == sorted(relative(world["results"][key]) for key in ("j6", "j7", "j8"))
    orphan = facts.write_fact("J6", "choice", {"slm": "granite-4.2-8b"})  # a fact no stored judgment decided
    config["arms"]["B3"]["choice"] = relative(orphan)
    _, divergences = verify.verify(config, only_j8)
    assert divergences == [f"arms.B3.choice ({relative(orphan)}): no stored J6 result wrote this fact"]
    orphan.write_text(json.dumps({"slm": "qwen3-8b"}))  # a fact edited in place
    _, divergences = verify.verify(config, only_j8)
    assert len(divergences) == 1 and "content sha256" in divergences[0] and divergences[0].startswith("arms.B3.choice")


def test_a_result_that_does_not_say_what_it_read_cannot_be_verified(world, capsys):
    bare = write_result("J8", {}, {"engine": "slm:qwen3-8b", "cost_per_request": {"100%": 0.001}})
    _, divergences = verify.verify(world["config"], {"j8": relative(bare)})
    assert len(divergences) == 1 and "its reads do not name what J8 needs to be recomputed" in divergences[0]
    missing = f"judgments/J8/{'0' * 64}/result.json"
    _, divergences = verify.verify(world["config"], {"j8": missing})
    assert len(divergences) == 1 and divergences[0].startswith(f"{missing}: ") and "No such file" in divergences[0]
    gone = write_result("J2", {"run": {"run_id": "nowhere", "manifest_sha256": "0" * 64}, "config": []}, {"mode": "run"})
    _, divergences = verify.verify(world["config"], {"format": {"B0": relative(gone)}})
    assert divergences == [f"{relative(gone)}: cannot be recomputed: no execution nowhere (runs/nowhere/manifest.json is missing)"]
    assert cli.main(["verify", "--plan", "no-plan.yaml", "--config", str(world["config_path"])]) == 2
    assert "bench verify: " in capsys.readouterr().err


def test_j5_is_recomputed_from_the_executions_it_clustered(tmp_path, monkeypatch):
    from test_j5 import build
    config, _, train, calib, _ = build(tmp_path, monkeypatch)
    path, _ = j5.run("curate-t", train, calib, config)
    assert verify.verify(config, {"j5": relative(path)}) == ([relative(path)], [])
    narrower = copy.deepcopy(config)
    narrower["clustering"]["k_max"] = 2  # a key J5 read: three call sites no longer get three clusters
    _, divergences = verify.verify(narrower, {"j5": relative(path)})
    assert len(divergences) == 1 and "clustering.k_max" in divergences[0]
