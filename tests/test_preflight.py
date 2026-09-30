"""`bench preflight`: every SPEC 7.1 check with its action, and P-4 proven against fake vLLM endpoints (one
that passes, one where the adapter changes nothing, one that diverges from HF-PEFT). The real P-4 runs on
the morning of Day 1."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bench import cli, paths
from bench.preflight import (FAIL, PASS, PENDING, UNDECIDED, PreflightError, check_agent_runs, check_call_sites, check_data,
                             check_lora_parity, check_pilot_spend, check_teacher_terms, check_throughput,
                             check_training_time, compare, lora_parity, parity_verdict, served_generate)
from test_train_fixtures import TINY, TINY_NAME, fake_adapter, make_s5_repo, save, served

STOP = [2]


def gen(tokens, alternatives=(99,)):
    """A generation whose top-k at each position is the chosen token and the alternatives."""
    return {"tokens": list(tokens), "top": [[t, *alternatives] for t in tokens]}


# ---------------------------------------------------------------- the parity rule

def test_identical_generations_match_exactly():
    assert compare(gen([5, 6, 7]), gen([5, 6, 7]), STOP) == {"exact": True, "close": True, "first_difference": None}


def test_a_numerical_near_tie_at_the_first_difference_still_matches():
    served = {"tokens": [5, 8, 9], "top": [[5], [8, 6], [9]]}
    reference = {"tokens": [5, 6, 7], "top": [[5], [6, 8], [7]]}
    assert compare(served, reference, STOP) == {"exact": False, "close": True, "first_difference": 1}


def test_a_token_outside_the_other_sides_top_k_is_a_divergence():
    served = {"tokens": [5, 8], "top": [[5], [8, 6]]}
    reference = {"tokens": [5, 6], "top": [[5], [6, 3]]}  # 8 is not among the reference's candidates
    assert compare(served, reference, STOP)["close"] is False


def test_stop_tokens_end_a_generation_and_an_early_stop_is_judged_as_a_near_tie():
    assert compare(gen([5, 6, 2, 0, 0]), gen([5, 6, 2]), STOP)["exact"] is True
    stops_early = {"tokens": [5, 2], "top": [[5], [2, 6]]}
    continues = {"tokens": [5, 6, 7], "top": [[5], [6, 2], [7]]}
    assert compare(stops_early, continues, STOP) == {"exact": False, "close": True, "first_difference": 1}
    without_top = {"tokens": [5], "top": [[5]]}  # the server gave no top-k where it stopped: not shown close
    assert compare(without_top, continues, STOP)["close"] is False


def _four(n, served_adapter, ref_adapter=None):
    base = [gen([10 + i, 20, 30]) for i in range(n)]
    adapter = [gen([40 + i, 50, 60]) for i in range(n)]
    return base, served_adapter(base, adapter), base, ref_adapter(base, adapter) if ref_adapter else adapter


def test_verdict_pass_when_the_adapter_changes_the_output_and_matches_peft():
    verdict = parity_verdict(*_four(3, lambda b, a: a), STOP)
    assert verdict["status"] == PASS and verdict["adapter_changes_output"] == 3 and verdict["adapter_matches_peft"] == 3


def test_verdict_when_vllm_ignores_the_adapter():
    verdict = parity_verdict(*_four(3, lambda b, a: b), STOP)
    assert verdict["status"] == FAIL and "ignores the adapter" in verdict["diagnosis"]
    assert "reserve" in verdict["action"]


def test_verdict_when_the_served_adapter_diverges_from_peft():
    verdict = parity_verdict(*_four(3, lambda b, a: [gen([40 + i, 51, 61]) for i in range(3)]), STOP)
    assert verdict["status"] == FAIL and "diverges from HF-PEFT" in verdict["diagnosis"]
    assert "reserve" in verdict["action"] and verdict["base_matches_hf"] == 3


def test_near_tie_changes_only_leave_p4_undecided_never_switch_the_base():
    """The served adapter equals the base everywhere; HF-PEFT's changes are near-ties, so every prompt
    'matches' by the top-k rule. That evidence cannot tell an applied adapter from an ignored one."""
    base = [{"tokens": [10, 20], "top": [[10, 40], [20]]}]
    adapter = [{"tokens": [40, 50], "top": [[40, 10], [50]]}]
    verdict = parity_verdict(base, base, base, adapter, STOP)
    assert verdict["adapter_matches_peft"] == 1 and verdict["status"] == UNDECIDED
    assert "near-ties" in verdict["diagnosis"] and "not to be switched" in verdict["action"]
    assert "reserve" not in verdict["action"]


def test_the_reviewers_near_tie_fixture_does_not_pass():
    """The server changes the output and 'matches' PEFT by the top-k rule, but no change is decisive."""
    ref_base = {"tokens": [10, 20], "top": [[10, 40], [20, 21]]}
    ref_adapter = {"tokens": [40, 50], "top": [[40, 10], [50]]}
    served_adapter = {"tokens": [10, 21], "top": [[10, 40], [21, 20]]}
    verdict = parity_verdict([ref_base], [served_adapter], [ref_base], [ref_adapter], STOP)
    assert verdict["adapter_changes_output"] == 1 and verdict["adapter_matches_peft"] == 1  # the old rule passed
    assert verdict["status"] == UNDECIDED and verdict["reproduced_decisive_changes"] == 0


def test_verdict_when_vllm_applies_the_adapter_on_only_some_decisive_prompts():
    base = [gen([10 + i, 20]) for i in range(3)]
    adapter = [gen([40 + i, 50]) for i in range(3)]
    served = [adapter[0], base[1], base[2]]  # changes one prompt of three where PEFT's change is decisive
    verdict = parity_verdict(base, served, base, adapter, STOP)
    assert verdict["status"] == FAIL and verdict["ignored_where_peft_is_decisive"] == 2
    assert "on 2 prompt(s)" in verdict["diagnosis"]


def test_an_ignored_adapter_is_caught_even_where_the_servers_numerics_make_it_look_close():
    """Prompt 0: the adapter is applied. Prompt 1: the served adapter is the served base, whose own top-k
    happens to hold PEFT's token (so the top-k rule calls it close), while HF-PEFT's change is decisive."""
    ref_base = [gen([10, 20]), {"tokens": [11, 21], "top": [[11, 99], [21]]}]
    ref_adapter = [gen([40, 50]), {"tokens": [41, 51], "top": [[41, 11], [51]]}]
    served_base = [gen([10, 20]), {"tokens": [11, 21], "top": [[11, 41], [21]]}]
    served_adapter = [gen([40, 50]), served_base[1]]
    verdict = parity_verdict(served_base, served_adapter, ref_base, ref_adapter, STOP)
    assert verdict["adapter_matches_peft"] == 2 and verdict["adapter_changes_output"] == 1  # the top-k rule alone passes
    assert verdict["status"] == FAIL and verdict["ignored_where_peft_is_decisive"] == 1


def test_verdict_when_the_serving_itself_disagrees_does_not_blame_the_lora():
    base = [gen([10, 20]), gen([11, 21])]
    other = [gen([70, 80]), gen([71, 81])]
    adapter = [gen([40, 50]), gen([41, 51])]
    verdict = parity_verdict(other, [gen([90, 91]), gen([92, 93])], base, adapter, STOP)
    assert verdict["status"] == FAIL and "even without the adapter" in verdict["diagnosis"]
    assert "not decided" in verdict["action"]


def test_undecided_still_reports_a_divergence_it_saw():
    """No decisive HF-PEFT change (undecided, as decided), but the served adapter diverges from HF-PEFT
    beyond the near-tie rule while the base matches: the diagnosis says so."""
    ref_base = {"tokens": [10, 20], "top": [[10, 40], [20]]}
    ref_adapter = {"tokens": [40, 50], "top": [[40, 10], [50]]}
    served_adapter = gen([77, 78])  # neither side's token in the other's top-k
    verdict = parity_verdict([ref_base], [served_adapter], [ref_base], [ref_adapter], STOP)
    assert verdict["status"] == UNDECIDED and verdict["adapter_matches_peft"] == 0
    assert "diverges from HF-PEFT on 1 prompt(s)" in verdict["diagnosis"]


def test_an_adapter_that_changes_nothing_even_in_peft_leaves_p4_undecided():
    base = [gen([10, 20])]
    verdict = parity_verdict(base, base, base, base, STOP)
    assert verdict["status"] == UNDECIDED and "changes nothing" in verdict["diagnosis"]
    assert "not decided" in verdict["action"]


def test_pass_needs_a_decisive_change_the_server_reproduces():
    verdict = parity_verdict(*_four(2, lambda b, a: a), STOP)
    assert verdict["status"] == PASS and verdict["reproduced_decisive_changes"] == 2


def test_the_verdict_needs_four_generations_per_prompt():
    with pytest.raises(PreflightError):
        parity_verdict([gen([1])], [], [gen([1])], [gen([1])], STOP)


# ---------------------------------------------------------------- P-4 against fake vLLM endpoints

class FakeVLLM:
    """An OpenAI-compatible chat endpoint answering greedily from a script {model: [tokens per prompt]},
    with top-k logprobs as vLLM's `return_tokens_as_token_ids` gives them."""

    def __init__(self, script, prompts, cards=None):
        self.script, self.prompts, self.requests, self.cards = script, prompts, [], cards or []
        self.token_name = lambda t: f"token_id:{t}"  # vLLM with return_tokens_as_token_ids
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                fake.requests.append({"path": self.path, "headers": dict(self.headers), "body": None})
                body = json.dumps({"object": "list", "data": fake.cards}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                i = fake.prompts.index(body["messages"])
                generation = fake.script[body["model"]][i]
                name = fake.token_name
                content = [{"token": name(t), "logprob": -0.1,
                            "top_logprobs": [{"token": name(x), "logprob": -1.0} for x in top]}
                           for t, top in zip(generation["tokens"], generation["top"])]
                answer = json.dumps({"choices": [{"message": {"role": "assistant", "content": "x"},
                                                  "logprobs": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def parity_repo(tmp_path, monkeypatch):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    fake_adapter("c0", config)
    prompts = [json.loads(line)["prompt"] for line in (paths.ROOT / "train/datasets/c0.jsonl").read_text().splitlines()]
    return config_path, config, prompts


def cards(adapter_sha256, root=None, base_root=TINY["repo"], name=None):
    """/v1/models as vLLM lists the base and an adapter (vllm/entrypoints/openai/models/serving.py), the
    adapter under its content-addressed name."""
    return [{"id": TINY_NAME, "object": "model", "root": base_root},
            {"id": name or f"c0-{adapter_sha256[:12]}", "object": "model", "root": root or f"/adapters/{adapter_sha256}",
             "parent": TINY_NAME}]


def _sha(name="c0"):
    return json.loads((paths.ROOT / f"train/adapters/{name}/manifest.json").read_text())["adapter_sha256"]


def _reference(prompts):
    calls = []

    def reference(base, got_prompts, template_kwargs, max_new_tokens, top_k):
        calls.append((base, got_prompts, template_kwargs, max_new_tokens, top_k))
        return {"stop_ids": STOP, "base": [gen([10 + i, 20, 30]) for i in range(len(prompts))],
                "adapter": [gen([40 + i, 50, 60]) for i in range(len(prompts))]}
    return reference, calls


SERVED = {
    "passes": lambda i: gen([40 + i, 50, 60]),
    "adapter changes nothing": lambda i: gen([10 + i, 20, 30]),
    "diverges from PEFT": lambda i: gen([40 + i, 51, 61]),
}


@pytest.mark.parametrize("fake, status, diagnosis", [
    ("passes", PASS, "reproduces HF-PEFT"),
    ("adapter changes nothing", FAIL, "vLLM ignores the adapter"),
    ("diverges from PEFT", FAIL, "diverges from HF-PEFT"),
])
def test_p4_against_a_fake_vllm(parity_repo, monkeypatch, fake, status, diagnosis):
    config_path, config, prompts = parity_repo
    server = FakeVLLM({TINY_NAME: [gen([10 + i, 20, 30]) for i in range(len(prompts))],
                       served("c0"): [SERVED[fake](i) for i in range(len(prompts))]}, prompts, cards(_sha()))
    try:
        config["roles"]["slm_candidates"][-1]["endpoint"].update({
            "base_url": server.base_url, "api_key_env": "TEST_SLM_KEY",
            "headers_env": {"Modal-Key": "TEST_MODAL_KEY", "Modal-Secret": "TEST_MODAL_SECRET"}})
        monkeypatch.setenv("TEST_SLM_KEY", "k-123")
        monkeypatch.setenv("TEST_MODAL_KEY", "mk-1")
        monkeypatch.setenv("TEST_MODAL_SECRET", "ms-1")
        reference, calls = _reference(prompts)
        verdict = lora_parity(save(config, config_path), "c0", "local", reference=reference)
    finally:
        server.close()
    assert verdict["status"] == status and diagnosis in verdict["diagnosis"]
    assert verdict["served"] == {"base": TINY_NAME, "adapter": served("c0")} and verdict["n_prompts"] == len(prompts)
    # the reference renders the same prompts with the serving kwargs, from the pinned base
    settings = config["preflight"]["lora_parity"]
    assert calls == [(TINY, prompts, {"enable_thinking": False}, settings["max_new_tokens"], settings["top_logprobs"])]
    # every request (the model list, then the generations) carries the key and the proxy-auth headers
    assert {(r["headers"]["Authorization"], r["headers"]["Modal-Key"], r["headers"]["Modal-Secret"])
            for r in server.requests} == {("Bearer k-123", "mk-1", "ms-1")}
    assert server.requests[0]["path"] == "/v1/models"
    generations = server.requests[1:]
    assert {r["path"] for r in generations} == {"/v1/chat/completions"}
    for r in generations:
        assert r["body"]["temperature"] == 0.0 and r["body"]["logprobs"] is True
        assert r["body"]["return_tokens_as_token_ids"] is True
        assert r["body"]["top_logprobs"] == settings["top_logprobs"] and r["body"]["max_tokens"] == settings["max_new_tokens"]
    assert len(generations) == 2 * len(prompts)


@pytest.mark.parametrize("listed, message", [
    (lambda sha: cards("f" * 64, name=f"c0-{sha[:12]}"), "not the adapter trained"),  # a root that is not the adapter
    (lambda sha: cards("f" * 64), "deploy them"),  # retrained, not redeployed: the new name is not served (404)
    (lambda sha: cards(sha, base_root="Qwen/Other"), "not 'trl-internal-testing"),
    (lambda sha: cards(sha)[:1], "deploy them"),
])
def test_p4_refuses_to_judge_an_adapter_the_server_does_not_serve(parity_repo, listed, message):
    config_path, config, prompts = parity_repo
    server = FakeVLLM({}, prompts, listed(_sha()))
    try:
        config["roles"]["slm_candidates"][-1]["endpoint"]["base_url"] = server.base_url
        failed = check_lora_parity(save(config, config_path), "c0", "local")
    finally:
        server.close()
    assert failed["status"] == FAIL and message in failed["evidence"]["error"] and "not decided" in failed["action"]
    assert [r["path"] for r in server.requests] == ["/v1/models"]  # asked what it serves, generated nothing


def test_a_bug_in_p4_is_raised_not_reported_as_an_unreachable_server(parity_repo, monkeypatch):
    from bench import preflight

    _, config, _ = parity_repo
    monkeypatch.setattr(preflight, "lora_parity", lambda *a: {}["status"])
    with pytest.raises(KeyError):
        check_lora_parity(config, "c0", "local")


def test_served_generate_refuses_a_server_that_does_not_return_token_ids(parity_repo):
    _, _, prompts = parity_repo

    server = FakeVLLM({"m": [gen([1])] * len(prompts)}, prompts)
    try:
        assert served_generate(server.base_url, {}, "m", prompts[0], 4, 2, 10) == gen([1])
        server.token_name = lambda t: f"piece{t}"  # a server that returns text pieces
        with pytest.raises(PreflightError, match="not a token id"):
            served_generate(server.base_url, {}, "m", prompts[0], 4, 2, 10)
    finally:
        server.close()


def test_p4_refuses_a_missing_proxy_auth_variable(parity_repo, monkeypatch):
    config_path, config, prompts = parity_repo
    server = FakeVLLM({}, prompts, cards(_sha()))
    try:
        config["roles"]["slm_candidates"][-1]["endpoint"].update({"base_url": server.base_url,
                                                                  "headers_env": {"Modal-Key": "UNSET_MODAL_KEY"}})
        monkeypatch.delenv("UNSET_MODAL_KEY", raising=False)
        failed = check_lora_parity(save(config, config_path), "c0", "local")
    finally:
        server.close()
    assert failed["status"] == FAIL and "UNSET_MODAL_KEY" in failed["evidence"]["error"] and not server.requests


def test_p4_refuses_a_dataset_that_is_not_the_adapters(parity_repo):
    config_path, config, prompts = parity_repo
    path = paths.ROOT / "train/datasets/c0.jsonl"
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")  # one row more than trained on
    failed = check_lora_parity(config, "c0", "local")
    assert failed["status"] == FAIL and "not the dataset the adapter was trained on" in failed["evidence"]["error"]


def test_p4_is_pending_until_asked_and_fails_with_its_reason_when_nothing_is_served(parity_repo):
    _, config, _ = parity_repo
    assert check_lora_parity(config, None, "modal")["status"] == PENDING
    failed = check_lora_parity(config, "c0", "local")  # no base_url for the candidate
    assert failed["status"] == FAIL and "base_url is not set" in failed["evidence"]["error"]
    assert "not decided" in failed["action"]


# ---------------------------------------------------------------- SPEC 7.1

def _agent_run(run_id, split, config, status="done", sites=("agent_ir", "extract_keywords"), arm="B0",
               hours=0.5, other_config=False):
    from datetime import datetime, timedelta, timezone

    from bench.contracts.config import config_sha256

    started = datetime(2026, 10, 1, 8, tzinfo=timezone.utc)
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "arm": arm, "split": split, "status": status, "call_sites_seen": list(sites),
        "config_sha256": "0" * 64 if other_config else config_sha256(config),
        "started_at": started.isoformat(), "finished_at": (started + timedelta(hours=hours)).isoformat()}))


def test_teacher_terms(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    assert check_teacher_terms(config)["status"] == PASS
    config["roles"]["production_llm"]["terms"] = None
    failed = check_teacher_terms(config)
    assert failed["status"] == FAIL and failed["evidence"]["missing"] == ["weights_license", "provider_terms", "checked_on"]
    assert "reserve" in failed["action"]


def test_agent_runs_and_the_call_site_record_count_only_b0_runs_on_this_configuration(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    assert check_agent_runs(config)["status"] == FAIL
    _agent_run("agent-B0-train-smoke", "train", config, other_config=True)  # a smoke run on another config
    _agent_run("agent-B3-train-1", "train", config, arm="B3")
    _agent_run("agent-B0-train-slow", "train", config, hours=4)  # over the 3 h of SPEC 7.1
    _agent_run("agent-B0-train-2", "train", config, status="failed", sites=("revise",))
    assert check_agent_runs(config)["status"] == FAIL
    assert check_call_sites(config)["evidence"]["b0_runs"] == {"train": 1, "calib": 0}  # only the slow one, done
    bad = paths.RUNS / "agent-B0-train-badtime"
    _agent_run(bad.name, "train", config)
    manifest = json.loads((bad / "manifest.json").read_text())
    (bad / "manifest.json").write_text(json.dumps({**manifest, "finished_at": "not a time"}))
    assert check_agent_runs(config)["evidence"]["runs_without_readable_times"] == [bad.name]  # reported, not dropped
    _agent_run("agent-B0-train-1", "train", config)
    passed = check_agent_runs(config)
    assert passed["status"] == PASS and passed["evidence"]["b0_runs_within_3h"] == ["agent-B0-train-1"]
    pending = check_call_sites(config)
    assert pending["status"] == PENDING and pending["evidence"]["without_done_b0_runs"] == ["calib"]
    _agent_run("agent-B0-calib-1", "calib", config, sites=("agent_ss",))
    done = check_call_sites(config)
    assert done["status"] == PASS
    assert done["evidence"]["call_sites"] == {"train": ["agent_ir", "extract_keywords"], "calib": ["agent_ss"]}


def test_unreadable_manifests_are_reported_not_skipped(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    for name in ("agent-B0-train-bad", "loadtest-bad"):
        (paths.RUNS / name).mkdir(parents=True)
        (paths.RUNS / name / "manifest.json").write_text("{not json")
    (paths.ROOT / "train" / "adapters" / "c9").mkdir(parents=True)
    (paths.ROOT / "train" / "adapters" / "c9" / "manifest.json").write_text("{")
    assert check_agent_runs(config)["evidence"]["unreadable_manifests"] == ["agent-B0-train-bad: JSONDecodeError"]
    assert check_call_sites(config)["evidence"]["unreadable_manifests"] == ["agent-B0-train-bad: JSONDecodeError"]
    assert check_throughput()["evidence"]["unreadable_manifests"] == ["loadtest-bad: JSONDecodeError"]
    assert check_training_time(config)["evidence"]["unreadable_manifests"] == ["c9: JSONDecodeError"]


def test_a_bug_in_the_data_check_is_raised_not_reported_as_bad_data(tmp_path, monkeypatch):
    from bench import data

    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(data, "load_splits", lambda: {}["train"])
    with pytest.raises(KeyError):
        check_data(config)


def test_data_check_passes_on_pinned_data_and_fails_on_overlapping_splits(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    assert check_data(config)["status"] == PASS
    splits = json.loads(paths.SPLITS.read_text())
    paths.SPLITS.write_text(json.dumps({**splits, "test": ["9", "1"]}))
    failed = check_data(config)
    assert failed["status"] == FAIL and failed["evidence"]["test"]["overlap_with_train_or_calib"] == ["1"]
    paths.SPLITS.unlink()
    assert "error" in check_data(config)["evidence"]


def test_pilot_spend_is_pending_on_what_other_fronts_provide(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    assert check_pilot_spend(config)["status"] == PENDING


def test_training_time_projects_the_measured_gpu_time_over_every_dataset(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    assert check_training_time(config)["status"] == PENDING
    local = fake_adapter("c0", config).parent / "manifest.json"
    manifest = json.loads(local.read_text())
    local.write_text(json.dumps({**manifest, "where": "local", "stats": {"train_seconds": 1, "examples_seen": 1}}))
    assert check_training_time(config)["status"] == PENDING  # a CPU run does not project GPU time
    local.write_text(json.dumps({**manifest, "where": "modal", "stats": {"train_seconds": 100, "examples_seen": 50}}))
    measured = check_training_time(config)
    epochs = config["train"]["sft"]["num_train_epochs"]
    assert measured["evidence"]["dataset_rows"] == 8 and measured["evidence"]["seconds_per_example"] == 2
    assert measured["evidence"]["projected_train_hours"] == round(2 * 8 * epochs / 3600, 2)
    assert measured["status"] == PASS
    config["preflight"]["schedule"]["train_hours_max"] = 0.001
    assert check_training_time(config)["status"] == FAIL


def test_throughput_lists_only_finished_load_runs_and_leaves_the_judgment_to_j8(tmp_path, monkeypatch):
    make_s5_repo(tmp_path, monkeypatch)
    for run_id, status in (("loadtest-a", "failed"), ("loadtest-b", "running"), ("loadtest-c", "done")):
        (paths.RUNS / run_id).mkdir(parents=True)
        (paths.RUNS / run_id / "manifest.json").write_text(json.dumps(
            {"run_id": run_id, "engine": "slm:x", "concurrency": 4, "status": status}))
    check = check_throughput()
    assert check["status"] == PENDING and [r["run_id"] for r in check["evidence"]["done_loadtest_runs"]] == ["loadtest-c"]
    assert check["evidence"]["needs"] == "J8 over these runs"


def test_bench_preflight_writes_the_report_and_exits_1_unless_everything_passes(tmp_path, monkeypatch, capsys):
    _, config_path, _ = make_s5_repo(tmp_path, monkeypatch)
    assert cli.main(["preflight", "--config", str(config_path)]) == 1
    report_path = next(paths.RUNS.glob("preflight-*/report.json"))
    report = json.loads(report_path.read_text())
    assert [c["id"] for c in report["checks"]] == ["agent_end_to_end", "call_sites_registered", "data_ids_and_gold",
                                                   "teacher_terms", "pilot_spend", "training_time", "throughput",
                                                   "lora_parity"]
    assert report["all_pass"] is False and all(c["action"] for c in report["checks"])
    assert str(paths.ROOT) not in report_path.read_text()
    assert "lora_parity" in capsys.readouterr().out


def test_p4_on_modal_asks_the_gpu_reference_for_the_adapter_by_its_sha256(parity_repo, monkeypatch):
    from test_train_fixtures import fake_modal_app

    config_path, config, prompts = parity_repo
    server = FakeVLLM({TINY_NAME: [gen([10 + i, 20, 30]) for i in range(len(prompts))],
                       served("c0"): [gen([40 + i, 50, 60]) for i in range(len(prompts))]}, prompts, cards(_sha()))
    reference, _ = _reference(prompts)
    calls = fake_modal_app(monkeypatch, "train",
                           peft_reference=lambda sha, *args: {**reference(*args), "function_seconds": 12.5})
    try:
        config["roles"]["slm_candidates"][-1]["endpoint"]["base_url"] = server.base_url
        verdict = lora_parity(save(config, config_path), "c0", "modal")
    finally:
        server.close()
    manifest = json.loads((paths.ROOT / "train/adapters/c0/manifest.json").read_text())
    (sha, base, got_prompts, kwargs, _, _), = calls["peft_reference"]
    assert sha == manifest["adapter_sha256"] and base == TINY and got_prompts == prompts
    assert verdict["status"] == PASS and verdict["reference_on"] == "modal" and verdict["reference_gpu_seconds"] == 12.5
