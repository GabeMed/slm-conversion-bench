"""The server behind an execution's SLM engines (bench.agent.served): read from the pod and the tunnel, recorded,
and refused when it is not what serve_plan(config) says, on fakes (no pod, no network)."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from bench import paths
from bench.agent import served
from bench.data import DataError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # runpod_apps/ sits beside bench/

PORT = 18001
BASE = f"http://127.0.0.1:{PORT}/v1"
PLAN = {"name": "qwen3-8b", "repo": "Qwen/Qwen3-8B", "revision": "r" * 40, "chat_template_kwargs": {"enable_thinking": False},
        "adapters": [{"cluster": "c0", "served_name": "qwen3-8b-c0", "sha256": "a" * 64, "r": 16}],
        "vllm_version": "0.30.0", "max_model_len": 32768, "min_containers": 0, "scaledown_window_s": 300}
ENGINES = ["production_llm", "slm:qwen3-8b+lora:qwen3-8b-c0"]


def config(base_url=BASE, runpod=True):
    out = {"roles": {"slm_candidates": [{"name": "qwen3-8b", "endpoint": {"kind": "vllm", "base_url": base_url,
                                                                          "api_key_env": "SLM_TEST_KEY"}}]},
           "serving": {"vllm_version": "0.30.0"}}
    if runpod:
        out["runpod"] = {"local_ports": {"qwen3-8b": PORT}}
    return out


LOCAL = "c" * 64  # this checkout's code archive
CODE = {"sha256": LOCAL, "commit": "1" * 40, "dirty": False}


def server(plan=None, pod_id="pod1", state_pod="pod1", models=("qwen3-8b", "qwen3-8b-c0"), version="0.30.0",
           pod_code=CODE, started_from=CODE):
    """observe, get and expected as a pod serving `plan` (PLAN by default) would answer them; `asked` records the
    GETs. `pod_code`: the code record the pod holds; `started_from`: the one its server recorded at its start."""
    plan = copy.deepcopy(PLAN) if plan is None else plan
    asked = []

    def observe(cfg, candidate):
        return {"pod_id": pod_id, "usd_per_hour": 1.79, "plan": plan, "pod_code": pod_code, "local_code_sha256": LOCAL,
                "server_state": None if state_pod is None else {"pod_id": state_pod, "vllm_version": version,
                                                                "code": started_from}}

    def get(url, key):
        asked.append((url, key))
        if url.endswith("/v1/models"):
            return {"data": [{"id": m} for m in models]}
        assert url == f"http://127.0.0.1:{PORT}/version"
        return {"version": version}
    return {"observe": observe, "get": get, "expected": lambda cfg, candidate: copy.deepcopy(PLAN),
            "knobs": ("min_containers", "scaledown_window_s")}, asked


def test_engines_without_an_slm_record_nothing():
    assert served.check_served(config(), ["production_llm", "cheap_alt"], "test") is None
    assert served.slm_models(["slm:a", "slm:a+lora:a-c0", "slm:b+lora:b-c1", "cheap_alt"]) == \
        {"a": ["a", "a-c0"], "b": ["b-c1"]}


def test_the_server_is_recorded(monkeypatch):
    monkeypatch.setenv("SLM_TEST_KEY", "vllm-key")
    fakes, asked = server()
    found = served.check_served(config(), ENGINES, "test", **fakes)
    assert found["qwen3-8b"]["checked"] is True
    assert {k: found["qwen3-8b"][k] for k in ("provider", "pod_id", "usd_per_hour", "plan", "models", "vllm_version")} == {
        "provider": "runpod", "pod_id": "pod1", "usd_per_hour": 1.79, "plan": PLAN,
        "models": ["qwen3-8b", "qwen3-8b-c0"], "vllm_version": "0.30.0"}
    assert len(found["qwen3-8b"]["plan_sha256"]) == 64 and found["qwen3-8b"]["server_state"]["pod_id"] == "pod1"
    assert asked == [(f"{BASE}/models", "vllm-key"), (f"http://127.0.0.1:{PORT}/version", "vllm-key")]
    assert {k: found["qwen3-8b"][k] for k in ("pod_code_sha256", "pod_code_commit", "pod_code_dirty")} == {
        "pod_code_sha256": LOCAL, "pod_code_commit": "1" * 40, "pod_code_dirty": False}
    assert found["qwen3-8b"]["server_state"]["code"] == CODE


OTHER = {"sha256": "0" * 64, "commit": "2" * 40, "dirty": True}


@pytest.mark.parametrize("change, message, recorded", [
    ({"pod_code": OTHER}, f"the code recorded on the pod \\(sha256 {'0' * 64}\\) is not this checkout's \\(sha256 {LOCAL}\\)", OTHER),
    ({"pod_code": None}, "the code recorded on the pod \\(sha256 None\\) is not this checkout's", {}),  # no record
    # the pod's files were replaced after the server started: the record is this checkout's, the server is not
    ({"started_from": OTHER}, f"its server was started from other code \\(sha256 {'0' * 64}\\) than this checkout's", CODE),
    ({"started_from": None}, "its server was started from other code \\(sha256 None\\)", CODE),  # a server of older code
])
def test_the_test_split_is_refused_when_the_pods_code_is_not_this_checkouts(change, message, recorded):
    fakes, _ = server(**change)
    with pytest.raises(DataError, match=f"refusing to run on the server of qwen3-8b \\(pod pod1\\): {message}"):
        served.check_served(config(), ENGINES, "test", **fakes)
    found = served.check_served(config(), ENGINES, "calib", **fakes)["qwen3-8b"]  # recorded, never refused, elsewhere
    assert found["checked"] is True
    assert {k: found[k] for k in ("pod_code_sha256", "pod_code_commit", "pod_code_dirty")} == {
        "pod_code_sha256": recorded.get("sha256"), "pod_code_commit": recorded.get("commit"), "pod_code_dirty": recorded.get("dirty")}


def test_the_test_split_is_refused_when_this_checkouts_code_is_unknown():
    fakes, _ = server()
    observe = fakes["observe"]
    fakes["observe"] = lambda cfg, candidate: {k: v for k, v in observe(cfg, candidate).items() if k != "local_code_sha256"}
    with pytest.raises(DataError, match="is not this checkout's \\(sha256 None\\)"):
        served.check_served(config(), ENGINES, "test", **fakes)


def test_the_modal_only_deploy_knobs_are_not_compared():
    fakes, _ = server(plan={**copy.deepcopy(PLAN), "min_containers": 1, "scaledown_window_s": 60})
    assert served.check_served(config(), ENGINES, "test", **fakes)["qwen3-8b"]["checked"] is True


@pytest.mark.parametrize("change, message", [
    ({"plan": {**PLAN, "revision": "s" * 40}}, "differs from serve_plan\\(config\\) in revision"),
    ({"plan": {**PLAN, "adapters": [{**PLAN["adapters"][0], "sha256": "b" * 64}]}}, "differs .* in adapters"),
    ({"plan": {k: v for k, v in PLAN.items() if k != "max_model_len"}}, "in max_model_len"),
    ({"state_pod": "pod0"}, "written by pod pod0, not by pod1"),  # a stale state from an earlier pod
    ({"state_pod": None}, "left no state"),
    ({"version": "0.29.0"}, "reports vLLM 0.29.0, serving.vllm_version is 0.30.0"),
    ({"models": ("qwen3-8b",)}, "/v1/models lacks qwen3-8b-c0"),
])
def test_a_server_that_is_not_the_configured_one_is_refused(change, message):
    fakes, _ = server(**change)
    for split in ("calib", "test"):
        with pytest.raises(DataError, match=message):
            served.check_served(config(), ENGINES, split, **fakes)


def test_a_pod_without_its_plan_is_refused():
    fakes, _ = server()
    fakes["observe"] = lambda cfg, candidate: {"pod_id": "pod1", "plan": None, "server_state": {"pod_id": "pod1"}}
    with pytest.raises(DataError, match="holds no plan.json"):
        served.check_served(config(), ENGINES, "calib", **fakes)


@pytest.mark.parametrize("cfg", [config(base_url="http://slm.example/v1"), config(runpod=False)])
def test_a_server_that_cannot_be_read_is_recorded_unchecked_and_refused_on_test(cfg):
    def never(*args):
        raise AssertionError("nothing is asked of a server that is not a RunPod tunnel")
    fakes = {"observe": never, "get": never, "expected": never, "knobs": ()}
    assert served.check_served(cfg, ENGINES, "calib", **fakes) == {
        "qwen3-8b": {"checked": False, "why": "its endpoint is not the local end of a RunPod tunnel "
                                              "(runpod.local_ports): its server cannot be read"}}
    with pytest.raises(DataError, match="refusing the test split: qwen3-8b's server cannot be checked"):
        served.check_served(cfg, ENGINES, "test", **fakes)


class FakeRemote:
    def __init__(self, files):
        self.files, self.read_paths = files, []

    def read(self, path):
        self.read_paths.append(path)
        return self.files.get(path)


class FakeProvider:
    def __init__(self, record, pod, files):
        from runpod_apps import pods as rp
        self.state = type("S", (), {"serving": lambda _, candidate: record})()
        self.api = type("A", (), {"get_pod": lambda _, pod_id: pod if pod and pod_id == pod["id"] else None})()
        self.remote_ = FakeRemote(files)
        self.held_code = lambda remote: rp.Provider.held_code(self, remote)  # the real reading of the record

    def remote(self, pod):
        return self.remote_


def test_observe_reads_the_recorded_pods_plan_and_server_state(monkeypatch):
    from runpod_apps import pods as rp
    from runpod_apps import serve as rserve
    state = {"pod_id": "pod1", "vllm_version": "0.30.0"}
    files = {rserve.PLAN: json.dumps(PLAN), "/runpod-volume/vllm-cache/bench-serving/qwen3-8b.json": json.dumps(state)}
    provider = FakeProvider({"id": "pod1"}, {"id": "pod1", "status": "RUNNING", "cost": 1.79}, files)
    monkeypatch.setattr(rp, "make_provider", lambda cfg: provider)
    local = hashlib.sha256(rp.code_archive()).hexdigest()
    assert served.observe_runpod(config(), "qwen3-8b") == {"pod_id": "pod1", "usd_per_hour": 1.79, "plan": PLAN,
                                                          "server_state": state, "pod_code": None,
                                                          "local_code_sha256": local}
    assert provider.remote_.read_paths == ["/bench/plan.json", "/runpod-volume/vllm-cache/bench-serving/qwen3-8b.json",
                                           "/bench/code.json"]
    files[rp.CODE_RECORD] = json.dumps(CODE)  # the record `up` (or a later sync) left
    assert served.observe_runpod(config(), "qwen3-8b")["pod_code"] == CODE
    files[rp.CODE_RECORD] = "not json"
    assert served.observe_runpod(config(), "qwen3-8b")["pod_code"] is None


@pytest.mark.parametrize("record, pod, message", [
    (None, None, "no pod serves qwen3-8b"),
    ({"id": "pod1"}, None, "pod1 of qwen3-8b is gone"),
    ({"id": "pod1"}, {"id": "pod1", "status": "EXITED"}, "pod1 of qwen3-8b is EXITED"),
])
def test_observe_refuses_a_pod_that_does_not_serve(monkeypatch, record, pod, message):
    from runpod_apps import pods as rp
    monkeypatch.setattr(rp, "make_provider", lambda cfg: FakeProvider(record, pod, {}))
    with pytest.raises(DataError, match=message):
        served.observe_runpod(config(), "qwen3-8b")


def test_the_repository_packages_are_importable_from_the_check():
    served._repo_imports()
    assert str(paths.ROOT) in sys.path
