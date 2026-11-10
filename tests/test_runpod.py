"""RunPod as the inference provider, without RunPod: the REST client against a fake API (create, poll,
terminate, volumes, the catalog's choice of data center, retries and errors), the local state and `down --all`,
the serving pod (`bench serve --on runpod up`), the load test and P-4's reference on pods, and the config
patch, with every pod's shell run on this machine (tests/fixtures/fake_runpod.py). No network."""
import copy
import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # runpod_apps/ sits beside bench/

from bench import cli, paths  # noqa: E402
from bench.contracts import facts  # noqa: E402
from fixtures.fake_runpod import FakeRunPod, LocalShell  # noqa: E402
from modal_apps import common  # noqa: E402
from runpod_apps import api as rapi  # noqa: E402
from runpod_apps import pods as rp  # noqa: E402
from runpod_apps import serve as rserve  # noqa: E402
from runpod_apps import settings as rs  # noqa: E402
from runpod_apps.settings import RunPodError  # noqa: E402
from runpod_apps.ssh import Remote, Result, Shell  # noqa: E402
from runpod_apps.state import State  # noqa: E402
from test_loadtest import SOURCE, FakeServer, _source_run  # noqa: E402
from test_train_fixtures import TINY, TINY_NAME, fake_adapter, make_s5_repo, save, served  # noqa: E402

KEY = "rk-secret-0123456789"
PATCH = paths.ROOT / "configs" / "runpod.patch.yaml"


def _apply_script():
    spec = importlib.util.spec_from_file_location("apply_config_patch", paths.ROOT / "scripts" / "apply_config_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPO = Path(__file__).resolve().parent.parent
APPLY = _apply_script()


def shipped_runpod_section():
    """The `runpod` section of the shipped config.yaml (configs/runpod.patch.yaml is applied to it)."""
    return yaml.safe_load((REPO / "config.yaml").read_text())["runpod"]


def unpatched_text():
    """The shipped config.yaml before configs/runpod.patch.yaml: each replacement undone, last first."""
    patch = yaml.safe_load(PATCH.read_text())
    undo = [{**change, "old": change["new"], "new": change["old"]} for change in reversed(patch["replace"])]
    return APPLY.patched_text((REPO / "config.yaml").read_text(), {"replace": undo})


# ---------------------------------------------------------------- the REST client

@pytest.fixture
def fake():
    server = FakeRunPod()
    yield server
    server.close()


def client(fake, sleeps=None):
    return rapi.RunPodClient(KEY, base=fake.url, backoff_s=0.5, sleep=(sleeps.append if sleeps is not None else lambda s: None))


def tagged(name, token="t0k3n", **fields):
    """A create body with its unique token in its name and in env BENCH_POD_TOKEN, as pods.pod_body makes it."""
    return {"name": f"{name}-{token}", "image": "img", "env": {"BENCH_POD_TOKEN": token},
            **(fields or {"cpu": {"id": "cpu5c", "vcpuCount": 4}})}


def test_a_pod_is_created_polled_until_it_runs_and_terminated(fake):
    api = client(fake)
    pod = api.create_pod(tagged("slm-bench-x", gpu={"id": "NVIDIA A100 80GB PCIe", "count": 1}))
    assert pod["status"] == "PROVISIONING" and rp.ssh_endpoint(pod) is None
    running = api.get_pod(pod["id"])
    assert running["status"] == "RUNNING" and rp.ssh_endpoint(running) == {"host": "203.0.113.7", "port": 40000}
    assert [p["id"] for p in api.list_pods(name="slm-bench-x-t0k3n")] == [pod["id"]] and api.list_pods(name="other") == []
    assert api.terminate_pod(pod["id"]) is True and api.terminate_pod(pod["id"]) is False  # already gone
    assert api.get_pod(pod["id"]) is None
    assert {r["authorization"] for r in fake.requests} == {f"Bearer {KEY}"}


def test_ssh_endpoint_falls_back_on_the_runtime_port_of_22():
    pod = {"ssh": {"direct": None}, "runtime": {"ports": [{"private": 8000, "public": 1, "ip": "x"},
                                                           {"private": 22, "public": 40123, "ip": "198.51.100.2"}]}}
    assert rp.ssh_endpoint(pod) == {"host": "198.51.100.2", "port": 40123}


def test_5xx_and_429_are_retried_429_after_its_retry_after(fake):
    sleeps = []
    fake.fail += [{"method": "GET", "path": "/v2/network-volumes", "status": 503, "times": 2},
                  {"method": "GET", "path": "/v2/catalog/gpus", "status": 429, "headers": {"Retry-After": "7"}}]
    api = client(fake, sleeps)
    assert api.list_volumes() == []
    assert api.gpu_availability("SECURE", "13.0")[0]["id"] == "NVIDIA A100 80GB PCIe"
    assert sleeps == [0.5, 1.0, 7.0]  # exponential backoff, then the server's own wait
    query = [r["query"] for r in fake.requests if r["path"] == "/v2/catalog/gpus"][-1]
    assert query == {"include": "AVAILABILITY", "product": "POD", "cloud": "SECURE", "minCudaVersion": "13.0"}


def test_a_client_error_is_not_retried_and_never_carries_the_key(fake):
    fake.fail.append({"method": "POST", "path": "/v2/network-volumes", "status": 400, "detail": f"bad token {KEY}"})
    with pytest.raises(rapi.ApiError) as raised:
        client(fake).create_volume("v", 10, "EU-RO-1")
    assert raised.value.status == 400 and KEY not in str(raised.value) and "bad token <redacted>" in str(raised.value)
    assert len([r for r in fake.requests if r["path"] == "/v2/network-volumes"]) == 1


def test_retries_end(fake):
    fake.fail.append({"method": "GET", "path": "/v2/pods", "status": 502, "times": 99})
    with pytest.raises(rapi.ApiError, match="HTTP 502"):
        client(fake).list_pods()
    assert len(fake.requests) == 5


def test_a_create_whose_answer_was_lost_is_adopted_never_made_twice(fake):
    """A 503 after the pod was made: the retry finds it by its token instead of creating a second one (it bills)."""
    fake.fail.append({"method": "POST", "path": "/v2/pods", "status": 503, "apply": True})
    pod = client(fake).create_pod(tagged("slm-bench-serve-x"))
    assert len(fake.pods) == 1 and pod["id"] in fake.pods
    assert len(fake.created()) == 1  # one POST only: the retry listed the pods instead


def test_a_create_is_adopted_after_its_last_attempt_too(fake):
    fake.fail.append({"method": "POST", "path": "/v2/pods", "status": 503, "times": 1, "apply": True})
    api = rapi.RunPodClient(KEY, base=fake.url, max_attempts=1, sleep=lambda s: None)
    pod = api.create_pod(tagged("slm-bench-serve-x"))
    assert list(fake.pods) == [pod["id"]] and len(fake.created()) == 1


def test_only_a_pod_with_the_creates_token_and_not_exited_is_adopted(fake):
    api = client(fake)
    exited = api.create_pod(tagged("slm-bench-serve-x"))  # same name and token, but it already exited
    fake.status[exited["id"]] = "EXITED"
    stranger = dict(tagged("slm-bench-serve-x"), env={"BENCH_POD_TOKEN": "other"})  # same name, another token
    with pytest.raises(RunPodError, match="no unique token"):
        api.create_pod(stranger)
    api.create_pod(dict(tagged("slm-bench-serve-x", token="zz"), name="slm-bench-serve-x-zz"))
    fake.fail.append({"method": "POST", "path": "/v2/pods", "status": 503})  # lost, never made
    made = api.create_pod(tagged("slm-bench-serve-x"))
    assert made["id"] not in (exited["id"],) and fake.pods[made["id"]]["env"]["BENCH_POD_TOKEN"] == "t0k3n"
    with pytest.raises(RunPodError, match="no unique token"):
        api.create_pod({"name": "slm-bench-bare", "image": "img", "cpu": {"id": "cpu5c", "vcpuCount": 4}})


def test_a_create_that_cannot_be_checked_says_a_pod_may_exist(fake):
    fake.fail += [{"method": "POST", "path": "/v2/pods", "status": 503, "times": 9, "apply": True},
                  {"method": "GET", "path": "/v2/pods", "status": 500, "times": 99}]
    with pytest.raises(rapi.ApiError, match="a pod named slm-bench-serve-x-t0k3n may exist: run `bench serve --on "
                                            "runpod down --all`"):
        client(fake).create_pod(tagged("slm-bench-serve-x"))
    fake.fail.clear()
    fake.fail.append({"method": "POST", "path": "/v2/pods", "status": 502, "times": 99})  # never made, listing works
    with pytest.raises(rapi.ApiError, match="may exist"):
        client(fake).create_pod(tagged("slm-bench-serve-y"))


@pytest.mark.parametrize("raw, length", [(b"<html>Bad gateway</html>", None), (b'{"pods": [', 400)])
def test_an_answer_that_is_not_json_is_retried_never_returned(fake, raw, length):
    """A 200 with an HTML page or a body cut short (IncompleteRead) is no answer: retried, then an ApiError."""
    fault = {"method": "GET", "path": "/v2/network-volumes", "status": 200, "raw": raw, "times": 1}
    if length:
        fault["length"] = length
    fake.fail.append(fault)
    assert client(fake).list_volumes() == []  # the retry got the real answer
    fake.fail.append({**fault, "times": 99})
    with pytest.raises(rapi.ApiError, match="no usable answer"):
        client(fake).list_volumes()


def test_pages_are_capped_and_a_cursor_that_does_not_advance_stops(monkeypatch):
    api = rapi.RunPodClient(KEY, base="http://127.0.0.1:9", sleep=lambda s: None)
    calls = []

    def stuck(method, path, query=None, **kw):
        calls.append(query["cursor"])
        return {"pods": [{"id": f"p{len(calls)}"}], "pagination": {"nextCursor": "same", "hasNextPage": True}}
    monkeypatch.setattr(api, "request", stuck)
    assert len(api.list_pods()) == 2 and calls == [None, "same"]
    counter = iter(range(10_000))

    def endless(method, path, query=None, **kw):
        n = next(counter)
        return {"pods": [{"id": f"p{n}"}], "pagination": {"nextCursor": f"c{n}", "hasNextPage": True}}
    monkeypatch.setattr(api, "request", endless)
    assert len(api.list_pods()) == rapi.MAX_PAGES == 100


def test_the_bodies_sent_and_the_fields_read_are_the_openapi_specs():
    """tests/fixtures/runpod_openapi.json is https://api.runpod.io/v2/openapi.json (fetched 2026-11-09)."""
    from fixtures.fake_runpod import OPENAPI, openapi_validator

    settings = shipped_runpod_section()
    volume = {"id": "vol1", "datacenter": "EU-RO-1"}
    env = {"VLLM_API_KEY": "{{ RUNPOD_SECRET_slm-bench-vllm-api-key }}"}
    bodies = [rp.pod_body(settings, volume, "serve", "vllm/vllm-openai:v0.30.0", 3600, "ssh-ed25519 AAAA k",
                          candidate="qwen3-8b", gpu="NVIDIA A100 80GB PCIe", env=env),
              rp.pod_body(settings, volume, "loadtest", "python:3.12-slim-bookworm", 3600, "ssh-ed25519 AAAA k",
                          candidate="qwen3-8b", cpu_vcpus=4, env=env)]
    for body in bodies:
        assert list(openapi_validator("CreatePodRequest").iter_errors(body)) == []
    assert list(openapi_validator("CreatePodRequest").iter_errors({**bodies[0], "gpuTypeId": "x"}))  # it does check
    assert list(openapi_validator("CreateNetworkVolumeRequest").iter_errors(
        {"name": "slm-bench", "size": 100, "dataCenter": "EU-RO-1"})) == []
    schemas = json.loads(OPENAPI.read_text())["components"]["schemas"]
    assert {"$ref": "#/components/schemas/ContainerConfig"} in schemas["Pod"]["allOf"]  # image, env, ports...
    pod = {k: v for part in schemas["Pod"]["allOf"] + [schemas["BaseContainerConfig"]]
           for k, v in (part.get("properties") or {}).items()}
    assert {"id", "name", "status", "cost", "env", "ssh", "runtime", "dataCenterId"} <= set(pod)
    assert {"direct"} <= set(schemas["PodSsh"]["properties"]) and {"host", "port"} <= set(schemas["PodSshEndpoint"]["properties"])
    assert {"uptime", "ports"} <= set(schemas["PodRuntime"]["properties"])
    assert {"private", "public", "ip"} <= set(schemas["PodRuntimePort"]["properties"])
    assert {"nextCursor", "hasNextPage"} <= set(schemas["Pagination"]["properties"])
    from fixtures.fake_runpod import FakeRunPod as Fake
    fake = Fake()
    try:  # the data center catalog the fake answers, and the field setup reads to skip a data center
        assert list(openapi_validator("ListDataCentersResponse").iter_errors({"dataCenters": fake.datacenters})) == []
    finally:
        fake.close()
    assert "networkVolumeTypes" in schemas["DataCenter"]["required"]
    assert {"id", "name", "size", "dataCenter"} <= set(schemas["NetworkVolume"]["properties"])
    assert "networkVolumes" in schemas["ListNetworkVolumesResponse"]["properties"]
    assert {"id", "dataCenters"} <= set(schemas["GpuType"]["properties"])
    assert "secrets" in schemas["ListSecretsResponse"]["properties"] and "name" in schemas["Secret"]["properties"]
    assert set(schemas["PodStatus"]["enum"]) >= set(rapi.DEAD) | {"RUNNING"}


def test_the_fake_refuses_a_body_the_spec_refuses(fake):
    with pytest.raises(rapi.ApiError, match="HTTP 400.*unexpected"):
        client(fake).create_pod({**tagged("slm-bench-x"), "gpuTypeId": "NVIDIA A100 80GB PCIe"})


def test_the_data_center_is_the_most_available_one_ties_by_id(fake):
    gpus = client(fake).gpu_availability("SECURE", "13.0")
    assert [d["id"] for d in rapi.best_datacenters(gpus, "NVIDIA A100 80GB PCIe")] == ["CA-MTL-1", "EU-RO-1", "US-TX-3"]
    assert rapi.best_datacenters(gpus, "NVIDIA B200") == []


# ---------------------------------------------------------------- a repo served on the fake

@pytest.fixture
def runpod_repo(tmp_path, monkeypatch, fake):
    """The tiny candidate with adapter c0, configured for RunPod (the patch's runpod section), a fake vLLM
    listening on the candidate's local port (where the tunnel would be), every pod a LocalShell."""
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    fake_adapter("c0", config)
    vllm = FakeServer([TINY_NAME, served("c0")])
    port = int(vllm.base_url.split(":")[-1].split("/")[0])
    config = copy.deepcopy(config)
    config["runpod"] = shipped_runpod_section()
    key = tmp_path / "keys" / "id_ed25519"
    key.parent.mkdir()
    key.write_text("PRIVATE")
    key.with_suffix(".pub").write_text("ssh-ed25519 AAAAoperator me@laptop\n")
    config["runpod"].update(ssh_key=str(key), local_ports={TINY_NAME: port})
    config["serving"]["unauthenticated"] = True
    config["roles"]["slm_candidates"][-1]["endpoint"].update(base_url=f"http://127.0.0.1:{port}/v1",
                                                            api_key_env="T_SLM_KEY", headers_env=None)
    monkeypatch.setenv("T_SLM_KEY", "from-the-secret")
    monkeypatch.setattr(common, "check_serving_template", lambda entry: None)  # needs transformers: env/train
    config = save(config, config_path)
    world = types.SimpleNamespace(config=config, config_path=config_path, fake=fake, vllm=vllm, port=port,
                                  shells={}, jobs={}, ssh=[], tmp=tmp_path)

    def shell_for(pod, endpoint):
        if pod["id"] not in world.shells:
            world.shells[pod["id"]] = LocalShell(tmp_path / "pods" / pod["id"], tmp_path / "volume", world.jobs,
                                                 lambda command: world.ssh.append(command) or 0)
        return world.shells[pod["id"]]

    def provider(config):
        return rp.Provider(config, api=client(fake), state=State(), shell_for=shell_for, sleep=lambda s: None, poll_s=0)

    monkeypatch.setattr(rp, "make_provider", provider)
    world.provider = lambda: provider(world.config)
    world.jobs.update(serve=serve_job(world), download=download_job, calibrate=calibrate_job(world),
                      **{"train-env": lambda shell, name, cmd: 0})
    yield world
    vllm.close()


def serve_job(world):
    def run(shell, name, command):
        """What runpod_apps.job serve leaves when vLLM is warm: the server's state on the volume, READY."""
        plan = json.loads(shell.local("/bench/plan.json").read_text())
        state = shell.local(f"/runpod-volume/vllm-cache/bench-serving/{plan['name']}.json")
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({**common.serving_state(plan, "NVIDIA A100 80GB PCIe"), "provider": "runpod",
                                     "versions": {"vllm": "0.30.0", "torch": "2.13.0", "transformers": "5.18.0", "flashinfer-python": "0.9.1"}}))
        shell.local("/bench/jobs/serve.ready").write_text(str(state))
        world.served_plan = plan
        return None  # vLLM keeps running
    return run


def calibrate_job(world):
    def run(shell, name, command):
        """What runpod_apps.job calibrate writes: slower through the client's tunnel than on the server."""
        spec_path, out_path = re.search(r"job calibrate (\S+) (\S+)", command).groups()
        spec = json.loads(shell.local(spec_path).read_text())
        world.calibrated = getattr(world, "calibrated", []) + [spec]
        extra = 0.0 if spec["base_url"] == "http://127.0.0.1:8000/v1" else 3.5
        levels = [{"concurrency": c, "requests": int(spec["requests"][str(c)]), "errors": 0,
                   "p50_ms": 100.0 * c + extra, "mean_ms": 110.0 * c + extra} for c in spec["concurrency"]]
        shell.local(out_path).write_text(json.dumps({"base_url": spec["base_url"], "levels": levels}))
        return 0
    return run


def download_job(shell, name, command):
    spec = json.loads(shell.local("/bench/download.json").read_text())
    marker = shell.local(rserve.complete_marker(spec["repo"], spec["revision"]))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("snapshot")
    return 0


def ready_volume(world):
    rserve.setup(world.provider())
    rserve.download(world.provider(), TINY_NAME)


def fake_tunnel(world):
    def tunnel(host, port, key, known_hosts, local_port, log):
        world.tunnels = getattr(world, "tunnels", []) + [(host, port, local_port)]
        return 424242
    return tunnel


# ---------------------------------------------------------------- setup, download, the state, down

def test_setup_creates_the_volume_where_the_gpu_is_and_keeps_it(runpod_repo):
    world = runpod_repo
    volume = rserve.setup(world.provider())
    assert (volume["datacenter"], volume["name"], volume["size_gb"], volume["existing"]) == ("CA-MTL-1", "slm-bench", 100, False)
    assert State().read()["volume"]["id"] == volume["id"] and len(world.fake.volumes) == 1
    again = rserve.setup(world.provider())
    assert again["existing"] is True and again["id"] == volume["id"] and len(world.fake.volumes) == 1


def test_setup_passes_over_a_data_center_without_network_volumes_for_the_next_one_with_the_gpu(runpod_repo):
    """CA-MTL-1 has the most A100s but holds no volumes: EU-RO-1, as available, is next."""
    world = runpod_repo
    world.fake.datacenters[0]["networkVolumeTypes"] = []
    volume = rserve.setup(world.provider())
    assert volume["datacenter"] == "EU-RO-1" and list(world.fake.volumes.values())[0]["dataCenter"] == "EU-RO-1"
    assert State().read()["volume"]["skipped_without_volumes"] == ["CA-MTL-1"]


def test_setup_fails_naming_the_way_out_when_no_data_center_with_the_gpu_holds_volumes(runpod_repo):
    world = runpod_repo
    for d in world.fake.datacenters:
        if d["id"] != "US-KS-2":
            d["networkVolumeTypes"] = []
    with pytest.raises(RunPodError, match=r"without network volumes \(CA-MTL-1, EU-RO-1, US-TX-3\): try later, or "
                                          r"set runpod.datacenter.*volumes: US-KS-2"):
        rserve.setup(world.provider())
    assert not world.fake.volumes and State().read()["volume"] is None


def test_a_pinned_data_center_is_used_as_is_without_asking_the_data_center_catalog(runpod_repo):
    world = runpod_repo
    world.config["runpod"]["datacenter"] = "US-TX-3"
    volume = rserve.setup(world.provider())
    assert volume["datacenter"] == "US-TX-3" and State().read()["volume"]["skipped_without_volumes"] == []
    assert not [r for r in world.fake.requests if r["path"] == "/v2/catalog/datacenters"]


def test_setup_adopts_a_volume_of_the_same_name_and_honours_a_pinned_data_center(runpod_repo):
    world = runpod_repo
    world.fake.volumes["v9"] = {"id": "v9", "name": "slm-bench", "size": 100, "dataCenter": "EU-RO-1", "type": "STANDARD"}
    world.config["runpod"]["datacenter"] = "EU-RO-1"
    volume = rserve.setup(world.provider())
    assert (volume["id"], volume["existing"]) == ("v9", True) and len(world.fake.volumes) == 1
    State().write({"volume": None, "pods": {}, "terminated": [], "downloaded": {}})
    world.config["runpod"]["datacenter"] = None
    world.fake.gpus[0]["dataCenters"] = [{"id": "EU-SE-1", "name": "Sweden", "availability": "NONE"}]
    with pytest.raises(RunPodError, match="no data center has NVIDIA A100 80GB PCIe available"):
        rserve.setup(world.provider())


def test_the_weights_download_on_a_cpu_pod_once_and_the_pod_is_terminated(runpod_repo):
    world = runpod_repo
    rserve.setup(world.provider())
    done = rserve.download(world.provider(), TINY_NAME)
    assert done == {"repo": TINY["repo"], "revision": TINY["revision"], "skipped": False}
    body, = world.fake.created()
    assert "gpu" not in body and body["cpu"] == {"id": "cpu5c", "vcpuCount": world.config["loadtest"]["client"]["cpu"]}
    assert body["mounts"] == {"network": [{"volumeId": "vol1", "path": "/runpod-volume"}]}
    assert body["dataCenterIds"] == ["CA-MTL-1"] and body["image"] == "python:3.12-slim-bookworm"
    assert not world.fake.pods and State().read()["pods"] == {}  # terminated, and recorded so
    command = next(c for c in world.shells[world.fake.terminated[0]].commands if c.startswith("# bench-job download"))
    assert "huggingface_hub==1.33.0 hf_xet==1.6.0" in command  # the serving lock's pins
    assert rserve.download(world.provider(), TINY_NAME)["skipped"] is True and len(world.fake.created()) == 1


def test_a_pod_is_recorded_before_anything_waits_on_it_and_down_all_finds_every_pod(runpod_repo, monkeypatch):
    world = runpod_repo
    rserve.setup(world.provider())
    provider = world.provider()
    monkeypatch.setattr(provider, "wait_ssh", lambda pod_id, timeout: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):  # the operator's Ctrl-C while the pod boots
        rserve.up(provider, TINY_NAME)
    assert not world.fake.pods  # up terminates a pod that did not come up
    pod = provider.create("serve", tagged("slm-bench-serve-x", gpu={"id": "NVIDIA A100 80GB PCIe"}), "x")
    assert State().read()["pods"][pod["id"]]["role"] == "serve"
    stray = client(world.fake).create_pod(tagged("slm-bench-loadtest-y", token="a1"))
    other = client(world.fake).create_pod(tagged("someone-else", token="b2"))
    with pytest.raises(RunPodError, match="--candidate"):
        rserve.down(world.provider())
    result = rserve.down(world.provider(), everything=True)
    assert {g["id"] for g in result["done"]} == {pod["id"], stray["id"]} and list(world.fake.pods) == [other["id"]]
    assert result["failed"] == [] and result["warnings"] == []
    assert State().read()["pods"] == {} and {t["id"] for t in State().read()["terminated"]} >= {pod["id"]}


class SilentSshd(Shell):
    """A pod whose sshd never answers: every command is ssh's own failure."""

    def __init__(self):
        self.commands = []

    def run(self, command, timeout_s=600):
        self.commands.append(command)
        return Result(255, "", "Warning: no known_hosts\nssh: connect to host 203.0.113.7 port 40000: "
                               "Connection timed out\n")


def clocked_provider(world, shell_for):
    """A Provider on the fake whose sleeps advance its clock (poll every 10 s): seconds pass without waiting."""
    now = [0.0]
    provider = rp.Provider(world.config, api=client(world.fake), state=State(), shell_for=shell_for,
                           sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0], poll_s=10)
    return provider, now


def test_a_running_pod_whose_sshd_never_answers_is_given_up_early_with_sshs_stderr_and_its_runtime(runpod_repo):
    world = runpod_repo
    ready_volume(world)
    sshd = SilentSshd()
    provider, now = clocked_provider(world, lambda pod, endpoint: sshd)
    with pytest.raises(RunPodError) as raised:
        rserve.up(provider, TINY_NAME)
    message = str(raised.value)
    assert "RUNNING for 600 s without SSH answering" in message and "1800 s startup timeout" in message
    assert "ssh exit 255: ssh: connect to host 203.0.113.7 port 40000: Connection timed out" in message
    assert "uptime 1800 s, ports [22->203.0.113.7:" in message
    assert 600 < now[0] < 700 and len(sshd.commands) > 1  # well before serving.startup_timeout_s
    assert not world.fake.pods  # the pod that did not come up is terminated


def test_ssh_is_tried_only_once_the_runtime_shows_22_and_a_pod_that_never_maps_it_is_given_up(runpod_repo, monkeypatch):
    world = runpod_repo
    rserve.setup(world.provider())
    view = world.fake._pod_view

    def no_port_22(pod):
        seen = view(pod)
        if seen["runtime"]:
            seen["runtime"]["ports"] = []
        return seen
    monkeypatch.setattr(world.fake, "_pod_view", no_port_22)
    world.config["runpod"]["ssh_wait_after_running_s"] = 120
    sshd = SilentSshd()
    provider, now = clocked_provider(world, lambda pod, endpoint: sshd)
    pod = provider.create("serve", tagged("slm-bench-serve-x", gpu={"id": "NVIDIA A100 80GB PCIe"}), "x")
    with pytest.raises(RunPodError, match=r"RUNNING for 120 s .*port 22 not mapped yet \(uptime 1800 s, ports \[\]\)"):
        provider.wait_ssh(pod["id"], 1800)
    assert sshd.commands == [] and 120 < now[0] < 200


def test_a_pod_whose_sshd_answers_is_reached_as_before_its_uptime_never_counting_against_it(runpod_repo):
    world = runpod_repo
    rserve.setup(world.provider())
    world.config["runpod"]["ssh_wait_after_running_s"] = 60  # the fake reports uptime 1800: only our clock counts
    shells = []
    provider, now = clocked_provider(world, lambda pod, endpoint: shells.append(endpoint) or
                                     LocalShell(world.tmp / "pods" / pod["id"], world.tmp / "volume"))
    pod = provider.create("serve", tagged("slm-bench-serve-x", gpu={"id": "NVIDIA A100 80GB PCIe"}), "x")
    ready = provider.wait_ssh(pod["id"], 1800)
    assert ready["pod"]["status"] == "RUNNING" and shells == [{"host": "203.0.113.7", "port": ready["pod"]["ssh"]["direct"]["port"]}]
    assert now[0] == 0  # RUNNING with 22 mapped at the first poll: SSH answered at once, no sleep
    assert State().read()["pods"][pod["id"]]["ssh"] == shells[0]


def test_down_all_terminates_the_recorded_pods_first_and_goes_on_past_any_failure(runpod_repo, capsys):
    world = runpod_repo
    rserve.setup(world.provider())
    provider = world.provider()
    first = provider.create("serve", tagged("slm-bench-serve-a", token="a1"), "a")
    second = provider.create("serve", tagged("slm-bench-serve-b", token="b2"), "b")
    stray = client(world.fake).create_pod(tagged("slm-bench-loadtest-c", token="c3"))
    world.fake.fail += [{"method": "DELETE", "path": f"/v2/pods/{first['id']}", "status": 400, "times": 99},
                        {"method": "GET", "path": "/v2/pods", "status": 500, "times": 99}]
    result = rserve.down(world.provider(), everything=True)
    assert [d["id"] for d in result["done"]] == [second["id"]]  # past the first one's failure
    assert result["failed"] and result["failed"][0].startswith(first["id"])
    assert "could not be listed" in result["warnings"][0]  # the listing failed: a warning, not a stop
    assert stray["id"] in world.fake.pods  # unrecorded and unlisted: the warning says to check the console
    assert cli.main(["serve", "down", "--on", "runpod", "--all", "--config", str(world.config_path)]) == 2
    err = capsys.readouterr().err
    assert "warning: the pods could not be listed" in err and "could not terminate every pod" in err


# ---------------------------------------------------------------- the serving pod

def test_up_serves_the_modal_plan_on_a_gpu_pod_behind_ssh(runpod_repo):
    world = runpod_repo
    ready_volume(world)
    served_now = rserve.up(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    assert served_now["base_url"] == f"http://127.0.0.1:{world.port}/v1" and served_now["reused"] is False
    body = world.fake.created("serve")[0]
    assert body["gpu"] == {"id": "NVIDIA A100 80GB PCIe", "count": 1, "minCudaVersion": "13.0"}
    assert body["image"] == "vllm/vllm-openai:v0.30.0" and body["cloud"] == "SECURE"
    assert body["dataCenterIds"] == ["CA-MTL-1"] and body["ports"] == ["22/tcp"]  # vLLM's port is not published
    assert body["env"]["VLLM_API_KEY"] == "{{ RUNPOD_SECRET_slm-bench-vllm-api-key }}"  # RunPod substitutes it
    # vLLM's POST /reset_prefix_cache (a cold level of the load test), reached through the tunnel only (above)
    assert body["env"]["VLLM_SERVER_DEV_MODE"] == "1"
    assert body["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAAoperator me@laptop"
    assert body["env"]["BENCH_DEADLINE_S"] == str(12 * 3600)
    assert body["entrypoint"] == ["bash", "-c"] and body["cmd"] == [rp.BOOTSTRAP]
    assert "from-the-secret" not in json.dumps(world.fake.requests)  # no credential value goes to RunPod
    # the plan is Modal's, so the vLLM command is identical: adapters at /adapters/<sha256>
    plan = common.serve_plan(world.config, TINY_NAME)
    assert world.served_plan == plan
    sha = facts.sha256_dir(paths.ROOT / "train/adapters/c0/adapter")
    assert f"{served('c0')}=/adapters/{sha}" in common.vllm_command(plan)
    assert facts.sha256_dir(world.tmp / "volume" / "adapters" / sha) == sha and served_now["uploaded_adapters"] == [sha]
    shell = world.shells[served_now["pod_id"]]
    assert (shell.pod_dir / "bench/code/runpod_apps/job.py").is_file()
    assert (shell.pod_dir / "bench/code/env/train/requirements.lock").is_file()
    record = State().serving(TINY_NAME)
    assert record["tunnel_pid"] == 424242 and record["local_port"] == world.port and record["observed_gpus"] == ["NVIDIA A100 80GB PCIe"]
    assert world.tunnels == [("203.0.113.7", 40001, world.port)]  # the pod's direct SSH, never the HTTPS proxy


def test_up_again_reuses_the_pod_and_reopens_only_a_dead_tunnel(runpod_repo, monkeypatch):
    world = runpod_repo
    ready_volume(world)
    rserve.up(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    monkeypatch.setattr(rserve, "is_tunnel", lambda pid: pid == 424242)
    again = rserve.up(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    assert again["reused"] is True and len(world.fake.created("serve")) == 1 and len(world.tunnels) == 1
    monkeypatch.setattr(rserve, "is_tunnel", lambda pid: False)  # the laptop slept: the tunnel died
    rserve.ensure_tunnel(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    assert len(world.tunnels) == 2


def test_up_serves_anew_when_the_recorded_pod_is_gone_and_keeps_a_warm_pod_when_only_the_tunnel_fails(runpod_repo):
    world = runpod_repo
    ready_volume(world)
    first = rserve.up(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    del world.fake.pods[first["pod_id"]]  # past its deadline, or terminated in the console
    second = rserve.up(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    assert second["reused"] is False and second["pod_id"] != first["pod_id"]
    assert list(State().read()["pods"]) == [second["pod_id"]]
    rserve.down(world.provider(), TINY_NAME)

    def busy(*args):
        raise RunPodError("local port 18001 is already in use")
    with pytest.raises(RunPodError, match="The server is up \\(pod .*billing\\)"):
        rserve.up(world.provider(), TINY_NAME, tunnel=busy)
    assert len(world.fake.pods) == 1 and State().serving(TINY_NAME)  # warm, recorded: `tunnel` or `down`


def test_the_pod_saves_only_the_secrets_its_jobs_need(tmp_path):
    lines = [line for line in rp.BOOTSTRAP.splitlines() if line.startswith(("names=", "( umask 077"))]
    script = "W=" + str(tmp_path) + "\n" + "\n".join(lines)
    env = {"PATH": "/usr/bin:/bin", "VLLM_API_KEY": "vk", "RUNPOD_API_KEY": "pod-scoped", "BENCH_UV_VERSION": "0.9.5"}
    subprocess.run(["bash", "-c", script], env=env, check=True)
    saved = (tmp_path / "env").read_text()
    assert "VLLM_API_KEY" in saved and "BENCH_UV_VERSION" in saved and "RUNPOD_API_KEY" not in saved
    assert oct((tmp_path / "env").stat().st_mode)[-3:] == "600"
    subprocess.run(["bash", "-c", script], env={"PATH": "/usr/bin:/bin", "RUNPOD_API_KEY": "pod-scoped"}, check=True)
    assert (tmp_path / "env").read_text() == ""  # nothing to save is nothing, never every variable


@pytest.mark.parametrize("missing", ["weights", "secret", "auth", "endpoint"])
def test_up_refuses_and_leaves_nothing_billing(runpod_repo, missing):
    world = runpod_repo
    rserve.setup(world.provider())
    if missing != "weights":
        rserve.download(world.provider(), TINY_NAME)
    if missing == "secret":
        world.fake.secrets = []
    elif missing == "auth":
        world.config["serving"]["unauthenticated"] = False
    elif missing == "endpoint":
        world.config["roles"]["slm_candidates"][-1]["endpoint"]["headers_env"] = {"Modal-Key": "MK"}
    message = {"weights": "is not on the volume: bench serve --on runpod download", "secret": "no RunPod secret",
               "auth": "unauthenticated is false", "endpoint": "Modal proxy headers"}[missing]
    with pytest.raises(RunPodError, match=message):
        rserve.up(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    assert not world.fake.pods and State().read()["pods"] == {}
    assert len(world.fake.created("serve")) == (1 if missing == "weights" else 0)


def test_a_server_that_exits_before_it_is_warm_is_an_error_with_its_log(runpod_repo):
    world = runpod_repo
    ready_volume(world)

    def crash(shell, name, command):
        shell.local("/bench/jobs/serve.log").write_text("vllm: CUDA out of memory\n")
        return 1
    world.jobs["serve"] = crash
    with pytest.raises(RunPodError, match="(?s)exited with 1 before it was ready.*CUDA out of memory"):
        rserve.up(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    assert not world.fake.pods


def test_status_and_the_cli(runpod_repo, capsys):
    world = runpod_repo
    ready_volume(world)
    rserve.up(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    rows = rserve.status(world.provider())["pods"]
    assert [(r["role"], r["status"], r["usd_per_hour"], r["spent_usd_estimate"]) for r in rows] == [("serve", "RUNNING", 1.09, 0.545)]
    assert cli.main(["serve", "status", "--on", "runpod", "--config", str(world.config_path)]) == 0
    assert "serve RUNNING $1.09/h ~$0.545" in capsys.readouterr().out
    assert cli.main(["serve", "up", "--on", "runpod", "--config", str(world.config_path)]) == 2
    assert "up needs --candidate" in capsys.readouterr().err
    assert cli.main(["serve", "down", "--on", "runpod", "--all", "--config", str(world.config_path)]) == 0
    assert "terminated" in capsys.readouterr().out and not world.fake.pods


def test_without_a_runpod_key_the_cli_says_so(runpod_repo, monkeypatch, capsys):
    monkeypatch.undo()  # the real make_provider
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    config_path = runpod_repo.config_path
    assert cli.main(["serve", "status", "--on", "runpod", "--config", str(config_path)]) == 2
    assert "RUNPOD_API_KEY is not set" in capsys.readouterr().err


# ---------------------------------------------------------------- the load test, and J8 on it

def aiperf_job(world, monkeypatch):
    from bench import loadtest

    def fake_aiperf(run_dir, cmd):
        (run_dir / "aiperf.log").write_text("Authorization: Bearer from-the-secret")  # AIPerf leaks headers
        (run_dir / loadtest.EXPORT).write_text(json.dumps({
            "request_latency": {"unit": "ms", "avg": 400.0, "p95": 900.0},
            "request_throughput": {"unit": "requests/sec", "avg": 2.5},
            "overall_usage_prompt_cache_read_pct": {"unit": "%", "avg": 61.0}}))
        return 0

    def run(shell, name, command):
        from runpod_apps import job

        args_path, run_dir = re.search(r"job aiperf (\S+) (\S+)", command).groups()
        monkeypatch.setattr(loadtest, "run_aiperf", fake_aiperf)
        monkeypatch.setattr(loadtest, "local_aiperf", lambda: sys.executable)
        monkeypatch.setenv("VLLM_API_KEY", "from-the-secret")  # the pod's, from the RunPod secret
        return job.main(["aiperf", str(shell.local(args_path)), str(shell.local(run_dir))])
    return run


def test_the_load_test_runs_beside_the_server_and_j8_prices_it_at_runpods_price(runpod_repo, monkeypatch):
    from bench import loadtest
    from bench.judge import j8
    from bench.judge.base import read_result
    from fixtures.fake import call, write_run

    world = runpod_repo
    ready_volume(world)
    rserve.up(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    world.jobs["aiperf"] = aiperf_job(world, monkeypatch)
    monkeypatch.delenv("T_SLM_KEY")  # the client's key is the pod's secret: none is read here
    engine = f"slm:{TINY_NAME}+lora:{served('c0')}"
    run_dirs = loadtest.loadtest(str(world.config_path), engine, _source_run(calls=SOURCE), "runpod", concurrency=[1, 4])
    manifests = [json.loads((d / "manifest.json").read_text()) for d in run_dirs]
    for m, level in zip(manifests, (1, 4)):
        assert (m["type"], m["status"], m["where"], m["engine"], m["concurrency"]) == ("loadtest", "done", "runpod", engine, level)
        assert m["gpu"] == "A100-80GB" and m["gpu"] in world.config["modal"]["gpu_prices"]["usd_per_s"]
        assert m["prefix_cache"] is True and m["sweep_id"] == manifests[0]["sweep_id"] and m["source_run_id"]
        assert m["observed"]["gpus"] == ["NVIDIA A100 80GB PCIe"] and m["observed"]["prompt_cache_read_pct"] == 61.0
        assert m["secrets_redacted_in"] == ["aiperf.log"]
        assert m["server_versions"] == {"vllm": "0.30.0", "torch": "2.13.0", "transformers": "5.18.0", "flashinfer-python": "0.9.1"}  # the serving image's, as it recorded them
        top = max(world.config["loadtest"]["concurrency"])
        assert m["tunnel_calibration"]["overhead_ms"] == {"1": 3.5, str(top): 3.5}  # client via tunnel - server
    direct, through = world.calibrated  # once per sweep: on the server against vLLM's port, then through the tunnel
    assert direct["base_url"] == "http://127.0.0.1:8000/v1" and through["base_url"] == f"http://127.0.0.1:{world.port}/v1"
    assert direct["concurrency"] == [1, max(world.config["loadtest"]["concurrency"])] and direct["model"] == TINY_NAME
    for d in run_dirs:
        assert "from-the-secret" not in (d / "aiperf.log").read_text() and (d / loadtest.EXPORT).is_file()
        assert not (d / "result.json").exists()
    # one CPU client pod in the server's data center, terminated; its key no longer authorized on the server
    client_body, = world.fake.created("loadtest")
    assert client_body["cpu"]["vcpuCount"] == world.config["loadtest"]["client"]["cpu"]
    assert client_body["dataCenterIds"] == world.fake.created("serve")[0]["dataCenterIds"]
    assert client_body["env"]["VLLM_API_KEY"] == "{{ RUNPOD_SECRET_slm-bench-vllm-api-key }}"
    assert [p["name"].rsplit("-", 1)[0] for p in world.fake.pods.values()] == [f"slm-bench-serve-{TINY_NAME}"]
    server = world.shells[State().serving(TINY_NAME)["id"]]
    assert "bench-loadtest" not in (server.pod_dir / "root/.ssh/authorized_keys").read_text()
    tunnel_cmd, = world.ssh
    assert f"-L 127.0.0.1:{world.port}:localhost:8000 -p 40001 root@203.0.113.7" in tunnel_cmd
    assert all(o in tunnel_cmd for o in ("Compression=no", "IPQoS=none", "ForwardAgent=no"))
    # J8 over the runs, with the patch's prices: the pod's price per second, CPU and memory bundled (0)
    patched = copy.deepcopy(world.config)
    patched["modal"]["gpu_prices"] = {"as_of": "2026-11-09", "usd_per_s": {"A100-80GB": 0.000497},
                                      "cpu_usd_per_core_s": 0, "memory_usd_per_gib_s": 0}
    patched["cost"].update(p95_slo_cap_ms=5000, slo_from=None)
    pilot = "agent-B0-calib-pilot"
    write_run(pilot, {"type": "agent", "arm": "B0", "split": "calib"},
              [call(pilot, str(q), "select_tables", latency_ms=3000) for q in (1, 2, 3)])
    result = read_result(j8.run([d.name for d in run_dirs], patched, slo_from=pilot), "J8")["result"]
    assert result["gpu"] == "A100-80GB" and result["gpu_prices_as_of"] == "2026-11-09"
    assert result["price_per_hour"] == pytest.approx(0.000497 * 3600)
    assert result["cost_per_request"]["100%"] == pytest.approx(0.000497 * 3600 / (2.5 * 3600))


def test_the_load_test_needs_the_server_up(runpod_repo):
    from bench import loadtest

    world = runpod_repo
    rserve.setup(world.provider())
    with pytest.raises(RunPodError, match="no pod serves"):
        loadtest.loadtest(str(world.config_path), f"slm:{TINY_NAME}", _source_run(calls=SOURCE), "runpod", concurrency=[1])
    assert not world.fake.pods


# ---------------------------------------------------------------- P-4 and the training time

def parity_job(reference):
    def run(shell, name, command):
        spec_path, out_path = re.search(r"job parity (\S+) (\S+)", command).groups()
        spec = json.loads(shell.local(spec_path).read_text())
        adapter = shell.local(f"/runpod-volume/adapters/{spec['adapter_sha256']}")
        assert facts.sha256_dir(adapter) == spec["adapter_sha256"]  # what the job checks in the pod
        out = reference(spec["base"], spec["prompts"], spec["template_kwargs"], spec["max_new_tokens"], spec["top_k"])
        serving = shell.local(spec["serving_state"])
        out.update(function_seconds=31.5, versions={"torch": "2.14.1", "transformers": "5.18.0", "peft": "0.21.1"},
                   **({"serving_state": json.loads(serving.read_text())} if serving.is_file() else {}))
        shell.local(out_path).write_text(json.dumps(out))
        return 0
    return run


def serving_for_p4(world, monkeypatch):
    """The candidate served on its pod (`up`), the server stop recorded and the tunnel faked; the served
    generations come from a FakeVLLM. The order of the jobs and of the served requests goes in world.events."""
    from runpod_apps import parity
    from test_preflight import FakeVLLM, _reference, _sha, cards, gen

    ready_volume(world)
    served_now = rserve.up(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    prompts = [json.loads(line)["prompt"] for line in (paths.ROOT / "train/datasets/c0.jsonl").read_text().splitlines()]
    server = FakeVLLM({TINY_NAME: [gen([10 + i, 20, 30]) for i in range(len(prompts))],
                       served("c0"): [gen([40 + i, 50, 60]) for i in range(len(prompts))]}, prompts, cards(_sha()))
    world.events = []
    serve = world.jobs["serve"]

    def stop(shell, name, command):
        assert "nvidia-smi" in command and "$JOB serve" in command  # the server's processes, then its VRAM
        world.events.append(("serve-stop", sum(1 for r in server.requests if r["body"])))
        shell.local("/bench/jobs/serve.ready").unlink()
        return 0

    def restart(shell, name, command):
        world.events.append(("serve", None))
        return serve(shell, name, command)

    world.jobs.update({"serve-stop": stop, "serve": restart})
    monkeypatch.setattr(parity, "open_tunnel", fake_tunnel(world))
    world.config["roles"]["slm_candidates"][-1]["endpoint"]["base_url"] = server.base_url
    reference, _ = _reference(prompts)
    return served_now, server, prompts, reference


def test_p4_on_runpod_runs_the_reference_in_the_serving_pod_between_two_servers(runpod_repo, monkeypatch):
    from bench.preflight import PASS, lora_parity
    from runpod_apps import parity

    world = runpod_repo
    served_now, server, prompts, reference = serving_for_p4(world, monkeypatch)
    world.jobs["parity"] = parity_job(reference)
    returned = []
    original = parity.peft_reference
    monkeypatch.setattr(parity, "peft_reference", lambda *a: returned.append(original(*a)) or returned[-1])
    try:
        verdict = lora_parity(world.config, "c0", "runpod")
    finally:
        server.close()
    assert verdict["status"] == PASS and verdict["reference_on"] == "runpod" and verdict["reference_gpu_seconds"] == 31.5
    assert verdict["reference_versions"]["peft"] == "0.21.1" and verdict["serving_versions"]["vllm"] == "0.30.0"
    shell = world.shells[served_now["pod_id"]]
    assert shell.started == ["serve", "serve-stop", "parity", "serve"]  # stopped, the reference, served again
    assert world.events[0] == ("serve-stop", 2 * len(prompts))  # every served generation came before the stop
    assert not world.fake.created("parity") and len(world.fake.created("serve")) == 1  # no pod of its own
    assert list(world.fake.pods) == [served_now["pod_id"]] and not world.fake.terminated[1:]  # the server lives on
    assert returned[0]["pod_id"] == served_now["pod_id"] and returned[0]["usd_per_hour"] == 1.09
    assert (shell.pod_dir / "bench/jobs/serve.ready").is_file()  # warm again
    assert State().serving(TINY_NAME)["observed_gpus"] == ["NVIDIA A100 80GB PCIe"]
    assert world.tunnels[-1] == ("203.0.113.7", 40001, world.port)  # health checked through the tunnel
    command = next(c for c in shell.commands if c.startswith("# bench-job parity"))
    assert "env/train/requirements.lock" in command and "/opt/bench-train" in command  # the train lock's environment


def _p4_args():
    manifest = json.loads((paths.ROOT / "train/adapters/c0/manifest.json").read_text())
    return manifest["adapter_sha256"], manifest["base"], manifest["chat_template_kwargs"]


def test_p4_on_runpod_serves_again_when_the_reference_fails(runpod_repo, monkeypatch):
    from runpod_apps import parity

    world = runpod_repo
    served_now, server, prompts, _ = serving_for_p4(world, monkeypatch)
    server.close()
    world.jobs["parity"] = lambda shell, name, command: 1
    sha, base, kwargs = _p4_args()
    with pytest.raises(RunPodError, match="job parity exited with 1"):
        parity.peft_reference(world.config, "c0", sha, base, prompts, kwargs, 8, 5)
    shell = world.shells[served_now["pod_id"]]
    assert shell.started == ["serve", "serve-stop", "parity", "serve"]
    assert (shell.pod_dir / "bench/jobs/serve.ready").is_file() and list(world.fake.pods) == [served_now["pod_id"]]


def test_p4_on_runpod_keeps_the_reference_when_the_server_does_not_come_back(runpod_repo, monkeypatch, capsys):
    """A server that does not come back is reported beside the reference's result, never in place of it."""
    from runpod_apps import parity

    world = runpod_repo
    served_now, server, prompts, reference = serving_for_p4(world, monkeypatch)
    server.close()
    world.jobs["parity"] = parity_job(reference)
    world.jobs["serve"] = lambda shell, name, command: 1  # vLLM exits before it is warm
    sha, base, kwargs = _p4_args()
    returned = parity.peft_reference(world.config, "c0", sha, base, prompts, kwargs, 8, 5)
    assert returned["base"] and returned["adapter"] and returned["function_seconds"] == 31.5  # the reference, kept
    assert returned["pod_id"] == served_now["pod_id"]
    assert re.search(f"(?s)the server of {TINY_NAME} did not come back.*job serve exited.*still billing", returned["restart_error"])
    assert "did not come back" in capsys.readouterr().err
    assert world.shells[served_now["pod_id"]].started == ["serve", "serve-stop", "parity", "serve"]


def test_p4_on_runpod_keeps_the_references_error_when_the_server_does_not_come_back_either(runpod_repo, monkeypatch):
    from runpod_apps import parity

    world = runpod_repo
    served_now, server, prompts, _ = serving_for_p4(world, monkeypatch)
    server.close()
    world.jobs["parity"] = lambda shell, name, command: 1
    world.jobs["serve"] = lambda shell, name, command: 1
    sha, base, kwargs = _p4_args()
    with pytest.raises(RunPodError, match="job parity exited with 1") as raised:  # the reference's error, not the restart's
        parity.peft_reference(world.config, "c0", sha, base, prompts, kwargs, 8, 5)
    notes = " ".join(getattr(raised.value, "__notes__", []))
    assert f"the server of {TINY_NAME} did not come back" in notes and "still billing" in notes


def test_p4_on_runpod_starts_no_second_server_when_the_stop_stopped_nothing(runpod_repo, monkeypatch):
    """Without nvidia-smi the stop exits 3 before it touches anything: the server still serves, and starting
    another beside it would hold the GPU twice. P-4 fails, saying why, and leaves the server alone."""
    from runpod_apps import parity

    world = runpod_repo
    served_now, server, prompts, reference = serving_for_p4(world, monkeypatch)
    server.close()
    world.jobs["serve-stop"] = lambda shell, name, command: parity.NO_GPU_VIEW
    world.jobs["parity"] = parity_job(reference)
    sha, base, kwargs = _p4_args()
    with pytest.raises(RunPodError, match="no nvidia-smi.*left serving, untouched"):
        parity.peft_reference(world.config, "c0", sha, base, prompts, kwargs, 8, 5)
    shell = world.shells[served_now["pod_id"]]
    assert shell.started == ["serve", "serve-stop"]  # no reference, no second server
    assert (shell.pod_dir / "bench/jobs/serve.ready").is_file()  # the first one, still up


def test_p4_on_runpod_stop_signals_only_processes_it_found_in_the_containers_proc():
    """nvidia-smi's PIDs may be the host's inside a container: the stop reads the GPU's memory from it, never
    whom to kill. Every kill names a process group or a PID found under /proc."""
    from runpod_apps import parity

    script = parity.STOP_SERVE.format(timeout=300, free=1024, work="/bench", no_gpu_view=parity.NO_GPU_VIEW)
    kills = [line for line in script.splitlines() if "kill " in line]
    assert kills and not any("nvidia-smi" in line for line in kills)
    assert "--query-compute-apps" not in script
    assert all(re.search(r'kill -(TERM|KILL) (-- "-\$g"|\$\(server\)|\$alive)', line) for line in kills)
    assert 'alive=$( { server; members; }' in script  # the server and every member of its process groups
    assert script.index(f"exit {parity.NO_GPU_VIEW}") < script.index("kill ")  # exit 3 before anything is touched


def test_p4_on_runpod_needs_a_serving_pod_and_fails_with_that_reason(runpod_repo):
    from bench.preflight import FAIL, check_lora_parity
    from test_preflight import FakeVLLM, _sha, cards, gen

    world = runpod_repo
    rserve.setup(world.provider())  # a volume, but no `up`
    prompts = [json.loads(line)["prompt"] for line in (paths.ROOT / "train/datasets/c0.jsonl").read_text().splitlines()]
    server = FakeVLLM({name: [gen([1, 2]) for _ in prompts] for name in (TINY_NAME, served("c0"))}, prompts, cards(_sha()))
    world.config["roles"]["slm_candidates"][-1]["endpoint"]["base_url"] = server.base_url
    try:
        failed = check_lora_parity(world.config, "c0", "runpod")
    finally:
        server.close()
    assert failed["status"] == FAIL and f"RunPodError: no running pod serves {TINY_NAME}" in failed["evidence"]["error"]
    assert not world.fake.created()  # no pod made for it


def test_p4_on_runpod_never_imports_modal(monkeypatch):
    from bench import preflight

    class NoModal:
        def find_spec(self, name, path=None, target=None):
            if name == "modal" or name.startswith("modal."):
                raise AssertionError(f"imported {name}")
            return None

    monkeypatch.delitem(sys.modules, "modal", raising=False)
    monkeypatch.delitem(sys.modules, "modal.exception", raising=False)
    monkeypatch.setattr(sys, "meta_path", [NoModal(), *sys.meta_path])
    errors = preflight._unreachable("runpod")
    assert RunPodError in errors and preflight.PreflightError in errors


def test_training_time_reads_tinker_manifests_when_training_on_tinker(tmp_path, monkeypatch):
    from bench.preflight import PASS, PENDING, check_training_time
    from test_preflight import count_tokens, trained

    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    count_tokens(monkeypatch, {"c0": 90_000, "c1": 100})
    trained("c1", config, where="tinker", gpu="none", tokens_per_second=50.0)
    config.pop("runpod", None)  # no runpod section: the registered backend, Modal
    assert check_training_time(config)["status"] == PENDING  # no Modal run yet
    config["runpod"] = {"train_backend": "tinker"}
    measured = check_training_time(config)
    assert measured["status"] == PASS and measured["evidence"]["backend"] == "tinker"
    assert measured["evidence"]["projected_train_hours"] == 1.0
    assert check_training_time(config, "modal")["status"] == PENDING
    assert "bench train --on modal" in check_training_time(config, "modal")["evidence"]["needs"]


def test_preflight_cli_takes_runpod_and_the_training_backend(tmp_path, monkeypatch):
    import bench.preflight as preflight

    _, config_path, _ = make_s5_repo(tmp_path, monkeypatch)
    seen = {}
    monkeypatch.setattr(preflight, "preflight", lambda *a: seen.setdefault("args", a) and 0)
    assert cli.main(["preflight", "--config", str(config_path), "--on", "runpod", "--train-backend", "tinker"]) == 0
    assert seen["args"] == (str(config_path), None, "runpod", "tinker")


# ---------------------------------------------------------------- the pods' script, the jobs, the archive

def test_the_boot_script_and_the_training_environment_are_valid_bash():
    for script in (rp.BOOTSTRAP, rp.TRAIN_ENV + 'echo "$PY"'):
        assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode == 0
    assert "PasswordAuthentication=no" in rp.BOOTSTRAP and "/adapters" in rp.BOOTSTRAP
    assert "umask 077" in rp.BOOTSTRAP and "RUNPOD_API_KEY|" not in rp.BOOTSTRAP  # the pod's own key is not saved
    assert "UV_PYTHON_PREFERENCE=only-managed" in rp.TRAIN_ENV


def test_a_pod_has_a_gpu_or_a_cpu_count():
    settings = shipped_runpod_section()
    volume = {"id": "v", "datacenter": "EU-RO-1"}
    with pytest.raises(RunPodError):
        rp.pod_body(settings, volume, "n", "i", 60, "k")
    with pytest.raises(RunPodError):
        rp.pod_body(settings, volume, "n", "i", 60, "k", gpu="NVIDIA A100 80GB PCIe", cpu_vcpus=4)


def test_jobs_run_detached_and_their_exit_code_is_the_completion_signal(tmp_path):
    shell = LocalShell(tmp_path / "pod", tmp_path / "volume")  # no interception: bash and nohup, as in a pod
    (tmp_path / "pod" / "bench" / "code").mkdir(parents=True)
    remote = Remote(shell, sleep=lambda s: __import__("time").sleep(0.05))
    remote.start_job("ok", 'echo "key=$VLLM_API_KEY" > /bench/out.txt')
    assert remote.wait_job("ok", 10, 0.05) == 0
    assert (tmp_path / "pod" / "bench" / "out.txt").read_text() == "key=from-the-secret\n"  # the saved environment
    remote.start_job("bad", "echo working; exit 3")
    with pytest.raises(RunPodError, match="(?s)job bad exited with 3.*working"):
        remote.wait_job("bad", 10, 0.05)
    remote.start_job("slow", "sleep 5")
    with pytest.raises(RunPodError, match="did not finish within 0 s"):
        remote.wait_job("slow", 0, 0.05)


def test_the_code_archive_holds_the_code_and_the_locks_only():
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(rp.code_archive()), mode="r:gz") as tar:
        names = set(tar.getnames())
    assert {"bench/preflight.py", "bench/contracts/calls.schema.json", "modal_apps/common.py", "runpod_apps/job.py",
            "env/train/requirements.lock", "env/train/serve-vllm.lock"} <= names
    assert not [n for n in names if n.startswith("tests/") or "__pycache__" in n or n.startswith("runs/")]
    assert rp.code_archive() == rp.code_archive()  # reproducible: its sha256 identifies the code


def test_the_serve_job_checks_the_vllm_version_and_the_weights_then_records_itself(tmp_path, monkeypatch):
    from runpod_apps import job

    plan = {"name": TINY_NAME, "repo": TINY["repo"], "revision": TINY["revision"], "vllm_version": "0.30.0"}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(job, "HF_COMPLETE", str(tmp_path / "complete"))
    monkeypatch.setattr(job, "READY", str(tmp_path / "serve.ready"))
    monkeypatch.setattr(common, "VLLM_CACHE", str(tmp_path / "vllm"))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.29.1"))
    with pytest.raises(common.SettingsError, match="vLLM 0.29.1"):
        job.serve(str(tmp_path / "plan.json"))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.30.0"))
    with pytest.raises(common.SettingsError, match="is not on the volume"):
        job.serve(str(tmp_path / "plan.json"))
    marker = Path(job.complete_marker(TINY["repo"], TINY["revision"]))
    marker.parent.mkdir(parents=True)
    marker.write_text("x")

    def start(plan, environ, launch):
        state = common.state_path(common.VLLM_CACHE, plan["name"])
        state.parent.mkdir(parents=True)
        state.write_text(json.dumps({"gpus": ["NVIDIA A100 80GB PCIe"], "auth": "vllm-key"}))
        return types.SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(common, "start_serving", start)
    assert job.versions(["pytest", "no-such-distribution"]) == {"pytest": pytest.__version__, "no-such-distribution": None}
    monkeypatch.setattr(job, "versions", lambda packages: {p: "x" for p in packages})
    monkeypatch.setenv("BENCH_IMAGE", "vllm/vllm-openai:v0.30.0")
    assert job.serve(str(tmp_path / "plan.json")) == 0
    state = json.loads(Path((tmp_path / "serve.ready").read_text()).read_text())
    assert state == {"gpus": ["NVIDIA A100 80GB PCIe"], "auth": "vllm-key", "provider": "runpod",
                     "image": "vllm/vllm-openai:v0.30.0", "pod_id": None, "vllm_version": "0.30.0",
                     "versions": {"vllm": "x", "torch": "x", "transformers": "x", "flashinfer-python": "x"}}


def test_a_tunnel_is_only_ever_an_ssh_process():
    import os

    from runpod_apps.ssh import close_tunnel, is_tunnel, ssh_options

    assert not is_tunnel(None) and not is_tunnel(os.getpid()) and close_tunnel(os.getpid()) is False
    options = " ".join(ssh_options("/k", 40001, Path("/kh/pod1")))
    assert "BatchMode=yes" in options and "StrictHostKeyChecking=accept-new" in options
    assert "UserKnownHostsFile=/kh/pod1" in options  # a host key per pod: a reused IP and port never mismatch
    assert "ForwardAgent=no" in options  # the operator's agent never reaches a pod
    from runpod_apps.ssh import TUNNEL_OPTIONS
    assert TUNNEL_OPTIONS == ["-o", "Compression=no", "-o", "IPQoS=none"]


def test_the_calibration_job_times_requests_at_each_concurrency(tmp_path, monkeypatch):
    from runpod_apps import job
    from test_loadtest import RecordingChat

    server = RecordingChat()
    try:
        spec = {"base_url": server.base_url, "model": "m", "concurrency": [1, 3], "requests": {"1": 2, "3": 3}}
        (tmp_path / "in.json").write_text(json.dumps(spec))
        monkeypatch.setenv("VLLM_API_KEY", "vk")
        assert job.main(["calibrate", str(tmp_path / "in.json"), str(tmp_path / "out.json")]) == 0
    finally:
        server.server.shutdown()
    levels = json.loads((tmp_path / "out.json").read_text())["levels"]
    assert [(lv["concurrency"], lv["requests"], lv["errors"]) for lv in levels] == [(1, 2, 0), (3, 3, 0)]
    assert all(lv["p50_ms"] > 0 for lv in levels) and len(server.bodies) == 5
    assert server.bodies[0]["max_tokens"] == 8 and server.bodies[0]["temperature"] == 0.0


def test_up_refuses_a_serving_pod_it_did_not_record_and_hours_that_are_not_hours(runpod_repo):
    world = runpod_repo
    ready_volume(world)
    for hours in (0, -1, float("inf"), float("nan")):
        with pytest.raises(RunPodError, match="--hours must be a finite number"):
            rserve.up(world.provider(), TINY_NAME, hours=hours, tunnel=fake_tunnel(world))
    unrecorded = client(world.fake).create_pod(tagged(f"slm-bench-serve-{TINY_NAME}", token="ab12",
                                                      gpu={"id": "NVIDIA A100 80GB PCIe", "count": 1}))
    world.fake.pods[unrecorded["id"]]["env"].update(BENCH_ROLE="serve", BENCH_CANDIDATE=TINY_NAME)
    with pytest.raises(RunPodError, match=f"{unrecorded['id']} already serve {TINY_NAME}.*down --all"):
        rserve.up(world.provider(), TINY_NAME, tunnel=fake_tunnel(world))
    assert len(world.fake.created("serve")) == 1  # the unrecorded one only: no second server


def test_the_state_keeps_every_concurrent_change(tmp_path, monkeypatch):
    import threading

    state = State(tmp_path / ".runpod" / "state.json")
    threads = [threading.Thread(target=state.add_pod, args=({"id": f"p{i}", "name": f"n{i}"}, "serve", f"c{i}"))
               for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(state.read()["pods"]) == sorted(f"p{i}" for i in range(20))
    assert not list((tmp_path / ".runpod").glob("*.tmp"))  # each write's own temporary file, renamed or removed


def train_lock_script(volume: Path, pod_id: str) -> str:
    """The build lock of rp.TRAIN_ENV, alone: from taking the lock to holding it (the heartbeat stopped at once)."""
    lines = rp.TRAIN_ENV.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith('L="$V.lock"'))
    end = next(i for i, line in enumerate(lines) if "trap 'kill $HEARTBEAT" in line)
    body = "\n".join(lines[start:end + 1]).replace("/runpod-volume", str(volume))
    return (f'V="{volume}/venvs/train-x"\nRUNPOD_POD_ID={pod_id}\n{body}\n'
            'cat "$L/owner"; kill $HEARTBEAT; trap - EXIT\n')


def test_a_dead_builders_lock_is_broken_after_its_heartbeat_goes_stale(tmp_path):
    import time

    lock = tmp_path / "venvs" / "train-x.lock"
    lock.mkdir(parents=True)
    (lock / "owner").write_text(f"pod-dead {int(time.time()) - 2000}\n")  # no heartbeat for 2000 s > 1200
    started = time.monotonic()
    done = subprocess.run(["bash", "-c", train_lock_script(tmp_path, "pod-new")], capture_output=True, text=True,
                          timeout=60, env={"PATH": "/usr/bin:/bin", "BENCH_LOCK_STALE_S": "1200"})
    assert done.returncode == 0 and time.monotonic() - started < 10
    assert "breaking" in done.stdout and "no heartbeat from pod-dead" in done.stdout
    assert done.stdout.strip().splitlines()[-1].startswith("pod-new ")  # the new builder holds it, beating


def test_a_lock_whose_pod_no_longer_runs_is_removed_from_the_local_side(runpod_repo):
    import hashlib

    world = runpod_repo
    rserve.setup(world.provider())
    provider = world.provider()
    alive = client(world.fake).create_pod(tagged("slm-bench-loadtest-x", token="aa"))
    pod = {"id": "podX", "ssh": {"direct": {"host": "h", "port": 1}}}
    remote = provider.remote(pod)
    sha = hashlib.sha256((rp.REPO / "env/train/requirements.lock").read_bytes()).hexdigest()[:16]
    lock = world.tmp / "volume" / "venvs" / f"train-{sha}.lock"
    lock.mkdir(parents=True)
    (lock / "owner").write_text(f"{alive['id']} 1\n")
    assert provider.clear_stale_train_lock(remote) is None and lock.is_dir()  # its pod runs: a builder at work
    (lock / "owner").write_text("pod-gone 1\n")
    assert provider.clear_stale_train_lock(remote) == "pod-gone" and not lock.exists()


# ---------------------------------------------------------------- the config patch

def test_the_config_patch_turns_the_pre_patch_config_into_the_shipped_one_and_says_what_j8_reads(tmp_path):
    from bench.judge import j8

    config_path = tmp_path / "config.yaml"
    config_path.write_text(unpatched_text())
    assert "runpod" not in yaml.safe_load(config_path.read_text())
    shown = APPLY.apply(REPO / "configs" / "runpod.patch.yaml", config_path)
    assert shown["written"] is False and config_path.read_text() == unpatched_text()
    assert "+  unauthenticated: true" in shown["diff"] and shown["sha256_before"] != shown["sha256_after"]
    APPLY.apply(REPO / "configs" / "runpod.patch.yaml", config_path, write=True)
    from bench.contracts.config import config_sha256, load_config

    config = load_config(config_path)
    assert config_sha256(config) == shown["sha256_after"]
    assert j8.container_price(config, "A100-80GB") == {"gpu": 0.000497, "cpu": 0, "memory": 0}
    assert config["serving"]["unauthenticated"] is True and config["serving"]["gpu"] == "A100-80GB"
    settings = rs.runpod_settings(config)
    for entry in config["roles"]["slm_candidates"]:
        assert rserve.check_endpoint(config, settings, entry["name"]) == settings["local_ports"][entry["name"]]
    assert "# Modal's price for the same GPU" in config_path.read_text()  # every comment kept, Modal's price too
    assert config_path.read_text() == (REPO / "config.yaml").read_text()  # the patch is what config.yaml ships
    with pytest.raises(APPLY.PatchError, match="occurs 0 times"):  # applied once only
        APPLY.apply(REPO / "configs" / "runpod.patch.yaml", config_path)
    assert APPLY.main([str(REPO / "configs" / "runpod.patch.yaml"), "--config", str(config_path)]) == 2


def test_runpod_settings_say_what_is_missing():
    with pytest.raises(RunPodError, match="apply configs/runpod.patch.yaml"):
        rs.runpod_settings({})
    with pytest.raises(RunPodError, match="lacks"):
        rs.runpod_settings({"runpod": {"cloud": "SECURE"}})
    settings = shipped_runpod_section()
    assert rs.gpu_id(settings, "A100-80GB") == "NVIDIA A100 80GB PCIe"
    with pytest.raises(RunPodError, match="H100"):
        rs.gpu_id(settings, "H100")
    with pytest.raises(RunPodError, match="RUNPOD_API_KEY is not set"):
        rs.api_key(settings, {})
