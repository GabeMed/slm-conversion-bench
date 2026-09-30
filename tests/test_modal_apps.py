"""The Modal apps: the serving plan and the vLLM command (design §5.1, "Serving"), the container helpers,
and a structure test of the SDK objects (images, functions, volumes and secrets by name) built without
deploying and without an account. The SDK part runs where `modal` is installed (env/train)."""
import importlib
import json
import sys
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # modal_apps/ sits beside bench/

from bench import paths  # noqa: E402
from bench.contracts import facts  # noqa: E402
from modal_apps import common  # noqa: E402
from test_train_fixtures import TINY, TINY_NAME, fake_adapter, make_s5_repo, rel, save, served  # noqa: E402


def _lock_pins(path):
    return dict(line.split("==", 1) for line in Path(path).read_text().splitlines() if "==" in line and not line.startswith("#"))


def test_the_images_pin_the_design_versions_and_the_config_says_the_same():
    train, serve = _lock_pins(common.TRAIN_LOCK), _lock_pins(common.SERVE_LOCK)
    assert (train["trl"], train["peft"], train["aiperf"], train["modal"]) == ("1.14.1", "0.21.1", "0.13.0", "1.6.0")
    from bench.contracts.config import load_config

    config = load_config(paths.ROOT / "config.yaml")
    assert serve["vllm"] == config["serving"]["vllm_version"] == "0.30.0"
    assert train["aiperf"] == config["loadtest"]["aiperf_version"]
    assert serve["transformers"] == train["transformers"]  # one chat-template renderer for training and serving


def test_lora_rank_is_the_smallest_vllm_accepts():
    assert [common.lora_rank(r) for r in (1, 8, 9, 16, 17, 320, 512)] == [1, 8, 16, 16, 32, 320, 512]
    with pytest.raises(common.SettingsError):
        common.lora_rank(513)


@pytest.fixture
def serving(tmp_path, monkeypatch):
    """A synthetic repo with the tiny candidate and one trained adapter, c0."""
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    fake_adapter("c0", config)
    monkeypatch.setenv("BENCH_CONFIG", str(config_path))
    monkeypatch.setenv("BENCH_CANDIDATE", TINY_NAME)
    for variable in (common.ENV, *common.DEPLOY_KNOBS.values()):
        monkeypatch.delenv(variable, raising=False)
    checked = []  # the template check itself needs transformers: tested apart, in env/train
    monkeypatch.setattr(common, "check_serving_template", lambda entry: checked.append(entry["name"]))
    config["_template_checked"] = checked
    return config_path, config


def test_the_serving_plan_and_the_vllm_command(serving):
    _, config = serving
    plan = common.serve_plan(config, TINY_NAME)
    sha = facts.sha256_dir(paths.ROOT / "train" / "adapters" / "c0" / "adapter")
    assert plan["adapters"] == [{"cluster": "c0", "served_name": f"c0-{sha[:12]}", "sha256": sha, "r": 8}]
    assert config["_template_checked"] == [TINY_NAME]  # the thinking-off check runs without training too
    assert (plan["min_containers"], plan["scaledown_window_s"]) == (0, 300)  # modal_apps/deploy.yaml
    cmd = common.vllm_command(plan)
    arg = {flag: cmd[i + 1] for i, flag in enumerate(cmd[:-1]) if flag.startswith("--")}
    assert cmd[:3] == ["vllm", "serve", TINY["repo"]]
    assert arg["--revision"] == arg["--tokenizer-revision"] == TINY["revision"]
    assert arg["--served-model-name"] == TINY_NAME  # the router's slm:<name>
    assert arg["--lora-modules"] == f"c0-{sha[:12]}=/adapters/{sha}"  # the router's slm:<name>+lora:<served_name>
    assert (arg["--max-loras"], arg["--max-cpu-loras"], arg["--max-lora-rank"]) == ("1", "1", "8")
    assert "--enable-lora" in cmd and "--enable-prefix-caching" in cmd and "--enable-prompt-tokens-details" in cmd
    assert json.loads(arg["--default-chat-template-kwargs"]) == {"enable_thinking": False}
    assert arg["--generation-config"] == "vllm" and arg["--max-model-len"] == str(config["serving"]["max_model_len"])
    # the agent's own <tool_call> text must reach it untouched: no tool or reasoning parser strips it
    assert not {"--enable-auto-tool-choice", "--tool-call-parser", "--reasoning-parser"} & set(cmd)
    no_cache = common.vllm_command({**plan, "prefix_caching": False, "adapters": [], "chat_template_kwargs": {}})
    assert "--no-enable-prefix-caching" in no_cache and "--enable-lora" not in no_cache
    assert "--default-chat-template-kwargs" not in no_cache


def test_the_adapters_fact_decides_what_is_served_once_it_is_this_candidates(serving):
    config_path, config = serving
    fake_adapter("c1", config)
    from bench.train import register_adapters

    fact_path, _ = register_adapters(config)
    config["arms"]["B4"]["adapters"] = rel(fact_path)
    config = save(config, config_path)
    (paths.ROOT / "train" / "adapters" / "stray").mkdir()
    assert [a["cluster"] for a in common.adapters_to_serve(config, TINY_NAME)] == ["c0", "c1"]
    assert common.adapters_to_serve(config, "qwen3-8b") == []  # another candidate: its own adapters (none)
    (paths.ROOT / "train" / "adapters" / "c1" / "adapter" / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(common.SettingsError, match="c1/adapter is not the adapter"):
        common.adapters_to_serve(config, TINY_NAME)


def test_settings_are_computed_locally_and_carried_into_the_container(serving, monkeypatch):
    local = common.load("serve")
    assert local["plan"]["name"] == TINY_NAME and local["volumes"]["adapters"] == "slm-bench-adapters"
    monkeypatch.setenv(common.ENV, json.dumps(local))
    monkeypatch.setenv("BENCH_CONFIG", "/nonexistent.yaml")  # a container never reads config.yaml
    assert common.load("serve") == local
    monkeypatch.delenv(common.ENV)
    monkeypatch.delenv("BENCH_CANDIDATE")
    monkeypatch.setenv("BENCH_CONFIG", str(serving[0]))
    with pytest.raises(common.SettingsError, match="BENCH_CANDIDATE"):
        common.load("serve")


def test_the_container_serves_only_adapters_whose_bytes_are_the_plans(serving, tmp_path):
    _, config = serving
    plan = common.serve_plan(config, TINY_NAME)
    volume = tmp_path / "volume"
    volume.mkdir()
    with pytest.raises(common.SettingsError, match="not on the adapters volume.*upload train/adapters/c0/adapter"):
        common.check_adapters(plan, str(volume))
    import shutil

    shutil.copytree(paths.ROOT / "train" / "adapters" / "c0" / "adapter", volume / plan["adapters"][0]["sha256"])
    common.check_adapters(plan, str(volume))
    (volume / plan["adapters"][0]["sha256"] / "adapter_config.json").write_text("{}")
    with pytest.raises(common.SettingsError):
        common.check_adapters(plan, str(volume))


class FakeVLLMServer:
    def __init__(self, healthy_after=0):
        self.healthy_after, self.gets, self.posts = healthy_after, 0, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self, status, body=b"{}"):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                fake.gets += 1
                self._answer(200 if fake.gets > fake.healthy_after else 503)

            def do_POST(self):
                fake.posts.append((self.headers.get("Authorization"), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self._answer(200, b'{"choices": []}')

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"


def test_wait_healthy_then_warm_up_the_base_and_every_adapter(serving):
    _, config = serving
    plan = common.serve_plan(config, TINY_NAME)
    server = FakeVLLMServer(healthy_after=2)
    try:
        common.wait_healthy(server.base, None, timeout_s=5, poll_s=0.01)
        assert server.gets == 3
        assert common.warm_up(server.base, plan, "k", plan["warmup_timeout_s"]) == [TINY_NAME, served("c0")]
        assert [(auth, body["model"], body["temperature"]) for auth, body in server.posts] == [
            ("Bearer k", TINY_NAME, 0.0), ("Bearer k", served("c0"), 0.0)]
    finally:
        server.server.shutdown()

    class Exited:
        returncode = 1

        def poll(self):
            return 1

    with pytest.raises(common.SettingsError, match="vllm exited with 1"):
        common.wait_healthy("http://127.0.0.1:9", Exited(), timeout_s=5, poll_s=0.01)


# ---------------------------------------------------------------- the SDK objects, offline

def _dockerfile(image, version="2025.06"):
    """The Dockerfile commands of every layer of an image, rendered offline by the pinned SDK (1.6.0)."""
    from modal._utils.async_utils import synchronizer

    layer, commands = synchronizer._translate_in(image), []
    while layer is not None:
        cells = dict(zip(layer._load.__code__.co_freevars, [c.cell_contents for c in layer._load.__closure__ or []]))
        if cells.get("dockerfile_function") is not None:
            commands = list(cells["dockerfile_function"](version).commands) + commands
        parents = [d for d in layer._deps() if type(d).__name__ == "_Image"]
        layer = parents[0] if parents else None
    return commands


def _spec(obj):
    from modal._utils.async_utils import synchronizer

    inner = synchronizer._translate_in(obj)
    return (getattr(inner, "_service_function", None) or inner)._spec


def _names(spec):
    return ({path: repr(v) for path, v in spec.volumes.items()}, sorted(repr(s) for s in spec.secrets))


@pytest.fixture
def apps(serving, monkeypatch):
    pytest.importorskip("modal")
    for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(name, raising=False)
    modules = ("modal_apps.serve_vllm", "modal_apps.train", "modal_apps.loadtest")
    for name in modules:
        sys.modules.pop(name, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = [importlib.import_module(name) for name in modules]
    yield serving[1], loaded
    for name in modules:
        sys.modules.pop(name, None)


def test_the_apps_build_their_objects_without_an_account(apps):
    config, (serve, train, load) = apps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # registered_functions: deprecated after 1.6, the pinned version
        assert (serve.app.name, sorted(serve.app.registered_functions)) == (f"slm-bench-serve-{TINY_NAME}", ["Server", "download"])
        assert (train.app.name, sorted(train.app.registered_functions)) == ("slm-bench-train", ["peft_reference", "train_adapter"])
        assert (load.app.name, sorted(load.app.registered_functions)) == ("slm-bench-loadtest", ["run_aiperf"])
    hf, adapters, vllm = ("modal.Volume.from_name('slm-bench-hf-cache')", "modal.Volume.from_name('slm-bench-adapters')",
                          "modal.Volume.from_name('slm-bench-vllm-cache')")
    key = ["modal.Secret.from_name('slm-bench-vllm-api-key')"]
    proxy = "modal.Secret.from_name('slm-bench-proxy-auth')"

    server = _spec(serve.Server)
    assert server.gpus == config["serving"]["gpu"] and server.cpu == config["serving"]["cpu"]
    offline = "Secret.from_dict([HF_HUB_OFFLINE])"  # the server's env: weights only from the volume, never downloaded
    assert _names(server) == ({common.HF_CACHE: hf, common.VLLM_CACHE: vllm, common.ADAPTERS: adapters}, sorted([offline, *key]))
    assert _names(_spec(serve.download)) == ({common.HF_CACHE: hf}, [])
    for fn in (train.train_adapter, train.peft_reference):
        spec = _spec(fn)
        assert spec.gpus == config["train"]["gpu"]
        assert _names(spec) == ({common.HF_CACHE: hf, common.ADAPTERS: adapters}, [])
    client = _spec(load.run_aiperf)
    assert client.gpus is None and client.cpu == config["loadtest"]["client"]["cpu"]
    # reads the server's state; holds the proxy-auth variables the endpoint's headers_env names
    assert _names(client) == ({common.HF_CACHE: hf, common.VLLM_CACHE: vllm}, sorted([proxy, *key]))
    assert load.image is train.image  # the load client runs in the training image (AIPerf is pinned there)
    # the server's autoscaling and auth, as Modal will deploy them
    from modal._utils.async_utils import synchronizer

    service = synchronizer._translate_in(serve.Server)._service_function
    loader = dict(zip(service._load.__code__.co_freevars, [c.cell_contents for c in service._load.__closure__]))
    assert (loader["min_containers"], loader["max_containers"], loader["scaledown_window"]) == (0, 1, 300)
    assert service._function_info._inner_server_info.unauthenticated is False  # Modal proxy auth


def test_the_images_install_the_locks_and_carry_the_plan(apps):
    config, (serve, train, _) = apps
    serve_commands = _dockerfile(serve.image)
    assert serve_commands[0] == f"FROM {config['serving']['base_image']}"
    assert any(f"--requirements /.uv/0/{common.SERVE_LOCK.name}" in c for c in serve_commands)
    carried = next(c for c in serve_commands if c.startswith(f"ENV {common.ENV}="))
    plan = json.loads(carried.split("=", 1)[1].strip("'"))["plan"]
    assert plan == serve.PLAN
    assert common.vllm_command(plan)[common.vllm_command(plan).index("--lora-modules") + 1].startswith(f"{served('c0')}=")
    train_commands = _dockerfile(train.image)
    assert any(f"--requirements /.uv/0/{common.TRAIN_LOCK.name}" in c for c in train_commands)
    assert json.loads(next(c for c in train_commands if c.startswith(f"ENV {common.ENV}=")).split("=", 1)[1].strip("'")) == train.SETTINGS


def test_serving_refuses_an_adapter_trained_on_other_weights_than_the_candidates(serving):
    config_path, config = serving
    config["roles"]["slm_candidates"][-1]["hf"]["revision"] = "b" * 40
    with pytest.raises(common.SettingsError, match="trained on other weights or template kwargs"):
        common.adapters_to_serve(save(config, config_path), TINY_NAME)


def test_a_finished_training_is_collected_and_a_running_one_is_never_started_twice(tmp_path):
    volume = tmp_path / "adapters"
    assert common.stored_training(str(volume), "p1", now=100.0, stale_after_s=50) is None
    common.mark_started(str(volume), "p1", now=100.0)
    with pytest.raises(common.SettingsError, match="still running"):
        common.stored_training(str(volume), "p1", now=120.0, stale_after_s=50)
    assert common.stored_training(str(volume), "p1", now=151.0, stale_after_s=50) is None  # a stale marker: it died
    adapter = volume / "tmp"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    sha = facts.sha256_dir(adapter)
    adapter.rename(volume / sha)
    common.store_training(str(volume), "p1", {"sha256": sha, "stats": {"global_step": 3}, "function_seconds": 9.0})
    assert not (volume / common.RESULTS / "p1.started").exists()
    stored = common.stored_training(str(volume), "p1", now=999.0, stale_after_s=50)
    assert stored == {"sha256": sha, "stats": {"global_step": 3}, "function_seconds": 9.0, "reused": True,
                      "files": {"adapter_model.safetensors": b"w"}}
    (volume / sha / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(common.SettingsError, match="not on the volume intact"):
        common.stored_training(str(volume), "p1", now=999.0, stale_after_s=50)


def test_the_plan_id_changes_with_anything_the_training_depends_on(tmp_path, monkeypatch):
    from bench.train import plan_id, training_plan

    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    plan, _ = training_plan(config, "c0")
    assert plan_id(plan) == plan_id(json.loads(json.dumps(plan)))
    assert plan_id(plan) != plan_id({**plan, "dataset": {**plan["dataset"], "sha256": "0" * 64}})
    assert plan_id(plan) != plan_id({**plan, "base": {**plan["base"], "revision": "b" * 40}})


def test_the_server_records_the_gpus_it_got_and_what_it_runs(serving):
    _, config = serving
    plan = common.serve_plan(config, TINY_NAME)
    state = common.serving_state(plan, "NVIDIA H200\n\n")  # asked for H100, got H200 (infra.md: Modal may do this)
    assert state["gpus"] == ["NVIDIA H200"] and state["vllm_command"] == common.vllm_command(plan)
    assert state["adapters"] == {served("c0"): plan["adapters"][0]["sha256"]} and state["revision"] == TINY["revision"]
    assert common.state_path("/root/.cache/vllm", TINY_NAME) == Path(f"/root/.cache/vllm/bench-serving/{TINY_NAME}.json")


def test_deploy_knobs_change_without_touching_the_registered_config(serving, monkeypatch):
    from bench.contracts.config import config_sha256, load_config

    config_path, config = serving
    before = config_sha256(load_config(config_path))
    monkeypatch.setenv("BENCH_MIN_CONTAINERS", "1")
    monkeypatch.setenv("BENCH_SCALEDOWN_WINDOW_S", "900")
    plan = common.serve_plan(config, TINY_NAME)
    assert (plan["min_containers"], plan["scaledown_window_s"]) == (1, 900)
    assert config_sha256(load_config(config_path)) == before
    assert not {"min_containers", "scaledown_window_s"} & set(config["serving"])  # not in config.yaml at all
    monkeypatch.setenv("BENCH_MIN_CONTAINERS", "one")
    with pytest.raises(common.SettingsError, match="BENCH_MIN_CONTAINERS must be an integer"):
        common.deploy_knobs()


def test_the_server_fails_closed_without_any_auth():
    assert common.check_auth({"unauthenticated": False}, {}) == "proxy"
    assert common.check_auth({"unauthenticated": False}, {"VLLM_API_KEY": "k"}) == "proxy+vllm-key"
    assert common.check_auth({"unauthenticated": True}, {"VLLM_API_KEY": "k"}) == "vllm-key"
    with pytest.raises(common.SettingsError, match="neither Modal proxy auth nor a VLLM_API_KEY"):
        common.check_auth({"unauthenticated": True}, {"VLLM_API_KEY": ""})


def test_the_shipped_config_serves_behind_proxy_auth_with_headers_for_every_candidate():
    from bench.contracts.config import load_config

    config = load_config(paths.ROOT / "config.yaml")
    assert config["serving"]["unauthenticated"] is False
    for entry in config["roles"]["slm_candidates"]:
        assert set(entry["endpoint"]["headers_env"]) == {"Modal-Key", "Modal-Secret"}
        assert entry["endpoint"]["api_key_env"]  # the server always holds vLLM's key: clients must send it


def test_a_failed_training_releases_its_marker(tmp_path):
    volume = tmp_path / "adapters"
    common.mark_started(str(volume), "p2", now=100.0)
    common.release_started(str(volume), "p2")
    assert common.stored_training(str(volume), "p2", now=101.0, stale_after_s=1000) is None  # not "still running"
    common.release_started(str(volume), "p2")  # idempotent


def test_the_serving_template_check_reads_the_candidates_pinned_tokenizer():
    pytest.importorskip("transformers")
    from bench.train import TrainError

    entry = {"name": TINY_NAME, "hf": dict(TINY), "chat_template_kwargs": {"enable_thinking": False}}
    common.check_serving_template(entry)
    with pytest.raises(TrainError, match="does not read"):
        common.check_serving_template({**entry, "chat_template_kwargs": {"reasoning_mode": "off"}})


def test_a_candidate_whose_template_ignores_its_kwargs_is_not_served(serving, monkeypatch):
    from bench.train import TrainError

    _, config = serving

    def refuse(entry):
        raise TrainError("the chat template does not read ['enable_thinking']")

    monkeypatch.setattr(common, "check_serving_template", refuse)
    with pytest.raises(common.SettingsError, match=f"{TINY_NAME}: the chat template does not read"):
        common.serve_plan(config, TINY_NAME)


# ---------------------------------------------------------------- the container sequences, run for real

class FakeVolume:
    def __init__(self):
        self.commits, self.reloads = 0, 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


def test_run_training_releases_its_marker_whatever_happens(tmp_path):
    root, commits = str(tmp_path / "adapters"), []
    started = Path(root) / common.RESULTS / "k.started"

    def fails():
        assert started.is_file() and commits == [1]  # marked and committed before training
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        common.run_training(root, "k", 1.0, fails, lambda: commits.append(1))
    assert not started.exists() and commits == [1, 1]  # released and committed in the finally
    result = common.run_training(root, "k", 2.0, lambda: {"sha256": "s", "files": {"a": b"x"}}, lambda: None)
    assert result["files"] == {"a": b"x"}
    assert json.loads((Path(root) / common.RESULTS / "k.json").read_text()) == {"sha256": "s"}  # stored without files


def test_start_serving_refuses_without_auth_before_launching_anything(serving, tmp_path):
    _, config = serving
    plan = {**common.serve_plan(config, TINY_NAME), "unauthenticated": True}
    launched = []
    with pytest.raises(common.SettingsError, match="neither Modal proxy auth nor a VLLM_API_KEY"):
        common.start_serving(plan, {}, launch=launched.append, adapters_root=str(tmp_path), state_root=str(tmp_path))
    assert not launched


def test_start_serving_launches_warms_up_and_records_what_it_got(serving, tmp_path):
    import shutil

    _, config = serving
    plan = common.serve_plan(config, TINY_NAME)
    volume = tmp_path / "adapters"
    shutil.copytree(paths.ROOT / "train" / "adapters" / "c0" / "adapter", volume / plan["adapters"][0]["sha256"])
    server, launched, commits = FakeVLLMServer(), [], []

    class Running:
        returncode = None

        def poll(self):
            return None

    def launch(cmd):
        launched.append(cmd)
        return Running()

    try:
        common.start_serving(plan, {"VLLM_API_KEY": "k"}, launch=launch, adapters_root=str(volume),
                             state_root=str(tmp_path / "cache"), commit=lambda: commits.append(1), base=server.base,
                             gpus=lambda: ["NVIDIA H200"])
    finally:
        server.server.shutdown()
    assert launched == [common.vllm_command(plan)] and commits == [1]
    assert [body["model"] for _, body in server.posts] == [TINY_NAME, served("c0")]  # warmed up, base and adapter
    state = json.loads(common.state_path(str(tmp_path / "cache"), TINY_NAME).read_text())
    assert state["gpus"] == ["NVIDIA H200"] and state["auth"] == "proxy+vllm-key"


def test_serving_refuses_a_served_name_its_bytes_do_not_give(serving):
    config_path, config = serving
    fake_adapter("c1", config, served_name="c1")  # the old, non content-addressed name
    with pytest.raises(common.SettingsError, match="served name 'c1' is not 'c1-"):
        common.adapters_to_serve(config, TINY_NAME)


def test_the_training_function_releases_its_marker_when_training_fails(apps, tmp_path, monkeypatch):
    from bench import train as bench_train
    from bench.train import run_identity, training_plan

    config, (_, train, _) = apps
    monkeypatch.setattr(train, "adapters", FakeVolume())
    monkeypatch.setattr(train, "hf_cache", FakeVolume())
    monkeypatch.setattr(common, "ADAPTERS", str(tmp_path / "volume"))

    def out_of_memory(*args):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(bench_train, "train_lora", out_of_memory)
    plan, raw = training_plan(config, "c1")
    with pytest.raises(RuntimeError, match="out of memory"):
        train.train_adapter.local(plan, raw, run_identity(config))
    assert not list((tmp_path / "volume" / common.RESULTS).glob("*.started")) and train.adapters.commits == 2
    assert train.hf_cache.commits == 2  # the weights it downloaded are kept even though training failed


def test_the_training_function_refuses_a_gpu_other_than_its_own(apps, monkeypatch):
    from bench.train import run_identity, training_plan

    config, (_, train, _) = apps
    plan, raw = training_plan(config, "c1")
    with pytest.raises(RuntimeError, match="this function runs on L40S, the plan asks H100"):
        train.train_adapter.local(plan, raw, {**run_identity(config), "gpu": "H100"})


def test_the_training_function_stores_its_own_run_and_a_second_call_collects_it(apps, tmp_path, monkeypatch):
    from bench import train as bench_train
    from bench.train import run_identity, training_plan

    config, (_, train, _) = apps
    monkeypatch.setattr(train, "adapters", FakeVolume())
    monkeypatch.setattr(train, "hf_cache", FakeVolume())
    monkeypatch.setattr(common, "ADAPTERS", str(tmp_path / "volume"))

    def trains(plan, rows, out, device):
        out.mkdir(parents=True)
        (out / "adapter_model.safetensors").write_bytes(b"w")
        return {"global_step": 2}

    monkeypatch.setattr(bench_train, "train_lora", trains)
    monkeypatch.setattr(common, "gpu_names", lambda: ["NVIDIA L40S"])
    plan, raw = training_plan(config, "c1")
    identity = run_identity(config)
    first = train.train_adapter.local(plan, raw, identity)
    assert first["reused"] is False and first["run"]["commit"] == identity["commit"]
    assert first["run"]["observed_gpus"] == ["NVIDIA L40S"] and first["run"]["started_at"] <= first["run"]["finished_at"]
    # collected later, at another commit and price: the stored run is reported, never today's
    later = {**identity, "commit": "d" * 40, "price_usd_per_s": 9.0, "price_as_of": "2027-01-01"}
    again = train.train_adapter.local(plan, raw, later)
    assert again["reused"] is True and again["run"] == first["run"] and again["files"] == first["files"]
    assert again["run"]["commit"] == identity["commit"] and "collection_seconds" in again


def test_the_modal_load_client_scrubs_the_key_and_the_proxy_headers(apps, tmp_path, monkeypatch):
    import shutil
    import sys as system

    from bench import loadtest as bench_loadtest
    from test_loadtest import FakeServer

    _, (_, _, load) = apps
    server = FakeServer(["c0"])
    monkeypatch.setattr(load, "vllm_cache", FakeVolume())
    monkeypatch.setattr(common, "VLLM_CACHE", str(tmp_path / "cache"))
    for name, value in {"VLLM_API_KEY": "vk-secret", "T_MK": "mk-secret", "T_MS": "ms-secret"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(shutil, "which", lambda name: system.executable)

    def aiperf_that_leaks(run_dir, cmd):  # as AIPerf 0.13.0 writes Modal-Key into its export
        (run_dir / "profile_export_aiperf.json").write_text(json.dumps({"headers": {"Modal-Key": "mk-secret"},
                                                                         "key": "vk-secret", "s": "ms-secret"}))
        return 0

    monkeypatch.setattr(bench_loadtest, "run_aiperf", aiperf_that_leaks)
    args = {"url": server.base_url[:-3], "base_url": server.base_url, "model": "c0", "concurrency": 1, "request_count": 1,
            "warmup": [], "tokenizer": TINY, "timeout_s": 5, "stream": False, "ready_timeout_s": 5,
            "headers_env": {"Modal-Key": "T_MK", "Modal-Secret": "T_MS"}, "candidate": TINY_NAME}
    try:
        result = load.run_aiperf.local(b"{}\n", args)
    finally:
        server.close()
    assert result["secrets_redacted_in"] == ["profile_export_aiperf.json"]
    everything = b"".join(result["files"].values())
    assert not any(secret in everything for secret in (b"vk-secret", b"mk-secret", b"ms-secret"))
    assert {(h["Authorization"], h["Modal-Key"], h["Modal-Secret"]) for h in server.headers} == {
        ("Bearer vk-secret", "mk-secret", "ms-secret")}
    assert "${VLLM_API_KEY}" in result["files"]["aiperf.yaml"].decode()


def test_the_server_class_starts_through_start_serving(apps, monkeypatch):
    """The Modal server's enter method is common.start_serving (auth first, fail closed), nothing else."""
    from modal._utils.async_utils import synchronizer

    _, (serve, _, _) = apps
    calls, volume = [], FakeVolume()
    monkeypatch.setattr(serve, "vllm_cache", volume)
    monkeypatch.setattr(common, "start_serving", lambda plan, environ, commit: calls.append((plan, commit)) or "vllm")
    cls = synchronizer._translate_in(serve.Server)._user_cls
    instance = cls.__new__(cls)
    cls.__dict__["start"]._get_raw_f()(instance)
    (plan, commit), = calls
    assert plan == serve.PLAN and instance.process == "vllm"
    assert commit == volume.commit  # its recorded state is committed to the vLLM-cache volume



def test_gpu_names_are_observed_or_unknown_never_an_empty_guess(monkeypatch):
    import subprocess as sp

    class Done:
        def __init__(self, returncode, stdout):
            self.returncode, self.stdout = returncode, stdout

    monkeypatch.setattr(sp, "run", lambda *a, **k: Done(0, "NVIDIA L40S\n\n"))
    assert common.gpu_names() == ["NVIDIA L40S"]
    monkeypatch.setattr(sp, "run", lambda *a, **k: Done(9, ""))
    assert common.gpu_names() is None  # nvidia-smi failed: not observed

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(sp, "run", missing)
    assert common.gpu_names() is None


def test_start_serving_checks_the_adapters_before_launching(serving, tmp_path):
    _, config = serving
    plan = common.serve_plan(config, TINY_NAME)
    launched = []
    with pytest.raises(common.SettingsError, match="not on the adapters volume"):
        common.start_serving(plan, {"VLLM_API_KEY": "k"}, launch=launched.append, adapters_root=str(tmp_path / "empty"),
                             state_root=str(tmp_path / "cache"))
    assert not launched


def test_start_serving_records_its_state_only_after_the_warm_up(serving, tmp_path):
    import shutil

    _, config = serving
    plan = common.serve_plan(config, TINY_NAME)
    volume = tmp_path / "adapters"
    shutil.copytree(paths.ROOT / "train" / "adapters" / "c0" / "adapter", volume / plan["adapters"][0]["sha256"])
    server = FakeVLLMServer()
    handler = server.server.RequestHandlerClass

    def refuse(self):
        self.send_response(500)
        self.send_header("Content-Length", "0")
        self.end_headers()

    handler.do_POST = refuse

    class Running:
        returncode = None

        def poll(self):
            return None

    try:
        with pytest.raises(Exception):
            common.start_serving(plan, {"VLLM_API_KEY": "k"}, launch=lambda cmd: Running(), adapters_root=str(volume),
                                 state_root=str(tmp_path / "cache"), base=server.base, gpus=lambda: ["NVIDIA L40S"])
    finally:
        server.server.shutdown()
    assert not common.state_path(str(tmp_path / "cache"), TINY_NAME).exists()  # a server that never warmed up
