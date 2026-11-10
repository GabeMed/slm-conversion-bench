"""The co-located load client's choices and samples (runpod_apps/colocated.py) on a fake /proc and cgroup: the
cores AIPerf is pinned to, what the probe records, and the CPU each side used over a level. No pod."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # runpod_apps/ sits beside bench/

from runpod_apps import colocated  # noqa: E402
from runpod_apps import job  # noqa: E402


def test_cpu_lists_read_and_written_in_the_kernels_format():
    assert colocated.parse_cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert colocated.parse_cpu_list("") == []
    assert colocated.format_cpu_list([11, 10, 3, 0, 1, 2, 8]) == "0-3,8,10-11"
    assert colocated.format_cpu_list([248, 249, 250, 251]) == "248-251"


def test_the_server_unpinned_the_last_four_allowed_cpus_are_taken_and_that_is_said():
    allowed = list(range(252))
    choice = colocated.choose_cores(allowed, {101: allowed, 102: allowed})
    assert choice["cores"] == [248, 249, 250, 251] and choice["server_pinned"] is False
    assert choice["server_cores"] == [] and "not pinned" in choice["rule"]


def test_the_cores_the_server_is_pinned_to_are_never_taken():
    allowed = list(range(16))
    choice = colocated.choose_cores(allowed, {101: allowed, 102: [12, 13, 14, 15]})
    assert choice["cores"] == [8, 9, 10, 11] and choice["server_pinned"] is True
    assert choice["server_cores"] == [12, 13, 14, 15] and "12-15" in choice["rule"]
    with pytest.raises(colocated.ColocatedError, match="3 CPU"):
        colocated.choose_cores([0, 1, 2, 3], {101: [3]})


def fake_proc(root: Path, processes, cpus=None):
    """processes: {pid: (ppid, cmdline, utime, stime, affinity)}; cpus: {cpu: [user, nice, system, idle]}."""
    for pid, (ppid, cmdline, utime, stime, affinity) in processes.items():
        d = root / str(pid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "cmdline").write_text(cmdline.replace(" ", "\0") + "\0")
        rest = ["S", str(ppid)] + ["0"] * 9 + [str(utime), str(stime)] + ["0"] * 6
        (d / "stat").write_text(f"{pid} (a (b) c) " + " ".join(rest))  # a comm with parentheses
        (d / "status").write_text(f"Name:\tx\nCpus_allowed_list:\t{affinity}\n")
    lines = ["cpu  1 1 1 1 0 0 0 0"] + [f"cpu{c} {u} {n} {s} {i} 0 0 0 0" for c, (u, n, s, i) in (cpus or {}).items()]
    (root / "stat").write_text("\n".join(lines) + "\n")


def fake_cgroup(root: Path, usage_usec=0, cpuset="0-15", cpu_max="1600000 100000"):
    cg = root / "fs/cgroup"
    cg.mkdir(parents=True, exist_ok=True)
    (cg / "cpuset.cpus.effective").write_text(cpuset + "\n")
    (cg / "cpu.max").write_text(cpu_max + "\n")
    (cg / "cpu.stat").write_text(f"usage_usec {usage_usec}\nuser_usec 0\n")


SERVER = {1: (0, "/bin/bash /start.sh", 0, 0, "0-15"),
          50: (1, "python3 -m runpod_apps.job serve /bench/plan.json", 100, 20, "0-15"),
          51: (50, "vllm serve /model --port 8000", 400, 50, "0-15"),
          52: (51, "VLLM::EngineCore", 3000, 200, "0-15"),
          60: (1, "sshd: root", 5, 5, "0-15")}


def test_the_probe_reads_the_cgroup_the_servers_tree_and_its_affinity(tmp_path, monkeypatch):
    proc, sysfs = tmp_path / "proc", tmp_path / "sys"
    fake_proc(proc, SERVER)
    fake_cgroup(sysfs)
    monkeypatch.setattr(colocated, "own_affinity", lambda: list(range(16)))
    monkeypatch.setattr(colocated.shutil, "which", lambda name: f"/usr/bin/{name}")
    found = colocated.probe(proc, sysfs)
    assert found["server_pids"] == [50, 51, 52]  # the server, vLLM and its EngineCore; not sshd, not init
    assert found["allowed_cpus"] == "0-15" and found["allowed_count"] == 16
    assert found["cgroup"] == {"cpuset_effective": "0-15", "cpu_max": "1600000 100000", "cpu_limit": 16.0}
    assert found["server_affinity"] == ["0-15"] and found["taskset"] == "/usr/bin/taskset"
    assert found["choice"]["cores"] == [12, 13, 14, 15] and found["choice"]["server_pinned"] is False
    json.dumps(found)  # what the job writes


def test_the_probe_falls_back_on_the_jobs_affinity_and_refuses_without_a_server(tmp_path, monkeypatch):
    proc, sysfs = tmp_path / "proc", tmp_path / "sys"
    fake_proc(proc, SERVER)
    (sysfs / "fs/cgroup").mkdir(parents=True)
    (sysfs / "fs/cgroup/cpu.max").write_text("max 100000\n")
    monkeypatch.setattr(colocated, "own_affinity", lambda: list(range(8)))
    found = colocated.probe(proc, sysfs)
    assert found["allowed_cpus"] == "0-7" and found["cgroup"]["cpu_limit"] is None
    assert found["choice"]["cores"] == [4, 5, 6, 7]
    empty = tmp_path / "empty"
    fake_proc(empty, {1: (0, "/bin/bash /start.sh", 0, 0, "0-7")})
    with pytest.raises(colocated.ColocatedError, match="no server process"):
        colocated.probe(empty, sysfs)


def test_the_sample_is_each_sides_cpu_over_the_level(tmp_path, monkeypatch):
    proc, sysfs = tmp_path / "proc", tmp_path / "sys"
    fake_proc(proc, SERVER, cpus={12: [100, 0, 0, 900], 13: [0, 0, 0, 1000]})
    fake_cgroup(sysfs, usage_usec=1_000_000)
    clock = iter([10.0, 20.0])
    sampler = colocated.Sampler([12, 13], proc, sysfs, clock=lambda: next(clock))
    sampler.tick = 100
    children = iter([1.0, 11.0])  # AIPerf's processes: 10 CPU seconds over the 10 s level
    monkeypatch.setattr(sampler, "_children_s", lambda: next(children))
    with sampler:
        fake_proc(proc, {**SERVER, 52: (51, "VLLM::EngineCore", 3000 + 900, 200 + 100, "0-15")},  # +10 s of CPU
                  cpus={12: [400, 0, 0, 1600], 13: [0, 0, 0, 2000]})
        fake_cgroup(sysfs, usage_usec=1_000_000 + 41_000_000)
    out = sampler.result
    assert out["wall_s"] == 10.0
    assert out["aiperf"] == {"cpu_s": 10.0, "cores": 1.0}
    assert out["server"] == {"pids": 3, "cpu_s": 10.0, "cores": 1.0}
    assert out["container"] == {"cpu_s": 41.0, "cores": 4.1}
    assert out["pinned_cores_host_busy"] == {"12": 0.3, "13": 0.0}
    assert out["cpu_limit"] == 16.0 and out["cgroup_cpu_max"] == "1600000 100000"  # the quota used, stated
    assert out["throttling"]["before"] is None  # this fake cgroup shows no counters
    assert out["server_affinity"] == {"50/50": "0-15", "51/51": "0-15", "52/52": "0-15"}
    check = out["contention_check"]  # vLLM unpinned here: it may run on the client's cores
    assert check["pass"] is False and [f[0] for f in check["failed"]] == ["a", "d"]
    assert check["metrics"]["container_share"] == round(4.1 / 16, 3) and check["metrics"]["server_cores_used"] == 1.0
    # (c) is AIPerf's own use, 1 core of its 2 (50%); core 12 being 30% busy host-wide is data only
    assert check["metrics"]["aiperf_pinned_share"] == 0.5 and check["metrics"]["pinned_cores_host_busy_mean"] == pytest.approx(0.15, abs=1e-3)
    assert check["metrics"]["cgroup_cpu_max"] == "1600000 100000" and check["metrics"]["allotted_vcpus"] == 16.0


def test_the_sample_says_none_for_what_the_machine_does_not_show(tmp_path):
    with colocated.Sampler([0], tmp_path / "noproc", tmp_path / "nosys") as sampler:
        pass
    out = sampler.result
    assert out["server"] == {"pids": 0, "cpu_s": None, "cores": None}
    assert out["container"] == {"cpu_s": None, "cores": None} and out["pinned_cores_host_busy"] is None
    assert out["aiperf"]["cpu_s"] >= 0
    assert [f[0] for f in out["contention_check"]["failed"]] == ["a", "b", "d"]  # unmeasured is no proof


def test_the_aiperf_job_samples_only_when_asked(tmp_path, monkeypatch):
    from bench import loadtest

    seen = []

    def fake_run_level(run_dir, args, key, headers, var, aiperf, measure=None):
        seen.append(measure)
        return {"returncode": 0}
    monkeypatch.setattr(loadtest, "run_level", fake_run_level)
    for extra in ({}, {"cpu_sample": {"cores": [1, 2]}}):
        (tmp_path / "args.json").write_text(json.dumps({"headers_env": {}, **extra}))
        assert job.main(["aiperf", str(tmp_path / "args.json"), str(tmp_path)]) == 0
    assert seen[0] is None and isinstance(seen[1](), colocated.Sampler) and seen[1]().cores == [1, 2]


def test_run_level_returns_the_sample_taken_around_aiperf_alone(tmp_path, monkeypatch):
    from bench import loadtest

    order = []

    class Measure:
        result = {"aiperf": "sampled"}

        def __enter__(self):
            order.append("enter")

        def __exit__(self, *exc):
            order.append("exit")

    monkeypatch.setattr(loadtest, "aiperf_command", lambda aiperf, tok: ["aiperf"])
    monkeypatch.setattr(loadtest, "wait_ready", lambda *a: (0.0, {"id": "m"}))
    monkeypatch.setattr(loadtest, "warm_up", lambda *a: order.append("warm-up"))
    monkeypatch.setattr(loadtest, "write_aiperf_config", lambda *a: None)
    monkeypatch.setattr(loadtest, "run_aiperf", lambda run_dir, cmd: order.append("aiperf") or 0)
    args = {"tokenizer": {}, "base_url": "http://x/v1", "model": "m", "ready_timeout_s": 1, "warmup": [],
            "timeout_s": 1, "url": "http://x", "concurrency": 1, "request_count": 1, "stream": False, "headers_env": {}}
    out = loadtest.run_level(tmp_path, args, None, {}, None, "aiperf", measure=Measure)
    assert out["cpu_sample"] == {"aiperf": "sampled"} and order == ["warm-up", "enter", "aiperf", "exit"]
    assert "cpu_sample" not in loadtest.run_level(tmp_path, args, None, {}, None, "aiperf")


def clean_sample(**changes):
    """A level's sample with no contention: no throttling, 4 of 16 vCPUs, the pinned cores 30% busy, the server's
    threads off them."""
    sample = {"throttling": {"before": {"source": "/sys/fs/cgroup/cpu.stat", "nr_throttled": 7, "throttled_usec": 900.0},
                             "after": {"source": "/sys/fs/cgroup/cpu.stat", "nr_throttled": 7, "throttled_usec": 900.0}},
              "container": {"cpu_s": 40.0, "cores": 4.0}, "cpu_limit": 16.0, "cpu_limit_from": "quota",
              "aiperf": {"cpu_s": 8.0, "cores": 0.8},
              "pinned_cores_host_busy": {"12": 0.3, "13": 0.3, "14": 0.3, "15": 0.3},
              "server_affinity": {"50/50": list(range(12)), "52/53": list(range(12))},
              "server": {"pids": 3, "cpu_s": 10.0, "cores": 1.0}}
    sample.update(changes)
    return sample


CORES = [12, 13, 14, 15]


def test_a_level_without_contention_passes_and_says_its_metrics_and_thresholds():
    check = colocated.contention_check(clean_sample(), CORES)
    assert check["pass"] is True and check["failed"] == []
    assert check["metrics"]["container_share"] == 0.25 and check["metrics"]["aiperf_pinned_share"] == 0.2
    assert check["metrics"]["server_pinned_core_overlap"] == [] and check["metrics"]["server_cores_used"] == 1.0
    assert check["thresholds"]["CONTAINER_SHARE_MAX"] == 0.75 and check["thresholds"]["PINNED_BUSY_MAX"] == 0.90


@pytest.mark.parametrize("changes, letter", [
    ({"throttling": {"before": {"nr_throttled": 7, "throttled_usec": 900.0},
                     "after": {"nr_throttled": 8, "throttled_usec": 900.0}}}, "a"),
    ({"throttling": {"before": {"nr_throttled": 7, "throttled_usec": 900.0},
                     "after": {"nr_throttled": 7, "throttled_usec": 1200.0}}}, "a"),
    ({"container": {"cpu_s": 120.0, "cores": 12.0}}, "b"),  # exactly 75% of 16
    ({"aiperf": {"cpu_s": 36.0, "cores": 3.6}}, "c"),  # AIPerf itself used 90% of its 4 cores
    ({"aiperf": {"cpu_s": None, "cores": None}}, "c"),
    ({"server_affinity": {"50/50": list(range(16))}}, "d"),
    ({"server_affinity": {}}, "d"),
])
def test_each_contention_fails_the_level_by_its_letter(changes, letter):
    check = colocated.contention_check(clean_sample(**changes), CORES)
    assert check["pass"] is False and [f[0] for f in check["failed"]] == [letter]


def test_throttling_is_read_from_cgroup_v2_or_v1(tmp_path):
    v2 = tmp_path / "v2/fs/cgroup"
    v2.mkdir(parents=True)
    (v2 / "cpu.stat").write_text("usage_usec 5\nnr_periods 10\nnr_throttled 2\nthrottled_usec 300\n")
    assert colocated.throttling(tmp_path / "v2") == {"source": "/sys/fs/cgroup/cpu.stat", "nr_throttled": 2,
                                                     "throttled_usec": 300.0}
    v1 = tmp_path / "v1/fs/cgroup/cpu,cpuacct"
    v1.mkdir(parents=True)
    (v1 / "cpu.stat").write_text("nr_periods 10\nnr_throttled 3\nthrottled_time 5000\n")
    assert colocated.throttling(tmp_path / "v1")["throttled_usec"] == 5.0
    assert colocated.throttling(tmp_path / "none") is None


def test_the_server_is_kept_off_the_clients_cores_thread_by_thread(tmp_path, monkeypatch):
    proc, sysfs = tmp_path / "proc", tmp_path / "sys"
    fake_proc(proc, SERVER)
    for pid, tid, cpus in ((52, 52, "0-15"), (52, 70, "0-15"), (51, 51, "0-11")):
        task = proc / str(pid) / "task" / str(tid)
        task.mkdir(parents=True)
        (task / "status").write_text(f"Cpus_allowed_list:\t{cpus}\n")
    fake_cgroup(sysfs)
    monkeypatch.setattr(colocated, "own_affinity", lambda: list(range(16)))
    calls = []
    found = colocated.probe(proc, sysfs, isolate_server=True, setter=lambda tid, cpus: calls.append((tid, sorted(cpus))))
    assert found["choice"]["cores"] == [12, 13, 14, 15]
    assert calls == [(50, list(range(12))), (52, list(range(12))), (70, list(range(12)))]  # 51 already off them
    assert found["server_isolated"] == {"threads_changed": 3, "before": ["0-15"], "away_from": "12-15"}
    assert colocated.probe(proc, sysfs)["server_isolated"] is None  # only when asked (the job asks)


def test_a_failed_level_is_reported_and_the_cli_exits_3_after_writing_the_sweep(tmp_path, monkeypatch, capsys):
    from bench import cli, loadtest

    dirs = []
    for index, (concurrency, passed) in enumerate(((1, True), (4, False))):
        d = tmp_path / f"loadtest-x-c{concurrency}"
        d.mkdir()
        check = colocated.contention_check(clean_sample() if passed else clean_sample(container={"cores": 13.0}), CORES)
        (d / "manifest.json").write_text(json.dumps({"run_id": d.name, "level_index": index,
                                                     "concurrency": concurrency, "contention_check": check}))
        dirs.append(d)
    assert loadtest.contention_failures(dirs[:1]) == []
    line, = loadtest.contention_failures(dirs)
    assert line.startswith("CONTENTION: level 2 (concurrency 4") and "failed b: the container used 13.0 of 16.0" in line
    monkeypatch.setattr(loadtest, "loadtest", lambda *a: dirs)
    monkeypatch.setattr(loadtest.paths, "ROOT", tmp_path)
    args = ["loadtest", "--engine", "e", "--source", "s", "--on", "runpod", "--client", "colocated"]
    assert cli.main(args) == loadtest.CONTENTION_EXIT == 3
    err = capsys.readouterr().err
    assert "CONTENTION: level 2" in err and "level 1" not in err
    monkeypatch.setattr(loadtest, "loadtest", lambda *a: dirs[:1])
    assert cli.main(args) == 0


def test_other_tenants_load_on_the_pinned_cores_is_data_not_a_failure():
    check = colocated.contention_check(clean_sample(pinned_cores_host_busy={"12": 0.99, "13": 0.95, "14": 0.9,
                                                                            "15": 0.97}), CORES)
    assert check["pass"] is True and check["metrics"]["pinned_cores_host_busy_mean"] == pytest.approx(0.9525, abs=1e-3)


def test_the_cores_are_the_least_host_busy_not_pinned_to_the_server():
    allowed = list(range(16))
    busy = {c: 0.5 for c in allowed} | {1: 0.0, 3: 0.01, 5: 0.0, 7: 0.02, 9: 0.03, 12: 0.0}
    choice = colocated.choose_cores(allowed, {101: allowed, 102: [12, 13]}, busy=busy)
    assert choice["cores"] == [1, 3, 5, 7] and "least host-busy" in choice["rule"]  # 12 is the server's
    tie = colocated.choose_cores(allowed, {101: allowed}, busy={c: 0.0 for c in allowed})
    assert tie["cores"] == [12, 13, 14, 15]  # ties to the higher core
    assert colocated.choose_cores(allowed, {101: allowed})["cores"] == [12, 13, 14, 15]  # no sample: the last 4


def test_the_probe_samples_the_hosts_load_once_and_chooses_by_it(tmp_path, monkeypatch):
    proc, sysfs = tmp_path / "proc", tmp_path / "sys"
    ticks = {c: [0, 0, 0, 1000] for c in range(16)}
    fake_proc(proc, SERVER, cpus=ticks)
    fake_cgroup(sysfs)
    monkeypatch.setattr(colocated, "own_affinity", lambda: list(range(16)))
    slept = []

    def sleep(seconds):  # over the sample, every core is busy but 2, 4, 6 and 8
        slept.append(seconds)
        fake_proc(proc, SERVER, cpus={c: [0, 0, 0, 2000] if c in (2, 4, 6, 8) else [900, 0, 0, 1100]
                                      for c in range(16)})
    found = colocated.probe(proc, sysfs, sample_s=2.0, sleep=sleep)
    assert slept == [2.0] and found["choice"]["cores"] == [2, 4, 6, 8]
    sample = found["choice"]["host_busy_sample"]
    assert sample["seconds"] == 2.0 and sample["chosen"] == {"2": 0.0, "4": 0.0, "6": 0.0, "8": 0.0}
    assert sample["allowed_least"] == [0.0, 0.0, 0.0, 0.0] and sample["allowed_mean"] == pytest.approx(0.675, abs=1e-3)
