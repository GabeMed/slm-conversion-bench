"""The AIPerf client co-located on the serving pod (`bench loadtest --on runpod --client colocated`): an opt-in
fallback for when the serving pod's data center has no CPU pod to give the load client its own machine. The
default (`--client pod`) is unchanged: runpod_apps/loadtest.py's CPU pod, through its SSH tunnel.

Co-located, AIPerf runs in the serving pod's container, pinned with `taskset` to 4 of the CPUs the container may
use, against vLLM on the loopback (no tunnel). What runs in the pod (`python -m runpod_apps.job cpuset`, and
the CPU sample around each level's AIPerf run) is here, standard library only, read from /proc and the cgroup:

- `probe`: the CPUs the container may use (the cgroup's `cpuset.cpus.effective`, intersected with the job's own
  affinity), `nproc` (the host's, in a container), the cgroup's CPU limit (`cpu.max`: RunPod gives a pod its
  vCPUs as a CFS quota), the processes of the server (`runpod_apps.job serve` and every process under it: vLLM,
  its EngineCore) and their affinity, and the cores chosen once per sweep (`choose_cores`: the 4 least
  host-busy over a 2 s /proc/stat sample, none the server is pinned to);
- `Sampler`: over one level's AIPerf run, the CPU seconds of AIPerf (every process under the job, reaped:
  RUSAGE_CHILDREN), of the server's processes (/proc/<pid>/stat deltas), of the whole container (the cgroup's
  usage) and, as data only, the host-wide busy share of each pinned core (/proc/stat: every tenant), with the
  no-contention check on them (`contention_check`).
"""
import json
import os
import resource
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

N_CORES = 4
BUSY_SAMPLE_S = 2.0  # the host-busy sample the cores are chosen by, once per sweep
SERVER_PATTERNS = ("runpod_apps.job serve", "vllm serve")  # parity.STOP_SERVE's: the server and vLLM
CGROUP_CPUSETS = ("fs/cgroup/cpuset.cpus.effective", "fs/cgroup/cpuset/cpuset.effective_cpus")  # v2, v1


class ColocatedError(RuntimeError):
    pass


def parse_cpu_list(text: str) -> List[int]:
    """"0-3,8,10-11" (the kernel's list format) as sorted CPU numbers."""
    cpus = set()
    for part in (text or "").strip().split(","):
        part = part.strip()
        if not part:
            continue
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return sorted(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Sorted CPU numbers in the kernel's list format, as `taskset -c` takes them."""
    ordered, parts = sorted(set(cpus)), []
    for cpu in ordered:
        if parts and cpu == parts[-1][1] + 1:
            parts[-1][1] = cpu
        else:
            parts.append([cpu, cpu])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in parts)


def choose_cores(allowed: List[int], server_affinity: Dict[int, List[int]], n: int = N_CORES,
                 busy: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
    """`n` cores for AIPerf among the container's `allowed` CPUs: none the server's processes are pinned to (a
    process is pinned when its affinity is narrower than `allowed`); of what is left, the `n` least busy on the
    host (`busy`: each core's host-wide busy share, sampled at the sweep's start; the cpuset is shared with
    other tenants), ties to the higher core number; without a sample, the last `n`. Fewer than `n` left is an
    error."""
    allowed_set = set(allowed)
    pinned = sorted({c for affinity in server_affinity.values() if set(affinity) < allowed_set for c in affinity})
    free = [c for c in sorted(allowed_set) if c not in pinned]
    if len(free) < n:
        raise ColocatedError(f"{len(free)} CPU(s) left for AIPerf beside the server (allowed {format_cpu_list(allowed)}, "
                             f"the server pinned to {format_cpu_list(pinned) or 'none'}): {n} needed")
    which = "the container's CPUs" + (f" that the server is not pinned to ({format_cpu_list(pinned)})" if pinned
                                       else " (the server is not pinned)")
    if busy:
        ranked = sorted(free, key=lambda c: (busy.get(c, float("inf")), -c))
        cores, rule = sorted(ranked[:n]), (f"the {n} least host-busy of {which}, by a /proc/stat sample at the "
                                           f"sweep's start, ties to the higher core")
    else:
        cores, rule = free[-n:], f"the last {n} of {which}: no host-busy sample"
    return {"cores": cores, "server_pinned": bool(pinned), "server_cores": pinned, "rule": rule}


def host_busy_sample(proc: Path, cpus: Iterable[int], seconds: float = 2.0,
                     sleep: Callable[[float], None] = time.sleep) -> Dict[int, float]:
    """Each core's host-wide busy share over `seconds` (/proc/stat deltas); a core not shown is left out."""
    cpus = list(cpus)
    before = host_cpu_ticks(proc, cpus)
    sleep(seconds)
    after = host_cpu_ticks(proc, cpus)
    return {c: round((after[c][0] - before[c][0]) / max(after[c][1] - before[c][1], 1), 3)
            for c in cpus if c in before and c in after}


# ---------------------------------------------------------------- /proc and the cgroup

def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text()
    except OSError:
        return None


def processes(proc: Path) -> Dict[int, Dict[str, Any]]:
    """{pid: {"ppid", "cmdline"}} of every process /proc shows."""
    found = {}
    for entry in proc.glob("[0-9]*"):
        stat, cmdline = _read(entry / "stat"), _read(entry / "cmdline")
        if stat is None or cmdline is None or ")" not in stat:
            continue
        found[int(entry.name)] = {"ppid": int(stat.rsplit(")", 1)[1].split()[1]),
                                  "cmdline": cmdline.replace("\0", " ").strip()}
    return found


def server_pids(proc: Path, exclude: Iterable[int] = ()) -> List[int]:
    """The server's processes: each whose command line names it (SERVER_PATTERNS), and every process under one."""
    table, skip = processes(proc), set(exclude)
    tree = {pid for pid, p in table.items() if pid not in skip and any(s in p["cmdline"] for s in SERVER_PATTERNS)}
    grew = True
    while grew:
        children = {pid for pid, p in table.items() if p["ppid"] in tree and pid not in tree and pid not in skip}
        tree |= children
        grew = bool(children)
    return sorted(tree)


def affinity(proc: Path, pid: int) -> Optional[List[int]]:
    for line in (_read(proc / str(pid) / "status") or "").splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return parse_cpu_list(line.split(":", 1)[1])
    return None


def cpu_ticks(proc: Path, pid: int) -> Optional[int]:
    """utime + stime of one process, in clock ticks."""
    stat = _read(proc / str(pid) / "stat")
    if not stat or ")" not in stat:
        return None
    fields = stat.rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])


def host_cpu_ticks(proc: Path, cpus: Iterable[int]) -> Dict[int, List[int]]:
    """{cpu: [busy, total]} ticks from /proc/stat (the host's counters, every tenant's work)."""
    wanted, out = set(cpus), {}
    for line in (_read(proc / "stat") or "").splitlines():
        name, *values = line.split()
        if name.startswith("cpu") and name[3:].isdigit() and int(name[3:]) in wanted:
            ticks = [int(v) for v in values]
            idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)  # idle + iowait
            out[int(name[3:])] = [sum(ticks[:8]) - idle, sum(ticks[:8])]
    return out


def cgroup_usage_s(sysfs: Path) -> Optional[float]:
    """The container's CPU seconds so far (cgroup v2 cpu.stat usage_usec, or v1 cpuacct.usage)."""
    for line in (_read(sysfs / "fs/cgroup/cpu.stat") or "").splitlines():
        if line.startswith("usage_usec "):
            return int(line.split()[1]) / 1e6
    v1 = _read(sysfs / "fs/cgroup/cpuacct/cpuacct.usage")
    return int(v1) / 1e9 if v1 and v1.strip().isdigit() else None


def cgroup_limits(sysfs: Path) -> Dict[str, Any]:
    """The cgroup's cpuset and CPU quota as the files say them, and the quota in CPUs."""
    cpuset = next((text.strip() for text in (_read(sysfs / p) for p in CGROUP_CPUSETS) if text is not None), None)
    cpu_max, cpus = _read(sysfs / "fs/cgroup/cpu.max"), None
    if cpu_max is not None:
        quota, _, period = cpu_max.strip().partition(" ")
        cpus = int(quota) / int(period) if quota.isdigit() and period.isdigit() else None
        cpu_max = cpu_max.strip()
    else:
        quota, period = (_read(sysfs / f"fs/cgroup/cpu/cpu.cfs_{k}_us") for k in ("quota", "period"))
        if quota and period and quota.strip().lstrip("-").isdigit() and int(quota) > 0:
            cpus, cpu_max = int(quota) / int(period), f"{quota.strip()} {period.strip()}"
    return {"cpuset_effective": cpuset, "cpu_max": cpu_max, "cpu_limit": cpus}


def own_affinity() -> Optional[List[int]]:
    getter = getattr(os, "sched_getaffinity", None)
    return sorted(getter(0)) if getter else None


def task_affinities(proc: Path, pids: Iterable[int]) -> Dict[str, List[int]]:
    """{"<pid>/<tid>": affinity} of every thread of `pids` (a process's threads may differ from its main one)."""
    out = {}
    for pid in pids:
        for task in sorted((proc / str(pid) / "task").glob("[0-9]*"), key=lambda t: int(t.name)):
            found = affinity(proc / str(pid) / "task", int(task.name))
            if found is not None:
                out[f"{pid}/{task.name}"] = found
        if not any(k.startswith(f"{pid}/") for k in out) and (found := affinity(proc, pid)) is not None:
            out[f"{pid}/{pid}"] = found  # no task directory shown: the process's own
    return out


def isolate(proc: Path, pids: Iterable[int], away: Iterable[int],
            setter: Optional[Callable[[int, set], None]] = None) -> Dict[str, Any]:
    """Every thread of the server's processes kept off the cores in `away` (the client's), its affinity otherwise
    as it was: what makes the server's affinity disjoint from the client's (criterion d). Returns the threads
    changed and the affinities before."""
    setter = setter or getattr(os, "sched_setaffinity", None)
    if setter is None:
        raise ColocatedError("this system cannot set a process's affinity (no sched_setaffinity)")
    off, changed, before = set(away), [], {}
    for key, cpus in task_affinities(proc, pids).items():
        if off & set(cpus):
            keep = set(cpus) - off
            if not keep:
                raise ColocatedError(f"thread {key} of the server may run only on the client's cores {sorted(off)}")
            setter(int(key.split("/")[1]), keep)
            changed.append(key)
            before[key] = format_cpu_list(cpus)
    return {"threads_changed": len(changed), "before": sorted(set(before.values())),
            "away_from": format_cpu_list(off)}


def probe(proc: Path = Path("/proc"), sysfs: Path = Path("/sys"), n: int = N_CORES,
          isolate_server: bool = False, setter: Optional[Callable[[int, set], None]] = None,
          sample_s: float = 0.0, sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """What the co-located client is chosen from, and the choice (see the module's docstring). No server
    process found is an error: the cores would be chosen blind and no level could show the server's CPU. With
    `isolate_server`, the server's threads that may run on the chosen cores are kept off them (`isolate`), and
    the server's affinity after that is recorded."""
    limits = cgroup_limits(sysfs)
    mine = own_affinity()
    allowed = parse_cpu_list(limits["cpuset_effective"]) if limits["cpuset_effective"] else list(mine or [])
    if mine is not None:
        allowed = sorted(set(allowed) & set(mine))
    if not allowed:
        raise ColocatedError("the CPUs this container may use could not be read (no cgroup cpuset, no affinity)")
    pids = server_pids(proc, exclude=[os.getpid()])
    if not pids:
        raise ColocatedError(f"no server process ({' or '.join(SERVER_PATTERNS)}) runs in this pod")
    server_affinity = {pid: a for pid in pids if (a := affinity(proc, pid)) is not None}
    busy = host_busy_sample(proc, allowed, sample_s, sleep) if sample_s else {}
    choice = choose_cores(allowed, server_affinity, n, busy)
    choice["host_busy_sample"] = {"seconds": sample_s, "source": "/proc/stat (host-wide, every tenant)",
                                  "chosen": {str(c): busy.get(c) for c in choice["cores"]},
                                  "allowed_mean": round(sum(busy.values()) / len(busy), 3) if busy else None,
                                  "allowed_least": sorted(busy.values())[:n] if busy else None}
    out = {"nproc": os.cpu_count(), "allowed_cpus": format_cpu_list(allowed), "allowed_count": len(allowed),
           "job_affinity": format_cpu_list(mine) if mine is not None else None, "cgroup": limits,
           "server_pids": pids,
           "server_affinity": sorted({format_cpu_list(a) for a in server_affinity.values()}),
           "choice": choice, "taskset": shutil.which("taskset"),
           "hostname": os.uname().nodename, "pod_id": os.environ.get("RUNPOD_POD_ID"), "server_isolated": None}
    if isolate_server:
        out["server_isolated"] = isolate(proc, pids, choice["cores"], setter)
        out["server_affinity_after"] = sorted({format_cpu_list(a) for a in task_affinities(proc, pids).values()})
    return out


# ---------------------------------------------------------------- the no-contention criterion, per level

CONTAINER_SHARE_MAX = 0.75  # (b) the container's CPU over the level, as a share of its allotted vCPUs: below
PINNED_BUSY_MAX = 0.90      # (c) AIPerf's own use of its pinned cores over the level: below
THROTTLE_FILES = (("fs/cgroup/cpu.stat", 1.0), ("fs/cgroup/cpu,cpuacct/cpu.stat", 1e-3),
                  ("fs/cgroup/cpu/cpu.stat", 1e-3))  # cgroup v2 (throttled_usec), v1 (throttled_time, ns)


def throttling(sysfs: Path) -> Optional[Dict[str, Any]]:
    """The cgroup's throttling counters: {"source", "nr_throttled", "throttled_usec"}; None when not shown."""
    for rel, to_usec in THROTTLE_FILES:
        fields = {}
        for line in (_read(sysfs / rel) or "").splitlines():
            name, _, value = line.partition(" ")
            if value.strip().isdigit():
                fields[name] = int(value)
        if "nr_throttled" in fields:
            spent = fields.get("throttled_usec", fields.get("throttled_time"))
            return {"source": f"/sys/{rel}", "nr_throttled": fields["nr_throttled"],
                    "throttled_usec": round(spent * to_usec, 3) if spent is not None else None}
    return None


def contention_check(sample: Dict[str, Any], cores: List[int]) -> Dict[str, Any]:
    """The level ran without contention unless one of these holds (a quantity not measured fails too: no proof):
    (a) the cgroup throttled over the level (nr_throttled or throttled_usec grew);
    (b) the container used >= CONTAINER_SHARE_MAX of its allotted vCPUs (the cgroup's quota) over the level;
    (c) AIPerf's own use of its pinned cores reached PINNED_BUSY_MAX over the level: its CPU seconds (every
        process under the job, reaped) / the pinned cores / the wall time (client-bound throughput). The
        cores' host-wide busy share (/proc/stat, every tenant) is recorded beside it, as data only;
    (d) a thread of the server may run on a pinned core (its affinity, at the level's start and end, is not
        disjoint from them), or no server process was seen. The server's CPU is recorded beside it."""
    failed = []
    before, after = (sample.get("throttling") or {}).get("before"), (sample.get("throttling") or {}).get("after")
    if before is None or after is None:
        failed.append("a: the cgroup's throttling counters were not readable")
    elif after["nr_throttled"] > before["nr_throttled"] or (after["throttled_usec"] or 0) > (before["throttled_usec"] or 0):
        failed.append(f"a: the cgroup throttled (nr_throttled {before['nr_throttled']} -> {after['nr_throttled']}, "
                      f"throttled_usec {before['throttled_usec']} -> {after['throttled_usec']})")
    used, limit = (sample.get("container") or {}).get("cores"), sample.get("cpu_limit")
    share = round(used / limit, 3) if used is not None and limit else None
    if share is None:
        failed.append("b: the container's CPU use or its allotted vCPUs were not readable")
    elif share >= CONTAINER_SHARE_MAX:
        failed.append(f"b: the container used {used} of {limit} vCPUs ({share:.0%} >= {CONTAINER_SHARE_MAX:.0%})")
    busy = sample.get("pinned_cores_host_busy") or {}
    host_mean = round(sum(busy.values()) / len(busy), 3) if busy else None
    aiperf_cores = (sample.get("aiperf") or {}).get("cores")
    own = round(aiperf_cores / len(cores), 3) if aiperf_cores is not None and cores else None
    if own is None:
        failed.append("c: AIPerf's CPU over the level was not readable")
    elif own >= PINNED_BUSY_MAX:
        failed.append(f"c: AIPerf used {own:.0%} of its {len(cores)} pinned cores (>= {PINNED_BUSY_MAX:.0%}): "
                      f"client-bound")
    affinities = sample.get("server_affinity") or {}
    overlap = sorted({c for cpus in affinities.values() for c in cpus} & set(cores))
    if not affinities:
        failed.append("d: no server thread's affinity was seen")
    elif overlap:
        failed.append(f"d: the server may run on the pinned cores {format_cpu_list(overlap)}")
    return {"pass": not failed, "failed": failed,
            "metrics": {"throttling_before": before, "throttling_after": after, "container_cores": used,
                        "allotted_vcpus": limit, "allotted_from": sample.get("cpu_limit_from"),
                        "cgroup_cpu_max": sample.get("cgroup_cpu_max"),
                        "container_share": share, "pinned_cores": cores, "aiperf_cores": aiperf_cores,
                        "aiperf_pinned_share": own, "pinned_cores_host_busy_mean": host_mean,
                        "server_threads": len(affinities), "server_pinned_core_overlap": overlap,
                        "server_cores_used": (sample.get("server") or {}).get("cores")},
            "thresholds": {"CONTAINER_SHARE_MAX": CONTAINER_SHARE_MAX, "PINNED_BUSY_MAX": PINNED_BUSY_MAX,
                           "c": "AIPerf's CPU seconds / pinned cores / wall; host-wide busy is data only",
                           "throttling": "any increase of nr_throttled or throttled_usec",
                           "server_affinity": "disjoint from the pinned cores"},
            "source": "runpod_apps/colocated.py:contention_check"}


class Sampler:
    """The CPU each side used over one AIPerf run (a context manager; `result` once it exits), and the
    no-contention criterion on it (`result["contention_check"]`). Cores are CPU seconds over wall seconds; a
    quantity the pod does not show (no /proc, no cgroup file) is None."""

    def __init__(self, cores: List[int], proc: Path = Path("/proc"), sysfs: Path = Path("/sys"),
                 clock: Callable[[], float] = time.monotonic):
        self.cores, self.proc, self.sysfs, self.clock = list(cores), proc, sysfs, clock
        self.result: Optional[Dict[str, Any]] = None
        self.tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100

    def _children_s(self) -> float:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return usage.ru_utime + usage.ru_stime

    def __enter__(self):
        self.pids = server_pids(self.proc, exclude=[os.getpid()])
        self.affinity0 = task_affinities(self.proc, self.pids)
        self.throttle0 = throttling(self.sysfs)
        self.started = self.clock()
        self.server0 = {pid: cpu_ticks(self.proc, pid) for pid in self.pids}
        self.children0, self.cgroup0 = self._children_s(), cgroup_usage_s(self.sysfs)
        self.host0 = host_cpu_ticks(self.proc, self.cores)
        return self

    def __exit__(self, *exc):
        wall = max(self.clock() - self.started, 1e-9)
        server1 = {pid: cpu_ticks(self.proc, pid) for pid in self.pids}
        both = [pid for pid in self.pids if self.server0.get(pid) is not None and server1.get(pid) is not None]
        server_s = sum(server1[p] - self.server0[p] for p in both) / self.tick if both else None
        aiperf_s = self._children_s() - self.children0
        cgroup1 = cgroup_usage_s(self.sysfs)
        container_s = cgroup1 - self.cgroup0 if cgroup1 is not None and self.cgroup0 is not None else None
        host1 = host_cpu_ticks(self.proc, self.cores)
        busy = {str(c): round((host1[c][0] - self.host0[c][0]) / max(host1[c][1] - self.host0[c][1], 1), 3)
                for c in self.cores if c in host1 and c in self.host0}
        throttle1 = throttling(self.sysfs)
        affinities = {**self.affinity0}
        for key, cpus in task_affinities(self.proc, self.pids).items():  # the union of start and end
            affinities[key] = sorted(set(affinities.get(key, [])) | set(cpus))
        limits = cgroup_limits(self.sysfs)
        limit, limit_from = limits["cpu_limit"], "the cgroup's CPU quota (cpu.max or cfs_quota_us/cfs_period_us)"
        if limit is None and limits["cpuset_effective"]:
            limit, limit_from = len(parse_cpu_list(limits["cpuset_effective"])), "the cgroup's cpuset (no quota)"

        def cores(seconds):
            return round(seconds / wall, 3) if seconds is not None else None
        self.result = {
            "method": "CPU seconds over the AIPerf run: AIPerf's processes reaped (RUSAGE_CHILDREN), the server's "
                      "/proc/<pid>/stat utime+stime deltas, the container's cgroup usage; the pinned cores' "
                      "host-wide busy share from /proc/stat; the cgroup's throttling counters (cpu.stat) and the "
                      "server threads' affinity (/proc/<pid>/task/<tid>/status) at the start and the end",
            "wall_s": round(wall, 3), "aiperf_affinity": format_cpu_list(own_affinity() or []) or None,
            "aiperf": {"cpu_s": round(aiperf_s, 3), "cores": cores(aiperf_s)},
            "server": {"pids": len(both), "cpu_s": round(server_s, 3) if server_s is not None else None,
                       "cores": cores(server_s)},
            "container": {"cpu_s": round(container_s, 3) if container_s is not None else None,
                          "cores": cores(container_s)},
            "cpu_limit": limit, "cpu_limit_from": limit_from if limit is not None else None,
            "cgroup_cpu_max": limits["cpu_max"],
            "pinned_cores_host_busy": busy or None,
            "throttling": {"before": self.throttle0, "after": throttle1},
            "server_affinity": {k: format_cpu_list(v) for k, v in sorted(affinities.items())} or None,
        }
        self.result["contention_check"] = contention_check(
            {**self.result, "server_affinity": affinities}, self.cores)
        return False


def write_probe(out_path: str) -> int:
    """`python -m runpod_apps.job cpuset <out.json>`: the probe, the server kept off the chosen cores."""
    Path(out_path).write_text(json.dumps(probe(isolate_server=True, sample_s=BUSY_SAMPLE_S)))
    return 0
