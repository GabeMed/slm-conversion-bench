"""The AIPerf client co-located on the serving pod (`bench loadtest --on runpod --client colocated`): an opt-in
fallback for when the serving pod's data center has no CPU pod to give the load client its own machine. The
default (`--client pod`) is unchanged: runpod_apps/loadtest.py's CPU pod, through its SSH tunnel.

Co-located, AIPerf runs in the serving pod's container, pinned with `taskset` to 4 of the CPUs the container may
use, against vLLM on the loopback (no tunnel). What runs in the pod (`python -m runpod_apps.job cpuset`, and
the CPU sample around each level's AIPerf run) is here, standard library only, read from /proc and the cgroup:

- `probe`: the CPUs the container may use (the cgroup's `cpuset.cpus.effective`, intersected with the job's own
  affinity), `nproc` (the host's, in a container), the cgroup's CPU limit (`cpu.max`: RunPod gives a pod its
  vCPUs as a CFS quota), the processes of the server (`runpod_apps.job serve` and every process under it: vLLM,
  its EngineCore) and their affinity, and the cores chosen (`choose_cores`);
- `Sampler`: over one level's AIPerf run, the CPU seconds of AIPerf (every process under the job, reaped:
  RUSAGE_CHILDREN), of the server's processes (/proc/<pid>/stat deltas), of the whole container (the cgroup's
  usage) and the host-wide busy share of each pinned core (/proc/stat), so the manifest shows the client and the
  server did not compete for CPU.
"""
import json
import os
import resource
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

N_CORES = 4
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


def choose_cores(allowed: List[int], server_affinity: Dict[int, List[int]], n: int = N_CORES) -> Dict[str, Any]:
    """`n` cores for AIPerf among the container's `allowed` CPUs: none the server's processes are pinned to (a
    process is pinned when its affinity is narrower than `allowed`), the last `n` of what is left; the server
    unpinned, the last `n` of `allowed`. Fewer than `n` left is an error."""
    allowed_set = set(allowed)
    pinned = sorted({c for affinity in server_affinity.values() if set(affinity) < allowed_set for c in affinity})
    free = [c for c in sorted(allowed_set) if c not in pinned]
    if len(free) < n:
        raise ColocatedError(f"{len(free)} CPU(s) left for AIPerf beside the server (allowed {format_cpu_list(allowed)}, "
                             f"the server pinned to {format_cpu_list(pinned) or 'none'}): {n} needed")
    if pinned:
        rule = f"the last {n} of the container's CPUs that the server is not pinned to ({format_cpu_list(pinned)})"
    else:
        rule = f"the last {n} of the container's CPUs: the server is not pinned"
    return {"cores": free[-n:], "server_pinned": bool(pinned), "server_cores": pinned, "rule": rule}


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


def probe(proc: Path = Path("/proc"), sysfs: Path = Path("/sys"), n: int = N_CORES) -> Dict[str, Any]:
    """What the co-located client is chosen from, and the choice (see the module's docstring). No server
    process found is an error: the cores would be chosen blind and no level could show the server's CPU."""
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
    return {"nproc": os.cpu_count(), "allowed_cpus": format_cpu_list(allowed), "allowed_count": len(allowed),
            "job_affinity": format_cpu_list(mine) if mine is not None else None, "cgroup": limits,
            "server_pids": pids,
            "server_affinity": sorted({format_cpu_list(a) for a in server_affinity.values()}),
            "choice": choose_cores(allowed, server_affinity, n), "taskset": shutil.which("taskset"),
            "hostname": os.uname().nodename, "pod_id": os.environ.get("RUNPOD_POD_ID")}


class Sampler:
    """The CPU each side used over one AIPerf run (a context manager; `result` once it exits). Cores are CPU
    seconds over wall seconds; a quantity the pod does not show (no /proc, no cgroup file) is None."""

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

        def cores(seconds):
            return round(seconds / wall, 3) if seconds is not None else None
        self.result = {
            "method": "CPU seconds over the AIPerf run: AIPerf's processes reaped (RUSAGE_CHILDREN), the server's "
                      "/proc/<pid>/stat utime+stime deltas, the container's cgroup usage; the pinned cores' "
                      "host-wide busy share from /proc/stat",
            "wall_s": round(wall, 3), "aiperf_affinity": format_cpu_list(own_affinity() or []) or None,
            "aiperf": {"cpu_s": round(aiperf_s, 3), "cores": cores(aiperf_s)},
            "server": {"pids": len(both), "cpu_s": round(server_s, 3) if server_s is not None else None,
                       "cores": cores(server_s)},
            "container": {"cpu_s": round(container_s, 3) if container_s is not None else None,
                          "cores": cores(container_s)},
            "pinned_cores_host_busy": busy or None,
        }
        return False


def write_probe(out_path: str) -> int:
    """`python -m runpod_apps.job cpuset <out.json>`."""
    Path(out_path).write_text(json.dumps(probe()))
    return 0
