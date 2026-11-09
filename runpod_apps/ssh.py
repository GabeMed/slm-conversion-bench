"""Driving a pod over SSH, behind a small interface the tests fake (`Shell`: run, put, get), and the local SSH
tunnel that makes a serving pod's vLLM port reachable as http://127.0.0.1:<port>.

Why SSH and not RunPod's HTTPS proxy: the proxy (<pod>-<port>.proxy.runpod.net) cuts every request at 100 s
(Cloudflare 524), and an agent call may take up to `params.timeout_s` (600 s). The pods publish only 22/tcp; vLLM
listens inside the pod and is reached through the tunnel, its own key (VLLM_API_KEY, a RunPod secret) the
auth layer behind SSH's.

Jobs (`Remote.start_job`) run detached in the pod (nohup), each writing its exit code to
`/bench/jobs/<name>.rc` when it ends: that file is the completion signal, polled over SSH, so a job outlives a
dropped connection and a sleeping laptop.
"""
import os
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional

from runpod_apps.settings import WORK, RunPodError

JOB_HEADER = "# bench-job "


class Result(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


class Shell:
    """A shell in one pod: `run` a bash command, `put` a local file or directory at a remote path, `get` one back."""

    def run(self, command: str, timeout_s: float = 600) -> Result:
        raise NotImplementedError

    def put(self, local: Path, remote: str) -> None:
        raise NotImplementedError

    def get(self, remote: str, local: Path) -> None:
        raise NotImplementedError


def ssh_options(key: str, port: int, known_hosts: Path) -> List[str]:
    """A new host key per pod (each pod has its own known_hosts file), never a password, never a prompt."""
    return ["-i", key, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={known_hosts}", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            "-o", "LogLevel=ERROR"]


class SshShell(Shell):
    def __init__(self, host: str, port: int, key: str, known_hosts: Path, user: str = "root"):
        self.host, self.port, self.key, self.known_hosts, self.user = host, port, key, known_hosts, user
        known_hosts.parent.mkdir(parents=True, exist_ok=True)

    def run(self, command: str, timeout_s: float = 600) -> Result:
        cmd = ["ssh", *ssh_options(self.key, self.port, self.known_hosts), "-p", str(self.port),
               f"{self.user}@{self.host}", "bash -c " + shlex.quote(command)]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return Result(255, "", f"ssh: no answer within {timeout_s} s")
        return Result(done.returncode, done.stdout, done.stderr)

    def _scp(self, source: str, target: str) -> None:
        cmd = ["scp", "-r", "-q", *ssh_options(self.key, self.port, self.known_hosts), "-P", str(self.port), source, target]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RunPodError(f"scp {source} -> {target} failed ({done.returncode}): {done.stderr.strip()[:300]}")

    def put(self, local: Path, remote: str) -> None:
        self._scp(str(local), f"{self.user}@{self.host}:{remote}")

    def get(self, remote: str, local: Path) -> None:
        self._scp(f"{self.user}@{self.host}:{remote}", str(local))


class Remote:
    """Files and detached jobs in one pod, through its Shell."""

    def __init__(self, shell: Shell, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.shell, self.sleep, self.clock = shell, sleep, clock

    def check(self, command: str, what: str, timeout_s: float = 600) -> str:
        result = self.shell.run(command, timeout_s)
        if result.returncode != 0:
            raise RunPodError(f"{what} failed in the pod ({result.returncode}): "
                              f"{(result.stderr or result.stdout).strip()[-500:]}")
        return result.stdout

    def exists(self, path: str) -> bool:
        return self.shell.run(f"test -e {shlex.quote(path)}").returncode == 0

    def read(self, path: str) -> Optional[str]:
        result = self.shell.run(f"cat {shlex.quote(path)}")
        return result.stdout if result.returncode == 0 else None

    def job_paths(self, name: str):
        return f"{WORK}/jobs/{name}.rc", f"{WORK}/jobs/{name}.log"

    def start_job(self, name: str, command: str) -> None:
        """Run `command` detached, from the code directory, with the pod's saved environment (/bench/env:
        the secrets RunPod substituted at boot, which an SSH session does not inherit)."""
        rc, log = self.job_paths(name)
        inner = (f". {WORK}/env 2>/dev/null; cd {WORK}/code || exit 97\n( {command}\n)\n"
                 f"echo $? > {rc}.tmp && mv {rc}.tmp {rc}")
        # the background `&` takes nohup alone: on a list (a && b &) the whole list would hold the session open
        script = (f"{JOB_HEADER}{name}\nmkdir -p {WORK}/jobs && rm -f {rc} || exit 1\n"
                  f"nohup bash -c {shlex.quote(inner)} > {log} 2>&1 < /dev/null &")
        self.check(script, f"starting job {name}")

    def job_rc(self, name: str) -> Optional[int]:
        text = self.read(self.job_paths(name)[0])
        return int(text.strip()) if text and text.strip().lstrip("-").isdigit() else None

    def log_tail(self, name: str, lines: int = 40) -> str:
        result = self.shell.run(f"tail -n {lines} {self.job_paths(name)[1]}")
        return result.stdout if result.returncode == 0 else ""

    def wait_job(self, name: str, timeout_s: float, poll_s: float = 10.0,
                 ready: Optional[str] = None) -> int:
        """Until the job's exit code is written, or, for a job that keeps running (a server), until its `ready`
        file exists. A job that ends with an error, or a server that ends before it is ready, raises with the
        end of its log."""
        started = self.clock()
        while True:
            if ready and self.exists(ready):
                return 0
            rc = self.job_rc(name)
            if rc is not None:
                if rc != 0 or ready:
                    raise RunPodError(f"job {name} exited with {rc}{' before it was ready' if ready else ''}; the end "
                                      f"of its log:\n{self.log_tail(name)}")
                return rc
            if self.clock() - started > timeout_s:
                raise RunPodError(f"job {name} did not finish within {timeout_s:.0f} s; the end of its log:\n"
                                  f"{self.log_tail(name)}")
            self.sleep(poll_s)


# ---------------------------------------------------------------- the local tunnel

def port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def open_tunnel(host: str, port: int, key: str, known_hosts: Path, local_port: int, log: Path,
                remote_port: int = 8000) -> int:
    """`ssh -N -L 127.0.0.1:<local_port>:localhost:<remote_port>` in the background, in its own session (it
    outlives this command); its pid."""
    if port_open(local_port):
        raise RunPodError(f"local port {local_port} is already in use: close what holds it, or change runpod.local_ports")
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ssh", "-N", *ssh_options(key, port, known_hosts), "-o", "ExitOnForwardFailure=yes",
           "-o", "ServerAliveCountMax=4", "-L", f"127.0.0.1:{local_port}:localhost:{remote_port}", "-p", str(port),
           f"root@{host}"]
    with open(log, "ab") as out:
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True)
    for _ in range(60):
        if port_open(local_port):
            return process.pid
        if process.poll() is not None:
            raise RunPodError(f"the SSH tunnel exited with {process.returncode} (see {log})")
        time.sleep(0.5)
    process.terminate()
    raise RunPodError(f"the SSH tunnel did not open port {local_port} within 30 s (see {log})")


def is_tunnel(pid: Optional[int]) -> bool:
    """The pid is alive and still an ssh process (a pid reused by another program is never killed)."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    command = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    return command.strip().startswith("ssh ") and " -N " in f" {command} "


def close_tunnel(pid: Optional[int]) -> bool:
    if not is_tunnel(pid):
        return False
    os.kill(pid, signal.SIGTERM)
    return True
