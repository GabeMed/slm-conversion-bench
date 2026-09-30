"""The real AIPerf v0.13.0 against a recording OpenAI-compatible server: the API key and the proxy-auth
headers reach the server from `${VAR}` references in aiperf.yaml (never AIPerf's command line), and no
credential value survives in any artifact (AIPerf writes Modal-Key verbatim into its export; bench scrubs
it). Runs where AIPerf is installed (env/train); needs no model server."""
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bench import cli, paths
from test_train_fixtures import TINY, make_s5_repo, save

CALLS = json.loads((Path(__file__).parent / "fixtures_calls.json").read_text())
SECRETS = {"T_API_KEY": "sk-aiperf-live-key", "T_MODAL_KEY": "mk-aiperf-live", "T_MODAL_SECRET": "ms-aiperf-live"}


class Recording:
    """/v1/models lists the model; /v1/chat/completions answers and records the headers it got."""

    def __init__(self, model):
        self.headers = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                fake.headers.append(("GET", dict(self.headers)))
                self._answer({"object": "list", "data": [{"id": model, "object": "model", "root": "r"}]})

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                fake.headers.append(("POST", dict(self.headers)))
                self._answer({"id": "x", "object": "chat.completion", "created": 0, "model": model,
                              "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                           "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


def test_credentials_reach_the_server_by_reference_and_leave_no_trace(tmp_path, monkeypatch):
    if not (Path(sys.executable).parent / "aiperf").exists() and not shutil.which("aiperf"):
        pytest.skip("AIPerf is not installed (env/train)")
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    server = Recording("tiny-qwen3")
    try:
        config["roles"]["slm_candidates"][-1]["endpoint"].update({
            "base_url": server.base_url, "api_key_env": "T_API_KEY",
            "headers_env": {"Modal-Key": "T_MODAL_KEY", "Modal-Secret": "T_MODAL_SECRET"}})
        config["loadtest"].update({"concurrency": [1], "request_count": 2, "warmup_request_count": 0})
        save(config, config_path)
        for name, value in SECRETS.items():
            monkeypatch.setenv(name, value)
        source = paths.RUNS / "agent-B0-train-fixture"
        source.mkdir(parents=True)
        (source / "manifest.json").write_text(json.dumps({"run_id": source.name, "arm": "B0", "split": "train",
                                                          "status": "done"}))
        (source / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CALLS))
        assert cli.main(["loadtest", "--config", str(config_path), "--engine", "slm:tiny-qwen3", "--source",
                         source.name, "--on", "local", "--tokenizer", TINY["repo"],
                         "--tokenizer-revision", TINY["revision"]]) == 0
    finally:
        server.server.shutdown()
    run_dir = next(paths.RUNS.glob("loadtest-*"))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "done"
    posts = [h for method, h in server.headers if method == "POST"]
    assert len(posts) == 2 and all((h["Authorization"], h["Modal-Key"], h["Modal-Secret"]) ==
                                   ("Bearer sk-aiperf-live-key", "mk-aiperf-live", "ms-aiperf-live") for h in posts)
    assert "${T_MODAL_KEY}" in (run_dir / "aiperf.yaml").read_text()  # the config holds references only
    assert "profile_export_aiperf.json" in manifest["secrets_redacted_in"]  # AIPerf had written Modal-Key there
    leaked = [p.name for p in run_dir.rglob("*") if p.is_file()
              and any(value.encode() in p.read_bytes() for value in SECRETS.values())]
    assert not leaked
