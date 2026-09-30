"""The CHESS patches of front F1 (vendor/chess/PATCHES.md), one behaviour per test: HTTP retry with
backoff (5b), `literal_eval` (7), seeded schema shuffling (8), bounded concurrency (9), `local`
embeddings (10b), no gold in the agent's state (13), no `.env` over the harness environment (14),
and the configuration keys they read. Agent environment only; no server, no download."""
import json
import os
import random
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("langchain_core")
import httpx  # noqa: E402
import openai  # noqa: E402
from langchain_core.exceptions import OutputParserException  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.output_parsers import JsonOutputParser  # noqa: E402

from bench import paths  # noqa: E402
from bench.agent import hooks, runner  # noqa: E402
from bench.contracts.calls import read_calls, validate_calls  # noqa: E402
from bench.contracts.config import ConfigError, load_config  # noqa: E402
from synthetic import GOLD, make_repo  # noqa: E402
from test_agent_wiring import ScriptedChess  # noqa: E402

SMOKE = paths.ROOT / "configs" / "smoke-local.yaml"
USAGE = {"prompt_tokens": 12, "completion_tokens": 3}
REQUEST = httpx.Request("POST", "http://engine.example/v1/chat/completions")


def status_error(cls, code):
    return cls(f"Error code: {code}", response=httpx.Response(code, request=REQUEST), body=None)


class ScriptedModel:
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def invoke(self, messages):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return AIMessage(content=item, response_metadata={"token_usage": USAGE})


@pytest.fixture
def calls(tmp_path, monkeypatch):
    """A run on the smoke configuration with a scripted model and a recorded, instant backoff."""
    hooks.configure(load_config(SMOKE))
    hooks.start_run("test-run", "B0", tmp_path / "calls.jsonl")
    hooks.set_question("1470")
    sleeps = []
    monkeypatch.setattr(hooks, "_sleep", sleeps.append)
    state = {"model": None, "sleeps": sleeps, "path": tmp_path / "calls.jsonl"}
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: state["model"])
    yield state
    hooks.end_run()


# ---------------------------------------------------------------- 5b · HTTP retry with backoff

def test_transport_errors_are_retried_with_backoff_and_every_attempt_is_logged(calls):
    calls["model"] = ScriptedModel([status_error(openai.RateLimitError, 429), status_error(openai.InternalServerError, 503),
                                    openai.APITimeoutError(request=REQUEST), '{"table_names": ["t"]}'])
    assert hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser()) == {"table_names": ["t"]}
    lines = read_calls(calls["path"])
    assert [line["attempt"] for line in lines] == [1, 2, 3, 4] and validate_calls(lines) == []
    assert [line["retry_of"] for line in lines[1:]] == [line["call_id"] for line in lines[:-1]]
    assert [line["error"].split(":")[0] for line in lines[:3]] == ["RateLimitError", "InternalServerError", "APITimeoutError"]
    ceilings = [2.0, 4.0, 8.0]  # base * 2^(n-1), config.yaml › retries.http_backoff_s
    assert all(c / 2 <= s <= c for s, c in zip(calls["sleeps"], ceilings)) and len(calls["sleeps"]) == 3
    assert hooks.take_harness_errors() == []


def test_backoff_is_capped_and_jittered(monkeypatch):
    config = load_config(SMOKE)
    config["retries"]["http_backoff_s"] = {"base": 2, "max": 5}
    hooks.configure(config)
    assert [hooks.backoff_s(n) for n in (1, 2, 3, 4)] == [2.0, 4.0, 5.0, 5.0]
    sleeps = []
    monkeypatch.setattr(hooks, "_sleep", sleeps.append)
    for _ in range(20):
        hooks._backoff(3)
    assert all(2.5 <= s <= 5.0 for s in sleeps) and len(set(sleeps)) > 1  # concurrent retries do not retry together


def test_an_engine_that_stays_unreachable_fails_the_harness(calls):
    attempts = load_config(SMOKE)["retries"]["http_max_attempts"]
    calls["model"] = ScriptedModel([openai.APIConnectionError(request=REQUEST)] * attempts)
    with pytest.raises(hooks.HarnessError, match="unreachable"):
        hooks.invoke_tool_call("filter_column", "t.c", [HumanMessage(content="q")], JsonOutputParser())
    lines = read_calls(calls["path"])
    assert len(lines) == attempts == calls["model"].calls and validate_calls(lines) == []
    assert len(calls["sleeps"]) == attempts - 1
    (error,) = hooks.take_harness_errors()
    assert "filter_column" in error and "APIConnectionError" in error


def test_parse_and_transport_retries_share_one_attempt_counter(calls):
    calls["model"] = ScriptedModel(["not json", openai.APIConnectionError(request=REQUEST), "still not json"])
    with pytest.raises(OutputParserException):
        hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    lines = read_calls(calls["path"])
    assert [line["attempt"] for line in lines] == [1, 2, 3] and validate_calls(lines) == []
    assert len(calls["sleeps"]) == 1  # parse_max_attempts (2) counts parse failures; the transport one had its own budget


def test_the_agents_calls_retry_transport_errors_too(calls):
    calls["model"] = ScriptedModel([status_error(openai.InternalServerError, 502), "<tool_call>select_tables</tool_call>"])
    response = hooks.invoke_agent_call("agent_ss", "ss:0", "state", lambda r: {"tool": "select_tables"})
    assert response == "<tool_call>select_tables</tool_call>"
    first, second = read_calls(calls["path"])
    assert (first["parsed_ok"], second["parsed_ok"], second["retry_of"]) == (False, True, first["call_id"])


def test_a_rejected_request_is_the_harnesss_failure_not_the_models(calls):
    calls["model"] = ScriptedModel([status_error(openai.AuthenticationError, 401)])
    with pytest.raises(hooks.HarnessError, match="refused"):
        hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    assert calls["model"].calls == 1 and len(read_calls(calls["path"])) == 1
    assert len(hooks.take_harness_errors()) == 1


def test_a_bad_request_is_the_models_failure_and_is_not_retried(calls):
    calls["model"] = ScriptedModel([status_error(openai.BadRequestError, 400)])  # e.g. the prompt exceeds the context
    with pytest.raises(openai.BadRequestError):
        hooks.invoke_tool_call("select_tables", "single", [HumanMessage(content="q")], JsonOutputParser())
    assert calls["model"].calls == 1 and calls["sleeps"] == [] and hooks.take_harness_errors() == []


def test_the_new_configuration_keys_are_checked():
    config = load_config(SMOKE)
    hooks.check_settings(config)
    for path, value, message in ((("retries", "http_max_attempts"), 0, "http_max_attempts"),
                                 (("agent", "max_workers"), 0, "max_workers"),
                                 (("arms", "B1", "few_shot", "k"), -1, "few_shot.k")):
        broken = json.loads(json.dumps(config))
        node = broken
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(ConfigError, match=message):
            hooks.configure(broken)
    broken = json.loads(json.dumps(config))
    broken["embeddings"] = {**broken["embeddings"], "provider": "local", "local": {"model": "m", "revision": "main"}}
    with pytest.raises(ConfigError, match="40-hex"):
        hooks.configure(broken)


# ---------------------------------------------------------------- 7 · literal_eval

@pytest.fixture
def chess():
    config = load_config(SMOKE)
    runner._prepare_chess(config, paths.bird_root(config))
    return config


def test_keywords_are_read_as_a_literal_never_executed(chess, tmp_path):
    from llm.parsers import PythonListOutputParser
    parser = PythonListOutputParser()
    assert parser.parse("```python\n['gas stations', 'CZE']\n```") == ["gas stations", "CZE"]
    marker = tmp_path / "executed"
    with pytest.raises(OutputParserException):
        parser.parse(f"__import__('pathlib').Path({str(marker)!r}).touch()")
    assert not marker.exists()
    with pytest.raises(OutputParserException):
        parser.parse("['gas stations', 'CZE'")  # malformed: a parse failure, which the call layer retries


def test_a_malformed_list_is_retried_not_a_crash(chess, calls):
    from llm.parsers import PythonListOutputParser
    calls["model"] = ScriptedModel(["['a', 'b'", "['a', 'b']"])
    assert hooks.invoke_tool_call("extract_keywords", "single", [HumanMessage(content="q")], PythonListOutputParser()) == ["a", "b"]
    first, second = read_calls(calls["path"])
    assert first["error"].startswith("OutputParserException") and second["parsed_ok"]


# ---------------------------------------------------------------- 8 · seeded schema shuffling

def _schema(config, root):
    runner._prepare_chess(config, root)
    from runner.database_manager import DatabaseManager
    manager = DatabaseManager(db_mode="dev", db_id="wide")
    return manager.get_database_schema_string(manager.get_db_schema(), {}, {}, include_value_description=True)


@pytest.fixture
def wide_db(tmp_path):
    import sqlite3
    db_dir = tmp_path / "dev_databases" / "wide"
    db_dir.mkdir(parents=True)
    connection = sqlite3.connect(db_dir / "wide.sqlite")
    for t in range(6):
        connection.execute(f"CREATE TABLE t{t} (id INTEGER PRIMARY KEY, {', '.join(f'c{c} TEXT' for c in range(8))})")
    connection.close()
    return tmp_path


def test_the_schema_order_is_seeded_and_ignores_the_global_random_state(wide_db):
    config = load_config(SMOKE)
    random.seed(1)
    first = _schema(config, wide_db)
    random.seed(2)
    assert _schema(config, wide_db) == first
    orders = set()
    for seed in range(5):
        config["seeds"]["schema_shuffle"] = seed
        orders.add(_schema(config, wide_db))
    assert len(orders) > 1  # the seed, from the configuration, is what decides the order
    assert first.index("CREATE TABLE t0") != 0 or first.index("c0") > first.index("c7")  # it is shuffled


# ---------------------------------------------------------------- 9 · bounded concurrency

def test_concurrency_is_bounded_and_an_empty_call_list_is_no_error(chess):
    from threading_utils import ordered_concurrent_function_calls
    config = load_config(SMOKE)
    config["agent"]["max_workers"] = 2
    hooks.configure(config)
    assert ordered_concurrent_function_calls([]) == []  # published: max_workers=0 raised (revise with nothing to fix)
    active, peak, lock = [0], [0], threading.Lock()

    def work(i):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with lock:
            active[0] -= 1
        return i
    results = ordered_concurrent_function_calls([{"function": work, "kwargs": {"i": i}} for i in range(8)])
    assert results == list(range(8)) and peak[0] == 2


# ---------------------------------------------------------------- 14 · .env

def test_a_dotenv_never_overrides_the_harness_environment(tmp_path):
    (tmp_path / ".env").write_text("DB_ROOT_PATH=/from/dotenv\nINDEX_SERVER_PORT=9\nBENCH_ONLY_IN_DOTENV=yes\n")
    script = f"""
import sys, dotenv.main
dotenv.main.find_dotenv = lambda *a, **k: {str(tmp_path / '.env')!r}
sys.path.insert(0, {str(paths.VENDOR_CHESS / 'src')!r})
import runner.database_manager, database_utils.db_catalog.preprocess, preprocess, os
print(os.environ["DB_ROOT_PATH"], os.environ["INDEX_SERVER_PORT"], os.environ.get("BENCH_ONLY_IN_DOTENV"))
"""
    env = {**os.environ, "DB_ROOT_PATH": "/from/harness", "INDEX_SERVER_PORT": "0", "ANONYMIZED_TELEMETRY": "False"}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=paths.ROOT)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split() == ["/from/harness", "0", "yes"]  # the harness wins; a .env only fills gaps


# ---------------------------------------------------------------- 13 and 10b, end to end

@pytest.fixture
def repo(tmp_path, monkeypatch):
    root, config_path, config = make_repo(tmp_path, monkeypatch)
    runner.preprocess(str(config_path), ["tiny"])
    return config_path, config


OTHER_SQL = "SELECT COUNT(id) FROM gas_t WHERE segment = 'Premium' AND country = 'CZE'"


class NotTheGold(ScriptedChess):
    def answer(self, text):
        answer = super().answer(text)
        return answer.replace(GOLD, OTHER_SQL)


def test_the_gold_never_enters_the_agents_state(monkeypatch, repo):
    from runner.run_manager import RunManager
    seen = []
    worker = RunManager.worker
    monkeypatch.setattr(RunManager, "worker", lambda self, task: seen.append(task.SQL) or worker(self, task))
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: NotTheGold())
    run_dir = runner.run_agent(str(repo[0]), "B0", "train", ids=["1", "2"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "done" and manifest["tool_errors"] == {}  # the selector tools tolerate its absence
    assert seen == [None, None]
    for call in read_calls(run_dir / "calls.jsonl"):
        assert all(GOLD not in m["content"] for m in call["prompt_messages"])
    leaked = [p for p in run_dir.rglob("*") if p.is_file() and GOLD in p.read_text(errors="ignore")]
    assert leaked == []  # not in questions.json, not in CHESS's own logs


@pytest.fixture(scope="module")
def tiny_sentence_transformer(tmp_path_factory):
    """A sentence-transformers model built here, offline: a one-layer BERT with a character vocabulary."""
    import torch
    from sentence_transformers import SentenceTransformer, models
    from transformers import BertConfig, BertModel, BertTokenizerFast
    root = tmp_path_factory.mktemp("tiny-st")
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijklmnopqrstuvwxyz0123456789") + \
        [f"##{c}" for c in "abcdefghijklmnopqrstuvwxyz0123456789"]
    (root / "hf").mkdir()
    (root / "hf" / "vocab.txt").write_text("\n".join(words) + "\n")
    torch.manual_seed(0)
    BertModel(BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=512)).save_pretrained(root / "hf")
    BertTokenizerFast(vocab_file=str(root / "hf" / "vocab.txt")).save_pretrained(root / "hf")
    word = models.Transformer(str(root / "hf"), max_seq_length=128)
    SentenceTransformer(modules=[word, models.Pooling(word.get_word_embedding_dimension())]).save(str(root / "st"))
    return str(root / "st")


def test_local_embeddings_are_pinned_unit_vectors(tiny_sentence_transformer):
    config = load_config(SMOKE)
    config["embeddings"].update(provider="local", local={"model": tiny_sentence_transformer, "revision": "0" * 40})
    hooks.configure(config)
    vectors = hooks.embeddings("entity").embed_documents(["gas stations", "CZE"])
    assert len(vectors) == 2 and all(abs(sum(x * x for x in v) - 1) < 1e-4 for v in vectors)
    assert hooks.embeddings("context").embed_query("gas stations") == pytest.approx(vectors[0], abs=1e-6)
    assert hooks.vector_db_dirname() == "context_vector_db_local"


def test_a_run_with_local_embeddings_uses_its_own_vector_db(monkeypatch, repo, tiny_sentence_transformer):
    import yaml
    config = yaml.safe_load(repo[0].read_text())
    config["embeddings"].update(provider="local", local={"model": tiny_sentence_transformer, "revision": "0" * 40})
    repo[0].write_text(yaml.safe_dump(config))
    db_dir = paths.bird_root(repo[1]) / "dev_databases" / "tiny"
    monkeypatch.setattr(hooks, "chat_model", lambda engine, temperature: ScriptedChess())
    with pytest.raises(runner.data.DataError, match="context_vector_db_local"):
        runner.run_agent(str(repo[0]), "B0", "train", ids=["1"])
    runner.preprocess(str(repo[0]), ["tiny"])
    stamp = json.loads((db_dir / "context_vector_db_local" / "STAMP.json").read_text())
    assert stamp["provider"] == "local" and stamp["local"]["model"] == tiny_sentence_transformer
    assert "local" not in json.loads((db_dir / "context_vector_db_fake" / "STAMP.json").read_text())  # other stamps unchanged
    run_dir = runner.run_agent(str(repo[0]), "B0", "train", ids=["1"])
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "done"
