"""C1: the record schema and the invariants across lines of calls.jsonl."""
import copy
import json
import uuid
from pathlib import Path

import pytest

from bench.contracts.calls import validate_call, validate_calls

FIXTURES = json.loads((Path(__file__).parent / "fixtures_calls.json").read_text())


def record(**changes):
    base = copy.deepcopy(FIXTURES[1])  # a real select_tables call from the local smoke run
    base.update(changes)
    return base


def test_real_records_are_valid():
    assert [validate_call(r) for r in FIXTURES] == [[], []]
    assert validate_calls(FIXTURES) == []


@pytest.mark.parametrize("field", ["call_id", "usage", "invocation_key", "prompt_messages", "error"])
def test_every_field_is_required(field):
    broken = record()
    del broken[field]
    assert validate_call(broken)


@pytest.mark.parametrize("changes", [
    {"call_site": "unit_test"},
    {"model_role": "teacher"},
    {"endpoint": "ollama"},
    {"attempt": 0},
    {"call_id": "not-a-uuid"},
    {"extra": 1},
    {"prompt_messages": []},
    {"usage": {"input": None, "cached_input": None, "output": None, "source": "api"}},
])
def test_invalid_values(changes):
    assert validate_call(record(**changes))


def test_retry_of_matches_attempt():
    assert validate_call(record(attempt=2, retry_of=None))
    assert validate_call(record(attempt=1, retry_of=str(uuid.uuid4())))
    assert validate_call(record(attempt=2, retry_of=str(uuid.uuid4()))) == []


def test_failure_carries_a_message_and_no_output():
    assert validate_call(record(parsed_ok=False, parsed_output=None, error=None))
    assert validate_call(record(parsed_ok=False, parsed_output={"table_names": []}, error="x"))
    assert validate_call(record(parsed_ok=True, error="x"))
    assert validate_call(record(parsed_ok=False, parsed_output=None, error="empty output")) == []


def test_failed_invocation_has_no_response_and_no_parse():
    assert validate_call(record(response_text=None))  # parsed_ok still true
    assert validate_call(record(response_text=None, parsed_ok=False, parsed_output=None, error="Timeout")) == []


def test_agent_action_is_a_name_or_done():
    agent = copy.deepcopy(FIXTURES[0])
    assert validate_call({**agent, "parsed_output": {"done": True}}) == []
    assert validate_call({**agent, "parsed_output": {"tool": "select_tables", "args": {}}})  # P-5
    assert validate_call({**agent, "parsed_output": {"done": False}})


def test_duplicate_call_id():
    assert "duplicate call_id" in " ".join(validate_calls([FIXTURES[1], FIXTURES[1]]))


def test_retry_chain():
    first = record(parsed_ok=False, parsed_output=None, error="OutputParserException: bad json")
    second = record(call_id=str(uuid.uuid4()), attempt=2, retry_of=first["call_id"])
    assert validate_calls([first, second]) == []
    assert "earlier line" in " ".join(validate_calls([second, first]))
    other = dict(second, invocation_key="other")
    assert "different invocation" in " ".join(validate_calls([first, other]))
    skipped = dict(second, attempt=3)
    assert "previous attempt" in " ".join(validate_calls([first, skipped]))
