"""C1 · calls.jsonl: one line per LLM invocation, retries and failures included.

The record shape is `calls.schema.json` (JSON Schema, language neutral). This module adds the
invariants that span lines of one file: unique `call_id`; one line per invocation and attempt; and
a retry chain whose every link points to the previous attempt of the same invocation.
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import jsonschema

SCHEMA_PATH = Path(__file__).with_name("calls.schema.json")

AGENT_CALL_SITES = ("agent_ir", "agent_ss", "agent_cg")
TOOL_CALL_SITES = ("extract_keywords", "filter_column", "select_tables", "select_columns",
                   "generate_candidate", "revise")
CALL_SITES = AGENT_CALL_SITES + TOOL_CALL_SITES

# `invocation_key` of a call site that makes one invocation per question.
SINGLE = "single"


@lru_cache(maxsize=1)
def _validator() -> jsonschema.protocols.Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


def validate_call(record: dict) -> List[str]:
    """Errors of one record against the schema; empty when valid."""
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in _validator().iter_errors(record)]


def validate_calls(records: Iterable[dict]) -> List[str]:
    """Errors of a whole calls.jsonl, per line and across lines."""
    errors: List[str] = []
    seen = {}
    identities = set()
    for n, record in enumerate(records, start=1):
        errors += [f"line {n}: {e}" for e in validate_call(record)]
        call_id = record.get("call_id")
        if call_id in seen:
            errors.append(f"line {n}: duplicate call_id {call_id}")
            continue
        identity = tuple(record.get(k) for k in ("run_id", "question_id", "call_site", "invocation_key", "attempt"))
        if identity in identities:
            errors.append(f"line {n}: a second line for the same invocation and attempt {identity[1:]}")
        identities.add(identity)
        retry_of = record.get("retry_of")
        if retry_of is not None:
            previous = seen.get(retry_of)
            if previous is None:
                errors.append(f"line {n}: retry_of {retry_of} does not point to an earlier line")
            else:
                same = ("run_id", "question_id", "call_site", "invocation_key")
                if any(previous.get(k) != record.get(k) for k in same):
                    errors.append(f"line {n}: retry_of links a different invocation")
                if record.get("attempt") != previous.get("attempt", 0) + 1:
                    errors.append(f"line {n}: attempt is not the previous attempt + 1")
        seen[call_id] = record
    return errors


def read_calls(path: Path) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
