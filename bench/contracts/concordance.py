"""C3 · agreement between two outputs of the same call site (J2, and B5 in clusters without gold).

Defined per call site on `parsed_output` (C1): keywords as a set; column filter as the same yes/no
decision; table and column selection as the same set; the agent's action as the same tool. Table
and column names compare case-insensitively and without backticks, as SQLite resolves them.
SQL generation and repair have gold and are judged by execution (J1), never by agreement.
"""
from typing import Any, FrozenSet, Tuple

from bench.contracts.calls import AGENT_CALL_SITES

GOLD_CALL_SITES = ("generate_candidate", "revise")


def _name(value: str) -> str:
    return value.strip().strip("`").strip().lower()


def _relevant(output: Any) -> str:
    return str(output["is_column_information_relevant"]).strip().lower()


def _column_pairs(output: Any) -> FrozenSet[Tuple[str, str]]:
    return frozenset((_name(table), _name(column))
                     for table, columns in output.items() if table != "chain_of_thought_reasoning"
                     for column in columns)


def agree(call_site: str, a: Any, b: Any) -> bool:
    """True when two parsed outputs of `call_site` agree. A None (unparsed) output never agrees."""
    if call_site in GOLD_CALL_SITES:
        raise ValueError(f"{call_site} has gold: judge it by execution, not agreement")
    if a is None or b is None:
        return False
    if call_site == "extract_keywords":
        return set(a) == set(b)
    if call_site == "filter_column":
        return _relevant(a) == _relevant(b)
    if call_site == "select_tables":
        return {_name(t) for t in a["table_names"]} == {_name(t) for t in b["table_names"]}
    if call_site == "select_columns":
        return _column_pairs(a) == _column_pairs(b)
    if call_site in AGENT_CALL_SITES:
        return a == b
    raise ValueError(f"unknown call site {call_site!r}")
