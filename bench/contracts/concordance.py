"""C3 · agreement between two outputs of the same call site (J2, and B5 in clusters without gold).

Two outputs agree when **the agent derives the same decision from them**, read with CHESS's own
rules (vendor/chess/src/workflow/agents/**), not when their text looks alike:
- keywords: the same set of strings (`extract_keywords.py` keeps the list as is);
- column filter: the same keep/drop decision, `value.lower() == "yes"` (`filter_column.py:67`);
- table selection: the same set of table names, as written (`select_tables.py:66-69` keys the
  schema by the raw name);
- column selection: the same set of (table, column), each with one surrounding pair of backticks
  removed, as `select_columns.py:aggregate_columns` does;
- the agent's action: the same tool, or both done.
SQL generation and repair have gold and are judged by execution (J1, J2), never by agreement.

Deliberately stricter than CHESS in three edge cases, so the error is on the side of disagreeing:
CHESS also de-duplicates columns and tables case-insensitively within one output, and drops the
columns of tables it did not select; agreement here compares the output as written.
"""
from typing import Any, FrozenSet, Tuple

from bench.contracts.calls import AGENT_CALL_SITES

GOLD_CALL_SITES = ("generate_candidate", "revise")


def _unquote(name: str) -> str:
    return name[1:-1] if name.startswith("`") else name


def _keeps_column(output: Any) -> bool:
    return output["is_column_information_relevant"].lower() == "yes"


def _column_pairs(output: Any) -> FrozenSet[Tuple[str, str]]:
    return frozenset((_unquote(table), _unquote(column))
                     for table, columns in output.items() if table != "chain_of_thought_reasoning"
                     for column in columns)


def agree(call_site: str, a: Any, b: Any) -> bool:
    """True when two parsed outputs of `call_site` lead the agent to the same decision.
    A None (unparsed) output never agrees."""
    if call_site in GOLD_CALL_SITES:
        raise ValueError(f"{call_site} has gold: judge it by execution, not agreement")
    if a is None or b is None:
        return False
    if call_site == "extract_keywords":
        return set(a) == set(b)
    if call_site == "filter_column":
        return _keeps_column(a) == _keeps_column(b)
    if call_site == "select_tables":
        return set(a["table_names"]) == set(b["table_names"])
    if call_site == "select_columns":
        return _column_pairs(a) == _column_pairs(b)
    if call_site in AGENT_CALL_SITES:
        return a == b
    raise ValueError(f"unknown call site {call_site!r}")
