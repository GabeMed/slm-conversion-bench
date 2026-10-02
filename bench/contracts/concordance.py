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

A parsed output need not have the shape its call site asks for (a cut-off answer still parses).
`agree` never raises on one: where CHESS has a reading for it, that reading is the decision (a
filter with no answer drops the column, a selection with no `table_names` selects no table), and
an output of a shape nothing can be read from agrees with nothing.
"""
from typing import Any, FrozenSet, Tuple

from bench.contracts.calls import AGENT_CALL_SITES

GOLD_CALL_SITES = ("generate_candidate", "revise")


def _unquote(name: str) -> str:
    return name[1:-1] if name.startswith("`") else name


def _keeps_column(output: Any) -> bool:
    """What CHESS does with a filter output (filter_column.py:65-72): the column stays only on a "yes",
    and an output it cannot read (a cut-off answer still parses) drops the column."""
    try:
        return output["is_column_information_relevant"].lower() == "yes"
    except (KeyError, TypeError, AttributeError):
        return False


def _column_pairs(output: Any) -> FrozenSet[Tuple[str, str]]:
    return frozenset((_unquote(table), _unquote(column))
                     for table, columns in output.items() if table != "chain_of_thought_reasoning"
                     for column in columns)


def _tables(output: Any) -> FrozenSet[str]:
    """select_tables.py:88 reads an output with no `table_names` as no table."""
    return frozenset(output.get("table_names", []))


def agree(call_site: str, a: Any, b: Any) -> bool:
    """True when two parsed outputs of `call_site` lead the agent to the same decision.
    A None (unparsed) output never agrees, nor does one of a shape no decision can be read from."""
    if call_site in GOLD_CALL_SITES:
        raise ValueError(f"{call_site} has gold: judge it by execution, not agreement")
    if a is None or b is None:
        return False
    try:
        if call_site == "extract_keywords":
            return set(a) == set(b)
        if call_site == "filter_column":
            return _keeps_column(a) == _keeps_column(b)
        if call_site == "select_tables":
            return _tables(a) == _tables(b)
        if call_site == "select_columns":
            return _column_pairs(a) == _column_pairs(b)
    except (TypeError, AttributeError):  # not a mapping, a null where a list goes, a name that is not text
        return False
    if call_site in AGENT_CALL_SITES:
        return a == b
    raise ValueError(f"unknown call site {call_site!r}")
