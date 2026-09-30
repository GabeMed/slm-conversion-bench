"""`bench eval <run_id>`: the only source of execution accuracy (J1 reads what this writes).

Each prediction is paired with its gold by `question_id`, never by position. Gold and prediction
run back to back on the pinned SQLite file (sha256 checked against data/MANIFEST.json), read-only
and with a timeout. Correct means the same set of rows, BIRD's rule; SQL that holds no statement (a
comment, blanks) is an execution error, never an empty answer.

The current moment is the pre-registered `eval.fixed_date` (midnight UTC), in prediction and gold
alike, so arms evaluated on different days, on machines in different time zones, stay comparable:
SQLite's date functions (and the functions behind CURRENT_TIMESTAMP / CURRENT_DATE / CURRENT_TIME,
which are also replaced in the text) are overridden on the connection, so that every value they read
as the current moment, however it was computed, is the fixed moment.
"""
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from bench import barrier, data, paths
from bench.contracts.calls import read_calls, validate_calls
from bench.contracts.config import config_sha256
from bench.provenance import git_state

# SQL as SQLite reads it: comments, string literals and quoted identifiers are whole tokens, so
# nothing inside them is rewritten (and an apostrophe in a comment opens no literal).
_TOKEN = re.compile(r"""(?P<comment>--[^\n]*|/\*.*?(?:\*/|$))
                      |(?P<string>'(?:[^']|'')*'|"(?:[^"]|"")*")
                      |(?P<ident>`[^`]*`|\[[^\]]*\])
                      |(?P<space>\s+)
                      |(?P<word>[A-Za-z_][A-Za-z_0-9$]*)
                      |(?P<other>.)""", re.DOTALL | re.VERBOSE)
# SQLite's date functions and the positions of the time value they read as 'now' (strftime's
# comes after the format; timediff reads two). As a format or a modifier 'now' means something
# else, and keeps it.
_TIME_VALUE = {"date": (0,), "time": (0,), "datetime": (0,), "julianday": (0,), "unixepoch": (0,),
               "strftime": (1,), "timediff": (0, 1)}


def _tokens(sql: str) -> List[Tuple[str, str]]:
    return [(m.lastgroup, m.group()) for m in _TOKEN.finditer(sql)]


def fixed_date(config: Dict[str, Any]) -> str:
    """The pre-registered date, refused when unset: without it, EX would depend on the day of the eval."""
    value = config["eval"].get("fixed_date")
    try:
        day = date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        day = None
    if day is None or day != value:  # also refuses what Python would normalise (20260930, 2026-W40-3)
        raise data.DataError(f"eval.fixed_date must be a pre-registered YYYY-MM-DD string, not {value!r}")
    return day


def fix_keywords(sql: str, day: str) -> Tuple[str, bool]:
    """`sql` with the keywords CURRENT_TIMESTAMP / CURRENT_DATE / CURRENT_TIME replaced by `day` at
    midnight, and whether any was. Nothing inside a comment, a literal or a quoted identifier is
    touched. The clock (`_FixedClock`) also overrides the functions these keywords compile to, so
    this is a second guard, kept by decision."""
    keywords = {"current_timestamp": f"'{day} 00:00:00'", "current_date": f"'{day}'", "current_time": "'00:00:00'"}
    out, replaced = [], False
    for kind, text in _tokens(sql):
        if kind == "word" and text.lower() in keywords:
            text, replaced = keywords[text.lower()], True
        out.append(text)
    return "".join(out), replaced


# SQLite's own zero-argument functions behind the keywords, and the function each equals at 'now'.
_KEYWORD_FUNCTIONS = {"current_date": "date", "current_time": "time", "current_timestamp": "datetime"}
# SQLite (date.c) reads these time values as the current moment, 'subsec' / 'subsecond' (3.42+) with
# fractional seconds; any case, as C text, so from a blob too and up to a NUL.
_MOMENTS = ("now", "subsec", "subsecond") if sqlite3.sqlite_version_info >= (3, 42, 0) else ("now",)


def _moment(value: Any) -> Optional[str]:
    if isinstance(value, bytes):
        value = value.split(b"\0", 1)[0].decode("utf-8", "replace")
    elif isinstance(value, str):
        value = value.split("\0", 1)[0]
    else:
        return None
    return value.lower() if value.lower() in _MOMENTS else None


class _FixedClock:
    """SQLite's date functions, installed over the built-ins of a connection, with the current
    moment fixed: a time value SQLite reads as the current moment (see `_moment`), or a time value
    left out (date(), strftime(fmt)), is the fixed stamp, and current_date() / current_time() /
    current_timestamp() (the keywords' functions, reachable by a quoted name) give the fixed moment.
    The result is computed by the real built-in on a connection of its own, so every other meaning
    is SQLite's. `readings` counts the times the current moment was read; `failure` keeps the error
    a wrapper raised, with SQLite's own message."""

    def __init__(self, day: str):
        self.stamp = f"{day} 00:00:00"
        self.readings = 0
        self.failure: Optional[str] = None
        self._builtins = sqlite3.connect(":memory:")

    def install(self, connection: sqlite3.Connection) -> None:
        for name, positions in _TIME_VALUE.items():
            connection.create_function(name, -1, self._guarded(self._function(name, positions)), deterministic=True)
        for name, function in _KEYWORD_FUNCTIONS.items():
            connection.create_function(name, -1, self._guarded(self._keyword(name, function)), deterministic=True)

    def _builtin(self, name: str, args: List[Any]) -> Any:
        return self._builtins.execute(f"SELECT {name}({', '.join('?' * len(args))})", args).fetchone()[0]

    def _guarded(self, call: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(*args: Any) -> Any:
            try:
                return call(*args)
            except Exception as e:  # SQLite would only say "user-defined function raised exception"
                self.failure = self.failure or f"{type(e).__name__}: {e}"
                raise
        return guarded

    def _keyword(self, name: str, function: str) -> Callable[..., Any]:
        def call(*args: Any) -> Any:
            if args:
                raise sqlite3.OperationalError(f"wrong number of arguments to function {name}()")
            self.readings += 1
            return self._builtin(function, [self.stamp])
        return call

    def _function(self, name: str, positions: Tuple[int, ...]) -> Callable[..., Any]:
        def call(*args: Any) -> Any:
            args = list(args)
            omitted = (len(args) == 0 if name != "strftime" else len(args) == 1 and args[0] is not None)
            if name != "timediff" and omitted:  # no time value: SQLite reads 'now'
                args.append(self.stamp)
                self.readings += 1
            for i in positions:
                moment = _moment(args[i]) if i < len(args) else None
                if moment is not None:
                    args[i] = self.stamp
                    self.readings += 1
                    if moment != "now" and name != "timediff":
                        args.insert(i + 1, "subsec")  # what the time value 'subsec' also asks for
            return self._builtin(name, args)
        return call

    def close(self) -> None:
        self._builtins.close()


def gold_has_limit(sql: str) -> bool:
    """LIMIT outside comments and literals: ties at the cut can make a right answer look wrong, for every arm."""
    return any(kind == "word" and text.lower() == "limit" for kind, text in _tokens(sql))


@contextmanager
def _utc() -> Iterator[None]:
    """SQLite's 'localtime' follows the process time zone; the evaluator runs in UTC."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = previous
        time.tzset()


def execute(db_path: Path, sql: str, timeout_s: float, day: str) -> Tuple[Optional[List[tuple]], Optional[str], bool]:
    """The rows of `sql` (or its error) at the fixed date, read-only, in UTC, within the timeout,
    and whether it read the current moment (a keyword in the text, or 'now' at run time)."""
    sql, keyword = fix_keywords(sql, day)
    clock = _FixedClock(day)
    connection = None
    deadline = time.monotonic() + timeout_s
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        clock.install(connection)
        connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        with _utc():
            cursor = connection.execute(sql)
            rows = cursor.fetchall()
        if cursor.description is None:  # a comment or blanks: [] here is not the zero rows of a query
            return None, "no statement", keyword
        return rows, None, keyword or clock.readings > 0
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        if time.monotonic() > deadline:
            error = "timeout"
        elif clock.failure:
            error = clock.failure
        elif "user-defined function raised exception" in error:  # an argument Python could not decode
            error = "a date function got text that is not valid UTF-8 (the evaluator cannot read it)"
        return None, error, keyword or clock.readings > 0
    finally:
        if connection is not None:
            connection.close()
        clock.close()


def run_gold(question: Dict[str, Any], path: Path, timeout_s: float, day: str) -> Tuple[Optional[List[tuple]], Optional[str], bool]:
    """The gold's rows (or error) at the fixed date, and whether it read the current moment."""
    return execute(path, question["SQL"], timeout_s, day)


def score_one(question: Dict[str, Any], predicted: Optional[str], path: Path, timeout_s: float,
              day: str, gold: Optional[Tuple[Optional[List[tuple]], Optional[str], bool]] = None) -> Dict[str, Any]:
    """One prediction against the gold of its own question, both at the fixed date; `gold` is the
    question's `run_gold`, when already executed in this eval."""
    executed_at = datetime.now(timezone.utc).isoformat()
    gold_rows, gold_error, gold_substituted = gold if gold is not None else run_gold(question, path, timeout_s, day)
    pred_substituted = False
    if predicted is None:
        pred_rows, pred_error = None, "no prediction"
    else:
        pred_rows, pred_error, pred_substituted = execute(path, predicted, timeout_s, day)
    return {
        "question_id": question["question_id"], "db_id": question["db_id"],
        "difficulty": question.get("difficulty"),
        "predicted_sql": predicted, "gold_sql": question["SQL"],
        "pred_error": pred_error, "gold_error": gold_error,
        "correct": gold_error is None and pred_error is None and set(pred_rows) == set(gold_rows),
        "pred_date_substituted": pred_substituted, "gold_date_substituted": gold_substituted,
        "gold_has_limit": gold_has_limit(question["SQL"]),
        "executed_at": executed_at,
    }


def score(predictions: Dict[str, Optional[str]], gold: Dict[str, dict], db_path: Callable[[str], Path],
          timeout_s: float, day: str) -> List[Dict[str, Any]]:
    """One result per prediction, each paired with the gold of the same question_id."""
    return [score_one({**gold[q], "question_id": q}, predictions[q], db_path(gold[q]["db_id"]), timeout_s, day)
            for q in sorted(predictions, key=int)]


def check_golds(questions: Dict[str, List[dict]], db_path: Callable[[str], Path], timeout_s: float,
                day: str) -> Dict[str, Any]:
    """Execute every gold of every split as the evaluator will (fixed date, read-only, timeout).
    `registered` is what reproduces on any machine, the fact `bench data` records and guards: the
    golds that fail with an execution error. `observed` is what this machine saw and is never
    compared: the golds that timed out, the golds with zero rows among those that finished, and the
    SQLite version."""
    errors, timeouts, empty = [], [], []
    for split, items in questions.items():
        for question in items:
            rows, error, _ = execute(db_path(question["db_id"]), question["SQL"], timeout_s, day)
            where = {"split": split, "question_id": question["question_id"], "db_id": question["db_id"]}
            if error == "timeout":
                timeouts.append(where)
            elif error is not None:
                errors.append({**where, "error": error})
            elif not rows:
                empty.append(where)
    return {"registered": {"fixed_date": day, "checked": {split: len(items) for split, items in questions.items()},
                           "errors": errors},
            "observed": {"sqlite_version": sqlite3.sqlite_version, "timeout_s": timeout_s,
                         "timeouts": timeouts, "empty": empty}}


def _source(source_run_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, dict]]:
    """The run to score, its configuration snapshot and the gold of its split, after every check
    both kinds of eval share: the snapshot is the one the run hashed, the barrier opens for that
    snapshot, a test run ran under the registration in force, the run is `done`, and a test run
    is scored by the analysis code that was registered."""
    source = paths.RUNS / source_run_id
    run_manifest = json.loads((source / "manifest.json").read_text())
    config = json.loads((source / "config.json").read_text())  # the run's own snapshot, not today's config.yaml
    if config_sha256(config) != run_manifest["config_sha256"]:  # validated when the run loaded it
        raise data.DataError(f"the configuration snapshot of {source_run_id} does not match its manifest")
    split = run_manifest["split"]
    barrier.ensure_split_allowed(split, config)
    if split == "test":
        in_force = (paths.ROOT / barrier.PREREG_HASH).read_text().strip()
        if run_manifest.get("prereg_hash") != in_force:
            raise data.DataError(f"run {source_run_id} ran under pre-registration {run_manifest.get('prereg_hash')}, "
                                 f"not the one in force ({in_force}): it is reported, never scored")
    if run_manifest["status"] != "done":
        raise data.DataError(f"run {source_run_id} is {run_manifest['status']!r}, not 'done': "
                             f"an incomplete or failed run is never scored")
    if split == "test":  # the run is scored by the analysis code that was registered, not by today's
        from bench.prereg import check_registered_analysis_code
        check_registered_analysis_code(paths.ROOT)
    return run_manifest, config, data.questions_for(config, split)


def _check_paired(question_ids: List[str], gold: Dict[str, dict], config: Dict[str, Any], split: str) -> List[str]:
    """Every scored question has a gold in the split and a database with the recorded hash."""
    unpaired = sorted(set(question_ids) - set(gold), key=int)
    if unpaired:
        raise data.DataError(f"predictions without a gold in the {split} split: {unpaired[:10]}")
    databases = sorted({gold[q]["db_id"] for q in question_ids})
    for db_id in databases:
        data.check_database(config, db_id)
    return databases


def _write(kind: str, source_run_id: str, run_manifest: Dict[str, Any], config: Dict[str, Any],
           databases: List[str], score_all: Callable[[str], List[Dict[str, Any]]], **extra: Any) -> Path:
    day = fixed_date(config)  # refused before the execution directory exists
    started = datetime.now(timezone.utc)
    run_id = f"{kind}-{source_run_id}-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
    out = paths.RUNS / run_id
    out.mkdir(parents=True)
    results = score_all(day)
    with open(out / "results.jsonl", "w") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    recorded = json.loads(paths.DATA_MANIFEST.read_text())["databases"]
    manifest = {
        "run_id": run_id, "type": "eval", **extra, "source_run_id": source_run_id,
        "source_type": run_manifest.get("type"), "arm": run_manifest.get("arm"), "engine": run_manifest.get("engine"),
        "mode": run_manifest.get("mode"), "split": run_manifest["split"],
        **git_state(), "source_commit": run_manifest["commit"], "config_sha256": run_manifest["config_sha256"],
        "prereg_hash": run_manifest.get("prereg_hash"),
        "fixed_date": day, "timeout_s": config["eval"]["timeout_s"], "sqlite_version": sqlite3.sqlite_version,
        "databases": {db: recorded[db]["sqlite"] for db in databases},
        "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        "n": len(results), "correct": sum(r["correct"] for r in results),
        "gold_errors": sum(r["gold_error"] is not None for r in results),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def evaluate(source_run_id: str) -> Path:
    """The eval execution of a run's final SQL per question (predictions.json)."""
    run_manifest, config, gold = _source(source_run_id)
    predictions: Dict[str, Optional[str]] = json.loads((paths.RUNS / source_run_id / "predictions.json").read_text())
    if set(predictions) != set(run_manifest["question_ids"]):
        raise data.DataError(f"the predictions of {source_run_id} are not exactly its question ids")
    databases = _check_paired(list(predictions), gold, config, run_manifest["split"])
    return _write("eval", source_run_id, run_manifest, config, databases, lambda day: score(
        predictions, gold, lambda db_id: paths.sqlite_path(config, db_id), config["eval"]["timeout_s"], day))


# The call sites whose output is a SQL query, and where their parser puts it (CHESS parsers.py).
PER_CALL_SQL = {"generate_candidate": "SQL", "revise": "refined_sql_query"}


def per_call_predictions(calls: List[dict]) -> List[Dict[str, Any]]:
    """One entry per generation or repair invocation, sorted by question, call site and key: the SQL
    of its last attempt that parsed, or None when no attempt did."""
    invocations: Dict[Tuple[str, str, str], Optional[dict]] = {}
    for record in calls:
        if record["call_site"] not in PER_CALL_SQL:
            continue
        key = (record["question_id"], record["call_site"], record["invocation_key"])
        best = invocations.setdefault(key, None)
        if record["parsed_ok"] and (best is None or record["attempt"] > best["attempt"]):
            invocations[key] = record
    entries = []
    for (question_id, call_site, invocation_key), record in sorted(
            invocations.items(), key=lambda item: (int(item[0][0]), item[0][1], item[0][2])):
        output = record["parsed_output"] if record else None
        sql = output.get(PER_CALL_SQL[call_site]) if isinstance(output, dict) else None
        entries.append({"question_id": question_id, "call_site": call_site, "invocation_key": invocation_key,
                        "call_id": record["call_id"] if record else None,
                        "attempt": record["attempt"] if record else None,
                        "sql": sql if isinstance(sql, str) else None})
    return entries


def evaluate_per_call(source_run_id: str) -> Path:
    """The eval execution of every generation and repair invocation in a run's calls.jsonl (an
    `agent` or a `replay` run), each against the gold of its question (design §5.1)."""
    run_manifest, config, gold = _source(source_run_id)
    calls = read_calls(paths.RUNS / source_run_id / "calls.jsonl")
    errors = validate_calls(calls)
    if errors:
        raise data.DataError(f"the calls.jsonl of {source_run_id} is not valid C1: {errors[:3]}")
    entries = per_call_predictions(calls)
    in_run = run_manifest.get("question_ids")
    if in_run is None:
        raise data.DataError(f"the manifest of {source_run_id} does not list its question ids")
    if not {e["question_id"] for e in entries} <= set(in_run):
        raise data.DataError(f"calls of {source_run_id} for questions outside its question ids")
    databases = _check_paired([e["question_id"] for e in entries], gold, config, run_manifest["split"])

    def score_calls(day: str) -> List[Dict[str, Any]]:
        results, golds = [], {}  # each question's gold runs once per eval, whatever its invocations
        for entry in entries:
            question = {**gold[entry["question_id"]], "question_id": entry["question_id"]}
            path, timeout_s = paths.sqlite_path(config, question["db_id"]), config["eval"]["timeout_s"]
            if entry["question_id"] not in golds:
                golds[entry["question_id"]] = run_gold(question, path, timeout_s, day)
            result = score_one(question, entry["sql"], path, timeout_s, day, gold=golds[entry["question_id"]])
            if entry["call_id"] is None:
                result["pred_error"] = "no attempt parsed"
            results.append({"question_id": entry["question_id"], "call_site": entry["call_site"],
                            "invocation_key": entry["invocation_key"], "correct": result["correct"],
                            "call_id": entry["call_id"], "attempt": entry["attempt"],
                            **{k: v for k, v in result.items() if k not in ("question_id", "correct")}})
        return results
    return _write("eval-per-call", source_run_id, run_manifest, config, databases, score_calls, per_call=True)
