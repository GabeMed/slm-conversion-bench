"""The exploratory tables of SPEC 6.1, computed from stored executions. Outside the registered code.

SPEC 6.1 declares these readings before the test: the per-call-site table "on the teacher's inputs"
with its five columns, the same quantities per cluster, the cuts (difficulty, database, replay against
live, path) and projection-tolerant EX. They are descriptive: no verdict, unadjusted 95% intervals that
resample questions and keep a question's calls together, every cell published. `bench report` does not
print them; this script does, and nothing here feeds a judgment, an allocation or a verdict.

What it reads: `runs/<id>/manifest.json`, `calls.jsonl` (streamed, the prompts never kept),
`results.jsonl` of the eval executions, the configuration snapshot of the teacher's run, and for
projection-tolerant EX the pinned SQLite files, read-only, through `bench.evaluate.execute`. It imports
from `bench` and changes nothing there. It refuses a run on the test split unless `--allow-test` is
given: the test is read only after the registered evaluation.

An input that does not exist prints as "not run": a column is never dropped and never imputed. A
quantity the SPEC names but does not fix prints as "not computed" with the open question (`OPEN`): no
rule is chosen here. The conventions this script does follow, and where each comes from, are in
`CONVENTIONS`, printed with the tables.

    python scripts/exploratory_tables.py --root <checkout with runs/ and data/> \\
        --teacher <B0 run> --teacher-per-call-eval <eval> \\
        --replay base=<replay>,<per-call eval> --replay adapter=<replay>,<per-call eval> \\
        --replay cheap=<replay>,<per-call eval> [--replay teacher=<replay>,<per-call eval>] \\
        [--teacher-pilot-replay <replay of the pilot on the production LLM>] \\
        --live B0=<run>,<eval>,<per-call eval> --live B4=<run>,<eval>,<per-call eval> ... \\
        [--compare B4:B0 ...] [--without 875] --out tables.md --json tables.json
"""
import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # run as a script: `bench` is the package beside scripts/
    sys.path.insert(0, str(REPO))

from bench.contracts import concordance  # noqa: E402  (read-only: the registered reading of an output)
from bench.contracts.calls import AGENT_CALL_SITES, CALL_SITES  # noqa: E402
from bench.contracts.concordance import GOLD_CALL_SITES  # noqa: E402

LABEL = "exploratory"
COLUMNS = ("trivial", "base", "adapter", "cheap", "teacher")
COLUMN_TITLES = {"trivial": "trivial rule (no language model)", "base": "base SLM", "adapter": "adapter (B4's replay)",
                 "cheap": "cheap alternative", "teacher": "teacher's own repetition"}
REPLAY_COLUMNS = COLUMNS[1:]
# the arm whose own execution stands beside each replay in the cross-table (see CONVENTIONS)
COLUMN_ARM = {"base": "B3", "adapter": "B4", "cheap": "B1", "teacher": "B0"}
SITE_ORDER = ("filter_column", "select_tables", "select_columns", "extract_keywords", "generate_candidate", "revise",
              "agent_ir", "agent_ss", "agent_cg")
SELECTION_SITES = ("select_tables", "select_columns")
DIFFICULTIES = ("simple", "moderate", "challenging")
CLUSTERS = ("c0", "c1", "c2", "c3")
RULE_OF_THUMB = (10_000, 100_000)  # the paper's 10k-100k rule (S2, p.9), as SPEC 6.1 reads it
# SPEC 6.1, "The call sites against the paper's conditional claim": the training rows of each call site,
# transcribed (the datasets do not label a row's call site). The three agent call sites share one figure.
TRAINING_ROWS_BY_SITE = {
    "filter_column": "12,000 (11,982 in c0, 18 in c1)", "select_tables": "833 (c1)", "select_columns": "833 (c1)",
    "extract_keywords": "834 (833 in c2, 1 in c1)", "generate_candidate": "826 (c1)", "revise": "6 (c1)",
    **{site: "9,164 for the three agent call sites together (c3; 11 distinct prompts)" for site in AGENT_CALL_SITES}}
SPEC_ROWS_BY_CLUSTER = {"c0": 11_982, "c1": 2_517, "c2": 833, "c3": 9_164}  # the same table, summed per cluster

NOT_RUN = "not run"
OPEN = {
    "Q1": "table and column selection, \"the recall of the teacher's set\": the SPEC does not say whether it is the mean "
          "of each call's recall or the share of all the teacher's items recovered, nor what a call whose teacher set "
          "is empty contributes",
    "Q2": "keyword extraction, the trivial rule \"the question's words without stop words, plus the hint's quoted "
          "spans\": no stop-word list, tokenisation or quoting rule is fixed",
    "Q3": "SQL generation and repair, the trivial rule \"the teacher's SQL for the nearest training question on the "
          "same database\": the distance that makes a question the nearest is not fixed, nor the training run it reads",
    "Q4": "the agent's steps, the trivial rule \"a lookup by prompt\": the source of the lookup (which training calls) "
          "and its answer for a prompt with more than one teacher answer, or none, are not fixed",
    "Q5": "the agent's steps, \"agreement per distinct prompt\": when a prompt's calls do not all agree, the SPEC does "
          "not say whether the prompt agrees. The prompts are therefore counted three ways (every call agrees, some "
          "do, none does) and no single share is printed",
}
CONVENTIONS = [
    "Agreement is contract C3 (`bench.contracts.concordance`): the decision CHESS reads from an output. An engine "
    "output that did not parse disagrees, and its SQL is wrong; a call whose teacher output did not parse is left "
    "out and counted (`bench.judge.j2`). For the column filter's kappa, a disagreeing output with no decision "
    "takes the label opposite to the teacher's.",
    "A replay is read within its own scope: its questions and its declared call sites. Inside that scope a "
    "teacher invocation the replay lacks is refused, as J2 refuses it.",
    "The output of an invocation is its last attempt's (`bench.judge.base.final`).",
    "EX at SQL generation and repair is per call, from `bench eval --per-call`; the question-level reading of the "
    "cross-table is J2's: a question is right at a call site when every one of its calls there is.",
    "Intervals are two-sided 95% percentile intervals (inverted CDF, as J4's quantile) over resampled questions, "
    "`stats.n_boot` resamples and the seed `seeds.bootstrap` of the teacher run's configuration snapshot, drawn "
    "with numpy's PCG64. A resample in which a ratio has no denominator is left out and counted.",
    "The Jaccard index of two empty keyword sets is 1 (they are the same set); such calls are counted.",
    "The cluster of a call is the router's assignment in the adapter replay (B4's), as J7 reads it, for every "
    "column. A cluster's rows list its call sites one by one; no metric is pooled across call sites.",
    "The trivial rule \"every candidate in the prompt\" reads the tables and columns of the CREATE TABLE blocks of "
    "the teacher's prompt; calls whose teacher set is not within what was read are counted.",
    "The cross-table sets each replay beside the arm that runs the same engine live: base SLM and B3, adapter "
    "and B4, cheap alternative and B1, the teacher's repetition and B0.",
    "The difficulty cut of the end-to-end comparisons takes the pairs given with --compare (by default every arm "
    "given against B0); the database cut covers each arm end to end and SQL generation, as counts.",
    "Calls per question counts invocations; attempts (retries included) are printed beside them.",
]


class TablesError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------ reading

def run_path(runs: Path, run_id: str, allow_test: bool) -> Path:
    """The directory of an execution, refused when it is on the test split and `allow_test` is not set.
    The id is checked before anything of the run is opened."""
    if not allow_test and "-test-" in run_id:
        raise TablesError(f"{run_id} is a test execution: pass --allow-test, after the registered evaluation")
    path = runs / run_id
    if not (path / "manifest.json").is_file():
        raise TablesError(f"no execution {run_id} under {runs}")
    return path


def read_manifest(runs: Path, run_id: str, allow_test: bool, **expected: Any) -> Dict[str, Any]:
    manifest = json.loads((run_path(runs, run_id, allow_test) / "manifest.json").read_text())
    if manifest.get("split") == "test" and not allow_test:
        raise TablesError(f"{run_id} is on the test split: pass --allow-test, after the registered evaluation")
    finished = manifest.get("status") == "done" or (manifest.get("type") == "eval" and "status" not in manifest)
    if not finished:
        raise TablesError(f"{run_id} is {manifest.get('status')!r}, not 'done'")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise TablesError(f"{run_id}: {key} is {manifest.get(key)!r}, expected {value!r}")
    return manifest


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def decision(call_site: str, output: Any) -> Any:
    """What `concordance.agree` compares, as a hashable value; None when no decision can be read (such an
    output agrees with nothing). Two outputs agree exactly when both decisions are read and equal."""
    try:
        if call_site == "extract_keywords":
            return frozenset(output)
        if call_site == "filter_column":
            return concordance._keeps_column(output)
        if call_site == "select_tables":
            return concordance._tables(output)
        if call_site == "select_columns":
            return concordance._column_pairs(output)
    except (TypeError, AttributeError):
        return None
    if call_site in AGENT_CALL_SITES:
        return json.dumps(output, sort_keys=True)
    return None  # the call sites with gold are judged by execution


_TABLE = re.compile(r"CREATE TABLE (.+?)\n\((.*?)\n\);", re.DOTALL)
_NAME = re.compile(r'`([^`]*)`|"([^"]*)"|\[([^\]]*)\]|(\S+)')
_CONSTRAINTS = ("foreign key", "primary key", "constraint", "unique")


def schema_candidates(prompt: str) -> Dict[str, List[str]]:
    """The tables of a prompt's schema and their columns, from its CREATE TABLE blocks (CHESS's DDL)."""
    tables: Dict[str, List[str]] = {}
    for block in _TABLE.finditer(prompt):
        columns = []
        for line in block.group(2).split("\n"):
            text = line.strip()
            if not text or text.lower().startswith(_CONSTRAINTS):
                continue
            columns.append(next(g for g in _NAME.match(text).groups() if g is not None))
        tables[_unquote(block.group(1).strip())] = columns
    return tables


def _unquote(name: str) -> str:
    return name[1:-1] if len(name) > 1 and name[0] in "`\"[" else name


def all_candidates(call_site: str, prompt: str) -> frozenset:
    tables = schema_candidates(prompt)
    if call_site == "select_tables":
        return frozenset(tables)
    return frozenset((table, column) for table, columns in tables.items() for column in columns)


Identity = Tuple[str, str, str]  # (question_id, call_site, invocation_key)


def load_invocations(calls_path: Path, teacher: bool = False) -> Dict[Identity, Dict[str, Any]]:
    """One compact entry per invocation of a calls.jsonl, streamed: whether its last attempt parsed, the
    decision read from it, its cluster and its attempts. For the teacher's run also the prompt's sha256 at
    the agent's steps and every candidate of the prompt at table and column selection. No prompt is kept."""
    found: Dict[Identity, Dict[str, Any]] = {}
    for call in read_jsonl(calls_path):
        site = call["call_site"]
        identity = (str(call["question_id"]), site, call["invocation_key"])
        entry = found.setdefault(identity, {"attempt": 0, "attempts": 0})
        entry["attempts"] += 1
        if teacher and call["attempt"] == 1:
            if site in AGENT_CALL_SITES:
                raw = json.dumps(call["prompt_messages"], sort_keys=True).encode()
                entry["prompt"] = hashlib.sha256(raw).hexdigest()
            elif site in SELECTION_SITES:
                entry["candidates"] = all_candidates(site, "\n".join(m["content"] for m in call["prompt_messages"]))
        if call["attempt"] > entry["attempt"]:
            parsed = bool(call["parsed_ok"])
            entry.update(attempt=call["attempt"], parsed=parsed, cluster=call.get("cluster"),
                         decision=decision(site, call["parsed_output"]) if parsed else None)
    return found


def load_per_call_eval(runs: Path, eval_id: str, source_run_id: str, allow_test: bool) -> Dict[Identity, Dict[str, Any]]:
    manifest = read_manifest(runs, eval_id, allow_test, type="eval", source_run_id=source_run_id)
    if not manifest.get("per_call"):
        raise TablesError(f"{eval_id} is an end-to-end evaluation, not a per-call one")
    rows: Dict[Identity, Dict[str, Any]] = {}
    for row in read_jsonl(runs / eval_id / "results.jsonl"):
        identity = (str(row["question_id"]), row["call_site"], row["invocation_key"])
        if identity in rows:
            raise TablesError(f"{eval_id}: two results for {identity}")
        rows[identity] = {"correct": bool(row["correct"]), "difficulty": row.get("difficulty"), "db_id": row.get("db_id")}
    return rows


def load_eval(runs: Path, eval_id: str, source_run_id: str, allow_test: bool) -> Tuple[Dict[str, Any], List[dict]]:
    manifest = read_manifest(runs, eval_id, allow_test, type="eval", source_run_id=source_run_id)
    if manifest.get("per_call") or manifest.get("partial"):
        raise TablesError(f"{eval_id} is not the end-to-end evaluation of a completed run")
    return manifest, list(read_jsonl(runs / eval_id / "results.jsonl"))


# ------------------------------------------------------------------------------------------ intervals

class Bootstrap:
    """Resamples of questions: each resample draws as many questions as there are, with replacement, and
    a question brings all its calls. The same seed gives the same resamples for the same questions."""

    def __init__(self, n_boot: int, seed: int):
        self.n_boot, self.seed = n_boot, seed

    def interval(self, per_question: Dict[str, Sequence[float]], statistic: Callable[[np.ndarray], np.ndarray]
                 ) -> Dict[str, Any]:
        """`per_question`: each question's sums; `statistic` maps rows of summed columns to one value each
        (NaN where it has no denominator). The value on the data, and the 2.5% and 97.5% quantiles."""
        questions = sorted(per_question, key=_question_key)
        if not questions:
            return {"value": None, "low": None, "high": None, "undefined_resamples": 0}
        sums = np.array([per_question[q] for q in questions], dtype=float)
        value = float(statistic(sums.sum(axis=0, keepdims=True))[0])
        rng = np.random.Generator(np.random.PCG64(self.seed))
        counts = rng.multinomial(len(questions), np.full(len(questions), 1 / len(questions)), size=self.n_boot)
        values = statistic(counts @ sums)
        defined = np.sort(values[~np.isnan(values)])
        out = {"value": None if math.isnan(value) else value, "low": None, "high": None,
               "undefined_resamples": int(self.n_boot - len(defined))}
        if len(defined):
            k = -(-len(defined) // 40)  # ceil(len * 0.025), in integers: the k-th smallest and the k-th largest
            out["low"], out["high"] = float(defined[k - 1]), float(defined[len(defined) - k])
        return out


def _question_key(question_id: str) -> Tuple[int, str]:
    return (int(question_id), "") if str(question_id).isdigit() else (1 << 62, str(question_id))


def ratio(numerator: int, denominator: int) -> Callable[[np.ndarray], np.ndarray]:
    def statistic(sums: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sums[:, denominator] > 0, sums[:, numerator] / sums[:, denominator], np.nan)
    return statistic


def kappa(sums: np.ndarray) -> np.ndarray:
    """Cohen's kappa of two binary labels from the columns (both yes, teacher yes only, engine yes only,
    both no); NaN where there is no call or chance agreement is 1."""
    a, b, c, d = sums[:, 0], sums[:, 1], sums[:, 2], sums[:, 3]
    n = a + b + c + d
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = (a + d) / n
        chance = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
        return np.where((n > 0) & (chance < 1), (observed - chance) / (1 - chance), np.nan)


def rate_metric(boot: Bootstrap, hits: Dict[str, List[int]], numerator: int = 0, denominator: int = 1) -> Dict[str, Any]:
    out = boot.interval(hits, ratio(numerator, denominator))
    out["numerator"] = int(sum(v[numerator] for v in hits.values()))
    out["denominator"] = int(sum(v[denominator] for v in hits.values()))
    return out


# ------------------------------------------------------------------------------------------ cells

def _add(per_question: Dict[str, List[float]], question: str, values: Sequence[float]) -> None:
    current = per_question.setdefault(question, [0.0] * len(values))
    for i, value in enumerate(values):
        current[i] += value


def not_run(why: str) -> Dict[str, Any]:
    return {"status": NOT_RUN, "why": why}


def open_question(key: str) -> Dict[str, Any]:
    return {"status": "not computed", "open_question": key}


def site_cell(site: str, teacher: Dict[Identity, dict], engine: Optional[Dict[Identity, dict]],
              identities: List[Identity], boot: Bootstrap, trivial: bool = False,
              engine_eval: Optional[Dict[Identity, dict]] = None) -> Dict[str, Any]:
    """One cell of the per-call-site table: `site` on `identities` (the teacher's invocations in scope),
    for a replay (`engine`, with `engine_eval` at the call sites with gold) or for the trivial rule."""
    if not identities:
        return {"status": "computed", "calls": 0, "questions": 0, "metrics": {}}
    cell: Dict[str, Any] = {"status": "computed", "calls": len(identities),
                            "questions": len({i[0] for i in identities}), "metrics": {}}
    if site in GOLD_CALL_SITES:
        if trivial:
            return {**open_question("Q3"), "calls": cell["calls"], "questions": cell["questions"]}
        if engine_eval is None:
            return {**not_run("no per-call evaluation of this replay"), "calls": cell["calls"], "questions": cell["questions"]}
        cell["metrics"]["ex"] = rate_metric(boot, _gold_hits(identities, engine, engine_eval))
        cell["unparsed"] = sum(not engine[i]["parsed"] for i in identities)
        return cell
    if site in AGENT_CALL_SITES:
        if trivial:
            return {**open_question("Q4"), "calls": cell["calls"], "questions": cell["questions"]}
        cell["prompts"] = _prompt_partition(teacher, engine, identities)
        cell["open_question"] = "Q5"
        return cell
    scored = [i for i in identities if teacher[i]["parsed"]]
    cell["teacher_unparsed"] = len(identities) - len(scored)
    if site == "extract_keywords" and trivial:
        return {**open_question("Q2"), "calls": cell["calls"], "questions": cell["questions"]}

    def engine_decision(identity: Identity) -> Any:
        if not trivial:
            return engine[identity]["decision"]
        return False if site == "filter_column" else teacher[identity].get("candidates")
    if not trivial:
        cell["unparsed"] = sum(not engine[i]["parsed"] for i in scored)
    agreement: Dict[str, List[float]] = {}
    for identity in scored:
        mine, theirs = engine_decision(identity), teacher[identity]["decision"]
        _add(agreement, identity[0], [mine is not None and theirs is not None and mine == theirs, 1])
    raw = rate_metric(boot, agreement)
    if site == "filter_column":
        table: Dict[str, List[float]] = {}
        for identity in scored:
            theirs = bool(teacher[identity]["decision"])
            mine = engine_decision(identity)
            mine = (not theirs) if mine is None else bool(mine)  # no decision: it disagrees
            _add(table, identity[0], [theirs and mine, theirs and not mine, mine and not theirs, not theirs and not mine,
                                      theirs, not theirs])
        cell["metrics"]["recall_kept"] = rate_metric(boot, table, 0, 4)
        cell["metrics"]["agreement_dropped"] = rate_metric(boot, table, 3, 5)
        cell["metrics"]["kappa"] = boot.interval(table, kappa)
        cell["metrics"]["raw_agreement"] = raw
        cell["counts"] = {name: int(sum(v[i] for v in table.values())) for i, name in enumerate(
            ("both_keep", "teacher_keeps_only", "engine_keeps_only", "both_drop"))}
    elif site in SELECTION_SITES:
        cell["metrics"]["recall"] = ({"status": "1 by construction"} if trivial else open_question("Q1"))
        cell["metrics"]["exact_set_agreement"] = raw
        if trivial:
            cell["teacher_set_outside_prompt"] = sum(
                teacher[i]["decision"] is None or not teacher[i]["decision"] <= teacher[i].get("candidates", frozenset())
                for i in scored)
    elif site == "extract_keywords":
        jaccard: Dict[str, List[float]] = {}
        both_empty = 0
        for identity in scored:
            mine, theirs = engine_decision(identity), teacher[identity]["decision"]
            if mine is None or theirs is None:
                index = 0.0
            elif not mine and not theirs:
                index, both_empty = 1.0, both_empty + 1
            else:
                index = len(mine & theirs) / len(mine | theirs)
            _add(jaccard, identity[0], [index, 1])
        cell["metrics"]["mean_jaccard"] = rate_metric(boot, jaccard)
        cell["metrics"]["mean_jaccard"].pop("numerator")
        cell["metrics"]["exact_set_agreement"] = raw
        cell["both_sets_empty"] = both_empty
    return cell


def _gold_hits(identities: List[Identity], calls: Dict[Identity, dict], results: Dict[Identity, dict]
               ) -> Dict[str, List[float]]:
    hits: Dict[str, List[float]] = {}
    for identity in identities:
        _add(hits, identity[0], [call_correct(identity, calls, results), 1])
    return hits


def call_correct(identity: Identity, calls: Dict[Identity, dict], results: Dict[Identity, dict]) -> bool:
    """J2's rule: an invocation whose last attempt did not parse has no SQL and is wrong; otherwise the
    per-call evaluation says, and one it does not cover is refused."""
    if not calls[identity]["parsed"]:
        return False
    if identity not in results:
        raise TablesError(f"the per-call evaluation has no result for {identity}")
    return results[identity]["correct"]


def _prompt_partition(teacher: Dict[Identity, dict], engine: Dict[Identity, dict], identities: List[Identity]
                      ) -> Dict[str, Any]:
    """The distinct prompts of the agent's steps, each with its calls and how many agree with the teacher,
    and the prompts counted by whether every call, some or none agrees. No rate over calls."""
    prompts: Dict[str, Dict[str, int]] = {}
    for identity in identities:
        entry = prompts.setdefault(teacher[identity]["prompt"], {"calls": 0, "agree": 0, "teacher_unparsed": 0})
        if not teacher[identity]["parsed"]:
            entry["teacher_unparsed"] += 1
            continue
        entry["calls"] += 1
        mine, theirs = engine[identity]["decision"], teacher[identity]["decision"]
        entry["agree"] += mine is not None and mine == theirs
    judged = [p for p in prompts.values() if p["calls"]]
    return {"distinct": len(prompts),
            "every_call_agrees": sum(p["agree"] == p["calls"] for p in judged),
            "some_calls_agree": sum(0 < p["agree"] < p["calls"] for p in judged),
            "no_call_agrees": sum(p["agree"] == 0 for p in judged),
            "by_prompt": [{"prompt_sha256": key, **prompts[key]} for key in sorted(prompts)]}


def replay_scope(teacher: Dict[Identity, dict], replay: Dict[Identity, dict], manifest: Dict[str, Any], name: str
                 ) -> List[Identity]:
    """The teacher's invocations a replay answers for: its questions and its declared call sites. Inside
    them, an invocation the replay lacks, or one it has that the teacher does not, is refused."""
    questions = {str(q) for q in manifest.get("question_ids") or ()}
    sites = set(manifest.get("call_sites") or CALL_SITES)
    scope = [i for i in teacher if i[0] in questions and i[1] in sites]
    missing = [i for i in scope if i not in replay]
    extra = set(replay) - set(scope)
    if missing or extra:
        raise TablesError(f"{name}: {len(missing)} of its source's invocations are missing and {len(extra)} are not the "
                          f"source's, e.g. {(missing or sorted(extra))[:2]}")
    return scope


def by_site(identities: Iterable[Identity]) -> Dict[str, List[Identity]]:
    out: Dict[str, List[Identity]] = {site: [] for site in SITE_ORDER}
    for identity in identities:
        out[identity[1]].append(identity)
    return out


def teacher_gold_cell(identities: List[Identity], teacher: Dict[Identity, dict], teacher_eval: Optional[Dict[Identity, dict]],
                      boot: Bootstrap, label: str) -> Dict[str, Any]:
    """The teacher's EX on the same invocations, from its own run (the per-call evaluation of B0)."""
    if teacher_eval is None:
        return not_run("no per-call evaluation of the teacher's run")
    if not identities:
        return {"status": "computed", "calls": 0, "questions": 0, "metrics": {}}
    return {"status": "computed", "source": label, "calls": len(identities), "questions": len({i[0] for i in identities}),
            "metrics": {"ex": rate_metric(boot, _gold_hits(identities, teacher, teacher_eval))}}


def column_cells(column: str, inputs: Dict[str, Any], select: Callable[[Identity], bool], boot: Bootstrap
                 ) -> Dict[str, Dict[str, Any]]:
    """Every call site's cell of one column, over the invocations `select` keeps (all of them for the
    per-call-site table, one cluster's for the roll-up)."""
    teacher = inputs["teacher_calls"]
    if column == "trivial":
        groups = by_site(i for i in teacher if select(i))
        return {site: site_cell(site, teacher, None, groups[site], boot, trivial=True) for site in SITE_ORDER}
    replay = inputs["replays"].get(column)
    if replay is None:
        cells = {site: not_run(f"no {COLUMN_TITLES[column]} replay given") for site in SITE_ORDER}
        if column == "teacher":
            cells = _teacher_fallback(inputs, select, boot, cells)
        return cells
    groups = by_site(i for i in replay["scope"] if select(i))
    declared = set(replay["manifest"].get("call_sites") or CALL_SITES)
    cells = {}
    for site in SITE_ORDER:
        if site not in declared:
            cells[site] = not_run("the replay does not cover this call site")
            continue
        cells[site] = site_cell(site, teacher, replay["calls"], groups[site], boot, engine_eval=replay["eval"])
        if column == "teacher" and site in GOLD_CALL_SITES:
            cells[site] = {"status": "computed", "calls": len(groups[site]), "questions": len({i[0] for i in groups[site]}),
                           "from_replay": cells[site],
                           "from_own_run": teacher_gold_cell(groups[site], teacher, inputs["teacher_eval"], boot,
                                                             "B0's run, the replay's invocations")}
    if column == "teacher":
        for site in GOLD_CALL_SITES:
            if cells[site]["status"] == NOT_RUN:
                own = by_site(i for i in teacher if select(i))[site]
                cells[site] = {"status": "computed", "calls": len(own), "questions": len({i[0] for i in own}),
                               "from_replay": cells[site],
                               "from_own_run": teacher_gold_cell(own, teacher, inputs["teacher_eval"], boot, "B0's run")}
    return cells


def _teacher_fallback(inputs: Dict[str, Any], select: Callable[[Identity], bool], boot: Bootstrap,
                      cells: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Without the teacher's replay of these calls: the self-agreement A of 6.4 measured on the pilot,
    labelled as such, where a pilot replay is given; and at the call sites with gold the teacher's EX on
    the same invocations, from its own run."""
    teacher = inputs["teacher_calls"]
    own = by_site(i for i in teacher if select(i))
    for site in GOLD_CALL_SITES:
        cells[site] = {"status": "computed", "calls": len(own[site]), "questions": len({i[0] for i in own[site]}),
                       "from_replay": not_run("no teacher replay given"),
                       "from_own_run": teacher_gold_cell(own[site], teacher, inputs["teacher_eval"], boot, "B0's run")}
    pilot = inputs.get("pilot")
    if pilot is None or inputs.get("pilot_unselectable"):
        return cells
    groups = by_site(pilot["scope"])
    declared = set(pilot["manifest"].get("call_sites") or CALL_SITES)
    for site in SITE_ORDER:
        if site in GOLD_CALL_SITES or site not in declared:
            continue
        cell = site_cell(site, pilot["teacher_calls"], pilot["calls"], groups[site], boot)
        kept = {name: metric for name, metric in cell.get("metrics", {}).items()
                if name in ("raw_agreement", "exact_set_agreement")}
        cells[site] = {**{k: v for k, v in cell.items() if k not in ("metrics", "counts")}, "metrics": kept,
                       "pilot": True, "source": "self-agreement A of SPEC 6.4, measured on the pilot (calibration), "
                                                "not on these calls"}
    return cells


# ------------------------------------------------------------------------------------------ tables

def per_call_site_table(inputs: Dict[str, Any], boot: Bootstrap) -> Dict[str, Any]:
    rows = {column: column_cells(column, inputs, lambda i: True, boot) for column in COLUMNS}
    return {"rows": {site: {column: rows[column][site] for column in COLUMNS} for site in SITE_ORDER}}


def cluster_table(inputs: Dict[str, Any], boot: Bootstrap) -> Dict[str, Any]:
    """The per-call-site quantities within each cluster. The cluster of an invocation is the router's
    assignment in the adapter replay; without that replay the roll-up is not run."""
    adapter = inputs["replays"].get("adapter")
    if adapter is None:
        return not_run("the cluster of a call is the router's assignment in the adapter (B4) replay, which was not given")
    assigned = {identity: adapter["calls"][identity]["cluster"] for identity in adapter["scope"]}
    clusters = sorted({c for c in assigned.values() if c is not None} | set(CLUSTERS))
    out: Dict[str, Any] = {"clusters": {}, "unassigned": sum(c is None for c in assigned.values())}
    inputs = {**inputs, "pilot_unselectable": True}  # the pilot's calls carry no assignment of this replay
    for cluster in clusters:
        members = [i for i, c in assigned.items() if c == cluster]
        columns = {column: column_cells(column, inputs, lambda i, c=cluster: assigned.get(i) == c, boot)
                   for column in COLUMNS}
        sites = [site for site in SITE_ORDER if any(i[1] == site for i in members)]
        out["clusters"][cluster] = {
            "calls": len(members), "questions": len({i[0] for i in members}),
            "training_rows": inputs["training_rows"].get(cluster),
            "rows": {site: {column: columns[column][site] for column in COLUMNS} for site in sites}}
    return out


def rule_reading(rows: Optional[int]) -> str:
    """Where a number of training rows stands against the paper's 10k-100k rule of thumb."""
    if rows is None:
        return "training rows not found"
    low, high = RULE_OF_THUMB
    if rows < low:
        return f"below 10k ({low / rows:.1f}x below)" if rows else "below 10k (no rows)"
    return "within 10k-100k" if rows <= high else "above 100k"


def live_table(inputs: Dict[str, Any], boot: Bootstrap) -> Dict[str, Any]:
    """Per arm and call site, in the arm's own execution: the counts, the path (calls per question and
    the questions that reach the call site) and, where gold exists, EX per call."""
    out: Dict[str, Any] = {}
    for arm, live in inputs["live"].items():
        questions = len(live["manifest"].get("question_ids") or ())
        groups = by_site(live["calls"])
        rows = {}
        for site in SITE_ORDER:
            identities = groups[site]
            reached = len({i[0] for i in identities})
            row: Dict[str, Any] = {
                "invocations": len(identities), "attempts": sum(live["calls"][i]["attempts"] for i in identities),
                "questions_reaching": reached, "questions_in_run": questions,
                "invocations_per_question_in_run": len(identities) / questions if questions else None,
                "invocations_per_reaching_question": len(identities) / reached if reached else None}
            if site in GOLD_CALL_SITES:
                if live["per_call_eval"] is None:
                    row["ex"] = not_run("no per-call evaluation of this run")
                elif identities:
                    row["ex"] = {"status": "computed", **rate_metric(boot, _gold_hits(identities, live["calls"], live["per_call_eval"]))}
                else:
                    row["ex"] = {"status": "computed", "value": None, "numerator": 0, "denominator": 0}
            rows[site] = row
        out[arm] = {"run_id": live["run_id"], "rows": rows}
    return out


def _ex_by(results: List[dict], key: str, values: Sequence[str], boot: Optional[Bootstrap]) -> Dict[str, Any]:
    out = {}
    for value in values:
        group = [r for r in results if r.get(key) == value]
        hits = {str(r["question_id"]): [bool(r["correct"]), 1] for r in group}
        cell: Dict[str, Any] = {"n": len(group), "correct": sum(bool(r["correct"]) for r in group)}
        if boot is not None and group:
            cell.update(rate_metric(boot, hits))
        out[value] = cell
    return out


def _databases(inputs: Dict[str, Any]) -> List[str]:
    found = set()
    for live in inputs["live"].values():
        found |= {r.get("db_id") for r in live["eval_results"] or ()}
    for rows in [inputs["teacher_eval"]] + [r["eval"] for r in inputs["replays"].values()]:
        found |= {row["db_id"] for row in (rows or {}).values()}
    return sorted(d for d in found if d is not None)


def difficulty_table(inputs: Dict[str, Any], boot: Bootstrap) -> Dict[str, Any]:
    """Difficulty (3): each arm's EX end to end, each end-to-end comparison (the paired difference), and
    SQL generation per column on the teacher's inputs and per arm in its own execution."""
    arms = {arm: (_ex_by(live["eval_results"], "difficulty", DIFFICULTIES, boot) if live["eval_results"] is not None
                  else not_run("no end-to-end evaluation of this run")) for arm, live in inputs["live"].items()}
    comparisons = {}
    for a, b in inputs["compare"]:
        name = f"{a} - {b}"
        sides = [inputs["live"].get(arm) for arm in (a, b)]
        if any(side is None or side["eval_results"] is None for side in sides):
            comparisons[name] = not_run("needs the end-to-end evaluation of both arms")
            continue
        right = [{str(r["question_id"]): r for r in side["eval_results"]} for side in sides]
        if set(right[0]) != set(right[1]):
            raise TablesError(f"{a} and {b} are not scored on the same questions")
        cells = {}
        for difficulty in DIFFICULTIES:
            ids = [q for q, r in right[0].items() if r.get("difficulty") == difficulty]
            diffs = {q: [int(bool(right[0][q]["correct"])) - int(bool(right[1][q]["correct"])), 1] for q in ids}
            cell: Dict[str, Any] = {"n": len(ids), "a_only": sum(v[0] == 1 for v in diffs.values()),
                                    "b_only": sum(v[0] == -1 for v in diffs.values())}
            if ids:
                cell.update(boot.interval(diffs, ratio(0, 1)))
            cells[difficulty] = cell
        comparisons[name] = {"status": "computed", "by_difficulty": cells}
    return {"arms": arms, "comparisons": comparisons, "sql_generation": _generation_cut(inputs, "difficulty", DIFFICULTIES, boot)}


def _generation_cut(inputs: Dict[str, Any], key: str, values: Sequence[str], boot: Optional[Bootstrap]) -> Dict[str, Any]:
    """`generate_candidate` EX per call by `key` (difficulty or database): the replays on the teacher's
    inputs, the teacher on its own run, and each arm in its own execution."""
    site = "generate_candidate"

    def cut(identities: List[Identity], calls: Dict[Identity, dict], results: Optional[Dict[Identity, dict]], label: Dict[Identity, dict]):
        if results is None:
            return not_run("no per-call evaluation")
        out = {}
        for value in values:
            group = [i for i in identities if (label.get(i) or {}).get(key) == value]
            hits = _gold_hits(group, calls, results)
            cell: Dict[str, Any] = {"calls": len(group), "questions": len({i[0] for i in group}),
                                    "correct": int(sum(v[0] for v in hits.values()))}
            if boot is not None and group:
                cell.update(rate_metric(boot, hits))
            out[value] = cell
        return {"status": "computed", "by": out}
    teacher, teacher_eval = inputs["teacher_calls"], inputs["teacher_eval"]
    labels = teacher_eval or {}
    on_inputs: Dict[str, Any] = {"trivial": open_question("Q3")}
    for column in REPLAY_COLUMNS:
        replay = inputs["replays"].get(column)
        if replay is None or site not in set(replay["manifest"].get("call_sites") or CALL_SITES):
            on_inputs[column] = not_run(f"no {COLUMN_TITLES[column]} replay of this call site")
            continue
        identities = by_site(replay["scope"])[site]
        on_inputs[column] = cut(identities, replay["calls"], replay["eval"], replay["eval"] or labels)
    on_inputs["teacher_own_run"] = cut(by_site(teacher)[site], teacher, teacher_eval, labels)
    own = {arm: cut(by_site(live["calls"])[site], live["calls"], live["per_call_eval"], live["per_call_eval"] or {})
           for arm, live in inputs["live"].items()}
    return {"on_the_teachers_inputs": on_inputs, "own_execution": own}


def database_table(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Database (11), counts only: each arm end to end, and SQL generation as in the difficulty cut."""
    databases = _databases(inputs)
    arms = {arm: (_ex_by(live["eval_results"], "db_id", databases, None) if live["eval_results"] is not None
                  else not_run("no end-to-end evaluation of this run")) for arm, live in inputs["live"].items()}
    return {"databases": databases, "arms": arms, "sql_generation": _generation_cut(inputs, "db_id", databases, None)}


def cross_table(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Replay against live, per arm and per call site with gold: questions by (right on the teacher's
    inputs, right in the arm's own execution). A question is right on the teacher's inputs at a call site
    when every one of its calls there is (J2's question-level reading)."""
    out = {}
    for column, arm in COLUMN_ARM.items():
        replay, live = inputs["replays"].get(column), inputs["live"].get(arm)
        if replay is None or replay["eval"] is None:
            out[arm] = not_run(f"needs the {COLUMN_TITLES[column]} replay and its per-call evaluation")
            continue
        if live is None or live["eval_results"] is None:
            out[arm] = not_run(f"needs {arm}'s own execution and its end-to-end evaluation")
            continue
        final = {str(r["question_id"]): bool(r["correct"]) for r in live["eval_results"]}
        sites = {}
        for site in GOLD_CALL_SITES:
            on_inputs: Dict[str, bool] = {}
            for identity in by_site(replay["scope"])[site]:
                ok = call_correct(identity, replay["calls"], replay["eval"])
                on_inputs[identity[0]] = on_inputs.get(identity[0], True) and ok
            both = [q for q in on_inputs if q in final]
            sites[site] = {"questions": len(both),
                           "replay_only_questions": len(set(on_inputs) - set(final)),
                           "right_both": sum(on_inputs[q] and final[q] for q in both),
                           "right_on_inputs_only": sum(on_inputs[q] and not final[q] for q in both),
                           "right_live_only": sum(not on_inputs[q] and final[q] for q in both),
                           "wrong_both": sum(not on_inputs[q] and not final[q] for q in both)}
        out[arm] = {"status": "computed", "replay": replay["run_id"], "live": live["run_id"], "by_call_site": sites}
    return out


# ------------------------------------------------------------------------------------------ projection

def ordered_projection(prediction: List[tuple], gold: List[tuple], max_checks: int = 20_000) -> Optional[bool]:
    """Whether some of the prediction's columns, kept in their order, give the gold's set of rows (SPEC
    6.1). None when the search was cut off after `max_checks` candidate projections."""
    gold_rows = set(gold)
    if not gold_rows or not prediction:
        return not gold_rows and not prediction
    width, k = len(prediction[0]), len(next(iter(gold_rows)))
    if k > width or any(len(row) != k for row in gold_rows):
        return False
    gold_columns = [{row[j] for row in gold_rows} for j in range(k)]
    try:
        prediction_columns = [{row[i] for row in prediction} for i in range(width)]
    except TypeError:
        return False
    fits = [[i for i in range(width) if prediction_columns[i] == gold_columns[j]] for j in range(k)]
    checks = 0

    def search(j: int, after: int, chosen: List[int]) -> Optional[bool]:
        nonlocal checks
        if j == k:
            checks += 1
            if checks > max_checks:
                return None
            return {tuple(row[i] for i in chosen) for row in prediction} == gold_rows
        for i in fits[j]:
            if i > after:
                found = search(j + 1, i, chosen + [i])
                if found or found is None:
                    return found
        return False
    return search(0, -1, [])


def projection_table(inputs: Dict[str, Any], boot: Bootstrap, root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Per arm: EX, and the share of questions right once the prediction's extra columns are dropped.
    Each wrong prediction that executed, and its gold, run again at the eval's fixed date."""
    from bench.evaluate import execute
    out: Dict[str, Any] = {}
    database_root = root / "data" / "bird_dev" / config["data"]["bird_dev_databases"]["root"] / "dev_databases"
    checked: Dict[Tuple[str, str], None] = {}
    for arm, live in inputs["live"].items():
        if live["eval_results"] is None:
            out[arm] = not_run("no end-to-end evaluation of this run")
            continue
        manifest = live["eval_manifest"]
        day, timeout_s = manifest["fixed_date"], manifest["timeout_s"]
        tolerant, not_reexecuted, cut_off, changed = [], [], [], []
        hits: Dict[str, List[float]] = {}
        for result in live["eval_results"]:
            question, right = str(result["question_id"]), bool(result["correct"])
            if not right and result["pred_error"] is None and result["gold_error"] is None:
                path = database_root / result["db_id"] / f"{result['db_id']}.sqlite"
                recorded = (manifest.get("databases") or {}).get(result["db_id"])
                if not path.is_file():
                    raise TablesError(f"{path} does not exist: projection-tolerant EX executes on the pinned databases "
                                      f"under --root")
                if (str(path), recorded) not in checked:
                    if recorded is not None and _sha256(path) != recorded:
                        raise TablesError(f"{path} is not the database {live['eval_id']} scored on")
                    checked[(str(path), recorded)] = None
                gold_rows, gold_error, _ = execute(path, result["gold_sql"], timeout_s, day)
                rows, error, _ = execute(path, result["predicted_sql"], timeout_s, day)
                if gold_error is not None or error is not None:
                    not_reexecuted.append(question)
                elif set(rows) == set(gold_rows):  # scored wrong, the same rows now: not a matter of columns
                    changed.append(question)
                else:
                    found = ordered_projection(rows, gold_rows)
                    if found is None:
                        cut_off.append(question)
                    elif found:
                        tolerant.append(question)
            hits[question] = [right, right or question in tolerant, 1]
        out[arm] = {"status": "computed", "eval": live["eval_id"], "n": len(hits),
                    "ex": rate_metric(boot, hits, 0, 2), "projection_tolerant_ex": rate_metric(boot, hits, 1, 2),
                    "right_only_once_extra_columns_are_dropped": sorted(tolerant, key=_question_key),
                    "not_reexecuted": not_reexecuted, "search_cut_off": cut_off, "same_rows_when_executed_again": changed}
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def without_table(inputs: Dict[str, Any], questions: List[str]) -> Dict[str, Any]:
    """Each arm's EX without the given questions (SPEC section 2: the test question that repeats a
    training one). Counts and the share, descriptive."""
    out = {}
    for arm, live in inputs["live"].items():
        if live["eval_results"] is None:
            out[arm] = not_run("no end-to-end evaluation of this run")
            continue
        kept = [r for r in live["eval_results"] if str(r["question_id"]) not in questions]
        correct = sum(bool(r["correct"]) for r in kept)
        out[arm] = {"status": "computed", "n": len(kept), "correct": correct, "ex": correct / len(kept) if kept else None,
                    "removed": len(live["eval_results"]) - len(kept)}
    return out


# ------------------------------------------------------------------------------------------ inputs

def training_rows(root: Path) -> Tuple[Dict[str, int], str]:
    """The training rows of each cluster, from train/datasets/manifest.json when this checkout has it."""
    path = root / "train" / "datasets" / "manifest.json"
    if not path.is_file():
        return {}, "train/datasets/manifest.json not found: rows per cluster not read"
    clusters = json.loads(path.read_text()).get("clusters") or {}
    return {cluster: found["n"] for cluster, found in clusters.items()}, "train/datasets/manifest.json"


def load_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    runs, allow = Path(args.root) / "runs", args.allow_test
    teacher_manifest = read_manifest(runs, args.teacher, allow, type="agent", arm="B0")
    split = teacher_manifest["split"]
    inputs: Dict[str, Any] = {
        "split": split, "teacher": args.teacher, "teacher_manifest": teacher_manifest,
        "teacher_calls": load_invocations(runs / args.teacher / "calls.jsonl", teacher=True),
        "teacher_eval": (load_per_call_eval(runs, args.teacher_per_call_eval, args.teacher, allow)
                         if args.teacher_per_call_eval else None),
        "teacher_per_call_eval_id": args.teacher_per_call_eval, "replays": {}, "live": {}, "pilot": None}
    for column, (run_id, eval_id) in args.replay.items():
        manifest = read_manifest(runs, run_id, allow, type="replay", source_run_id=args.teacher)
        calls = load_invocations(runs / run_id / "calls.jsonl")
        inputs["replays"][column] = {
            "run_id": run_id, "eval_id": eval_id, "manifest": manifest, "calls": calls,
            "scope": replay_scope(inputs["teacher_calls"], calls, manifest, run_id),
            "eval": load_per_call_eval(runs, eval_id, run_id, allow) if eval_id else None}
    if args.teacher_pilot_replay:
        manifest = read_manifest(runs, args.teacher_pilot_replay, allow, type="replay")
        source = manifest["source_run_id"]
        read_manifest(runs, source, allow, type="agent", arm="B0")
        source_calls = (inputs["teacher_calls"] if source == args.teacher
                        else load_invocations(runs / source / "calls.jsonl", teacher=True))
        calls = load_invocations(runs / args.teacher_pilot_replay / "calls.jsonl")
        inputs["pilot"] = {"run_id": args.teacher_pilot_replay, "source_run_id": source, "manifest": manifest,
                           "calls": calls, "teacher_calls": source_calls,
                           "scope": replay_scope(source_calls, calls, manifest, args.teacher_pilot_replay)}
    live = dict(args.live)
    if "B0" not in live:
        live = {"B0": (args.teacher, None, args.teacher_per_call_eval), **live}
    for arm, (run_id, eval_id, per_call_eval_id) in live.items():
        manifest = read_manifest(runs, run_id, allow, type="agent", arm=arm.split("/")[0], split=split)
        eval_manifest, eval_results = load_eval(runs, eval_id, run_id, allow) if eval_id else (None, None)
        inputs["live"][arm] = {
            "run_id": run_id, "eval_id": eval_id, "per_call_eval_id": per_call_eval_id, "manifest": manifest,
            "calls": inputs["teacher_calls"] if run_id == args.teacher else load_invocations(runs / run_id / "calls.jsonl"),
            "eval_manifest": eval_manifest, "eval_results": eval_results,
            "per_call_eval": load_per_call_eval(runs, per_call_eval_id, run_id, allow) if per_call_eval_id else None}
    inputs["compare"] = args.compare or [(arm, "B0") for arm in inputs["live"] if arm != "B0"]
    inputs["training_rows"], inputs["training_rows_source"] = training_rows(Path(args.root))
    return inputs


def build(args: argparse.Namespace) -> Dict[str, Any]:
    inputs = load_inputs(args)
    config = json.loads((Path(args.root) / "runs" / args.teacher / "config.json").read_text())
    n_boot = args.n_boot or config["stats"]["n_boot"]
    seed = args.seed if args.seed is not None else config["seeds"]["bootstrap"]
    boot = Bootstrap(n_boot, seed)
    replays = {column: {"replay": r["run_id"], "per_call_eval": r["eval_id"]} for column, r in inputs["replays"].items()}
    replays.update({column: NOT_RUN for column in REPLAY_COLUMNS if column not in replays})
    live = {arm: {"run": found["run_id"], "eval": found["eval_id"] or NOT_RUN,
                  "per_call_eval": found["per_call_eval_id"] or NOT_RUN} for arm, found in inputs["live"].items()}
    on_inputs = {"teacher_run": args.teacher, "teacher_per_call_eval": args.teacher_per_call_eval or NOT_RUN,
                 "replays": replays,
                 "teacher_pilot_replay": (f"{inputs['pilot']['run_id']} (of {inputs['pilot']['source_run_id']})"
                                          if inputs["pilot"] else NOT_RUN)}
    tables = {
        "per_call_site": {"inputs": on_inputs, **per_call_site_table(inputs, boot)},
        "per_cluster": {"inputs": {**on_inputs, "training_rows": inputs["training_rows_source"]}, **cluster_table(inputs, boot)},
        "own_execution": {"inputs": live, "arms": live_table(inputs, boot)},
        "difficulty": {"inputs": {"live": live, **on_inputs}, **difficulty_table(inputs, boot)},
        "database": {"inputs": {"live": live, **on_inputs}, **database_table(inputs)},
        "replay_against_live": {"inputs": {"live": live, "replays": replays}, "arms": cross_table(inputs)},
        "projection_tolerant_ex": {"inputs": live, "arms": projection_table(inputs, boot, Path(args.root), config)},
    }
    if args.without:
        tables["without_questions"] = {"inputs": live, "questions": args.without, "arms": without_table(inputs, args.without)}
    return {"label": LABEL, "split": inputs["split"], "interval": {"level": 0.95, "sides": 2, "unit": "question",
                                                                    "n_boot": n_boot, "seed": seed, "numpy": np.__version__},
            "training_rows": inputs["training_rows"], "open_questions": OPEN, "conventions": CONVENTIONS, "tables": tables}


# ------------------------------------------------------------------------------------------ rendering

def _number(value: Optional[float], digits: int = 3) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def _metric(name: str, metric: Dict[str, Any]) -> str:
    if metric.get("status") == "not computed":
        return f"{name}: not computed ({metric['open_question']})"
    if "status" in metric and "value" not in metric:
        return f"{name}: {metric['status']}"
    text = f"{name} {_number(metric['value'])}"
    if metric.get("low") is not None:
        text += f" [{_number(metric['low'])}, {_number(metric['high'])}]"
    if "numerator" in metric:
        text += f" ({metric['numerator']}/{metric['denominator']})"
    if metric.get("undefined_resamples"):
        text += f" ({metric['undefined_resamples']} resamples undefined)"
    return text


METRIC_NAMES = {"recall_kept": "recall of the columns the teacher kept", "agreement_dropped": "agreement on the columns it dropped",
                "kappa": "Cohen's kappa", "raw_agreement": "raw agreement", "recall": "recall of the teacher's set",
                "exact_set_agreement": "exact-set agreement (the raw agreement)", "mean_jaccard": "mean Jaccard",
                "ex": "EX per call"}


def render_cell(cell: Dict[str, Any]) -> str:
    if cell["status"] == NOT_RUN:
        return f"{NOT_RUN} ({cell['why']})"
    if cell["status"] == "not computed":
        return f"not computed ({cell['open_question']}); calls {cell['calls']}, questions {cell['questions']}"
    if "from_own_run" in cell:
        return "<br>".join(f"{title}: {render_cell(cell[key])}" for key, title in
                           (("from_own_run", "teacher's own run"), ("from_replay", "teacher's replay")))
    parts = []
    if cell.get("pilot"):
        parts.append(f"PILOT, not these calls: {cell['source']}")
    elif cell.get("source"):
        parts.append(cell["source"])
    parts += [_metric(METRIC_NAMES[name], metric) for name, metric in cell.get("metrics", {}).items()]
    if "prompts" in cell:
        p = cell["prompts"]
        parts.append(f"{p['distinct']} distinct prompts: every call agrees on {p['every_call_agrees']}, some on "
                     f"{p['some_calls_agree']}, none on {p['no_call_agrees']} (Q5; no rate over calls)")
    parts.append(f"calls {cell['calls']}, questions {cell['questions']}")
    for key, title in (("unparsed", "engine outputs that did not parse"), ("teacher_unparsed", "teacher outputs that did not parse, left out"),
                       ("both_sets_empty", "calls with both sets empty"), ("teacher_set_outside_prompt", "teacher sets not within the prompt's candidates")):
        if cell.get(key):
            parts.append(f"{title}: {cell[key]}")
    return "<br>".join(parts)


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    clean = [[str(c).replace("|", "\\|").replace("\n", " ") for c in row] for row in [header] + rows]
    return ["| " + " | ".join(clean[0]) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in clean[1:]]


def _inputs(found: Any) -> List[str]:
    return ["Inputs: `" + json.dumps(found, sort_keys=True) + "`", ""]


def _rate(cell: Dict[str, Any]) -> str:
    if cell.get("status") == NOT_RUN:
        return f"{NOT_RUN} ({cell['why']})"
    if cell.get("status") == "not computed":
        return f"not computed ({cell['open_question']})"
    return _metric("", cell).strip() if cell.get("value") is not None or "numerator" in cell else "no call"


def render(result: Dict[str, Any]) -> str:
    tables, interval = result["tables"], result["interval"]
    lines = [f"# Exploratory tables (SPEC 6.1), split `{result['split']}`", "",
             "Every table here is exploratory and descriptive. None enters a registered judgment or allocation. "
             f"Intervals are unadjusted, two-sided 95%, over {interval['n_boot']} resamples of questions (seed "
             f"{interval['seed']}, numpy {interval['numpy']}); a question's calls stay together. Fidelity to the teacher is not correctness "
             "(SPEC 6.2).", ""]

    site_rows = tables["per_call_site"]["rows"]
    lines += [f"## Per call site, on the teacher's inputs ({LABEL})", ""] + _inputs(tables["per_call_site"]["inputs"])
    lines += _table(["call site", "training rows (SPEC 6.1)"] + [COLUMN_TITLES[c] for c in COLUMNS],
                    [[site, TRAINING_ROWS_BY_SITE[site]] + [render_cell(site_rows[site][c]) for c in COLUMNS] for site in SITE_ORDER])
    lines += ["", "Warnings published with the table (SPEC 6.1): exact-set agreement on keywords is rarely met even by the "
              "teacher, so a low value there says little about the decision. The agent's steps have about 11 distinct "
              "prompts (the prompt does not depend on the question), so their effective n is the number of steps, not of "
              "calls. Repair has a small n (the teacher hardly repairs) and 6 training rows, and is read as such. Raw "
              "agreement is not the primary metric where the teacher's answers are imbalanced or the teacher's "
              "self-agreement is below 95%.", ""]

    cluster = tables["per_cluster"]
    lines += [f"## Per cluster, on the teacher's inputs ({LABEL})", ""] + _inputs(cluster["inputs"])
    if cluster.get("status") == NOT_RUN:
        lines += [f"{NOT_RUN}: {cluster['why']}", ""]
    else:
        lines += _table(["cluster", "calls", "questions", "training rows", "against the 10k-100k rule of thumb", "SPEC 6.1 rows"],
                        [[name, found["calls"], found["questions"],
                          found["training_rows"] if found["training_rows"] is not None else "not found",
                          rule_reading(found["training_rows"]), SPEC_ROWS_BY_CLUSTER.get(name, "not listed")]
                         for name, found in cluster["clusters"].items()])
        lines += ["", f"Calls of the adapter replay with no cluster: {cluster['unassigned']}. A cluster's rows list its call "
                  "sites; no metric is pooled across call sites.", ""]
        rows = [[name, site] + [render_cell(found["rows"][site][c]) for c in COLUMNS]
                for name, found in cluster["clusters"].items() for site in found["rows"]]
        lines += _table(["cluster", "call site"] + [COLUMN_TITLES[c] for c in COLUMNS], rows) + [""]

    own = tables["own_execution"]
    lines += [f"## In each arm's own execution: counts, path and EX per call site ({LABEL})", ""] + _inputs(own["inputs"])
    rows = []
    for arm, found in own["arms"].items():
        for site in SITE_ORDER:
            row = found["rows"][site]
            rows.append([arm, site, row["invocations"], row["attempts"], f"{row['questions_reaching']} of {row['questions_in_run']}",
                         _number(row["invocations_per_question_in_run"], 2), _number(row["invocations_per_reaching_question"], 2),
                         _rate(row["ex"]) if "ex" in row else "no gold"])
    lines += _table(["arm", "call site", "invocations", "attempts (retries included)", "questions that reach it",
                     "invocations per question of the run", "invocations per question that reaches it", "EX per call"], rows) + [""]

    difficulty = tables["difficulty"]
    lines += [f"## Difficulty: end to end ({LABEL})", ""] + _inputs(difficulty["inputs"])
    lines += _table(["arm"] + list(DIFFICULTIES),
                    [[arm] + ([_rate(found)] * 3 if found.get("status") == NOT_RUN else
                              [_by_cell(found[d]) for d in DIFFICULTIES]) for arm, found in difficulty["arms"].items()]) + [""]
    lines += [f"### Difficulty: paired differences in EX ({LABEL})", ""]
    rows = []
    for name, found in difficulty["comparisons"].items():
        if found.get("status") == NOT_RUN:
            rows.append([name] + [_rate(found)] * 3)
        else:
            rows.append([name] + [(f"{_metric('difference', c)}; n {c['n']}, first only {c['a_only']}, second only {c['b_only']}"
                                   if c["n"] else "no question") for c in (found["by_difficulty"][d] for d in DIFFICULTIES)])
    lines += _table(["comparison"] + list(DIFFICULTIES), rows) + [""]
    lines += _generation_lines("Difficulty", difficulty["sql_generation"], DIFFICULTIES)

    database = tables["database"]
    lines += [f"## Database: end to end, counts only ({LABEL})", ""] + _inputs(database["inputs"])
    lines += _table(["arm"] + database["databases"],
                    [[arm] + ([_rate(found)] * len(database["databases"]) if found.get("status") == NOT_RUN else
                              [f"{found[d]['correct']} of {found[d]['n']}" for d in database["databases"]])
                     for arm, found in database["arms"].items()]) + [""]
    lines += _generation_lines("Database", database["sql_generation"], database["databases"], counts_only=True)

    cross = tables["replay_against_live"]
    lines += [f"## Replay against live, per arm: questions right on the teacher's inputs by right in the arm's own execution ({LABEL})",
              ""] + _inputs(cross["inputs"])
    rows = []
    for arm, found in cross["arms"].items():
        if found["status"] == NOT_RUN:
            rows.append([arm, "every call site with gold", _rate(found), "", "", "", ""])
            continue
        for site, c in found["by_call_site"].items():
            rows.append([arm, site, c["questions"], c["right_both"], c["right_on_inputs_only"], c["right_live_only"], c["wrong_both"]])
    lines += _table(["arm", "call site", "questions in both", "right in both", "right on the teacher's inputs only",
                     "right in its own execution only", "wrong in both"], rows) + [""]

    projection = tables["projection_tolerant_ex"]
    lines += [f"## Projection-tolerant EX per arm ({LABEL})", ""] + _inputs(projection["inputs"])
    rows = []
    for arm, found in projection["arms"].items():
        if found["status"] == NOT_RUN:
            rows.append([arm, _rate(found), "", "", "", ""])
        else:
            rows.append([arm, found["n"], _rate(found["ex"]), _rate(found["projection_tolerant_ex"]),
                         len(found["right_only_once_extra_columns_are_dropped"]),
                         f"{len(found['not_reexecuted'])} / {len(found['search_cut_off'])} / "
                         f"{len(found['same_rows_when_executed_again'])}"])
    lines += _table(["arm", "questions", "EX", "EX once the prediction's extra columns are dropped",
                     "questions right only that way",
                     "left as scored: not executed again / search cut off / the gold's rows when executed again"], rows) + [""]

    if "without_questions" in tables:
        without = tables["without_questions"]
        lines += [f"## EX without question(s) {', '.join(without['questions'])} ({LABEL})", ""] + _inputs(without["inputs"])
        lines += _table(["arm", "questions", "right", "EX", "removed"],
                        [[arm, _rate(found), "", "", ""] if found["status"] == NOT_RUN else
                         [arm, found["n"], found["correct"], _number(found["ex"]), found["removed"]]
                         for arm, found in without["arms"].items()]) + [""]

    lines += ["## Open questions: quantities the SPEC names and does not fix (not computed)", ""]
    lines += [f"- **{key}**: {text}." for key, text in result["open_questions"].items()]
    lines += ["", "## Conventions followed", ""] + [f"- {text}" for text in result["conventions"]] + [""]
    return "\n".join(lines)


def _by_cell(cell: Dict[str, Any]) -> str:
    return f"{_rate(cell)}; n {cell['n']}" if cell["n"] else "no question"


def _generation_lines(cut: str, found: Dict[str, Any], values: Sequence[str], counts_only: bool = False) -> List[str]:
    def row(name: str, cell: Dict[str, Any]) -> List[str]:
        if cell.get("status") != "computed":
            return [name] + [_rate(cell)] * len(values)
        if counts_only:
            return [name] + [f"{cell['by'][v]['correct']} of {cell['by'][v]['calls']} calls ({cell['by'][v]['questions']} questions)"
                             for v in values]
        return [name] + [(f"{_rate(cell['by'][v])}; calls {cell['by'][v]['calls']}, questions {cell['by'][v]['questions']}"
                          if cell["by"][v]["calls"] else "no call") for v in values]
    titles = {**COLUMN_TITLES, "teacher_own_run": "teacher, its own run"}
    suffix = ", counts only" if counts_only else ""
    lines = [f"### {cut}: SQL generation on the teacher's inputs, EX per call{suffix} ({LABEL})", ""]
    lines += _table(["column"] + list(values), [row(titles[name], cell) for name, cell in found["on_the_teachers_inputs"].items()]) + [""]
    lines += [f"### {cut}: SQL generation in each arm's own execution, EX per call{suffix} ({LABEL})", ""]
    lines += _table(["arm"] + list(values), [row(arm, cell) for arm, cell in found["own_execution"].items()]) + [""]
    return lines


# ------------------------------------------------------------------------------------------ command line

def _pairs(values: List[str], fields: int, names: Optional[Sequence[str]] = None) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    for value in values:
        name, _, rest = value.partition("=")
        parts = [p or None for p in rest.split(",")] if rest else []
        if not name or not parts or parts[0] is None or len(parts) > fields or name in out:
            raise TablesError(f"cannot read {value!r}: NAME=RUN[,EVAL...] with at most {fields} ids, each name once")
        if names is not None and name not in names:
            raise TablesError(f"{name!r} is not one of {', '.join(names)}")
        out[name] = tuple(parts + [None] * (fields - len(parts)))
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=str(REPO), help="the checkout that holds runs/, data/ and train/datasets")
    parser.add_argument("--teacher", required=True, help="the B0 agent run the replays resend")
    parser.add_argument("--teacher-per-call-eval", help="its per-call evaluation (bench eval --per-call)")
    parser.add_argument("--replay", action="append", default=[], metavar="COLUMN=RUN[,PER_CALL_EVAL]",
                        help=f"a replay of the teacher's calls; COLUMN is one of {', '.join(REPLAY_COLUMNS)}")
    parser.add_argument("--teacher-pilot-replay", help="the teacher's replay of the pilot (SPEC 6.4's A), shown and "
                        "labelled as the pilot's when no teacher replay of these calls is given")
    parser.add_argument("--live", action="append", default=[], metavar="ARM=RUN[,EVAL[,PER_CALL_EVAL]]",
                        help="an arm's own execution, its end-to-end evaluation and its per-call evaluation; a second "
                             "execution of an arm takes a suffix (B5/exploratory)")
    parser.add_argument("--compare", action="append", default=[], metavar="A:B",
                        help="an end-to-end comparison for the difficulty cut (default: every arm given against B0)")
    parser.add_argument("--without", action="append", default=[], metavar="QUESTION_ID",
                        help="also print each arm's EX without this question (SPEC section 2: 875 on the test)")
    parser.add_argument("--allow-test", action="store_true", help="read executions on the test split")
    parser.add_argument("--n-boot", type=int, help="resamples (default: stats.n_boot of the teacher run's configuration)")
    parser.add_argument("--seed", type=int, help="seed (default: seeds.bootstrap of the teacher run's configuration)")
    parser.add_argument("--out", help="write the tables here, as Markdown (default: standard output)")
    parser.add_argument("--json", help="write every cell here, as JSON")
    args = parser.parse_args(argv)
    args.replay = _pairs(args.replay, 2, REPLAY_COLUMNS)
    args.live = _pairs(args.live, 3)
    compare = []
    for value in args.compare:
        a, _, b = value.partition(":")
        if not a or not b:
            raise TablesError(f"cannot read --compare {value!r}: A:B")
        compare.append((a, b))
    args.compare = compare
    return args


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        result = build(args)
    except TablesError as e:
        print(f"exploratory tables: {e}", file=sys.stderr)
        return 2
    text = render(result)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, sort_keys=True, default=_plain) + "\n")
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)
    return 0


def _plain(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON: {type(value).__name__}")


if __name__ == "__main__":
    sys.exit(main())
