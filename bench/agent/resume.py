"""`bench run --resume-from <run_id>`: a new run that keeps what an agent run answered before it
stopped on an infrastructure failure, and answers only the questions left.

The source must be an agent run that ended `failed` or `interrupted`, with the same type, arm,
split, engine, facts, few-shot prefix, batch mode and configuration (`config_sha256`, byte for byte),
on the same splits and data manifest. On train and calib the commit may differ (both are recorded).

On test the rule is fixed before the registration and stricter (`check_test_source`): the source is
the latest attempt of its configuration in the published registry, whose manifest there is the one
read here; it ran under the pre-registration in force, with the same configuration, on a clean tree
at the registered code (its commit holds the registered commit's tree outside prereg/ and registry/);
and the outputs it copies still hash to what its manifest recorded when it ended (`predictions_sha256`,
`calls_sha256`). This session must itself be clean and at the registered code
(`barrier.ensure_execution_allowed`). Anything else is refused: the configuration is then run again
from its first question, which on test needs a stated defect (`--redo-reason`, `runner.open_run`).

A question of the source is complete when it has a prediction and neither raised (`failures`) nor
met a harness error (`harness_errors`): the runner removes the prediction of a question that was in
flight when the run was stopped (its calls were refused), and keeps that of a question whose harness
failed, which is therefore not an answer. A complete question's C1 lines are copied unchanged, every
attempt, `run_id` included (the source's: C1 forbids any other field on a line and no reader of
calls.jsonl compares a line's `run_id` with the run's; `resumed_question_ids` says which lines came
from where), with its prediction and its `tool_errors`. Every other question is answered again from
the start. The calls the source made for questions that are not complete are part of no answer and
are not copied: they are counted (calls, usage) under `resumed_from.discarded`, so what they cost
stays visible. CHESS's own outputs of the complete questions are copied under
chess/resumed/<source run_id>/ (inspection only: nothing downstream reads chess/).

A source killed outright (SIGKILL, a lost machine) never closed its manifest: it still says `running`.
It is resumed once its session's lock is free (`runner.hold_session`): its completed questions are those
its session journaled as they ended (answers*.jsonl, `runner.journal_writer`), with their C1 lines from
calls.jsonl and each worker's calls.w<k>.jsonl (a last line cut off by the kill is dropped). No manifest
recorded the sha256 of those outputs, so they are computed before anything is read or copied and
recorded in the new manifest, and on test in its intent, as `unregistered_source_hashes`.

The new run's manifest has the usual fields, `started_at`/`finished_at` being this session's (as its
run_id), plus `resumed_from` (the source's run_id, commit, dirty, config_sha256, status, times,
workers), `resumed_question_ids`, and `sessions`: every working session that produced the run's
answers, each with started_at, finished_at, commit and workers (the source's own sessions first, if
it was itself resumed). The working time of the run is the sum of its sessions, not
finished_at - started_at, which covers the last one only.

`bench replay --resume-from <replay run_id>` resumes a replay the same way, by invocation rather than by
question (the section at the end of this module).
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bench import data, paths

RESUMABLE = ("failed", "interrupted")
KILLED = "running"  # a manifest still `running` whose session no longer holds its lock: killed outright
# What a killed source's manifest, as the registry published it when it started, must share with the
# local one (the local one gained the resume's own fields since, if it was itself resumed).
STARTED_IDENTITY = ("run_id", "type", "arm", "engine", "split", "status", "prereg_hash", "commit", "dirty",
                    "config_sha256", "started_at", "question_ids")
USAGE_FIELDS = ("input", "cached_input", "output", "reasoning", "cache_write")


def check_test_source(source: Dict[str, Any], prereg_hash: str, attempts: List[Dict[str, Any]],
                      published: Optional[Dict[str, Any]]) -> None:
    """The rule for resuming on test (the module's docstring), once the barrier has found this session
    clean and at the registered code. `attempts` are the configuration's attempts in the published
    registry (`registry.attempts_of`), `published` the source's manifest there."""
    from bench import barrier
    run_id = source["run_id"]

    def refuse(why: str) -> None:
        raise barrier.TestSplitLocked(f"refusing to resume {run_id} on the test split: {why}; run the configuration "
                                      f"again from its first question with --redo-reason instead")
    if published is None:
        refuse("the published registry holds no manifest of it")
    killed = source.get("status") == KILLED
    keys = STARTED_IDENTITY if killed else set(published) | set(source)
    differ = sorted(key for key in keys if published.get(key) != source.get(key))
    if differ:
        refuse(f"its manifest differs from the one the registry published ({', '.join(differ[:5])})")
    if source.get("prereg_hash") != prereg_hash:
        refuse(f"it ran under pre-registration {source.get('prereg_hash')}, not the one in force ({prereg_hash})")
    if source.get("dirty") is not False:
        refuse("it ran on a working tree that was not clean")
    registered = barrier.registered_commit()
    if not isinstance(source.get("commit"), str) or registered is None or \
            not barrier.same_code(source["commit"], registered):
        refuse(f"it ran at commit {source.get('commit')!r}, not at the registered code")
    if not attempts or attempts[-1]["run_id"] != run_id:
        refuse(f"it is not the latest attempt of its configuration ({attempts[-1]['run_id'] if attempts else 'none'} is)")
    if killed:  # no hash was recorded when it ended: they are recorded now, before anything runs
        return
    run_dir = paths.RUNS / run_id
    for key, name in (("predictions_sha256", "predictions.json"), ("calls_sha256", "calls.jsonl")):
        now = data.sha256_file(run_dir / name) if (run_dir / name).is_file() else None
        if key not in source or now != source[key]:
            refuse(f"its {name} does not hash to what its manifest recorded when it ended ({key})")


def load_source(source_run_id: str, split: str) -> Dict[str, Any]:
    """The source's manifest, after the checks that need nothing of the new run."""
    path = paths.RUNS / source_run_id / "manifest.json"
    if not path.is_file():
        raise data.DataError(f"no run {source_run_id}")
    source = json.loads(path.read_text())
    if source.get("type") != "agent":
        raise data.DataError(f"{source_run_id} is a {source.get('type')!r} run: only an agent run is resumed")
    check_resumable(source, "run")
    return source


def check_resumable(source: Dict[str, Any], what: str) -> None:
    """Failed, interrupted, or killed: still `running`, with its session's lock free (the process that
    held it is gone). A session that still holds it, or left none, is refused."""
    run_id, status = source["run_id"], source.get("status")
    if status == KILLED:
        from bench.agent.runner import session_alive
        alive = session_alive(paths.RUNS / run_id)
        if alive:
            raise data.DataError(f"{run_id} is still running (its session holds its lock): it is not resumed")
        if alive is None:
            raise data.DataError(f"{run_id} is 'running' and left no session lock, so nobody can tell its session "
                                 f"is gone: it is not resumed")
        return
    if status not in RESUMABLE:
        raise data.DataError(f"{run_id} is {status!r}: only a failed, interrupted or killed {what} is resumed")


def _whole_lines(path: Path) -> List[str]:
    """The lines of a file a killed session was appending to: a last line cut off mid-write (no newline,
    or not JSON) is dropped; any other line that is not JSON is refused."""
    text = path.read_text()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    whole = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except ValueError:
            if i == len(lines) - 1:
                break  # cut off by the kill
            raise data.DataError(f"{path.name} of a killed session has a broken line {i + 1}") from None
        whole.append(line)
    return whole


def _killed_files(run_dir: Path) -> List[Path]:
    """What a killed session wrote that a resume reads: its C1 lines (calls.jsonl and each worker's) and
    its answered questions (answers*.jsonl), and predictions.json if it had come that far."""
    def worker(path: Path) -> Tuple[int, str]:
        found = re.search(r"\.w(\d+)\.jsonl$", path.name)
        return (int(found.group(1)) + 1 if found else 0, path.name)
    names = [p for pattern in ("calls*.jsonl", "answers*.jsonl", "predictions.json") for p in run_dir.glob(pattern)]
    return sorted(names, key=lambda p: (p.name.split(".")[0], *worker(p)))


def unregistered_source_hashes(source: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """For a killed source, the sha256 of every output its session left, by file name, computed before
    the resume reads or copies anything: no manifest recorded them when it ended, so the new manifest (and
    on test the intent, published before the first question) records them, as hashes no earlier record
    vouches for. None for a source that ended."""
    if source.get("status") != KILLED:
        return None
    return {path.name: data.sha256_file(path) for path in _killed_files(paths.RUNS / source["run_id"])}


def _killed_session_end(source: Dict[str, Any]) -> Optional[str]:
    """When a killed session last wrote an output: the end of its working time, as well as anyone can know it."""
    from datetime import datetime, timezone
    times = [path.stat().st_mtime for path in _killed_files(paths.RUNS / source["run_id"])]
    return datetime.fromtimestamp(max(times), timezone.utc).isoformat() if times else source.get("started_at")


def check_same(source: Dict[str, Any], expected: Dict[str, Any]) -> None:
    """Every field of `expected` equals the source's: the new run is the same measurement."""
    for key, value in expected.items():
        if source.get(key) != value:
            raise data.DataError(f"{source['run_id']}: {key} is {source.get(key)!r}, this run's is {value!r}: "
                                 f"a resumed run must be the same run")
    from bench.contracts.config import config_sha256
    snapshot = json.loads((paths.RUNS / source["run_id"] / "config.json").read_text())
    if config_sha256(snapshot) != source["config_sha256"]:
        raise data.DataError(f"the configuration snapshot of {source['run_id']} does not match its manifest")


def source_answers(source: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The source's questions that ended normally, {question_id: {"prediction", "tool_errors"}}: for a
    source that ended, from predictions.json less the questions that failed; for a killed one, from the
    questions its session journaled as they ended (answers*.jsonl)."""
    run_dir = paths.RUNS / source["run_id"]
    if source.get("status") == KILLED:
        answers = {}
        for path in sorted(run_dir.glob("answers*.jsonl")):
            for line in _whole_lines(path):
                entry = json.loads(line)
                answers[str(entry["question_id"])] = {"prediction": entry["prediction"],
                                                      "tool_errors": entry.get("tool_errors") or None}
        return answers
    predictions = json.loads((run_dir / "predictions.json").read_text())
    failed = set(source.get("failures") or {}) | set(source.get("harness_errors") or {})
    tool_errors = source.get("tool_errors") or {}
    return {q: {"prediction": predictions[q], "tool_errors": tool_errors.get(q)}
            for q in source["question_ids"] if q in predictions and q not in failed}


def complete_questions(source: Dict[str, Any], answers: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    """The source's questions that ended normally, in the run's order."""
    answers = source_answers(source) if answers is None else answers
    return [q for q in source["question_ids"] if q in answers]


def _usage_totals(calls: List[Dict[str, Any]]) -> Dict[str, int]:
    totals = {name: 0 for name in USAGE_FIELDS}
    for call in calls:
        for name in USAGE_FIELDS:
            value = (call.get("usage") or {}).get(name)
            if isinstance(value, (int, float)):
                totals[name] += value
    return totals


def read_calls_of(source: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """The source's C1 lines, as written and parsed, refused unless valid."""
    from bench.contracts.calls import validate_calls
    run_dir = paths.RUNS / source["run_id"]
    if source.get("status") == KILLED:  # its lines, and its workers' that no merge reached
        lines = [line for path in _killed_files(run_dir) if path.name.startswith("calls")
                 for line in _whole_lines(path)]
    else:
        lines = [line for line in (run_dir / "calls.jsonl").read_text().splitlines() if line.strip()]
    try:
        calls = [json.loads(line) for line in lines]
    except ValueError as e:
        raise data.DataError(f"{source['run_id']}/calls.jsonl is unreadable: {e}") from e
    errors = validate_calls(calls)
    if errors:
        raise data.DataError(f"{source['run_id']}/calls.jsonl breaks C1: {errors[:3]}")
    return lines, calls


def copy_source(source: Dict[str, Any], source_calls: Tuple[List[str], List[Dict[str, Any]]], run_dir: Path,
                complete: List[str], outcome: Dict[str, Dict],
                answers: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Copy the complete questions' C1 lines, predictions, tool errors and CHESS outputs into the new
    run (before it answers anything: its own lines are appended after). Returns what the source spent
    on the questions that are not complete (`discarded`)."""
    source_dir = paths.RUNS / source["run_id"]
    lines, calls = source_calls
    keep = set(complete)
    copied = [line for line, call in zip(lines, calls) if call["question_id"] in keep]
    (run_dir / "calls.jsonl").write_text("".join(line + "\n" for line in copied))

    answers = source_answers(source) if answers is None else answers
    with open(run_dir / "answers.jsonl", "a") as journal:  # copied answers survive this session being killed too
        for q in complete:
            outcome["predictions"][q] = answers[q]["prediction"]
            if answers[q]["tool_errors"]:
                outcome["tool_errors"][q] = answers[q]["tool_errors"]
            journal.write(json.dumps({"question_id": q, **answers[q]}, ensure_ascii=False) + "\n")

    chess = source_dir / "chess"
    if chess.is_dir():
        target = run_dir / "chess" / "resumed" / source["run_id"]
        for path in sorted(chess.rglob("*.json")):
            if path.name.split("_", 1)[0] in keep and path.parent.name != "logs":
                target.mkdir(parents=True, exist_ok=True)
                (target / path.name).write_bytes(path.read_bytes())

    discarded = [call for call in calls if call["question_id"] not in keep]
    return {"n_calls": len(discarded), "usage": _usage_totals(discarded),
            "question_ids": sorted({c["question_id"] for c in discarded}, key=lambda q: int(q) if q.isdigit() else -1)}


def _resumed_from(source: Dict[str, Any], **what: Any) -> Dict[str, Any]:
    return {**{key: source.get(key) for key in ("run_id", "commit", "dirty", "config_sha256", "status",
                                               "started_at", "finished_at", "workers", "stopped_by", "prereg_hash",
                                               "end_reason", "predictions_sha256", "calls_sha256")}, **what}


def _sessions(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    previous = source.get("sessions") or [{key: source.get(key) for key in ("started_at", "finished_at", "commit", "workers")}]
    sessions = [dict(session) for session in previous]
    if source.get("status") == KILLED:  # its last session never closed: it ended with its last output
        sessions[-1].update({"finished_at": _killed_session_end(source), "killed": True})
    return sessions


def manifest_fields(source: Dict[str, Any], complete: List[str], discarded: Dict[str, Any]) -> Dict[str, Any]:
    """What the new run's manifest records of the resume. The last session (this one) is closed by
    `runner.finish`, with the run's `finished_at`."""
    return {"resumed_from": _resumed_from(source, discarded=discarded), "resumed_question_ids": complete,
            "sessions": _sessions(source)}


def open_session(manifest: Dict[str, Any], workers: int) -> None:
    """This run's own session, from its real start; `finished_at` is set when the run ends."""
    manifest["sessions"].append({"started_at": manifest["started_at"], "finished_at": None,
                                 "commit": manifest["commit"], "workers": workers})


def working_seconds(manifest: Dict[str, Any]) -> Optional[float]:
    """The time a run worked: the sum of its sessions when it has them, else finished_at - started_at."""
    from datetime import datetime
    sessions = manifest.get("sessions") or [manifest]
    try:
        return sum((datetime.fromisoformat(s["finished_at"]) - datetime.fromisoformat(s["started_at"])).total_seconds()
                   for s in sessions)
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------- `bench replay --resume-from <run_id>`
#
# A replay that stopped on an infrastructure failure (`failed` with `harness_errors`, or `interrupted`)
# is resumed as a new replay of the same source, engine or arm, configuration, few-shot prefix, call
# sites and questions. What is copied is the invocation (question_id, call_site, invocation_key), not
# the question: the invocations of one question run in parallel, and those that ended before another
# met the harness's failure are answers. An invocation is complete when its last attempt ended it as
# `hooks._invocation` ends one without the harness: an answer (parsed, or an agent call's text, which
# the agent reads itself), the parser's own failure (not retried), the parse retries exhausted, or a
# rejection that is the model's (`rejections.is_model_rejection`, read back from the recorded error).
# Every other last attempt (a transport failure, a harness error, a parse failure with retries left:
# the run stopped it) leaves the invocation incomplete, and it is sent again from its first attempt;
# so does an invocation whose rejection cannot be read back as the model's (the safe side: it costs
# one more call, never a wrong copy). A complete invocation's C1 lines are copied unchanged, every
# attempt, `run_id` included, as `bench run --resume-from` copies a question's; an incomplete one's
# are not copied and are counted under `resumed_from.discarded` (calls and usage), so its cost stays
# visible and no judgment reads a call twice. On test it follows `bench run --resume-from`'s rule there
# (`check_test_source`; the replay's split is its source run's).

REPLAY_IDENTITY = ("split", "source_run_id", "source_config_sha256", "source_calls_sha256", "engine", "arm",
                   "call_sites", "question_ids", "n_invocations", "facts", "few_shot", "few_shot_k", "few_shot_sha256")
_PARSE_FAILURE = re.compile(r"^(empty output$|OutputParserException:)")
_STATUS = re.compile(r"Error code: (\d{3})")

Identity = Tuple[str, str, str]


def load_replay_source(failed_run_id: str, split: str) -> Dict[str, Any]:
    """The failed replay's manifest, refused unless it stopped on an infrastructure failure (on test, the
    rule of `check_test_source` applies on top, in `runner.open_run`)."""
    path = paths.RUNS / failed_run_id / "manifest.json"
    if not path.is_file():
        raise data.DataError(f"no run {failed_run_id}")
    source = json.loads(path.read_text())
    if source.get("type") != "replay":
        raise data.DataError(f"{failed_run_id} is a {source.get('type')!r} run: `bench replay --resume-from` resumes a replay")
    check_resumable(source, "replay")
    if source["status"] == "failed" and not source.get("harness_errors"):
        raise data.DataError(f"{failed_run_id} failed without a harness error (c1_errors {source.get('c1_errors')}, "
                             f"problems {source.get('problems')}): only an infrastructure failure is resumed")
    return source


def _ended(attempts: List[Dict[str, Any]], parse_max_attempts: int) -> bool:
    """Whether an invocation's attempts end it without the harness (see above)."""
    from bench.agent.rejections import is_model_rejection
    from bench.contracts.calls import AGENT_CALL_SITES
    last = attempts[-1]
    if last["response_text"] is None:  # the call raised (or answered no text): only the model's rejection ends it
        found = _STATUS.search(last.get("error") or "")
        return bool(found) and is_model_rejection(int(found.group(1)), last["error"])
    if last["parsed_ok"] or last["call_site"] in AGENT_CALL_SITES:
        return True
    if _PARSE_FAILURE.match(last.get("error") or ""):
        return sum(a["response_text"] is not None and not a["parsed_ok"] for a in attempts) >= parse_max_attempts
    return True  # the parser raised something else: the model's failure, not retried


def _summary(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"n_invocations": len({(c["question_id"], c["call_site"], c["invocation_key"]) for c in calls}),
            "n_calls": len(calls), "usage": _usage_totals(calls)}


def plan_replay(source: Dict[str, Any], source_calls: Tuple[List[str], List[Dict[str, Any]]],
                expected: List[Identity], parse_max_attempts: int) -> Dict[str, Any]:
    """What a resumed replay keeps of the failed one, before the new run exists (a refusal leaves
    none). `expected` is every invocation the new replay sends. Returns `copied` (the complete
    invocations), `lines` (their C1 lines, as written), `model_failures` (the model's failures among
    them per call site, as `replay._attempt` counts them) and `fields` (the manifest's record of it)."""
    from bench.contracts.calls import AGENT_CALL_SITES
    lines, calls = source_calls
    attempts: Dict[Identity, List[Dict[str, Any]]] = {}
    for call in calls:
        attempts.setdefault((call["question_id"], call["call_site"], call["invocation_key"]), []).append(call)
    outside = sorted(set(attempts) - set(expected))
    if outside:
        raise data.DataError(f"{source['run_id']} has invocations this replay does not send, e.g. {outside[:2]}: "
                             f"a resumed replay must be the same replay")
    for chain in attempts.values():
        chain.sort(key=lambda c: c["attempt"])
    complete = {identity for identity, chain in attempts.items() if _ended(chain, parse_max_attempts)}
    failures: Dict[str, int] = {}
    for identity in sorted(complete):
        last = attempts[identity][-1]
        if not last["parsed_ok"] and not (last["call_site"] in AGENT_CALL_SITES and last["response_text"] is not None):
            failures[identity[1]] = failures.get(identity[1], 0) + 1  # the call raised the model's failure
    keep = [(line, call) for line, call in zip(lines, calls)
            if (call["question_id"], call["call_site"], call["invocation_key"]) in complete]
    discarded = [call for call in calls if (call["question_id"], call["call_site"], call["invocation_key"]) not in complete]
    return {"copied": complete, "lines": [line for line, _ in keep], "model_failures": failures, "fields": {
        "resumed_from": _resumed_from(source, copied=_summary([call for _, call in keep]), discarded={
            **_summary(discarded), "question_ids": sorted({c["question_id"] for c in discarded},
                                               key=lambda q: int(q) if q.isdigit() else -1)}),
        "sessions": _sessions(source)}}


def copy_replay(plan: Dict[str, Any], run_dir: Path) -> None:
    """The complete invocations' C1 lines, first in the new replay's calls.jsonl (its own are appended)."""
    (run_dir / "calls.jsonl").write_text("".join(line + "\n" for line in plan["lines"]))
