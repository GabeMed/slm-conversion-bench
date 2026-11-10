"""B1's "same few-shot and cache" (REQ-002; design §5.1): the prefix `cheap_alt` receives.

For each call site, `arms.B1.few_shot.k` examples are sampled with `seeds.few_shot` from the
teacher's successful invocations (`parsed_ok`) in a `done` B0 run on the `train` split
(`arms.B1.few_shot.source_run`), the same data the SLM learns from. Each example is the prompt as
the teacher received it and its answer, a user/assistant pair; the pairs are one fixed prefix,
which a provider can cache. `hooks` prepends it whenever the engine is `cheap_alt` (B1, the
clusters B5 allocates to it, a replay on it); the router and `assign` never see it.

The prompts of the agent's call sites (`agent_ir`, `agent_ss`, `agent_cg`) carry no state: the step
is told only by the invocation key (`ir:0`, `ir:1`, ...). Examples drawn from any step taught the
student the teacher's later steps at the first one, so an agent call site has one prefix per step,
its examples drawn from the teacher's invocations of that same step (the invocation key, its `@n`
repeat suffix dropped). When a step has fewer than k, the shortfall comes from the nearest steps
(by index distance; the steps at one distance sampled together); a step the teacher never reached
takes the prefix of the nearest step it did reach (a tie: the lower one). A tool call site's prompt
carries its own state: it keeps one prefix for all its invocations (step `*`), as before.

The prefix is built once per execution and recorded in its manifest: `few_shot_k`, the sha256 of
each call site's prefixes (`few_shot_sha256`), and under `few_shot` the source run, the sha256 of its
calls.jsonl and of the splits it ran on (which must be today's), the sampling rule, and the examples
chosen per call site and step, every one a question of the current train split.
"""
import hashlib
import json
import random
import re
from typing import Any, Dict, List, Tuple

from bench import data, paths
from bench.agent.hooks import HarnessError
from bench.contracts.calls import AGENT_CALL_SITES, CALL_SITES, read_calls, validate_calls
from bench.contracts.facts import canonical


ANY_STEP = "*"  # the one prefix of a tool call site
SAMPLING = "agent call sites: per step (invocation key), nearest steps fill a shortfall; tool call sites: per call site"
_STEP = re.compile(r"^(?P<name>[a-z]+):(?P<index>\d+)$")

Prefix = Dict[str, Dict[str, List[Dict[str, str]]]]  # {call site: {step: messages}}


def step_of(call_site: str, invocation_key: str) -> str:
    """The step whose examples an invocation gets: its invocation key on an agent call site, without
    the `@n` of a repeat; `*` on a tool call site."""
    if call_site not in AGENT_CALL_SITES:
        return ANY_STEP
    step = invocation_key.split("@", 1)[0]
    if not _STEP.match(step):
        raise HarnessError(f"{call_site} invocation key {invocation_key!r} is not <step name>:<index>")
    return step


def _index(step: str) -> int:
    return int(_STEP.match(step)["index"])


def resolve(steps: Dict[str, Any], call_site: str, invocation_key: str) -> Any:
    """The prefix of one invocation among its call site's `steps`: the call site's one prefix if it
    has one (a tool call site; k = 0), else its own step's, else (a step the teacher never reached)
    the nearest one's, the lower on a tie."""
    if ANY_STEP in steps:
        return steps[ANY_STEP]
    if not steps:  # no prefix at all (no few-shot): the key is not read, whatever its shape
        return []
    step = step_of(call_site, invocation_key)
    if step in steps:
        return steps[step]
    target = _index(step)
    return steps[min(steps, key=lambda s: (abs(_index(s) - target), _index(s)))]


def _sample(rng: random.Random, candidates: List[dict], step: str, k: int) -> List[dict]:
    """k of `candidates`: those of `step` first; a shortfall from the nearest steps, a whole distance
    at a time, sampled within the last one it needs."""
    if step == ANY_STEP:
        return rng.sample(candidates, k)
    target, tiers = _index(step), {}
    for c in candidates:
        tiers.setdefault(abs(_index(step_of(c["call_site"], c["invocation_key"])) - target), []).append(c)
    chosen: List[dict] = []
    for distance in sorted(tiers):
        need = k - len(chosen)
        tier = tiers[distance]
        chosen += tier if len(tier) <= need else rng.sample(tier, need)
        if len(chosen) == k:
            break
    return chosen


def digests(prefix: Prefix) -> Dict[str, str]:
    """The sha256 of each call site's prefixes (`few_shot_sha256` in the manifest of every execution
    that uses it): of its messages when it has one prefix for all invocations (a tool call site, or
    k = 0: the digest it had before steps), else of {step: messages}, every step's prefix at once."""
    return {site: hashlib.sha256(canonical(steps[ANY_STEP] if set(steps) == {ANY_STEP} else steps)).hexdigest()
            for site, steps in prefix.items()}


def build(config: Dict[str, Any]) -> Tuple[Prefix, Dict[str, Any]]:
    """({call site: {step: messages}}, provenance)."""
    settings = config["arms"]["B1"]["few_shot"]
    k, seed, source = settings["k"], config["seeds"]["few_shot"], settings["source_run"]
    if k == 0:
        return {site: {ANY_STEP: []} for site in CALL_SITES}, {"k": 0, "source_run": None}
    if not source:
        raise HarnessError("arms.B1.few_shot.source_run is not set: cheap_alt needs its few-shot examples "
                           "(a done B0 run on train), or k: 0")
    run_dir = paths.RUNS / source
    if not (run_dir / "manifest.json").exists():
        raise HarnessError(f"few-shot source run {source} does not exist")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    shape = (manifest.get("type"), manifest.get("arm"), manifest.get("split"), manifest.get("status"))
    if shape != ("agent", "B0", "train", "done"):
        raise HarnessError(f"few-shot source run {source} is (type, arm, split, status) = {shape}, "
                           f"not a done B0 agent run on train")
    splits_sha256 = data.sha256_file(paths.SPLITS)
    if manifest.get("splits_sha256") != splits_sha256:
        raise HarnessError(f"few-shot source run {source} ran on other splits (data/splits.json has changed since): "
                           f"its train answers may no longer be train")
    train = set(json.loads(paths.SPLITS.read_text())["train"])
    raw = (run_dir / "calls.jsonl").read_bytes()
    calls = read_calls(run_dir / "calls.jsonl")
    errors = validate_calls(calls)
    if errors:
        raise HarnessError(f"few-shot source run {source}: calls.jsonl is not valid C1: {errors[:3]}")

    prefix, examples = {}, {}
    for site in CALL_SITES:
        candidates = sorted((c for c in calls if c["call_site"] == site and c["parsed_ok"]),
                            key=lambda c: (int(c["question_id"]), c["invocation_key"]))
        if len(candidates) < k:
            raise HarnessError(f"few-shot source run {source} has {len(candidates)} successful {site} "
                               f"invocations, fewer than k = {k}")
        prefix[site], examples[site] = {}, {}
        for step in sorted({step_of(site, c["invocation_key"]) for c in candidates}):
            seeded = f"{seed}:{site}" if step == ANY_STEP else f"{seed}:{site}:{step}"
            chosen = _sample(random.Random(seeded), candidates, step, k)
            outside = sorted({c["question_id"] for c in chosen} - train, key=int)
            if outside:
                raise HarnessError(f"few-shot source run {source}: {site} examples of questions not in the current "
                                   f"train split: {outside}")
            messages = []
            for example in chosen:
                messages += example["prompt_messages"] + [{"role": "assistant", "content": example["response_text"]}]
            prefix[site][step] = messages
            examples[site][step] = [[c["question_id"], c["invocation_key"]] for c in chosen]
    provenance = {"k": k, "seed": seed, "source_run": source, "source_calls_sha256": hashlib.sha256(raw).hexdigest(),
                  "source_splits_sha256": splits_sha256, "sampling": SAMPLING, "examples": examples}
    return prefix, provenance
