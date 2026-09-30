"""B1's "same few-shot and cache" (REQ-002; design §5.1): the prefix `cheap_alt` receives.

For each call site, `arms.B1.few_shot.k` examples are sampled with `seeds.few_shot` from the
teacher's successful invocations (`parsed_ok`) in a `done` B0 run on the `train` split
(`arms.B1.few_shot.source_run`), the same data the SLM learns from. Each example is the prompt as
the teacher received it and its answer, a user/assistant pair; the pairs of a call site are one
fixed prefix, which a provider can cache. `hooks` prepends it whenever the engine is `cheap_alt`
(B1, the clusters B5 allocates to it, a replay on it); the router and `assign` never see it.

The prefix is built once per execution and recorded in its manifest: the source run, the sha256
of its calls.jsonl, the examples chosen and the sha256 of each call site's prefix.
"""
import hashlib
import json
import random
from typing import Any, Dict, List, Tuple

from bench import paths
from bench.agent.hooks import HarnessError
from bench.contracts.calls import CALL_SITES, read_calls, validate_calls
from bench.contracts.facts import canonical


def build(config: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Any]]:
    """({call site: messages}, provenance)."""
    settings = config["arms"]["B1"]["few_shot"]
    k, seed, source = settings["k"], config["seeds"]["few_shot"], settings["source_run"]
    if k == 0:
        return {site: [] for site in CALL_SITES}, {"k": 0, "source_run": None}
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
    raw = (run_dir / "calls.jsonl").read_bytes()
    calls = read_calls(run_dir / "calls.jsonl")
    errors = validate_calls(calls)
    if errors:
        raise HarnessError(f"few-shot source run {source}: calls.jsonl is not valid C1: {errors[:3]}")

    prefix, examples, digests = {}, {}, {}
    for site in CALL_SITES:
        candidates = sorted((c for c in calls if c["call_site"] == site and c["parsed_ok"]),
                            key=lambda c: (int(c["question_id"]), c["invocation_key"]))
        if len(candidates) < k:
            raise HarnessError(f"few-shot source run {source} has {len(candidates)} successful {site} "
                               f"invocations, fewer than k = {k}")
        chosen = random.Random(f"{seed}:{site}").sample(candidates, k)
        messages = []
        for example in chosen:
            messages += example["prompt_messages"] + [{"role": "assistant", "content": example["response_text"]}]
        prefix[site] = messages
        examples[site] = [[c["question_id"], c["invocation_key"]] for c in chosen]
        digests[site] = hashlib.sha256(canonical(messages)).hexdigest()
    provenance = {"k": k, "seed": seed, "source_run": source, "source_calls_sha256": hashlib.sha256(raw).hexdigest(),
                  "examples": examples, "prefix_sha256": digests}
    return prefix, provenance
