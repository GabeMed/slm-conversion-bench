"""B1's "same few-shot and cache" (REQ-002; design §5.1): the prefix `cheap_alt` receives.

For each call site, `arms.B1.few_shot.k` examples are sampled with `seeds.few_shot` from the
teacher's successful invocations (`parsed_ok`) in a `done` B0 run on the `train` split
(`arms.B1.few_shot.source_run`), the same data the SLM learns from. Each example is the prompt as
the teacher received it and its answer, a user/assistant pair; the pairs of a call site are one
fixed prefix, which a provider can cache. `hooks` prepends it whenever the engine is `cheap_alt`
(B1, the clusters B5 allocates to it, a replay on it); the router and `assign` never see it.

The prefix is built once per execution and recorded in its manifest: `few_shot_k`, the sha256 of
each call site's prefix (`few_shot_sha256`), and under `few_shot` the source run, the sha256 of its
calls.jsonl and of the splits it ran on (which must be today's), and the examples chosen, every one
a question of the current train split.
"""
import hashlib
import json
import random
from typing import Any, Dict, List, Tuple

from bench import data, paths
from bench.agent.hooks import HarnessError
from bench.contracts.calls import CALL_SITES, read_calls, validate_calls
from bench.contracts.facts import canonical


def digests(prefix: Dict[str, List[Dict[str, str]]]) -> Dict[str, str]:
    """The sha256 of each call site's prefix (`few_shot_sha256` in the manifest of every execution that uses it)."""
    return {site: hashlib.sha256(canonical(messages)).hexdigest() for site, messages in prefix.items()}


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
        chosen = random.Random(f"{seed}:{site}").sample(candidates, k)
        outside = sorted({c["question_id"] for c in chosen} - train, key=int)
        if outside:
            raise HarnessError(f"few-shot source run {source}: {site} examples of questions not in the current train "
                               f"split: {outside}")
        messages = []
        for example in chosen:
            messages += example["prompt_messages"] + [{"role": "assistant", "content": example["response_text"]}]
        prefix[site] = messages
        examples[site] = [[c["question_id"], c["invocation_key"]] for c in chosen]
    provenance = {"k": k, "seed": seed, "source_run": source, "source_calls_sha256": hashlib.sha256(raw).hexdigest(),
                  "source_splits_sha256": splits_sha256, "examples": examples}
    return prefix, provenance
