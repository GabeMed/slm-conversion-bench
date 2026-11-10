"""`bench preflight --base-check <cluster> --on runpod|local`: is the base we serve the base Tinker trained on?

Tinker trains on its own copy of the base, which it does not pin to a revision (a Tinker manifest's
stats.run.differences["base.revision"]); the adapter is served on the pinned revision. This check scores, by
teacher forcing, the training rows of the last optimizer steps with HF transformers + PEFT on the pinned
revision, (1) with the adapter and (2) with it disabled, and compares (1) with the losses Tinker logged for
those steps (`--tinker-loss STEP=LOSS`, from `bench train --on tinker`'s log).

    rows      the rows of the last steps of training, back from the last until they hold at least `min_rows`:
              `step_rows` of bench/train_tinker.py with the run's schedule and train.seed, so the exact rows (and
              order) Tinker trained on; tokenized and masked by `encode_rows` (the Tinker path: `check_rows`, the
              candidate's chat_template_kwargs, the loss on exactly the completion tokens)
    loss      per row, the summed cross-entropy of its completion tokens; per step, the token mean over the
              step's rows (what Tinker's `loss:sum` is, its weights 1 / completion tokens of the step)
    compared  Tinker logs step s's loss from the weights before update s; the final adapter differs from them
              by the updates s..last, which at the end of the cosine schedule have a learning rate near zero.
              So the final adapter on the pinned base must reproduce the late steps' logged losses up to
              numerics; a served base that is not Tinker's would not (the adapter's deltas land on other weights)

Verdict: pass when the pooled ratio sum(n_s * ours_s) / sum(n_s * tinker_s) over the logged steps given is within
`RATIO_TOLERANCE` of 1 and the base alone is at least `BASE_OVER_ADAPTER` times the adapter's loss on the same
rows (the adapter was loaded and does something); fail otherwise; undecided without a logged loss of a selected
step. Diagnostic only: never part of the registered analysis.
"""
import json
import math
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from bench import paths
from bench.preflight import FAIL, PASS, UNDECIDED, PreflightError

MIN_ROWS = 100
RATIO_TOLERANCE = 0.10    # bf16 HF vs Tinker's numerics: a few %; another base under the adapter: far more
BASE_OVER_ADAPTER = 1.5   # the base alone, against the adapter, on the rows the adapter was trained on


def last_steps(manifest: Dict[str, Any], n_rows: int, min_rows: int = MIN_ROWS) -> List[Dict[str, int]]:
    """[{step (1-based), index (row of the dataset)}] for the last steps of training holding >= min_rows rows."""
    from bench.train_tinker import schedule, step_rows

    sched = schedule(n_rows, manifest["hyperparameters"]["sft"])
    recorded = ((manifest.get("stats") or {}).get("run") or {}).get("schedule")
    if recorded and {k: recorded.get(k) for k in sched} != sched:
        raise PreflightError(f"the manifest's schedule {recorded} is not the one recomputed from the dataset {sched}")
    steps = step_rows(n_rows, sched, manifest["hyperparameters"]["seed"])
    chosen: List[Dict[str, int]] = []
    for number in range(len(steps), 0, -1):
        chosen = [{"step": number, "index": i} for i in steps[number - 1]] + chosen
        if len(chosen) >= min_rows:
            break
    return chosen


def teacher_forced(base: Dict[str, str], adapter_dir: str, rows: List[dict], template_kwargs: dict,
                   max_length: int, max_context: int, device: str) -> Dict[str, Any]:
    """Per row, the completion tokens and their summed cross-entropy with the adapter and with it disabled."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bench.train_tinker import encode_rows

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(base["repo"], revision=base["revision"])
    token_stats, encoded = encode_rows(tokenizer, rows, template_kwargs, max_length, max_context)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base["repo"], revision=base["revision"], dtype=dtype).to(device)
    model = PeftModel.from_pretrained(model, adapter_dir).eval()
    loaded = time.monotonic()
    out: Dict[str, Any] = {"tokens": token_stats["tokens"], "completion_tokens": [sum(e["mask"]) for e in encoded]}
    for which, context in (("adapter", nullcontext()), ("base", model.disable_adapter())):
        with context, torch.no_grad():
            out[which] = [row_nll(model, e, device) for e in encoded]
    out.update(load_seconds=round(loaded - started, 3), score_seconds=round(time.monotonic() - loaded, 3))
    return out


def row_nll(model: Any, encoded: Dict[str, List], device: str) -> float:
    """The summed cross-entropy (float32) of a row's masked targets, `inputs` fed whole."""
    import torch

    inputs = torch.tensor([encoded["inputs"]], device=device)
    logits = model(input_ids=inputs).logits[0].float()
    targets = torch.tensor(encoded["targets"], device=device)
    mask = torch.tensor(encoded["mask"], device=device, dtype=torch.bool)
    nll = torch.nn.functional.cross_entropy(logits[mask], targets[mask], reduction="sum")
    return float(nll)


def summarise(selected: List[Dict[str, int]], scored: Dict[str, Any],
              tinker: Dict[int, float]) -> Dict[str, Any]:
    """Per step and pooled token means, and the verdict against Tinker's logged losses."""
    n = scored["completion_tokens"]
    by_step: Dict[int, Dict[str, float]] = {}
    for k, row in enumerate(selected):
        s = by_step.setdefault(row["step"], {"rows": 0, "completion_tokens": 0, "adapter_nll": 0.0, "base_nll": 0.0})
        s["rows"] += 1
        s["completion_tokens"] += n[k]
        s["adapter_nll"] += scored["adapter"][k]
        s["base_nll"] += scored["base"][k]
    steps = []
    for number in sorted(by_step):
        s = by_step[number]
        steps.append({"step": number, "rows": s["rows"], "completion_tokens": s["completion_tokens"],
                      "adapter_loss": s["adapter_nll"] / s["completion_tokens"],
                      "base_loss": s["base_nll"] / s["completion_tokens"], "tinker_loss": tinker.get(number)})
    total = sum(n)
    adapter_loss, base_loss = sum(scored["adapter"]) / total, sum(scored["base"]) / total
    logged = [s for s in steps if s["tinker_loss"] is not None]
    unknown = sorted(set(tinker) - set(by_step))
    result: Dict[str, Any] = {"n_rows": len(selected), "completion_tokens": total, "adapter_loss": adapter_loss,
                              "base_loss": base_loss, "base_over_adapter": base_loss / adapter_loss if adapter_loss else math.inf,
                              "steps": steps, "thresholds": {"ratio_tolerance": RATIO_TOLERANCE,
                                                             "base_over_adapter": BASE_OVER_ADAPTER}}
    if unknown:
        result["ignored_tinker_steps"] = unknown  # logged steps outside the rows scored
    if not logged:
        return {**result, "status": UNDECIDED,
                "action": "give Tinker's logged loss of a selected step: --tinker-loss STEP=LOSS"}
    ratio = (sum(s["completion_tokens"] * s["adapter_loss"] for s in logged)
             / sum(s["completion_tokens"] * s["tinker_loss"] for s in logged))
    same = abs(ratio - 1) <= RATIO_TOLERANCE
    adapted = result["base_over_adapter"] >= BASE_OVER_ADAPTER
    result.update(ratio_to_tinker=ratio, status=PASS if same and adapted else FAIL)
    result["action"] = ("" if same and adapted else
                        "the adapter does not change the pinned base as it changed Tinker's: check the adapter loaded"
                        if not adapted else
                        "the pinned base under the adapter does not reproduce Tinker's losses: the served base may not "
                        "be the one Tinker trained on (record it; see prereg/DEVIATIONS.md)")
    return result


def parse_tinker_losses(values: Optional[List[str]]) -> Dict[int, float]:
    losses = {}
    for value in values or []:
        step, sep, loss = value.partition("=")
        try:
            losses[int(step)] = float(loss)
        except ValueError:
            raise PreflightError(f"--tinker-loss takes STEP=LOSS (e.g. 100=0.1453), not {value!r}") from None
        if not sep:
            raise PreflightError(f"--tinker-loss takes STEP=LOSS (e.g. 100=0.1453), not {value!r}")
    return losses


def base_check(config: Dict[str, Any], cluster: str, on: str, tinker: Dict[int, float], restart: bool = True,
               min_rows: int = MIN_ROWS, reference: Optional[Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    from bench.train import adapters_dir, datasets_dir, parse_dataset, sha256_bytes

    manifest_path = adapters_dir() / cluster / "manifest.json"
    if not manifest_path.is_file():
        raise PreflightError(f"no adapter for cluster {cluster} (bench train --cluster {cluster})")
    manifest = json.loads(manifest_path.read_text())
    raw = (datasets_dir() / f"{cluster}.jsonl").read_bytes()
    if sha256_bytes(raw) != manifest["dataset"]["sha256"]:
        raise PreflightError(f"train/datasets/{cluster}.jsonl is not the dataset the adapter was trained on")
    rows = parse_dataset(raw, f"{cluster}.jsonl")
    selected = last_steps(manifest, len(rows), min_rows)
    sft = manifest["hyperparameters"]["sft"]
    run = (manifest.get("stats") or {}).get("run") or {}
    args = (manifest["base"], [rows[r["index"]] for r in selected], manifest["chat_template_kwargs"],
            sft["max_length"], run.get("max_context") or sft["max_length"])
    if reference is not None:
        scored = reference(*args)
    elif on == "local":
        scored = teacher_forced(args[0], str(manifest_path.parent / "adapter"), *args[1:], device="cpu")
    elif on == "runpod":
        sys.path.insert(0, str(paths.ROOT))
        from runpod_apps import parity

        scored = parity.base_identity(config, cluster, manifest["adapter_sha256"], *args, restart=restart)
    else:
        raise PreflightError(f"--base-check runs on local or runpod, not {on!r}")
    result = summarise(selected, scored, tinker)
    result.update({"cluster": cluster, "slm": manifest["slm"], "base": manifest["base"],
                   "adapter_sha256": manifest["adapter_sha256"], "where_trained": manifest.get("where"),
                   "tinker_train_loss": (manifest.get("stats") or {}).get("train_loss"), "on": on,
                   "rows": [{**r, "completion_tokens": scored["completion_tokens"][k],
                             "adapter_nll": scored["adapter"][k], "base_nll": scored["base"][k]}
                            for k, r in enumerate(selected)],
                   **{k: scored[k] for k in ("tokens", "function_seconds", "load_seconds", "score_seconds",
                                             "observed_gpus", "versions", "pod_id", "usd_per_hour", "restart_error",
                                             "server_left_stopped")
                      if k in scored}})
    if scored.get("pod_code"):  # RunPod: the code the job ran, this checkout's by sha256 (runpod_apps/parity.py)
        code = scored["pod_code"]
        result.update({"pod_code_sha256": code.get("sha256"), "pod_code_commit": code.get("commit"),
                       "pod_code_dirty": code.get("dirty"), "pod_code_uploaded": code.get("uploaded")})
    return result


def cli(args) -> int:
    from bench.contracts.config import load_config
    from bench.preflight import _unreachable
    from bench.provenance import git_state

    try:
        config = load_config(args.config)
        result = base_check(config, args.base_check, args.on, parse_tinker_losses(args.tinker_loss),
                            restart=not args.no_restart)
    except _unreachable(args.on) as e:  # a missing input, the pod, or its job failing: said, never a verdict
        print(f"bench preflight --base-check: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc)
    run_dir = paths.RUNS / f"basecheck-{args.base_check}-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(json.dumps({"type": "base_check", **git_state(), **result}, indent=2) + "\n")
    for s in result["steps"]:
        tinker = f"{s['tinker_loss']:.4f}" if s["tinker_loss"] is not None else "-"
        print(f"step {s['step']:4}  rows {s['rows']:3}  tokens {s['completion_tokens']:5}  "
              f"adapter {s['adapter_loss']:.4f}  tinker {tinker}  base {s['base_loss']:.4f}")
    print(f"pooled: adapter {result['adapter_loss']:.4f}, base {result['base_loss']:.4f} "
          f"(x{result['base_over_adapter']:.2f}), ratio to Tinker {result.get('ratio_to_tinker', float('nan')):.3f}")
    print(f"{result['status']}  {result['action']}")
    print(run_dir)
    return 0 if result["status"] == PASS else 1
