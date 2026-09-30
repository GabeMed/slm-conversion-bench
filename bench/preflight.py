"""`bench preflight`: the preconditions of SPEC 7.1 on the morning of Day 1, each a check with the action
the SPEC gives when it fails, plus P-4, the LoRA parity gate (design §2, P-4).

A check is `pass`, `fail`, or `pending` (this command cannot decide it yet: the evidence is missing, and
the check says which). The command writes `runs/preflight-<timestamp>/report.json` and exits 0 only when
every check passes.

**P-4 (`--parity <cluster>`).** The server must first list the adapter at the path of the adapter trained
(`/adapters/<sha256>`, vLLM's `root` in /v1/models) and the base at its pinned repository. Then, with
greedy decoding on the first N prompts of the cluster's training set, the adapter served by vLLM must
(a) change the base's output wherever HF-PEFT's change is decisive (not a near-tie), and on at least one
prompt, and (b) match HF-PEFT's output on every prompt. "Match" is the rule vLLM's own tests use for greedy parity
(`check_logprobs_close`): the token ids agree one by one, and at the first disagreement each side's token
is among the other side's top-k (a numerical near-tie, after which the sequences are compared no further).
The served base is compared with HF too, so a failure says whether the adapter or the serving itself
(template, revision, dtype) disagrees, which changes the action. The reference runs in the Modal GPU
function `modal_apps/train.py:peft_reference` on Day 1, or locally for a small model.
"""
import json
import re
import urllib.error
import urllib.request
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from bench import paths

TERMS = ("weights_license", "provider_terms", "checked_on")
PASS, FAIL, PENDING = "pass", "fail", "pending"


class PreflightError(RuntimeError):
    pass


def teacher_terms_missing(config: Dict[str, Any]) -> List[str]:
    """The fields of `roles.production_llm.terms` still unset (D6: both must allow training on outputs)."""
    terms = config["roles"]["production_llm"].get("terms") or {}
    return [k for k in TERMS if not terms.get(k)]


def _check(check_id: str, precondition: str, status: str, evidence: Any, action: str) -> Dict[str, Any]:
    return {"id": check_id, "precondition": precondition, "status": status, "evidence": evidence, "action": action}


# ---------------------------------------------------------------- SPEC 7.1, one check per row

def _agent_manifests() -> List[Dict[str, Any]]:
    manifests = []
    for path in sorted(paths.RUNS.glob("agent-*/manifest.json")) if paths.RUNS.is_dir() else []:
        try:
            manifests.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return manifests


def check_agent_runs() -> Dict[str, Any]:
    done = [m["run_id"] for m in _agent_manifests() if m.get("status") == "done" and m.get("split") in ("train", "calib")]
    return _check(
        "agent_end_to_end", "the agent answers one question end to end within 3 h (SPEC 7.1)",
        PASS if done else FAIL, {"done_runs": done[-3:], "note": "the 3 h bound is the operator's clock"},
        "use an equivalent controller (same call sites, prompts, order, data flow and repair loop), declared in "
        "the report; if not even that runs by the end of the morning, the fallback of SPEC 7.4")


def check_call_sites() -> Dict[str, Any]:
    seen: Dict[str, set] = {"train": set(), "calib": set()}
    runs: Dict[str, List[str]] = {"train": [], "calib": []}
    for m in _agent_manifests():
        if m.get("status") == "done" and m.get("split") in seen:
            seen[m["split"]] |= set(m.get("call_sites_seen") or [])
            runs[m["split"]].append(m["run_id"])
    missing = [split for split in seen if not runs[split]]
    return _check(
        "call_sites_registered",
        "the set of call sites seen in the training logs and the calibration rounds is registered, with the "
        "id written on every call (SPEC 7.1, REQ-001)",
        PENDING if missing else PASS,
        {"call_sites": {k: sorted(v) for k, v in seen.items()}, "runs": {k: len(v) for k, v in runs.items()},
         "without_done_runs": missing},
        "fix the harness before going on")


def check_data(config: Dict[str, Any]) -> Dict[str, Any]:
    from bench import data
    from bench.barrier import TestSplitLocked
    from bench.data import DataError

    evidence: Dict[str, Any] = {}
    try:
        splits = data.load_splits()
        for split in ("train", "calib"):
            questions = data.questions_for(config, split)
            without_gold = sorted(set(splits[split]) - {q for q, item in questions.items() if item.get("SQL")}, key=int)
            evidence[split] = {"questions": len(splits[split]), "without_gold": without_gold[:20]}
        test, excluded = set(splits["test"]), set(splits["excluded"])
        overlap = sorted((set(splits["train"]) | set(splits["calib"])) & (test | excluded))
        evidence["test"] = {"questions": len(test), "overlap_with_train_or_calib": overlap,
                            "excluded_in_test": sorted(test & excluded),
                            "note": "the test gold is read only under the pre-registration (bench/barrier.py); "
                                    "bench data pinned its sha256"}
        ok = (all(not evidence[s]["without_gold"] for s in ("train", "calib")) and test and not overlap
              and not (test & excluded))
        status = PASS if ok else FAIL
    except (DataError, TestSplitLocked, OSError, ValueError, KeyError) as e:  # the data layer's errors are the evidence
        status, evidence["error"] = FAIL, f"{type(e).__name__}: {e}"
    return _check("data_ids_and_gold", "the Mini-Dev ids match the dev set and the corrected gold is reachable "
                  "(SPEC 7.1)", status, evidence, "fix the mapping; without gold there is no test")


def check_teacher_terms(config: Dict[str, Any]) -> Dict[str, Any]:
    missing = teacher_terms_missing(config)
    terms = config["roles"]["production_llm"].get("terms") or {}
    return _check("teacher_terms", "the licence and terms of the teacher and its provider allow training on its "
                  "outputs, with link and date (SPEC 7.1, D6)", FAIL if missing else PASS,
                  {"terms": terms, "missing": missing},
                  "switch to the operational reserve (roles.production_llm)")


def check_pilot_spend(config: Dict[str, Any]) -> Dict[str, Any]:
    return _check("pilot_spend", "the pilot's spend, projected to the core, fits the configured ceiling (SPEC 7.1)",
                  PENDING, {"needs": "J3 over the pilot run and a spend ceiling in config.yaml (neither is F3's)"},
                  "cut extensions and shrink the pilot before touching the core")


def _train_manifests() -> List[Dict[str, Any]]:
    root = paths.ROOT / "train" / "adapters"
    return [json.loads(p.read_text()) for p in sorted(root.glob("*/manifest.json"))] if root.is_dir() else []


def _loadtest_manifests() -> List[Dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(paths.RUNS.glob("loadtest-*/manifest.json"))] if paths.RUNS.is_dir() else []


def check_training_time(config: Dict[str, Any]) -> Dict[str, Any]:
    """GPU seconds per example seen, measured on Modal, times the examples every cluster's dataset will
    show (rows x epochs), against the overnight budget."""
    budget_h = config["preflight"]["schedule"]["train_hours_max"]
    precondition = "the measured training time confirms the schedule: every adapter trains overnight (SPEC 7.1, 7.3)"
    gpu_runs = [m for m in _train_manifests() if m.get("where") == "modal" and m["stats"].get("examples_seen")]
    evidence: Dict[str, Any] = {"train_hours_max": budget_h}
    if not gpu_runs:
        evidence["needs"] = "one adapter trained on Modal (bench train --on modal) to measure seconds per example"
        return _check("training_time", precondition, PENDING, evidence, "apply the cuts of SPEC 7.3")
    seconds_per_example = (sum(m["stats"]["train_seconds"] for m in gpu_runs)
                           / sum(m["stats"]["examples_seen"] for m in gpu_runs))
    epochs = config["train"]["sft"]["num_train_epochs"]
    datasets = sorted((paths.ROOT / "train" / "datasets").glob("*.jsonl"))
    rows = sum(sum(1 for line in p.read_text().splitlines() if line.strip()) for p in datasets)
    projected_h = seconds_per_example * rows * epochs / 3600
    evidence.update({"seconds_per_example": round(seconds_per_example, 3), "dataset_rows": rows, "epochs": epochs,
                     "projected_train_hours": round(projected_h, 2)})
    return _check("training_time", precondition, FAIL if projected_h > budget_h else PASS, evidence,
                  "apply the cuts of SPEC 7.3")


def check_throughput() -> Dict[str, Any]:
    """The load runs that finished are the evidence; whether their throughput within the p95 confirms the
    schedule is J8's judgment, which this command does not make."""
    done = [{"run_id": m["run_id"], "engine": m.get("engine"), "concurrency": m["concurrency"]}
            for m in _loadtest_manifests() if m.get("status") == "done"]
    needs = "J8 over these runs" if done else "a finished load test (bench loadtest), then J8 over it"
    return _check("throughput", "the measured throughput confirms the schedule (SPEC 7.1)", PENDING,
                  {"done_loadtest_runs": done, "needs": needs}, "apply the cuts of SPEC 7.3")


# ---------------------------------------------------------------- P-4: LoRA parity

Generation = Dict[str, List]  # {"tokens": [token ids], "top": [[top-k token ids] per generated position]}


def _upto_stop(tokens: Sequence[int], stop_ids: Sequence[int]) -> List[int]:
    out = []
    for token in tokens:
        if token in stop_ids:
            break
        out.append(token)
    return out


_STOP = "<stop>"


def compare(served: Generation, reference: Generation, stop_ids: Sequence[int]) -> Dict[str, Any]:
    """Greedy parity of two generations of the same prompt (vLLM tests' `check_logprobs_close`)."""
    s, r = _upto_stop(served["tokens"], stop_ids), _upto_stop(reference["tokens"], stop_ids)

    def chosen(tokens, i):
        return tokens[i] if i < len(tokens) else _STOP

    def in_top(generation, i, token):
        if i >= len(generation["top"]):
            return False  # no top-k at that position: the near-tie cannot be shown
        top = generation["top"][i]
        return any(t in stop_ids for t in top) if token == _STOP else token in top

    for i in range(max(len(s), len(r))):
        a, b = chosen(s, i), chosen(r, i)
        if a != b:
            return {"exact": False, "close": in_top(reference, i, a) and in_top(served, i, b), "first_difference": i}
    return {"exact": True, "close": True, "first_difference": None}


def parity_verdict(served_base: List[Generation], served_adapter: List[Generation], ref_base: List[Generation],
                   ref_adapter: List[Generation], stop_ids: Sequence[int]) -> Dict[str, Any]:
    n = len(served_base)
    if not n or not (len(served_adapter) == len(ref_base) == len(ref_adapter) == n):
        raise PreflightError("P-4 needs the four generations of every prompt")
    per_prompt = []
    for i in range(n):
        changes = _upto_stop(served_adapter[i]["tokens"], stop_ids) != _upto_stop(served_base[i]["tokens"], stop_ids)
        peft_change = compare(ref_adapter[i], ref_base[i], stop_ids)
        per_prompt.append({
            "adapter_changes_output": changes,
            "peft_changes_output": not peft_change["exact"],
            # HF-PEFT's change is decisive when it is not a near-tie: there the served adapter must change too
            "ignored_where_peft_is_decisive": not peft_change["close"] and not changes,
            "adapter_vs_peft": compare(served_adapter[i], ref_adapter[i], stop_ids),
            "base_vs_hf": compare(served_base[i], ref_base[i], stop_ids),
        })
    changed = sum(p["adapter_changes_output"] for p in per_prompt)
    ignored = sum(p["ignored_where_peft_is_decisive"] for p in per_prompt)
    peft_changed = sum(p["peft_changes_output"] for p in per_prompt)
    adapter_close = sum(p["adapter_vs_peft"]["close"] for p in per_prompt)
    base_close = sum(p["base_vs_hf"]["close"] for p in per_prompt)
    switch = ("switch the base to the reserve (roles.slm_reserve) and train again; if no base passes, B4 and B5 "
              "with an SLM are invalid and B0-B3 stay valid, declared (design P-4)")
    if adapter_close == n and changed and not ignored:
        status, diagnosis, action = PASS, "the served adapter changes the base's output and reproduces HF-PEFT", ""
    elif not changed and not peft_changed:
        status, diagnosis = FAIL, "uninformative: the adapter changes nothing even in HF-PEFT (untrained, or prompts outside its cluster)"
        action = "train the adapter, or check its prompts, and run P-4 again; P-4 is not decided"
    elif adapter_close < n and base_close < n:
        status, diagnosis = FAIL, "vLLM and HF disagree even without the adapter: the serving differs (template, revision, dtype)"
        action = "fix the serving (the chat template kwargs, the pinned revision) and run P-4 again; P-4 is not decided"
    elif not changed or ignored:
        status, diagnosis = FAIL, (f"vLLM ignores the adapter on {ignored or n} prompt(s): its output is the base's, "
                                   "while HF-PEFT's differs")
        action = switch
    else:
        status, diagnosis, action = FAIL, "the served adapter diverges from HF-PEFT while the served base matches HF", switch
    return {"status": status, "diagnosis": diagnosis, "action": action, "n_prompts": n,
            "adapter_changes_output": changed, "peft_changes_output": peft_changed, "ignored_where_peft_is_decisive": ignored,
            "adapter_matches_peft": adapter_close, "base_matches_hf": base_close, "per_prompt": per_prompt}


_TOKEN_ID = re.compile(r"^token_id:(\d+)$")


def _token_id(token: str) -> int:
    match = _TOKEN_ID.match(token)
    if not match:
        raise PreflightError(f"the server returned token {token!r}, not a token id: is it vLLM "
                             "(return_tokens_as_token_ids)?")
    return int(match.group(1))


def served_models(base_url: str, api_key: str, timeout_s: float) -> Dict[str, Dict[str, Any]]:
    """/v1/models of an OpenAI-compatible server, by id (vLLM gives each adapter's path as `root`)."""
    request = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return {card["id"]: card for card in json.loads(response.read()).get("data") or []}


def check_served(cards: Dict[str, Dict[str, Any]], base_name: str, repo: str, served_name: str, sha256: str) -> None:
    """The server serves the base from its pinned repository and the adapter from the directory of the
    adapter trained (named by its sha256): otherwise P-4 would judge another adapter than the one it records."""
    base, adapter = cards.get(base_name), cards.get(served_name)
    if base is None or adapter is None:
        raise PreflightError(f"the server lists {sorted(cards)}, not {base_name} and {served_name}: deploy them")
    if base.get("root") != repo:
        raise PreflightError(f"the server serves {base_name} from {base.get('root')!r}, not {repo!r}: redeploy")
    if Path(str(adapter.get("root"))).name != sha256 or adapter.get("parent") != base_name:
        raise PreflightError(f"the server serves {served_name} from {adapter.get('root')!r} over {adapter.get('parent')!r}, "
                             f"not the adapter trained (/adapters/{sha256}) over {base_name}: redeploy")


def served_generate(base_url: str, api_key: str, model: str, messages: List[dict], max_tokens: int,
                    top_logprobs: int, timeout_s: float) -> Generation:
    """One greedy generation from an OpenAI-compatible vLLM server, as token ids with their top-k."""
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens,
            "logprobs": True, "top_logprobs": top_logprobs, "return_tokens_as_token_ids": True}
    request = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise PreflightError(f"{model}: HTTP {e.code} {e.read()[:300]!r}") from None
    content = ((answer["choices"][0].get("logprobs") or {}).get("content")) or []
    return {"tokens": [_token_id(e["token"]) for e in content],
            "top": [[_token_id(t["token"]) for t in e.get("top_logprobs") or []] for e in content]}


def peft_generate(base: Dict[str, str], adapter_dir: str, prompts: List[List[dict]], template_kwargs: dict,
                  max_new_tokens: int, top_k: int, device: str) -> Dict[str, Any]:
    """HF-PEFT's greedy generations of every prompt, with the adapter and with it disabled (the reference
    of P-4), rendered with the same template kwargs the server applies."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base["repo"], revision=base["revision"])
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base["repo"], revision=base["revision"], dtype=dtype).to(device)
    model = PeftModel.from_pretrained(model, adapter_dir).eval()
    eos = model.generation_config.eos_token_id
    stop_ids = sorted({*(eos if isinstance(eos, list) else [eos] if eos is not None else []),
                       *([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])})
    out: Dict[str, Any] = {"stop_ids": stop_ids, "base": [], "adapter": []}
    for messages in prompts:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", **template_kwargs).to(device)
        for which, context in (("adapter", nullcontext()), ("base", model.disable_adapter())):
            with context, torch.no_grad():
                result = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                        top_p=None, top_k=None, output_logits=True, return_dict_in_generate=True)
            tokens = result.sequences[0, inputs["input_ids"].shape[1]:].tolist()
            top = [torch.topk(torch.log_softmax(step[0].float(), dim=-1), top_k).indices.tolist() for step in result.logits]
            out[which].append({"tokens": tokens, "top": top})
    return out


def lora_parity(config: Dict[str, Any], cluster: str, on: str,
                reference: Optional[Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    """P-4 on the adapter of `cluster`: the served base and adapter against the HF-PEFT reference."""
    import os

    from bench.train import adapters_dir, candidate, datasets_dir, parse_dataset

    manifest_path = adapters_dir() / cluster / "manifest.json"
    if not manifest_path.is_file():
        raise PreflightError(f"no adapter for cluster {cluster} (bench train --cluster {cluster})")
    manifest = json.loads(manifest_path.read_text())
    entry = candidate(config, manifest["slm"])
    settings = config["preflight"]["lora_parity"]
    rows = parse_dataset((datasets_dir() / f"{cluster}.jsonl").read_bytes(), f"{cluster}.jsonl")
    prompts = [row["prompt"] for row in rows[:settings["n_prompts"]]]
    endpoint = entry["endpoint"]
    if not endpoint.get("base_url"):
        raise PreflightError(f"roles.slm_candidates[{entry['name']}].endpoint.base_url is not set: serve it first")
    api_key = os.environ.get(endpoint["api_key_env"], "") if endpoint.get("api_key_env") else "EMPTY"
    if not api_key:
        raise PreflightError(f"environment variable {endpoint['api_key_env']} is not set")

    check_served(served_models(endpoint["base_url"], api_key, settings["timeout_s"]), entry["name"], manifest["base"]["repo"],
                 manifest["served_name"], manifest["adapter_sha256"])

    def served(model):
        return [served_generate(endpoint["base_url"], api_key, model, p, settings["max_new_tokens"],
                                settings["top_logprobs"], settings["timeout_s"]) for p in prompts]

    served_base, served_adapter = served(entry["name"]), served(manifest["served_name"])
    args = (manifest["base"], prompts, manifest["chat_template_kwargs"], settings["max_new_tokens"], settings["top_logprobs"])
    if reference is not None:
        ref = reference(*args)
    elif on == "local":
        ref = peft_generate(manifest["base"], str(manifest_path.parent / "adapter"), *args[1:], device="cpu")
    elif on == "modal":
        import sys

        sys.path.insert(0, str(paths.ROOT))
        from modal_apps import train as modal_train

        with modal_train.app.run():
            ref = modal_train.peft_reference.remote(manifest["adapter_sha256"], *args)
    else:
        raise PreflightError(f"--on must be local or modal, not {on!r}")
    verdict = parity_verdict(served_base, served_adapter, ref["base"], ref["adapter"], ref["stop_ids"])
    verdict.update({"cluster": cluster, "slm": manifest["slm"], "adapter_sha256": manifest["adapter_sha256"],
                    "served": {"base": entry["name"], "adapter": manifest["served_name"]}, "reference_on": on,
                    "reference_gpu_seconds": ref.get("function_seconds")})
    return verdict


def _unreachable() -> tuple:
    """What makes P-4 undecidable rather than a bug of this code: a missing input, a server or the Modal
    reference that cannot be reached or refuses."""
    from bench.train import TrainError

    errors = (PreflightError, TrainError, urllib.error.URLError, OSError)
    try:
        import modal.exception

        errors += (modal.exception.Error,)
    except ImportError:
        pass
    return errors


def check_lora_parity(config: Dict[str, Any], cluster: Optional[str], on: str) -> Dict[str, Any]:
    precondition = ("P-4: the served adapter changes the base's greedy output and reproduces HF-PEFT's "
                    "(design §2, P-4)")
    if cluster is None:
        return _check("lora_parity", precondition, PENDING,
                      {"needs": "the served base with its first adapter: bench preflight --parity <cluster>"},
                      "run it before the full training and the load test")
    try:
        verdict = lora_parity(config, cluster, on)
    except _unreachable() as e:  # a missing adapter, server or reference is a failed check, with its reason
        return _check("lora_parity", precondition, FAIL, {"error": f"{type(e).__name__}: {e}"},
                      "serve the base and the adapter, then run P-4 again; P-4 is not decided")
    return _check("lora_parity", precondition, verdict["status"], verdict, verdict["action"])


# ---------------------------------------------------------------- the command

def run_checks(config: Dict[str, Any], parity_cluster: Optional[str] = None, on: str = "modal") -> List[Dict[str, Any]]:
    return [check_agent_runs(), check_call_sites(), check_data(config), check_teacher_terms(config),
            check_pilot_spend(config), check_training_time(config), check_throughput(),
            check_lora_parity(config, parity_cluster, on)]


def preflight(config_path: str, parity_cluster: Optional[str] = None, on: str = "modal") -> int:
    from bench.contracts.config import config_sha256, load_config
    from bench.provenance import git_state, scrub

    config = load_config(config_path)
    started = datetime.now(timezone.utc)
    checks = run_checks(config, parity_cluster, on)
    run_dir = paths.RUNS / f"preflight-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
    run_dir.mkdir(parents=True)
    report = {"type": "preflight", "config_sha256": config_sha256(config), **git_state(),
              "started_at": started.isoformat(), "checks": checks,
              "all_pass": all(c["status"] == PASS for c in checks)}
    (run_dir / "report.json").write_text(scrub(json.dumps(report, indent=2, default=str)) + "\n")
    for c in checks:
        print(f"{c['status']:8} {c['id']:24} {c['action'] if c['status'] != PASS else ''}")
    print(run_dir)
    return 0 if report["all_pass"] else 1


def cli(args) -> int:
    try:
        return preflight(args.config, args.parity, args.on)
    except PreflightError as e:
        import sys

        print(f"bench preflight: {e}", file=sys.stderr)
        return 2
