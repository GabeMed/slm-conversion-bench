"""`bench train --cluster <c> --on tinker`: S5 on Thinking Machines' Tinker, an alternative backend to Modal (faster
to a first adapter; Qwen3-8B only). The registered configuration trains on Modal (`--on modal`, bench/train.py), so
an adapter trained here and used in the registered run is recorded in prereg/DEVIATIONS.md (SPEC §6).

A Tinker run and a Modal run of the same plan pair by `plan_sha256` (the plan alone, whatever the backend: a
Tinker manifest records it, and `plan_sha256(manifest)` computes it from either) and differ in `where`, so their
adapters can be compared.

The same plan as bench/train.py (`training_plan`: the chosen base at its pinned revision, the dataset, the
`choice` and `centroids` facts, train.seed/lora/sft from config.yaml), trained through Tinker's SDK with the
optimizer loop on this machine:

    rows        tokenized with the candidate's HF tokenizer at the pinned revision and the serving
                `chat_template_kwargs`, through `check_rows`, the checks the TRL path runs: the prompt renders as
                a prefix of prompt + completion, nothing is truncated, and the loss falls on exactly the
                completion tokens serving would produce
    loss        `cross_entropy`, which Tinker sums over weighted tokens; each completion token weighs
                1 / (completion tokens of its optimizer step), the token mean TRL takes over one step's
                accumulated batches (`num_items_in_batch`)
    steps       per_device_train_batch_size x gradient_accumulation_steps rows per optimizer step, one
                `forward_backward` each; ceil(rows / that) steps per epoch, as transformers' Trainer counts them
    schedule    cosine with linear warm-up (transformers' `get_cosine_schedule_with_warmup`), computed here per
                step; AdamW with the defaults the TRL path trains with (configs/train-tinker.yaml)
    export      `save_weights_external`, downloaded and converted to a PEFT adapter by tinker-cookbook's
                `weights.download` and `weights.build_lora_adapter` (adapter_config.json +
                adapter_model.safetensors, what vLLM's `--lora-modules` and HF PEFT load); the converter reads
                the base's config.json and safetensors headers only, so a header-only base at the pinned revision
                (the real headers by HTTP range request, in sparse files of the shards' sizes; no weights) is
                built apart from the HF cache, with the cookbook checked, before anything is paid for
                (`base_snapshot`); the manifest records it (stats.run.base_snapshot: headers-only)

What Tinker cannot honour is never silent: configs/train-tinker.yaml's `accept_differences` must name every
difference this plan has (`differences`), and the manifest records them; a hyper-parameter this module does not
know refuses. After export, the adapter's rank, alpha and target modules must be config.yaml's, or it is not
installed.

Writes what bench/train.py writes (train/adapters/<cluster>/adapter/ and manifest.json, `where: tinker`), and
registers the `adapters` fact the same way. It never replaces an existing adapter. Tinker bills per token: the
manifest records the tokens trained, and no price (none is kept in the repository). The moment the export
answers, its Tinker path and expiry are kept in train/adapters/.tinker/<key>.json, so running the same command
again downloads them instead of training twice (until they expire, which it says); with an adapter installed
from the same plan it only registers. Not covered: a run interrupted before the export starts over (no mid-run
checkpoint), and two processes on the same cluster are not locked against each other.

A plan with train.sft.max_steps set (the one-step probe of RUNBOOK step 9) is a probe: its adapter goes to
train/adapters/.probe/<cluster>/, never served nor registered, and never in the way of the full run.

`tinker` and `tinker-cookbook` are imported only here (env/tinker); `bench train --on modal|local` never needs them.
"""
import json
import math
import os
import random
import shutil
import struct
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from bench import paths
from bench.contracts.facts import sha256_dir
from bench.train import (TrainError, _token_ids, _write_json, adapters_dir, check_rows,
                         parse_dataset, plan_id, register_adapters, training_plan, write_manifest)

SETTINGS = "configs/train-tinker.yaml"
PROBES = ".probe"  # train/adapters/.probe/<cluster>: runs cut short by train.sft.max_steps
# every hyper-parameter of config.yaml's train section this module honours or records; another one refuses
HYPERPARAMETERS = {"seed", "precision", "lora", "sft"}
LORA = {"r", "alpha", "dropout", "target_modules"}
SFT = {"learning_rate", "num_train_epochs", "max_steps", "per_device_train_batch_size", "gradient_accumulation_steps",
       "max_length", "lr_scheduler_type", "warmup_steps", "gradient_checkpointing", "logging_steps"}
OPTIMIZER = {"beta1", "beta2", "eps", "weight_decay", "grad_clip_norm"}
SWITCHES = ("train_attn", "train_mlp")  # create_lora_training_client's module groups; train_unembed is always off


class TinkerMissing(TrainError):
    pass


# ---------------------------------------------------------------- the settings and what Tinker cannot honour

def load_settings(path: str) -> Dict[str, Any]:
    """configs/train-tinker.yaml, checked: a standalone file (not a `--config`, never merged into config.yaml)."""
    try:
        settings = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise TrainError(f"cannot read the Tinker settings {path}: {e}") from None
    problems = []
    if not isinstance(settings, dict):
        raise TrainError(f"{path} must be a mapping")
    if not isinstance(settings.get("sdk_version"), str):
        problems.append("sdk_version must be the exact tinker version (a string)")
    models = settings.get("models")
    if not isinstance(models, dict) or not all(
            isinstance(m, dict) and isinstance(m.get("tinker_id"), str) and isinstance(m.get("max_context"), int)
            for m in models.values()):
        problems.append("models must map an HF repo to {tinker_id, max_context}")
    groups = settings.get("lora_groups")
    if not isinstance(groups, dict) or set(groups) != set(SWITCHES) or not all(
            isinstance(g, list) and g and all(isinstance(m, str) for m in g) for g in groups.values()):
        problems.append(f"lora_groups must list the modules of each of {list(SWITCHES)}")
    optimizer = settings.get("optimizer")
    if not isinstance(optimizer, dict) or set(optimizer) != OPTIMIZER or not all(
            isinstance(v, (int, float)) for v in optimizer.values()):
        problems.append(f"optimizer must set exactly {sorted(OPTIMIZER)}")
    accepted = settings.get("accept_differences")
    if not isinstance(accepted, list) or not set(accepted) <= set(DIFFERENCES):
        problems.append(f"accept_differences must be a list of names among {sorted(DIFFERENCES)}")
    ttl = settings.get("export_ttl_s")
    if ttl is not None and (not isinstance(ttl, int) or ttl < 3600):
        problems.append("export_ttl_s must be null (never expires) or at least 3600 seconds")
    if problems:
        raise TrainError(f"{path}: " + "; ".join(problems))
    return settings


# What a Tinker run differs in from the TRL run of the same plan, by name; `differences` says which apply.
DIFFERENCES = {
    "base.revision": "Tinker trains its own copy of the base, which it does not pin to a revision; the adapter is "
                     "served on the pinned revision",
    "lora.alpha": "not settable in Tinker: the exported adapter's lora_alpha is checked against train.lora.alpha "
                  "after training (not installed if it differs)",
    "lora.dropout": "Tinker trains without LoRA dropout",
    "lora.init": "the LoRA matrices are initialised by Tinker under train.seed, not by PEFT",
    "precision": "the service's numerics, not train.precision",
    "sample_order": "rows shuffled each epoch by random.Random(f'{train.seed}:{epoch}'), not by TRL's sampler",
}


def differences(plan: Dict[str, Any]) -> List[str]:
    hp = plan["hyperparameters"]
    found = ["base.revision", "lora.alpha", "lora.init", "precision", "sample_order"]
    if hp["lora"]["dropout"]:
        found.append("lora.dropout")
    return sorted(found)


def tinker_plan(plan: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """What Tinker runs for `plan`, decided before any call; refuses what it cannot honour or record."""
    hp = plan["hyperparameters"]
    unknown = ([f"train.{k}" for k in set(hp) - HYPERPARAMETERS] + [f"train.lora.{k}" for k in set(hp["lora"]) - LORA]
               + [f"train.sft.{k}" for k in set(hp["sft"]) - SFT])
    if unknown:
        raise TrainError(f"the Tinker path does not know how to honour {sorted(unknown)}: train on Modal")
    repo = plan["base"]["repo"]
    model = settings["models"].get(repo)
    if model is None:
        raise TrainError(f"Tinker does not offer {repo} (the base of {plan['slm']}) for training; it offers "
                         f"{sorted(settings['models'])} (configs/train-tinker.yaml): train on Modal")
    lora, sft = hp["lora"], hp["sft"]
    targets, groups = set(lora["target_modules"]), settings["lora_groups"]
    switches = {s: bool(targets & set(groups[s])) for s in SWITCHES}
    covered = {m for s in SWITCHES if switches[s] for m in groups[s]}
    if covered != targets:
        raise TrainError(f"Tinker trains LoRA on whole module groups ({groups}); train.lora.target_modules "
                         f"{sorted(targets)} is not a union of them")
    if sft["lr_scheduler_type"] != "cosine":
        raise TrainError(f"the Tinker path implements train.sft.lr_scheduler_type cosine only, not "
                         f"{sft['lr_scheduler_type']!r}")
    missing = [d for d in differences(plan) if d not in settings["accept_differences"]]
    if missing:
        raise TrainError("Tinker cannot honour " + "; ".join(f"{d} ({DIFFERENCES[d]})" for d in missing)
                         + ": list each in accept_differences of configs/train-tinker.yaml to train anyway")
    return {"backend": "tinker", "sdk_version": settings["sdk_version"], "base_model": model["tinker_id"],
            "max_context": model["max_context"],
            "lora": {"rank": lora["r"], "seed": hp["seed"], **switches, "train_unembed": False},
            "optimizer": dict(settings["optimizer"]), "loss_fn": "cross_entropy",
            "differences": {d: DIFFERENCES[d] for d in differences(plan)}}


# ---------------------------------------------------------------- steps and schedule (as transformers' Trainer)

def schedule(n_rows: int, sft: Dict[str, Any]) -> Dict[str, int]:
    """Rows per optimizer step, steps per epoch (the last one partial), total and warm-up steps."""
    rows_per_step = sft["per_device_train_batch_size"] * sft["gradient_accumulation_steps"]
    per_epoch = max(math.ceil(n_rows / rows_per_step), 1)
    total = sft["max_steps"] if sft.get("max_steps") else math.ceil(sft["num_train_epochs"] * per_epoch)
    warmup = sft["warmup_steps"]
    if warmup < 0 or (warmup >= 1 and warmup != int(warmup)):
        raise TrainError(f"train.sft.warmup_steps must be a fraction below 1 or a whole number of steps, not {warmup}")
    warmup = int(warmup) if warmup >= 1 else math.ceil(total * warmup)
    return {"rows_per_step": rows_per_step, "steps_per_epoch": per_epoch, "total_steps": total, "warmup_steps": warmup}


def learning_rate(step: int, sched: Dict[str, int], peak: float) -> float:
    """transformers' cosine schedule with warm-up (num_cycles 0.5) at optimizer step `step` (0-based)."""
    warmup, total = sched["warmup_steps"], sched["total_steps"]
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def step_rows(n_rows: int, sched: Dict[str, int], seed: int) -> List[List[int]]:
    """The row indices of each optimizer step: every epoch a fresh shuffle cut into steps; epochs repeat until
    total_steps (max_steps may run past the epochs, as in the Trainer)."""
    steps: List[List[int]] = []
    epoch = 0
    while len(steps) < sched["total_steps"]:
        order = list(range(n_rows))
        random.Random(f"{seed}:{epoch}").shuffle(order)
        steps += [order[i:i + sched["rows_per_step"]] for i in range(0, n_rows, sched["rows_per_step"])]
        epoch += 1
    return steps[:sched["total_steps"]]


# ---------------------------------------------------------------- rows to Tinker datums

def encode_rows(tokenizer: Any, rows: List[dict], template_kwargs: dict, max_length: int,
                max_context: int) -> Tuple[Dict[str, int], List[Dict[str, List]]]:
    """Each row as next-token arrays: `inputs` = full[:-1], `targets` = full[1:], `mask` 1 on the targets that
    are completion tokens. Tokenized as the TRL path tokenizes (`check_rows`, with its refusals), and the masked
    targets are checked to be exactly the completion tokens serving produces."""
    token_stats, completions = check_rows(tokenizer, rows, template_kwargs, max_length)
    encoded = []
    for i, row in enumerate(rows):
        prompt = _token_ids(tokenizer, row["prompt"], template_kwargs, generation_prompt=True)
        full = _token_ids(tokenizer, row["prompt"] + row["completion"], template_kwargs, generation_prompt=False)
        if len(full) - 1 > max_context:
            raise TrainError(f"row {i + 1} has {len(full)} tokens, over Tinker's context for this base ({max_context})")
        mask = [0] * (len(prompt) - 1) + [1] * (len(full) - len(prompt))
        targets = full[1:]
        if [t for t, m in zip(targets, mask) if m] != completions[i]:
            raise TrainError(f"row {i + 1}: the loss would not fall on exactly the completion tokens")
        encoded.append({"inputs": full[:-1], "targets": targets, "mask": mask})
    return token_stats, encoded


def step_data(encoded: List[Dict[str, List]], indices: List[int]) -> List[Dict[str, Any]]:
    """One optimizer step's rows with their loss weights: 1 / (completion tokens of the step) on each completion
    token, so Tinker's weighted sum is the token mean TRL takes over the step."""
    n = sum(sum(encoded[i]["mask"]) for i in indices)
    return [{"inputs": encoded[i]["inputs"], "targets": encoded[i]["targets"],
             "weights": [m / n for m in encoded[i]["mask"]]} for i in indices]


# ---------------------------------------------------------------- the service (lazy: env/tinker only)

def _sdk() -> Any:
    try:
        import tinker
    except ImportError:
        raise TinkerMissing("`--on tinker` needs the Tinker SDK, which is optional: install env/tinker "
                            "(uv pip install -r env/tinker/requirements.lock) and set TINKER_API_KEY") from None
    return tinker


def _tokenizer(plan: Dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(plan["base"]["repo"], revision=plan["base"]["revision"])


def train_on_tinker(sdk: Any, run: Dict[str, Any], encoded: List[Dict[str, List]], sched: Dict[str, int],
                    sft: Dict[str, Any], name: str, ttl_s: Optional[int], log: Callable[[str], None],
                    exported: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    """The optimizer loop, client side; returns the exported weights' Tinker path and the run's numbers, which it
    hands to `exported` as soon as the export answers (before the session closes: what was paid for is recorded
    whatever happens next)."""
    service = sdk.ServiceClient(user_metadata={"created_by": "bench train", "adapter": name})
    status = "errored"
    try:
        offered = {m.model_name: m for m in service.create_rest_client().get_server_capabilities().result().supported_models}
        model = offered.get(run["base_model"])
        if model is None or model.trainable is False:
            raise TrainError(f"the Tinker service does not train {run['base_model']} today (it offers "
                             f"{sorted(n for n in offered if n)})")
        longest = max(len(e["inputs"]) for e in encoded)
        if model.max_context_length and longest > model.max_context_length:
            raise TrainError(f"a row has {longest} input tokens, over the service's context for "
                             f"{run['base_model']} ({model.max_context_length})")
        client = service.create_lora_training_client(base_model=run["base_model"], **run["lora"]).result()
        info = client.get_info()
        if info.lora_rank is not None and info.lora_rank != run["lora"]["rank"]:
            raise TrainError(f"Tinker created a LoRA of rank {info.lora_rank}, not {run['lora']['rank']}")
        losses, tokens, completion_tokens, examples = [], 0, 0, 0
        started = time.perf_counter()
        for step, indices in enumerate(step_rows(len(encoded), sched, run["lora"]["seed"])):
            data = step_data(encoded, indices)
            datums = [sdk.Datum(model_input=sdk.ModelInput.from_ints(d["inputs"]),
                                loss_fn_inputs={"target_tokens": d["targets"], "weights": d["weights"]}) for d in data]
            lr = learning_rate(step, sched, sft["learning_rate"])
            backward = client.forward_backward(datums, run["loss_fn"])  # both submitted, then awaited, in order
            optim = client.optim_step(sdk.AdamParams(learning_rate=lr, **run["optimizer"]))
            # the step's token-mean loss (its weights sum to 1); a missing metric must not cost the run
            losses.append(backward.result().metrics.get("loss:sum", math.nan))
            optim.result()
            tokens += sum(len(d["inputs"]) for d in data)
            completion_tokens += sum(sum(encoded[i]["mask"]) for i in indices)
            examples += len(indices)
            if (step + 1) % sft["logging_steps"] == 0 or step + 1 == sched["total_steps"]:
                log(f"step {step + 1}/{sched['total_steps']}: loss {losses[-1]:.4f}, lr {lr:.3g}")
        train_seconds = time.perf_counter() - started
        stats = {"external_weights": client.save_weights_external(name, ttl_seconds=ttl_s).result().path,
                 "training_run": str(info.model_id), "global_step": len(losses), "examples_seen": examples,
                 "train_loss": sum(losses) / len(losses) if all(map(math.isfinite, losses)) else None,
                 "train_seconds": round(train_seconds, 3),
                 # what was run through the model (Tinker bills per token) and what the loss fell on
                 "tokens_seen": tokens, "completion_tokens_seen": completion_tokens,
                 "tokens_per_second": round(tokens / train_seconds, 3) if train_seconds else None}
        exported(stats)
        status = "success"
    finally:
        # best effort: a failed close must neither lose a recorded export nor hide the error that ended the run
        try:
            service.close(status).result()
        except Exception as e:  # noqa: BLE001 (any SDK or transport error)
            log(f"closing the Tinker session ({status}) failed: {type(e).__name__}: {e}")
    return stats


BASE_SNAPSHOT = "headers-only"  # what the manifest records of the base the converter read (stats.run.base_snapshot)
HEADERS_ONLY_NOTE = "HEADERS-ONLY.txt"
SPARSE_PROBE_BYTES = 64 << 20


class Hub:
    """What `header_only_base` asks of the HF Hub, at one pinned revision (the tests stand a local directory in)."""

    def __init__(self, repo: str, revision: str):
        self.repo, self.revision = repo, revision

    def files(self) -> List[str]:
        from huggingface_hub import HfApi

        return HfApi().list_repo_files(self.repo, revision=self.revision)

    def download(self, name: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.repo, name, revision=self.revision)

    def read(self, name: str, start: int, end: int) -> Tuple[bytes, int]:
        """Bytes start..end (inclusive) of `name`, by an HTTP range request, and the file's whole size."""
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

        response = get_session().get(hf_hub_url(self.repo, name, revision=self.revision), timeout=60,
                                     headers={**build_hf_headers(), "Range": f"bytes={start}-{end}"})
        hf_raise_for_status(response)
        content_range = response.headers.get("Content-Range", "")
        if response.status_code != 206 or not content_range.startswith(f"bytes {start}-"):
            raise TrainError(f"the Hub did not answer a range request for {name} (HTTP {response.status_code}, "
                             f"Content-Range {content_range!r})")
        return response.content, int(content_range.rsplit("/", 1)[1])


def _check_sparse(directory: Path) -> None:
    """A file extended by truncate takes no disk here, or the header-only base would take the base's whole size."""
    probe = directory / "sparse.probe"
    probe.touch()
    os.truncate(probe, SPARSE_PROBE_BYTES)
    used = os.stat(probe).st_blocks * 512
    probe.unlink()
    if used >= SPARSE_PROBE_BYTES // 2:
        raise TrainError(f"{directory}'s file system does not keep sparse files ({used} bytes on disk for an empty "
                         f"{SPARSE_PROBE_BYTES}-byte file): the header-only base would take the base's whole size")


def header_shapes(directory: Path) -> Dict[str, Dict[str, Tuple[Tuple[int, ...], str]]]:
    """Each shard's tensors, shapes and dtypes, read as the converter reads them (safe_open: the header only)."""
    from safetensors import safe_open

    shards = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as f:
            shards[path.name] = {k: (tuple(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype()) for k in f.keys()}
    return shards


def header_only_base(repo: str, revision: str, root: Path, hub: Optional[Hub] = None) -> Path:
    """A directory the export's converter reads as the base at its pinned revision, without its weights:
    config.json (and model.safetensors.index.json when the repo has one) as the Hub serves them, and each
    top-level *.safetensors shard as a sparse file: the shard's real 8-byte length and JSON header (two range
    requests), extended to the shard's real size with nothing written, so safetensors opens it (it checks the
    size against the header) and lists the real names, shapes and dtypes. Its tensor data reads as zeros, so it
    is kept at <root>/<repo with -- for />/<revision>, apart from the HF cache, with HEADERS-ONLY.txt saying so.
    Built once (in a temporary directory, renamed into place when complete and checked), then reused."""
    hub = hub or Hub(repo, revision)
    target = root / repo.replace("/", "--") / revision
    if (target / HEADERS_ONLY_NOTE).is_file():
        if not header_shapes(target):
            raise TrainError(f"{target} holds no safetensors header: delete it to build it again")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{revision}.") as work:
        build = Path(work)
        _check_sparse(build)
        files = hub.files()
        shards = sorted(n for n in files if n.endswith(".safetensors") and "/" not in n)
        if "config.json" not in files or not shards:
            raise TrainError(f"{repo}@{revision} has no config.json or no top-level *.safetensors shard")
        index = "model.safetensors.index.json"
        for name in ["config.json"] + ([index] if index in files else []):
            shutil.copyfile(hub.download(name), build / name)
        for name in shards:
            prefix, size = hub.read(name, 0, 7)
            if len(prefix) != 8:
                raise TrainError(f"{name}: the Hub answered {len(prefix)} bytes for its first 8")
            (length,) = struct.unpack("<Q", prefix)
            if not 0 < length <= min(100 << 20, size - 8):  # safetensors refuses headers over 100 MB
                raise TrainError(f"{name}: a header of {length} bytes in a file of {size} is not safetensors")
            header, _ = hub.read(name, 8, 8 + length - 1)
            if len(header) != length:
                raise TrainError(f"{name}: the Hub answered {len(header)} bytes of its {length}-byte header")
            end = max((t["data_offsets"][1] for k, t in json.loads(header).items() if k != "__metadata__"), default=0)
            if 8 + length + end != size:
                raise TrainError(f"{name}: its header describes {8 + length + end} bytes, the Hub serves {size}")
            (build / name).write_bytes(prefix + header)
            os.truncate(build / name, size)
        shapes = header_shapes(build)
        if (build / index).is_file():
            weight_map = json.loads((build / index).read_text())["weight_map"]
            if weight_map != {k: shard for shard, tensors in shapes.items() for k in tensors}:
                raise TrainError(f"{repo}@{revision}: the shards' headers disagree with {index}")
        (build / HEADERS_ONLY_NOTE).write_text(
            f"{repo}@{revision}, for tinker-cookbook's weights.build_lora_adapter, which reads config.json and the "
            "safetensors headers only (bench/train_tinker.py, header_only_base). Each *.safetensors file here is "
            "sparse: the shard's real header, then zeros to its real size. These are NOT the weights: never load "
            "a model from this directory.\n")
        os.replace(build, target)
        build.mkdir()  # for TemporaryDirectory to remove
    return target


def base_snapshot(plan: Dict[str, Any], root: Optional[Path] = None) -> str:
    """What the export's converter reads, fetched before anything is paid for, so a missing package, a full disk
    or the HF Hub fails here and not after training. The converter (tinker-cookbook 0.6.1,
    weights.build_lora_adapter) reads the base's config.json and, through safetensors' safe_open, the headers of
    its *.safetensors files (get_model_state_keys, get_model_state_shapes: tensor names and shapes, no tensor
    data; no fallback to model.safetensors.index.json): `header_only_base` gives it those, under a megabyte
    instead of the whole base (~16 GB for Qwen3-8B). The manifest says so (stats.run.base_snapshot)."""
    try:
        from huggingface_hub import constants
        from tinker_cookbook import weights  # noqa: F401 (the export's converter: missing, it fails here)
    except ImportError as e:
        raise TinkerMissing(f"`--on tinker` exports through tinker-cookbook, which is missing ({e}): install "
                            "env/tinker (uv pip install -r env/tinker/requirements.lock)") from None
    root = root or Path(constants.HF_HOME) / "bench-headers-only"
    try:
        return str(header_only_base(plan["base"]["repo"], plan["base"]["revision"], root))
    except Exception as e:  # noqa: BLE001 (network, auth, disk: all refuse before training)
        raise TrainError(f"cannot fetch the headers of {plan['base']['repo']}@{plan['base']['revision']}, which "
                         f"the export's converter reads: {type(e).__name__}: {e}") from None


def export_adapter(external_weights: str, base_dir: str, out: Path) -> None:
    """Tinker's exported weights to a PEFT adapter at `out`, for the base in `base_dir` (`base_snapshot`)."""
    from tinker_cookbook import weights

    with tempfile.TemporaryDirectory(dir=out.parent) as work:
        raw = weights.download(tinker_path=external_weights, output_dir=str(Path(work) / "raw"))
        weights.build_lora_adapter(base_model=base_dir, adapter_path=raw, output_path=str(out))


def check_adapter(out: Path, plan: Dict[str, Any], external_weights: str) -> Dict[str, Any]:
    """The converted adapter is what config.yaml asked for (rank, alpha, modules) and names the pinned base as the
    TRL path's adapter does; returns what Tinker trained, for the manifest."""
    files = sorted(p.name for p in out.iterdir())
    if files != ["adapter_config.json", "adapter_model.safetensors"]:
        raise TrainError(f"the converted adapter has {files}, not adapter_config.json and adapter_model.safetensors")
    path = out / "adapter_config.json"
    adapter_config = json.loads(path.read_text())
    lora = plan["hyperparameters"]["lora"]
    trained = {"r": adapter_config.get("r"), "lora_alpha": adapter_config.get("lora_alpha"),
               "target_modules": sorted(adapter_config.get("target_modules") or []),
               "rank_pattern": adapter_config.get("rank_pattern") or {},
               "alpha_pattern": adapter_config.get("alpha_pattern") or {}}
    wanted = {"r": lora["r"], "lora_alpha": lora["alpha"], "target_modules": sorted(lora["target_modules"]),
              "rank_pattern": {}, "alpha_pattern": {}}
    if trained != wanted:
        raise TrainError(f"Tinker trained {trained}, config.yaml asks {wanted}: the adapter is not installed "
                         f"(the weights stay at {external_weights})")
    # the converter names the local snapshot it read; the adapter names the base as PEFT saves it
    adapter_config.update(base_model_name_or_path=plan["base"]["repo"], revision=plan["base"]["revision"])
    path.write_text(json.dumps(adapter_config, indent=2) + "\n")
    return trained


# ---------------------------------------------------------------- bench train --on tinker

PLAN_KEYS = ("cluster", "slm", "base", "chat_template_kwargs", "facts", "dataset", "hyperparameters")


def plan_sha256(manifest: Dict[str, Any]) -> str:
    """The plan's identity from any adapter manifest (Modal, local or Tinker: write_manifest copies the whole
    plan into each), so the runs of one plan on different backends pair."""
    return plan_id({k: manifest[k] for k in PLAN_KEYS})


def _pending(key: str) -> Path:
    return adapters_dir() / ".tinker" / f"{key}.json"


def train_cluster(config_path: str, cluster: str, settings_path: str = SETTINGS,
                  log: Callable[[str], None] = lambda line: print(line, file=sys.stderr)) -> Dict[str, Any]:
    from bench.contracts.config import load_config
    from bench.provenance import git_state

    config = load_config(config_path)
    plan, raw = training_plan(config, cluster)
    settings = load_settings(settings_path)
    run = tinker_plan(plan, settings)
    rows = parse_dataset(raw, plan["dataset"]["path"])
    sched = schedule(len(rows), plan["hyperparameters"]["sft"])
    run["schedule"] = sched
    key = plan_id({**plan, "run": run})
    # a run cut short (train.sft.max_steps, e.g. the one-step probe) is a probe: kept apart, never served or
    # registered (register_adapters compares no hyper-parameters), and never in the way of the full run
    probe = bool(plan["hyperparameters"]["sft"].get("max_steps"))
    target = (adapters_dir() / PROBES if probe else adapters_dir()) / cluster
    where = rel(target)
    result = {"probe": probe, "volume": config["modal"]["volumes"]["adapters"], "runpod": "runpod" in config}
    if (target / "manifest.json").is_file():
        manifest = json.loads((target / "manifest.json").read_text())
        if manifest.get("tinker_key") != key or sha256_dir(target / "adapter") != manifest.get("adapter_sha256"):
            raise TrainError(f"{where} holds another adapter (where: {manifest.get('where')}); "
                             "`--on tinker` never replaces one: move it away first")
        fact, missing = (None, []) if probe else register_adapters(config)  # already trained: nothing to spend
        return {**result, "manifest": target / "manifest.json", "fact": fact, "missing": missing, "collected": True}
    if target.exists():
        raise TrainError(f"{where} exists without a manifest: move it away first")

    sdk = _sdk()
    if sdk.__version__ != settings["sdk_version"]:
        raise TrainError(f"tinker {sdk.__version__} is installed, configs/train-tinker.yaml pins {settings['sdk_version']}")
    token_stats, encoded = encode_rows(_tokenizer(plan), rows, plan["chat_template_kwargs"],
                                       plan["hyperparameters"]["sft"]["max_length"], run["max_context"])
    pending = _pending(key)
    collected = pending.is_file()
    if collected:  # trained and exported by an earlier run of the same plan: download it again
        record = json.loads(pending.read_text())
        expires = record.get("expires_at")
        if expires and datetime.now(timezone.utc) >= datetime.fromisoformat(expires):
            raise TrainError(f"the weights {record['stats']['external_weights']} expired on Tinker at {expires} "
                             f"(export_ttl_s): delete {rel(pending)} to train again")
        base_dir = base_snapshot(plan)
        log(f"collecting {record['stats']['external_weights']} (trained {record['started_at']})")
    else:
        base_dir = base_snapshot(plan)  # before anything is paid for
        started = datetime.now(timezone.utc).isoformat()
        ttl = settings.get("export_ttl_s")

        def keep(stats: Dict[str, Any]) -> None:  # the moment the export answers: a re-run collects, never retrains
            now = datetime.now(timezone.utc)
            record.update(stats=stats, started_at=started, finished_at=now.isoformat(), exported_at=now.isoformat(),
                          expires_at=(now + timedelta(seconds=ttl)).isoformat() if ttl else None, **git_state())
            pending.parent.mkdir(parents=True, exist_ok=True)
            _write_json(pending, record)

        record: Dict[str, Any] = {}
        train_on_tinker(sdk, run, encoded, sched, plan["hyperparameters"]["sft"], f"{cluster}-{key[:12]}", ttl, log,
                        keep)
    external = record["stats"]["external_weights"]
    with tempfile.TemporaryDirectory(dir=adapters_dir()) as work:
        converted = Path(work) / "adapter"
        try:
            export_adapter(external, base_dir, converted)
        except Exception as e:  # noqa: BLE001 (the cookbook's download and conversion errors, disk, network)
            raise TrainError(f"the trained weights {external} could not be downloaded or converted "
                             f"({type(e).__name__}: {e}); they are recorded in {rel(pending)}: run the same command "
                             f"again to retry, or delete that file to train again") from None
        trained = check_adapter(converted, plan, external)
        target.mkdir(parents=True)
        shutil.move(str(converted), str(target / "adapter"))
    adapter = target / "adapter"
    # the converter read a header-only base (`base_snapshot`); recorded here, outside the key a re-run collects by
    stats = {**record["stats"], **token_stats, "backend": "tinker", "device": "tinker",
             "run": {**run, "base_snapshot": BASE_SNAPSHOT},
             "trained": trained, "collected_from_earlier_run": collected, "probe": probe,
             "versions": {"tinker": sdk.__version__, "tinker_cookbook": _cookbook_version()}}
    # Tinker bills tokens, not GPU-seconds: no GPU, no cost here (the tokens are in stats; no price in the repo)
    billing = {"gpu": None, "gpu_seconds": None, "cost_usd": None, "price_usd_per_s": None, "price_as_of": None}
    code = {"commit": record["commit"], "dirty": record["dirty"], "tinker_key": key, "plan_sha256": plan_id(plan),
            **{k: record[k] for k in ("dirty_paths", "dirty_paths_total") if k in record}}  # an older record has none
    path = write_manifest(config, plan, stats, adapter, "tinker", billing, code,
                          {"started_at": record["started_at"], "finished_at": record["finished_at"]})
    pending.unlink()
    fact, missing = (None, []) if probe else register_adapters(config)
    return {**result, "manifest": path, "fact": fact, "missing": missing, "collected": False}


def rel(path: Path) -> str:
    return str(path.relative_to(paths.ROOT))


def _cookbook_version() -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("tinker-cookbook")
    except PackageNotFoundError:
        return None


def cli(args) -> int:
    try:
        result = train_cluster(args.config, args.cluster, args.tinker_config)
    except TrainError as e:
        print(f"bench train: {e}", file=sys.stderr)
        return 2
    manifest = json.loads(result["manifest"].read_text())
    print(result["manifest"].relative_to(paths.ROOT))
    if result["collected"]:
        print("already trained on Tinker from this plan: nothing was trained")
    if result["probe"]:
        print(f"a probe (train.sft.max_steps): kept apart, neither served nor registered; check its manifest's "
              f"stats.trained (rank, alpha, modules) before a full run")
        return 0
    print(f"plan {manifest['plan_sha256']} (bench.train_tinker.plan_sha256 gives the same from the manifest of a "
          f"Modal run of this plan; the registered configuration trains on Modal: using this adapter in the "
          f"registered run goes in prereg/DEVIATIONS.md). " + (
              f"bench serve --on runpod up --candidate {manifest['slm']} uploads it to the RunPod volume" if result.get("runpod")
              else f"The server loads it from the adapters volume: modal volume put {result['volume']} "
                   f"train/adapters/{args.cluster}/adapter /{manifest['adapter_sha256']}"))
    if result["fact"]:
        print(f"adapters fact: {result['fact'].relative_to(paths.ROOT)} (point arms.B4.adapters and arms.B5.adapters at it)")
    else:
        print(f"clusters still without an adapter on this choice and centroids: {result['missing']}")
    return 0
