"""`bench train --cluster <c>` (S5, SPEC §4): one LoRA adapter per cluster, on the base the `choice` fact names.

Reads `train/datasets/<cluster>.jsonl`, one row per line in TRL's conversational prompt-completion format
(`{"prompt": [messages], "completion": [{"role": "assistant", ...}]}`, loss on the completion only),
trains with TRL/PEFT (`train_lora`: the same code on a local CPU and in the Modal GPU function of
`modal_apps/train.py`), and writes

    train/adapters/<cluster>/adapter/        the PEFT adapter, exactly what vLLM serves; its identity is
                                             `facts.sha256_dir` of this directory, and it is served under the
                                             content-addressed name `<cluster>-<sha256[:12]>`, so a server that
                                             was not redeployed answers 404 instead of serving the old adapter

On Modal the run is detached and its result stored on the adapters volume under `modal_key` (the plan, the
GPU and the training code with its lock), so re-running the same command collects
it; changing any of that code in between trains again.
    train/adapters/<cluster>/manifest.json   base and revision, hyper-parameters, dataset sha256, the facts it
                                             was trained on, where it ran, GPU-seconds and cost, and the tokens
                                             it trained per second (what `bench preflight` projects the
                                             largest dataset's hours from)

The adapters train at the same time: one `bench train --cluster <c> --on modal` process per cluster, each
its own Modal app run and GPU container. The processes share only `train/adapters/`, where each writes its
own cluster's directory.

When every cluster of the centroids has an adapter trained on the same `choice` and `centroids`, the set is
registered as the `adapters` fact (`judgments/S5/<sha256>/adapters.json`).

Training reads only `train/datasets/`, never the do-not-train area (SPEC §3), and refuses to run while the
teacher's terms are not recorded (D6). A row is rendered with the candidate's `chat_template_kwargs`, the
ones the server applies, and must render its prompt as a prefix of prompt + completion (so the model learns
the tokens it will be asked to continue at serving time) and fit in `train.sft.max_length`: a row is never
silently truncated.
"""
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bench import paths
from bench.contracts.facts import read_fact, sha256_dir, write_fact

JUDGMENT = "S5"
SOURCE_ROOT = Path(__file__).resolve().parent.parent  # the code that ships to the Modal images
MESSAGE_ROLES = ("system", "user", "assistant")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
# a served name reaches vLLM's `--lora-modules name=path` and the router's `slm:<c>+lora:<name>`
SERVED_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrainError(RuntimeError):
    pass


def datasets_dir() -> Path:
    return paths.ROOT / "train" / "datasets"


def adapters_dir() -> Path:
    return paths.ROOT / "train" / "adapters"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def candidate(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """The `roles.slm_candidates` entry named `name`, with its weights pinned to a commit."""
    for entry in config["roles"].get("slm_candidates") or []:
        if entry["name"] == name:
            hf = entry.get("hf") or {}
            if not isinstance(hf.get("repo"), str) or not _COMMIT.match(str(hf.get("revision") or "")):
                raise TrainError(f"roles.slm_candidates[{name}].hf needs repo and a 40-hex revision")
            return entry
    raise TrainError(f"no slm candidate named {name!r}")


def _trained_on(config: Dict[str, Any]) -> Tuple[Tuple[dict, str], Tuple[dict, str]]:
    """The `choice` and `centroids` facts B4 points at: S5 trains on the same ones, and the router
    refuses adapters trained on any other."""
    settings = (config.get("arms") or {}).get("B4") or {}
    facts = []
    for name in ("choice", "centroids"):
        if not settings.get(name):
            raise TrainError(f"arms.B4.{name} is not set: S5 trains on the {name} fact B4 uses")
        facts.append(read_fact(str(paths.ROOT / settings[name]), name))
    return facts[0], facts[1]


def _message_problems(where: str, message: Any) -> List[str]:
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        return [f"{where} must be {{role, content}}"]
    problems = []
    if message["role"] not in MESSAGE_ROLES:
        problems.append(f"{where}.role must be one of {MESSAGE_ROLES}")
    if not isinstance(message["content"], str):
        problems.append(f"{where}.content must be a string")
    return problems


def row_problems(row: Any) -> List[str]:
    """What keeps a dataset row from being one prompt-completion pair (design §5.1, "Dados de treino")."""
    if not isinstance(row, dict):
        return ["a row must be an object"]
    prompt, completion = row.get("prompt"), row.get("completion")
    if not isinstance(prompt, list) or not prompt:
        return ["prompt must be a non-empty list of messages"]
    if not isinstance(completion, list) or len(completion) != 1:
        return ["completion must be a list of exactly one message"]
    problems = [p for i, m in enumerate(prompt) for p in _message_problems(f"prompt[{i}]", m)]
    problems += _message_problems("completion[0]", completion[0])
    if not problems and prompt[-1]["role"] == "assistant":
        problems.append("the prompt must not end with an assistant message")
    if not problems and completion[0]["role"] != "assistant":
        problems.append("the completion must be an assistant message")
    return problems


def parse_dataset(raw: bytes, name: str) -> List[Dict[str, Any]]:
    """Rows of a dataset file, each reduced to its prompt and completion; any malformed row refuses all."""
    rows = []
    for n, line in enumerate(raw.decode().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise TrainError(f"{name}:{n}: not JSON ({e})") from None
        problems = row_problems(row)
        if problems:
            raise TrainError(f"{name}:{n}: " + "; ".join(problems))
        rows.append({"prompt": row["prompt"], "completion": row["completion"]})
    if not rows:
        raise TrainError(f"{name} has no rows")
    return rows


def training_plan(config: Dict[str, Any], cluster: str) -> Tuple[Dict[str, Any], bytes]:
    """Everything one training run depends on, decided locally before anything runs, and the dataset."""
    from bench.preflight import teacher_terms_missing

    missing = teacher_terms_missing(config)
    if missing:
        raise TrainError("the teacher's terms are not recorded (D6: training on its outputs must be allowed): "
                         f"roles.production_llm.terms.{', '.join(missing)}")
    (choice, choice_sha), (centroids, centroids_sha) = _trained_on(config)
    if cluster not in centroids["clusters"]:
        raise TrainError(f"{cluster!r} is not a cluster of the centroids fact: {sorted(centroids['clusters'])}")
    slm = choice["slm"]
    entry = candidate(config, slm)
    if not SERVED_NAME.match(cluster) or cluster == slm:
        raise TrainError(f"cluster {cluster!r} cannot be a served adapter name (letters, digits, '.', '_', '-'; "
                         "not the base's name)")
    gpu_cost(config, config["train"]["gpu"], 0)  # refuses a GPU without a price before any GPU time is spent
    path = datasets_dir() / f"{cluster}.jsonl"
    if not path.is_file():
        raise TrainError(f"no dataset for cluster {cluster}: train/datasets/{cluster}.jsonl")
    raw = path.read_bytes()
    rows = parse_dataset(raw, path.name)
    train = config["train"]
    plan = {
        "cluster": cluster, "slm": slm,
        "base": {"repo": entry["hf"]["repo"], "revision": entry["hf"]["revision"]},
        "chat_template_kwargs": dict(entry.get("chat_template_kwargs") or {}),
        "facts": {"choice": choice_sha, "centroids": centroids_sha},
        "dataset": {"path": f"train/datasets/{cluster}.jsonl", "sha256": sha256_bytes(raw), "rows": len(rows)},
        "hyperparameters": {"seed": train["seed"], "precision": train["precision"],
                            "lora": dict(train["lora"]), "sft": dict(train["sft"])},
    }
    return plan, raw


# ---------------------------------------------------------------- training (CPU or GPU, same code)

def _token_ids(tokenizer: Any, messages: List[dict], template_kwargs: dict, generation_prompt: bool) -> List[int]:
    # as TRL's SFT tokenizes a conversational prompt-completion row (trl/data_utils.py:_tokenize)
    return list(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                              add_generation_prompt=generation_prompt, **template_kwargs)["input_ids"])


def check_template_reads(tokenizer: Any, template_kwargs: dict) -> None:
    """A kwarg the chat template never reads is silently ignored (thinking would stay on)."""
    template = tokenizer.chat_template or ""
    unread = [k for k in template_kwargs if k not in template]
    if unread:
        raise TrainError(f"the chat template does not read {unread}: the candidate's chat_template_kwargs would do nothing")


def check_rows(tokenizer: Any, rows: List[dict], template_kwargs: dict,
               max_length: int) -> Tuple[Dict[str, int], List[List[int]]]:
    """Every row trains on what serving will ask: its prompt, rendered with the generation prompt and the
    serving template kwargs, is a prefix of prompt + completion; and the row fits in max_length. Returns
    the token counts and, per row, the completion tokens (what the loss must fall on)."""
    check_template_reads(tokenizer, template_kwargs)
    longest, tokens, completions = 0, 0, []
    for i, row in enumerate(rows):
        prompt = _token_ids(tokenizer, row["prompt"], template_kwargs, generation_prompt=True)
        full = _token_ids(tokenizer, row["prompt"] + row["completion"], template_kwargs, generation_prompt=False)
        if full[:len(prompt)] != prompt:
            raise TrainError(f"row {i + 1}: the chat template does not render the prompt as a prefix of prompt + "
                             f"completion with {template_kwargs}; the adapter would learn tokens serving never asks for")
        if len(full) > max_length:
            raise TrainError(f"row {i + 1} has {len(full)} tokens, over train.sft.max_length = {max_length}")
        if len(full) == len(prompt):
            raise TrainError(f"row {i + 1}: the completion renders to no tokens")
        longest = max(longest, len(full))
        tokens += len(full)
        completions.append(full[len(prompt):])
    # tokens: what one epoch trains on, every row whole (the prompt is computed too, though the loss is not on it)
    return {"longest_row_tokens": longest, "tokens": tokens, "completion_tokens": sum(map(len, completions))}, completions


def check_loss_tokens(dataset: Any, completions: List[List[int]]) -> None:
    """What TRL will train on, row by row, is exactly the completion tokens serving would produce: the loss
    on the completion only, rendered with the serving kwargs (thinking off), nothing dropped or cut."""
    if len(dataset) != len(completions):
        raise TrainError(f"TRL kept {len(dataset)} of {len(completions)} rows")
    for i, example in enumerate(dataset):
        if "labels" in example:
            under_loss = [t for t, label in zip(example["input_ids"], example["labels"]) if label != -100]
        elif "completion_mask" in example:
            under_loss = [t for t, m in zip(example["input_ids"], example["completion_mask"]) if m]
        else:
            raise TrainError("TRL's dataset has neither labels nor a completion mask: the loss cannot be checked")
        if under_loss != completions[i]:
            raise TrainError(f"row {i + 1}: TRL would train on {len(under_loss)} tokens that are not the "
                             f"{len(completions[i])} completion tokens serving produces")


def check_target_modules(model: Any, target_modules: List[str]) -> None:
    """PEFT is silent about a listed module that matches nothing, which would train a partial adapter."""
    present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = [m for m in target_modules if m not in present]
    if missing:
        raise TrainError(f"the base has no module named {missing}: set train.lora.target_modules for this base")


def train_lora(plan: Dict[str, Any], rows: List[dict], out_dir: Path, device: str) -> Dict[str, Any]:
    """Train one adapter and save it (PEFT files only) to out_dir. The same on a CPU and on a GPU."""
    with tempfile.TemporaryDirectory() as trainer_dir:
        return _train_lora(plan, rows, out_dir, device, trainer_dir)


def _train_lora(plan: Dict[str, Any], rows: List[dict], out_dir: Path, device: str, trainer_dir: str) -> Dict[str, Any]:
    import peft
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base, hp = plan["base"], plan["hyperparameters"]
    lora, sft = hp["lora"], hp["sft"]
    kwargs = plan["chat_template_kwargs"]
    # SFTTrainer wraps the model in PEFT before the Trainer seeds: without this the LoRA init is unseeded
    transformers.set_seed(hp["seed"])
    tokenizer = AutoTokenizer.from_pretrained(base["repo"], revision=base["revision"])
    token_stats, completions = check_rows(tokenizer, rows, kwargs, sft["max_length"])
    bf16 = device == "cuda" and hp["precision"] == "bf16"
    model = AutoModelForCausalLM.from_pretrained(base["repo"], revision=base["revision"],
                                                 dtype=torch.bfloat16 if bf16 else torch.float32)
    check_target_modules(model, lora["target_modules"])
    # the kwargs travel with every row: TRL renders prompt and prompt + completion with them
    dataset = Dataset.from_list([{**row, "chat_template_kwargs": kwargs} if kwargs else row for row in rows])
    args = SFTConfig(
        output_dir=trainer_dir, seed=hp["seed"], data_seed=hp["seed"],
        learning_rate=sft["learning_rate"], num_train_epochs=sft["num_train_epochs"],
        max_steps=sft["max_steps"] if sft.get("max_steps") else -1,
        per_device_train_batch_size=sft["per_device_train_batch_size"],
        gradient_accumulation_steps=sft["gradient_accumulation_steps"], max_length=sft["max_length"],
        lr_scheduler_type=sft["lr_scheduler_type"], warmup_steps=sft["warmup_steps"],
        gradient_checkpointing=sft["gradient_checkpointing"], logging_steps=sft["logging_steps"],
        bf16=bf16, use_cpu=device == "cpu", completion_only_loss=True, packing=False,
        save_strategy="no", report_to=[], disable_tqdm=True,
        include_num_input_tokens_seen="non_padding",  # counted by the trainer: the throughput below
    )
    peft_config = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                             target_modules=list(lora["target_modules"]), bias="none", task_type="CAUSAL_LM",
                             revision=base["revision"])
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer,
                         peft_config=peft_config)
    check_loss_tokens(trainer.train_dataset, completions)
    started = time.perf_counter()
    result = trainer.train()
    train_seconds = time.perf_counter() - started
    trainer.model.save_pretrained(str(out_dir))  # adapter_config.json and adapter_model.safetensors
    tokens_seen = trainer.state.num_input_tokens_seen
    return {
        "device": device, "dtype": "bfloat16" if bf16 else "float32", "train_seconds": round(train_seconds, 3),
        # the tokens the trainer ran through the model (every row whole, each epoch) and their rate
        "tokens_seen": tokens_seen, "tokens_per_second": round(tokens_seen / train_seconds, 3),
        "global_step": result.global_step, "train_loss": result.training_loss,
        "examples_seen": result.global_step * sft["per_device_train_batch_size"] * sft["gradient_accumulation_steps"],
        **token_stats,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "trl": trl.__version__, "peft": peft.__version__},
    }


# ---------------------------------------------------------------- records

def gpu_cost(config: Dict[str, Any], gpu: Optional[str], seconds: float) -> Dict[str, Any]:
    if gpu is None:
        return {"gpu": None, "gpu_seconds": 0.0, "cost_usd": 0.0, "price_usd_per_s": None, "price_as_of": None}
    prices = config["modal"]["gpu_prices"]
    if gpu not in prices["usd_per_s"]:
        raise TrainError(f"no price for GPU {gpu} in modal.gpu_prices")
    price = prices["usd_per_s"][gpu]
    return {"gpu": gpu, "gpu_seconds": round(seconds, 3), "cost_usd": round(seconds * price, 4),
            "price_usd_per_s": price, "price_as_of": prices["as_of"]}


def _write_json(path: Path, value: Any) -> None:
    """Whole or not there: the clusters train at the same time, and one registering the set reads the
    others' manifests while they are being written."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def served_name(cluster: str, sha256: str) -> str:
    """The name vLLM serves an adapter under: its cluster and the start of its sha256 (content-addressed)."""
    return f"{cluster}-{sha256[:12]}"


def write_manifest(config: Dict[str, Any], plan: Dict[str, Any], stats: Dict[str, Any], adapter: Path, where: str,
                   billing: Dict[str, Any], code: Dict[str, Any], times: Dict[str, Any]) -> Path:
    """`billing` (GPU, seconds, price, cost), `code` (commit) and `times` (started_at, finished_at) are the
    training's own: for a result collected from an earlier run, that run's, never today's."""
    from bench.contracts.config import config_sha256

    sha = sha256_dir(adapter)
    manifest = {
        **{k: plan[k] for k in ("cluster", "slm", "base", "chat_template_kwargs", "facts", "dataset", "hyperparameters")},
        "served_name": served_name(plan["cluster"], sha), "adapter_sha256": sha, "where": where, **billing,
        "stats": stats, "config_sha256": config_sha256(config), **code, **times,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    path = adapter.parent / "manifest.json"
    _write_json(path, manifest)
    return path


def _install_adapter(cluster: str, trained: Path) -> Path:
    """Move a finished adapter to train/adapters/<cluster>/adapter/, replacing the previous one whole
    (only once training succeeded: a failed run leaves the previous adapter in place)."""
    import shutil

    target = adapters_dir() / cluster
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.move(str(trained), str(target / "adapter"))
    return target / "adapter"


def register_adapters(config: Dict[str, Any]) -> Tuple[Optional[Path], List[str]]:
    """The `adapters` fact once every centroid cluster has an adapter trained on B4's choice and centroids,
    over the candidate's current pinned weights and template kwargs (the fact records neither, so this is
    where a change of revision is caught); otherwise None and the clusters still missing one."""
    (choice, choice_sha), (centroids, centroids_sha) = _trained_on(config)
    entry = candidate(config, choice["slm"])
    trained_for = {"slm": choice["slm"], "facts": {"choice": choice_sha, "centroids": centroids_sha},
                   "base": dict(entry["hf"]), "chat_template_kwargs": dict(entry.get("chat_template_kwargs") or {})}
    adapters, missing = {}, []
    for cluster in sorted(centroids["clusters"]):
        manifest_path = adapters_dir() / cluster / "manifest.json"
        if not manifest_path.is_file():
            missing.append(cluster)
            continue
        manifest = json.loads(manifest_path.read_text())
        if {k: manifest.get(k) for k in trained_for} != trained_for:
            missing.append(cluster)  # another choice, other centroids, other weights or template kwargs: retrain
            continue
        sha = sha256_dir(manifest_path.parent / "adapter")
        if sha != manifest["adapter_sha256"]:
            raise TrainError(f"train/adapters/{cluster}/adapter changed after training (sha256 {sha}, "
                             f"manifest {manifest['adapter_sha256']})")
        if manifest["served_name"] != served_name(cluster, sha):
            raise TrainError(f"train/adapters/{cluster}: served name {manifest['served_name']!r} is not "
                             f"{served_name(cluster, sha)!r}, the one its bytes give")
        adapters[cluster] = {"served_name": manifest["served_name"], "sha256": sha}
    if missing:
        return None, missing
    # base_revision and chat_template_kwargs: what the set was trained on (required by validate_fact; the
    # router checks them against the candidate)
    payload = {"slm": choice["slm"], "choice": choice_sha, "centroids": centroids_sha, "adapters": adapters,
               "base_revision": entry["hf"]["revision"], "chat_template_kwargs": trained_for["chat_template_kwargs"]}
    return write_fact(JUDGMENT, "adapters", payload), []


def plan_id(plan: Dict[str, Any]) -> str:
    """The identity of a training: everything it depends on (the dataset by its sha256)."""
    return sha256_bytes(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode())


# The code a training runs (bench.train and what it imports, the Modal training app and its shared module)
# and the lock it installs. Only these: an edit elsewhere (the load test, the report, a merge of another
# front) must not make a detached training's result uncollectable and bill a second training.
TRAINING_CODE = ("bench/__init__.py", "bench/train.py", "bench/paths.py", "bench/contracts/*.py",
                 "modal_apps/__init__.py", "modal_apps/common.py", "modal_apps/train.py", "env/train/requirements.lock")


def code_sha256(root: Path = SOURCE_ROOT) -> str:
    """The identity of the code and image a Modal training runs (TRAINING_CODE), by content (a dirty tree
    is not its commit)."""
    matched = {pattern: list(root.glob(pattern)) for pattern in TRAINING_CODE}
    missing = [pattern for pattern, found in matched.items() if not found]
    if missing:  # an identity that silently skips the code it names would not change when that code appears
        raise TrainError(f"the training code is incomplete: nothing at {', '.join(missing)}")
    files = sorted({file for found in matched.values() for file in found})
    digest = hashlib.sha256()
    for file in files:
        digest.update(f"{file.relative_to(root).as_posix()}\0{sha256_bytes(file.read_bytes())}\n".encode())
    return digest.hexdigest()


def run_identity(config: Dict[str, Any]) -> Dict[str, Any]:
    """What a Modal training runs on besides its plan: the GPU with its dated price, the code and image, the
    commit. Part of the stored result's key, and recorded with it, so a reused result reports its own."""
    from bench.provenance import git_state

    billing = gpu_cost(config, config["train"]["gpu"], 0)
    return {"gpu": billing["gpu"], "price_usd_per_s": billing["price_usd_per_s"], "price_as_of": billing["price_as_of"],
            "code_sha256": code_sha256(), **git_state()}


def modal_key(plan: Dict[str, Any], identity: Dict[str, Any]) -> str:
    """The key of a stored Modal training: the plan, the GPU and the code (a price or a commit alone does not
    change what is trained)."""
    return plan_id({**plan, "run": {k: identity[k] for k in ("gpu", "code_sha256")}})


def precheck(plan: Dict[str, Any], rows: List[dict]) -> Dict[str, int]:
    """The template checks of train_lora, run here with the candidate's tokenizer before any GPU starts."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(plan["base"]["repo"], revision=plan["base"]["revision"])
    return check_rows(tokenizer, rows, plan["chat_template_kwargs"], plan["hyperparameters"]["sft"]["max_length"])[0]


def _scratch() -> Path:
    adapters_dir().mkdir(parents=True, exist_ok=True)
    return adapters_dir()


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def train_cluster(config_path: str, cluster: str, on: str) -> Dict[str, Any]:
    """`bench train`: train locally (CPU, or a local GPU) or in the Modal GPU function; write the adapter
    and its manifest; register the set when it is complete."""
    from bench.contracts.config import load_config

    config = load_config(config_path)
    plan, raw = training_plan(config, cluster)
    started = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory(dir=_scratch()) as work:
        trained = Path(work) / "adapter"
        if on == "local":
            stats = train_lora(plan, parse_dataset(raw, plan["dataset"]["path"]), trained, _device())
        elif on == "modal":
            precheck(plan, parse_dataset(raw, plan["dataset"]["path"]))
            sys.path.insert(0, str(paths.ROOT))
            os.environ["BENCH_CONFIG"] = str(Path(config_path).resolve())  # the Modal app reads this configuration
            from modal_apps import train as modal_train

            # detached: the training goes on if this machine sleeps; running the same command again
            # collects the stored result of the same plan instead of training twice
            identity = run_identity(config)
            with modal_train.app.run(detach=True):
                result = modal_train.train_adapter.remote(plan, raw, identity)
            for rel, content in result["files"].items():
                (trained / rel).parent.mkdir(parents=True, exist_ok=True)
                (trained / rel).write_bytes(content)
            if sha256_dir(trained) != result["sha256"]:
                raise TrainError("the adapter downloaded from Modal is not the one trained there (sha256 differs)")
            # the run that trained it (for a result collected from an earlier run, that run's): its GPU, price,
            # seconds and commit, never today's
            run = result["run"]
            if (run["gpu"], run["code_sha256"]) != (identity["gpu"], identity["code_sha256"]):
                raise TrainError("the stored result was trained on another GPU or code than this plan asks")
            stats = {**result["stats"], "collected_from_earlier_run": bool(result.get("reused")),
                     "observed_gpus": run.get("observed_gpus"),
                     # a collecting call boots a GPU container too: its seconds, apart from the training's
                     "collection_seconds": result.get("collection_seconds")}
            seconds = result["function_seconds"]
            billing = {"gpu": run["gpu"], "gpu_seconds": round(seconds, 3),
                       "cost_usd": round(seconds * run["price_usd_per_s"], 4),
                       "price_usd_per_s": run["price_usd_per_s"], "price_as_of": run["price_as_of"]}
            code = {"commit": run["commit"], "dirty": run["dirty"], "code_sha256": run["code_sha256"],
                    "modal_key": modal_key(plan, identity)}
            times = {"started_at": run["started_at"], "finished_at": run["finished_at"]}
        else:
            raise TrainError(f"--on must be local or modal, not {on!r}")
        adapter = _install_adapter(cluster, trained)
    if on == "local":
        from bench.provenance import git_state

        billing, code = gpu_cost(config, None, 0), git_state()  # a local run is not billed
        times = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat()}
    manifest = write_manifest(config, plan, stats, adapter, on, billing, code, times)
    fact, missing = register_adapters(config)
    return {"manifest": manifest, "fact": fact, "missing": missing, "modal_key": code.get("modal_key")}


def cli(args) -> int:
    try:
        result = train_cluster(args.config, args.cluster, args.on)
    except TrainError as e:
        print(f"bench train: {e}", file=sys.stderr)
        return 2
    print(result["manifest"].relative_to(paths.ROOT))
    if result["modal_key"]:
        print(f"stored on the adapters volume under {result['modal_key']}: the same command collects it only with "
              "the same plan, GPU and code (a change to the training code or its lock trains again)")
    if result["fact"]:
        print(f"adapters fact: {result['fact'].relative_to(paths.ROOT)} (point arms.B4.adapters and arms.B5.adapters at it)")
    else:
        print(f"clusters still without an adapter on this choice and centroids: {result['missing']}")
    return 0
