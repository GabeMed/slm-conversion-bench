# Runbook: the two days of the experiment

The order below is the one SPEC.md §7 fixes. Every step names the command, what it must show, and
what to do when it does not. Nothing here is paid until step 2.

**The days need not be consecutive (SPEC D26).** The project may pause between phases while the next
phase's funding is not in place: after step 5 (before K1) and after step 12 (before the test). The order
does not change, and nothing on the test split runs before step 12.

- `<...>` is a value a previous step printed (a run id, a result path).
- **Run the blocks in `bash`**, from the repository root, in the agent environment unless a step
  says otherwise. They use bash arrays; zsh splits words differently.
- **`bench judge`, `bench verify` and `bench report` run in the analysis environment**
  (`.venv-analysis`, Python 3.12, `env/analysis`), whatever the step: its `sum()` of floats is
  compensated, so a sum over a set does not move with `PYTHONHASHSEED` and a judgment recomputes to
  the same bytes (under the agent environment's 3.11, J6's score moves in the last digit), and its
  numpy is the pinned one J5's floats were tested with.
- **`bench eval` always runs in the agent environment** (`.venv-agent`), whatever the step, never in
  the analysis one. An eval records the SQLite version it executed with (3.53.1 in `.venv-agent`,
  3.50.4 in `.venv-analysis`), and J1 refuses a split whose evals do not share it (the instrument:
  fixed date, timeout, SQLite version, pre-registration). Every eval so far ran in `.venv-agent`, so
  one eval elsewhere would leave its split without EX. In short: `bench run`, `bench replay` and
  `bench eval` in `.venv-agent`; `bench judge`, `bench verify` and `bench report` in `.venv-analysis`.
- **Paths written into `config.yaml` are relative to the repository root** (`judgments/J6/<sha>/choice.json`).
  The commands print absolute paths: cut the prefix. The configuration is registered and published.

## Before Day 1 (no paid call)

1. **Accounts and credentials.**
   - In the shell: the variables `config.yaml` names under each role's `endpoint.api_key_env` and
     `endpoint.headers_env`, and the embeddings key. None is written to a file of this repository.
   - In Modal: a token, and the two secrets `modal.secrets` names. `slm-bench-vllm-api-key` holds
     `VLLM_API_KEY`; `slm-bench-proxy-auth` holds the proxy-auth variables of `headers_env`.
2. **A spending limit on each provider account:** a prepaid balance, or the provider's own limit
   where it has one, set to the API budget for the teacher plus the cheap alternative. The harness
   has no spend cap of its own; step 5 projects the spend by hand, and this limit is what stops a
   projection that was wrong. A balance below a phase's projection pauses the project before that
   phase (D26).
3. **Fill every `before runs` value of `config.yaml`:**
   - each API role's `endpoint.base_url`, `provider` (name, quantization, checkpoint) and, for the
     teacher, `terms` (licence, provider terms, date);
   - `reasoning` for each API role, and the provider's own switch for it in `params.extra_body`;
   - `prices.as_of` and one `prices.table` entry per API model, **with its `provider`**: J3 refuses
     a price that names no provider, and a call whose provider differs from its price's;
   - `modal.gpu_prices` (GPU, CPU core and GiB of memory per second, with `as_of`);
   - `selection.footprint_gb` and `selection.triage`.
4. **Data and preprocessing:** `bench data`, then `bench preprocess --db <id>` for every database.
5. **The two SLM candidates on Modal**, each once:
   ```sh
   BENCH_CANDIDATE=<name> modal run -m modal_apps.serve_vllm::download
   BENCH_CANDIDATE=<name> modal deploy -m modal_apps.serve_vllm
   ```
   Set the candidate's `endpoint.base_url` to the printed URL + `/v1`.

**A served candidate bills its GPU while a container is up.** Deployed as above it scales to zero
when idle. Add `BENCH_MIN_CONTAINERS=1` to the deploy only while runs are using the server (a cold
start answers 503), and deploy without it again afterwards.

## Day 1

### 1. Stage 0: the rules are public before the pilot is read

```sh
git add config.yaml SPEC.md
git commit -m "Stage 0: the configuration and the verdict rules"
git push origin main
git rev-parse HEAD > prereg/STAGE0
ots stamp prereg/STAGE0
git add prereg/STAGE0 prereg/STAGE0.ots
git commit -m "Stage 0: timestamp"
git push origin main
gh api -X POST repos/{owner}/{repo}/rulesets --input - <<'EOF'
{"name": "main is append-only", "target": "branch", "enforcement": "active",
 "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
 "rules": [{"type": "non_fast_forward"}, {"type": "deletion"}]}
EOF
```

`ots` is the OpenTimestamps client (`pip install opentimestamps-client`). The stamped commit holds
the whole repository: the configuration, the SPEC and the code that computes every verdict. After
it, `config.yaml` changes only as step 12 lists, and the analysis code does not change at all;
step 12 checks both.

### 2. Preflight

```sh
bench preflight
```

- **`engines` must pass** for every engine at every temperature: text, `usage`,
  `finish_reason: stop`, reasoning in the declared state. A failed check prints its own action.
- Four checks cannot pass yet, and nothing waits on them here:
  - `agent_end_to_end` and `call_sites_registered` count only B0 runs made with the configuration
    as it is now. Read them in a `bench preflight` run right after step 4's B0 run, before
    `config.yaml` is edited again;
  - `pilot_spend` stays pending: step 5 decides it by hand;
  - `training_time` and P-4 are decided in step 9.

### 3. Smoke test: two questions per arm, configuration only

```sh
bench run --arm B0 --split train --limit 2
bench run --arm B1 --split train --limit 2 --config configs/b1-k0.yaml
bench run --arm B2 --split train --limit 2 --engine production_llm
bench run --arm B2 --split train --limit 2 --engine cheap_alt
bench replay <the B0 run> --engine slm:<candidate>
```

Run the last line once per candidate. In each run's `calls.jsonl`, every line has
`usage.source: api`, `usage.cached_input` not null, `finish_reason: stop` and `parsed_ok: true`. A
line that does not is a configuration to fix, never a result. This test says nothing about time or
cost.

### 4. The pilot (calibration split)

```sh
PILOT=($(python -c "from bench.contracts.config import load_config; from bench.data import pilot_ids; print(*pilot_ids(load_config('config.yaml')))"))
bench run --arm B0 --split calib --workers 8
bench eval <B0 calib>
bench eval <B0 calib> --per-call
bench call-sites <B0 calib>
bench preflight
bench replay <B0 calib> --engine slm:<candidate> --workers 8
bench eval <each zero-shot replay> --per-call
bench judge j6 --zeroshot <replay>=<its eval> --zeroshot <replay>=<its eval> --teacher-eval <B0 per-call eval>
```

- `--workers` is the number of questions answered at once: choose it by the provider's rate limit.
  On a replay it is the number of questions resent at once; it changes no output, only how full the
  SLM server is kept.
- The replay and its eval run once per candidate (zero-shot).
- J6 prints the choice fact. Write its path into `arms.B3.choice`, `arms.B4.choice` and
  `arms.B5.choice`, and set `cost.slo_from` to `<B0 calib>`: the SLO is the stricter of the cap and
  that run's p95 per call.
- Stop the candidate J6 did not choose: `modal app stop slm-bench-serve-<name>`.

Then, on the pilot questions:

```sh
bench run --arm B3 --split calib --ids "${PILOT[@]}"
bench eval <B3 pilot>
bench replay <B0 calib> --engine production_llm --ids "${PILOT[@]}" \
      --call-sites agent_ir agent_ss agent_cg extract_keywords filter_column select_tables select_columns
```

The last replay is the teacher against itself: it ties the agreement bar to the teacher (J7's
`--teacher-self-replay`). Its `--call-sites` are every call site except the two with gold,
`generate_candidate` and `revise`.

### 5. The projection: does the rest fit the budget and the two days?

```sh
bench judge j3 --run <B0 calib> --eval <its eval>
```

`per_question.standard` in the result is the teacher's USD per question.

| Remaining execution | Questions | Engine |
|---|---|---|
| K1, the teacher on train | 834 | production LLM |
| B0 on test | 498 | production LLM |
| B5 on test, worst case (every cluster stays on the LLM) | 498 | production LLM |
| B2-production on test | 498, one call each | production LLM |
| B1 on test, the calib replay, the two B1 pilot replays | 498, 200, 50 + 50 | cheap alternative |
| B2-cheap on test | 498, one call each | cheap alternative |

- **Cost:** USD per question × questions, per engine; K1 through `--via-claude-code` makes no API call (D25). For B2 use its smoke
  run's cost per question. The total API spend must stay under the budget (Before Day 1, item 2).
- **Time:** the pilot run's wall time per question (`finished_at` − `started_at`, ÷ 200) × the
  questions left, at the same `--workers`. Through `--via-claude-code`, K1 runs at its own pace (step 6), not the pilot's.
- **Cache:** the share of `usage.cached_input` in the input tokens. If it is zero, the provider is
  not caching: fix that before K1, since it changes every cost of the report.

**The rule, before K1 starts:**

1. What must run is the core of SPEC §7.2: K1 (the teacher's logs), K2 (B0, B1, B2 and B3 on test),
   K3 (S2 to S5), K4 (B4, B5 and the per-call-site evaluation) and K5 (the report).
2. Over the budget or the two days, cut in the order of SPEC §7.3: the extensions, last first, then
   S6's retraining round. Never K2, K5, B5 or the per-call-site evaluation.
3. **If the core alone does not fit, stop here: do not start K1.** What changes then (the budget,
   the teacher's provider, the number of training questions) changes the registered design, and is
   decided before any more is spent.

A call can hold a worker for up to about 105 minutes in the worst case: 7 attempts of up to 900 s
(transport and parse retries share one attempt counter, each with its own budget), plus the
backoff. If a run stalls, Ctrl-C: no further model call is made, what was answered is merged, and
the run ends `interrupted`. On train and calib the questions left can be run as another run (`--ids`).

A run that stopped on an infrastructure failure (an embeddings or network error, a dead worker:
status `failed` or `interrupted`) is resumed on train and calib with `bench run --arm <arm> --split
<split> --workers 8 --resume-from <run_id>`, with the same arm, split, engine and configuration (by
hash; the commit may differ). It is a new run: it copies the questions the source completed (their C1
lines unchanged, every attempt, and their predictions), answers the others from the start (those in
flight or failed included), and ends `done` like any run, so it is the run eval, replay and the
judgments read. Its manifest records the source (`resumed_from`, with the calls the source spent on
questions it did not complete under `discarded`), `resumed_question_ids`, and `sessions`, whose sum is
the run's working time. On test the rule is stricter (step 13).
A replay stopped the same way (`failed` with `harness_errors`, or `interrupted`) is resumed with
`bench replay <source> --engine <engine>|--arm <arm> [--call-sites ...] [--ids ...] --workers 8 --resume-from
<replay run_id>`, with the same source, engine or arm, call sites, ids and configuration: it copies the
invocations that ended (not the questions), sends the others, and records `resumed_from.copied` and `.discarded`. On test, as step 13 says.

### 6. K1: the teacher's logs on train

```sh
bench run --arm B0 --split train --workers 8
bench call-sites <B0 train>
git add registry/call_sites.json
git commit -m "The call sites observed on train and calib"
```

Accept K1 with at least 90% of the questions answered end to end (`predictions.json`).

**The registered run sends K1 through Claude Code (SPEC D25)**: add `--via-claude-code` to the
`bench run` above. It needs the provider's permission for that use. Only K1 may: its
answers become training data and no judgment reads its cost or latency. The manifest records the
CLI's version in `via_claude_code`.

- **In `bash`, with the API key loaded** (`source ~/.config/slm-bench/env`): the harness checks it is
  set, and the wrapper drops it from every `claude` process, so no call goes to the paid API.
- **`--workers 4`.** Speed is not the limit: 4 workers spent a whole usage window in about 25 minutes.
- **The Claude account's usage limit pauses the run**, which tries again every 10 minutes and goes on when
  the window reopens (stderr prints each pause). About 5,300 calls fit in a window, so K1 (about 88,000
  calls) takes about 17 windows: several days. It is one run; Ctrl-C stops it as any run.
- **The same limit covers every other use of that Claude account.** Run K1 when it is not otherwise needed.
- **Check after the first window:** every `calls.jsonl` line `finish_reason: stop`, and the share of
  `parsed_ok: false` on `agent_ss` near the probes' (1 in 94, SPEC §11.3).

**The fallback is the Message Batches API at half price** (`--batch` instead). Each
CHESS step waits for its batch, 2–8 minutes a step, so it needs many `--workers`; run a small `--ids` set
first. Cache reads inside a batch are best-effort.

### 7. B1's few-shot k

Set `arms.B1.few_shot.source_run` to `<B0 train>`. Then, on the pilot questions:

```sh
bench replay <B0 calib> --engine cheap_alt --ids "${PILOT[@]}" --config configs/b1-k0.yaml
bench replay <B0 calib> --engine cheap_alt --ids "${PILOT[@]}" --config configs/b1-k3.yaml
bench eval <each of the two> --per-call
bench judge b1k --k0 <k0 replay>=<its eval> --k3 <k3 replay>=<its eval> --teacher-eval <B0 per-call eval>
```

Write the `k` of the printed `choice.json` into `arms.B1.few_shot.k`. `bench prereg` and the report
both refuse a k the rule did not choose.

### 8. S2 and S3: curate, cluster, write the datasets

```sh
bench curate --source <B0 train>
bench embed --source <curate run>
bench embed --source <B0 calib>
bench judge j5 --curated <curate run> --embed <its embed> --calib-embed <the calib embed>
bench datasets --curated <curate run> --j5 <J5 result>
```

- The curate manifest's `counts.cap` shows what the cap per question and call site left out.
- J5 prints the centroids fact. Write its path into `arms.B4.centroids` and `arms.B5.centroids`.
- More than 10 clusters do not train at once on Modal's Starter plan (10 GPUs): train in two waves.

### 9. S5: train, smallest cluster first (training environment)

```sh
bench train --cluster <the smallest> --on modal
BENCH_MIN_CONTAINERS=1 BENCH_CANDIDATE=<the chosen candidate> modal deploy -m modal_apps.serve_vllm
bench preflight --parity <the smallest>
```

The deploy makes the server list the first adapter, which P-4 needs.

- **P-4 fails:** the action it prints (the serving, or the reserve candidate). Do not train the rest.
- **`training_time` fails** (above `preflight.schedule.train_hours_max`): lower
  `curation.max_per_question_call_site`, then repeat step 8 and this step.

Then the rest, in parallel, overnight:

```sh
for c in <the other clusters>; do bench train --cluster "$c" --on modal & done; wait
BENCH_CANDIDATE=<the chosen candidate> modal deploy -m modal_apps.serve_vllm
```

The last `bench train` prints the adapters fact. Write its path into `arms.B4.adapters` and
`arms.B5.adapters` **before** that deploy: the server serves the adapters the fact names.

#### Training on Tinker (alternative backend)

`--on tinker` trains the same plan on Thinking Machines' Tinker, usually sooner than a Modal GPU. It writes
the same outputs (adapter and manifest, `where: tinker`). The registered configuration trains on Modal, so
using a Tinker adapter in the registered run is recorded in `prereg/DEVIATIONS.md` (SPEC §6).

```sh
uv venv .venv-tinker --python 3.12
uv pip install --python .venv-tinker/bin/python -r env/tinker/requirements.lock -e ".[test]"
TINKER_API_KEY=... bench train --cluster <c> --on tinker
modal volume put slm-bench-adapters train/adapters/<c>/adapter /<adapter_sha256>   # the command it prints
```

- Settings besides config.yaml's `train` section: `configs/train-tinker.yaml`.
- Qwen3-8B only (Tinker's lineup, 2026-11-09). Granite-4.2-8B and Nemotron-3-Nano-4B are refused.
- Tinker sets no alpha, dropout, target modules or revision. Each difference is listed in
  `accept_differences` and recorded in the manifest. An adapter whose exported rank, alpha or modules differ
  from config.yaml is not installed.
- Tinker bills per token. The manifest records the tokens, not a price.
- Before any paid call it fetches the base's config.json and safetensors headers at the pinned revision
  (under 1 MB), not its weights: the export's converter reads only those. They are kept as sparse files of
  the shards' real sizes in `~/.cache/huggingface/bench-headers-only/`, apart from the HF cache. Never load a
  model from there. The manifest records `stats.run.base_snapshot: headers-only`.
- The exported weights stay on Tinker for `export_ttl_s`. Running the same command again collects them. Once
  they expire, it says so; delete the `train/adapters/.tinker/` file it names to train again.
- A run interrupted before the export starts over: there is no mid-run checkpoint. Two processes on the same
  cluster are not locked against each other: run one.

**The probe, before the first full Tinker run (required).** This path has not yet run against the service.
Run one step on the smallest cluster, with a `--config` that extends config.yaml and sets
`train.sft.max_steps: 1`. A run with `max_steps` set is a probe: its adapter goes to
`train/adapters/.probe/<c>/`, never served, never registered, never in the full run's way. Check:

- it finishes: the service accepts the schedule's first step, which warm-up sends at learning rate 0;
- the manifest's `stats.trained`: rank 16, alpha 32 and the seven modules (otherwise it stops before installing);
- no row is over the service's context for Qwen3-8B (checked before the first step).

Still unverified after it: determinism, the revision of Tinker's Qwen3-8B, and vLLM loading the converted
adapter (P-4 checks it on the first full adapter, as for Modal).

**Comparing backends.** Train a cluster on Tinker, move `train/adapters/<c>` out of `train/adapters/`
(serving and registration read everything there), then train the same cluster on Modal. Both manifests
give the same plan hash (`plan_sha256` in Tinker's; `bench.train_tinker.plan_sha256` computes it from
Modal's) and differ in `where`. To compare the adapters, serve each in turn and replay the calib split on
it (`bench replay <B0 calib> --engine slm:<candidate>+lora:<served name>`, then `bench eval --per-call`).
`--on tinker` never replaces an adapter; `--on modal` does.

`bench train --on tinker` needs `TINKER_API_KEY` set in the shell (a key named otherwise, such as
`TINKER_MACHINES_API_KEY`, is not read: export it as `TINKER_API_KEY`).

### RunPod: the inference provider (after K1)

RunPod can replace Modal for **inference**: the SLM servers, the load test's client and P-4's HF-PEFT
reference (`runpod_apps/`). Training stays on Tinker (`bench train --on tinker`, above). The pods run what the
Modal apps run: the serving plan and the `vllm serve` command come from `modal_apps/common.py`, unchanged, the
adapters at `/adapters/<sha256>`, the weights at the pinned revision, offline.

**How it differs from Modal.** A pod bills by the second from creation to termination, CPU and RAM included
(an A100 80GB PCIe Secure pod: $1.79/h on 2026-11-09). There is no scale-to-zero: **`down` when done.** A pod publishes
SSH only. vLLM is reached through an SSH tunnel to `http://127.0.0.1:<port>/v1`, not through RunPod's HTTPS
proxy: the proxy ends every request at 100 s, and an agent call may take 600. vLLM's own key is the auth
behind SSH.

**Apply the configuration after K1, never before:** a running `--resume-from` checks `config.yaml`'s hash.

```sh
python scripts/apply_config_patch.py configs/runpod.patch.yaml           # the diff, nothing written
python scripts/apply_config_patch.py configs/runpod.patch.yaml --write
```

The patch is applied (2026-11-09). It makes five changes:
- `modal.gpu_prices`: RunPod's A100 80GB PCIe price, $0.000497 a second, with CPU and memory at 0 because the
  pod's price includes them;
- `serving.gpu` and `train.gpu`: `A100-80GB`, the reference GPU since 2026-11-09 (L40S Secure was not
  available on RunPod where network volumes are), and the triage's footprints;
- each candidate's `endpoint`: `base_url` at the tunnel's port, `headers_env: null`;
- `serving.unauthenticated: true`;
- a new `runpod` section.

**After the patch, Modal is not a fallback.** The endpoints point to `127.0.0.1` (the tunnels) and Modal's
proxy auth is off (`unauthenticated: true`, no `headers_env`), so `modal deploy` and `--on modal` no longer
match the configuration. To go back to Modal, restore the file and check the hash:

```sh
git checkout <the commit before the patch> -- config.yaml   # or `git checkout -- config.yaml` if not yet committed
python -c "from bench.contracts.config import load_config, config_sha256; print(config_sha256(load_config('config.yaml')))"
```

Its header explains each change. This is a change after Stage 0 to the serving provider, the GPU price
J8 reads, and (with `train_backend: tinker`) the training backend. Before `bench prereg` (step 12), add it
to `prereg/DEVIATIONS.md` and commit it:

```
- 2026-11-xx · provider changed after Stage 0: config.yaml (modal.gpu_prices, the candidates' endpoints,
  serving.unauthenticated, runpod): inference on RunPod (A100 80GB PCIe Secure, $1.79/h, CPU and RAM bundled) instead
  of Modal; adapters trained on Tinker instead of Modal; J8's cost is also reported at Modal's price
  (2026-09-30), as sensitivity
```

For that sensitivity, run J8 again on the same load tests with a `--config` that extends config.yaml and
sets `modal.gpu_prices` back to Modal's values. The patch keeps them in a comment. Report both costs.

**Once, before the first pod** (no pod is created yet):

1. **The API key.** RunPod console > Settings > API Keys, with read/write on pods and network volumes. Export it
   as `RUNPOD_API_KEY` in the shell, never in a file of this repository. No command prints it.
2. **An SSH key** at `runpod.ssh_key` (default `~/.ssh/id_ed25519`, created with `ssh-keygen -t ed25519`).
   Each pod authorizes its `.pub` through `PUBLIC_KEY`, so it need not be registered in the account.
   Host keys are trusted on first use: the first connection to a pod accepts its host key
   (`StrictHostKeyChecking=accept-new`) into a file of its own, `.runpod/known_hosts/<pod id>`, and any later
   change is refused. A pod's IP and port come from RunPod's authenticated API, which is what that first trust
   rests on. Agent forwarding is never on (`ForwardAgent=no`).
3. **vLLM's key as a RunPod secret:** console > Secrets, named `slm-bench-vllm-api-key`
   (`runpod.secrets.vllm_api_key`). Its value is the one exported locally as `SLM_VLLM_API_KEY` (the
   candidates' `api_key_env`). The pods receive it as `VLLM_API_KEY` through
   `{{ RUNPOD_SECRET_slm-bench-vllm-api-key }}`, so this machine never sends it to RunPod.
   **At least 8 characters** (`SECRET_MIN_LENGTH`, `bench/provenance.py`): a shorter value is never
   redacted from a record, and every run on test refuses to start with one (`registry.check_keys`, before
   its intent). Nothing checks it earlier, neither `up` nor a run on train or calib, so choose a long one
   from the start: changing it later means changing the RunPod secret and restarting the servers.
4. **A spend limit** on the RunPod account (Settings > Billing; the default is $80/h) and a prepaid balance
   no larger than this phase needs.
5. **The volume and its data center:**
   ```sh
   bench serve --on runpod setup
   ```
   This creates the network volume `slm-bench` (100 GB) in the data center with the most A100 80GB PCIe
   available now among those that hold network volumes, or in `runpod.datacenter` when one is set. A
   network volume lives in one data center and attaches only to pods created there, so every pod goes to
   that data center. The volume bills per GB-month while it exists. It holds the weights, the adapters and
   the training environment.
6. **The weights,** once per candidate, from a CPU pod that is terminated afterwards:
   ```sh
   bench serve --on runpod download --candidate <name>
   ```

**Serving** (in the training environment: computing the plan reads the candidate's tokenizer, as
`modal deploy` does):

```sh
bench serve --on runpod up --candidate <name>       # prints http://127.0.0.1:<port>/v1
bench serve --on runpod status                      # the pods, $/h, the spend so far, the tunnels
bench serve --on runpod tunnel --candidate <name>   # reopens the tunnel after the laptop slept
bench serve --on runpod down --candidate <name>
bench serve --on runpod down --all                  # every pod, recorded or named slm-bench-*
```

`up` does the following, in order:
- creates the GPU pod;
- uploads this checkout's code and the adapters missing from the volume;
- starts vLLM, which checks its own version, the adapters' bytes and the auth, then warms up;
- opens the local tunnel.

A pod that does not come up is terminated. `.runpod/state.json` records every pod the moment it is created.
Each pod also terminates itself after `runpod.max_pod_hours` (12 h for a server, 8 h for a job; `up --hours`
overrides). That is a backstop, not the plan.

**Cost discipline.** Run `status` before stepping away and `down` when done. `down --all` terminates the pods
this machine recorded first, then any pod the API lists as `slm-bench-*`. It tries every pod whatever fails
before it, prints a warning when the listing itself fails, and exits non-zero if any pod could not be
terminated. `up` refuses to start a second server for a candidate the API already lists one for.

Each pod's name ends with a token unique to its creation, which is also in its environment. When a create's
answer is lost, the retry adopts only the pod with that token, never another of the same name nor one that
exited. If the pods cannot even be listed, the error says a pod may exist: run `down --all`.

At the end of the day, check the console's Pods page, and the network volume if it is no longer needed.

- **Step 3 (smoke) and step 4 (pilot):** `up` each candidate before `bench replay ... --engine slm:<name>`.
  `bench preflight`'s `engines` check calls every candidate, so both must be up while it runs.
- **Step 9:** P-4's reference runs in the serving pod, on its GPU, with the adapter uploaded from
  `train/adapters/` when it is missing:
  ```sh
  bench train --cluster <the smallest> --on tinker
  bench serve --on runpod down --candidate <chosen> ; bench serve --on runpod up --candidate <chosen>
  bench preflight --parity <the smallest> --on runpod
  ```
  A server serves the adapters that exist when it starts, so restart it after training: `up` on a pod
  that is still running only checks (and reopens) its tunnel, and loads no new adapter. No other pod is
  created. The preflight first collects the served generations through the tunnel. Then it stops vLLM in the
  pod and waits until the GPU's memory is free (`nvidia-smi`). It runs the HF-PEFT reference there, in the
  train lock's environment, which the first run installs onto the volume. Finally it starts the server again
  as `up` does and checks its health through the tunnel, even when the reference failed. The server is down
  for the reference and the warm-up after it, so run nothing else against it meanwhile. If the server does
  not come back, the error says so and the pod is still billing: `down`, then `up`. `training_time`
  reads the Tinker manifests (`runpod.train_backend: tinker`; `--train-backend` overrides).
- **After the last `bench train`** (the adapters fact written into `config.yaml`), and again before
  step 13: `down`, then `up`, for the chosen candidate. That `up` is the one that serves every adapter;
  a pod already up would keep the adapters it started with.
  ```sh
  bench serve --on runpod down --candidate <chosen> ; bench serve --on runpod up --candidate <chosen> --hours <h>
  ```
- **Step 10:** `bench loadtest ... --on runpod`. AIPerf runs in a CPU pod in the server's data center. That
  pod reaches vLLM through its own SSH tunnel with a key pair made for this load test, which the server
  authorizes and then removes. The runs come back into `runs/<id>/` with `where: runpod` and `gpu: A100-80GB`,
  the key of the price J8 reads.
  - **The tunnel's overhead.** Before each sweep, a calibration times the same small sample twice: 8 short
    greedy requests at concurrency 1, and one per slot at the sweep's highest concurrency. The first run is
    on the server against vLLM's own port, the second from the client through its tunnel. Each run's
    manifest records both in `tunnel_calibration`, with `overhead_ms` (client minus server, p50, per
    concurrency): the report states that overhead next to J8's latencies.
  - The tunnel runs without compression or QoS marking. OpenSSH has no `TCP_NODELAY` option; the clients
    set it themselves.
  - The manifests also record `server_versions`: the serving image's vllm, torch, transformers and
    flashinfer. P-4's verdict records them too (`serving_versions`), with the reference's own
    (`reference_versions`).
- **The training environment's build lock.** The first job builds the environment on the volume. It holds a
  lock there with its pod id and a heartbeat. A lock without a heartbeat for `runpod.lock_stale_minutes`
  (20) is broken by the next job, and the local side removes at once a lock whose pod no longer runs.

## Day 2

### 10. The load test and the SLM's cost

```sh
bench loadtest --engine slm:<candidate> --source <B0 calib> --on modal
bench loadtest --engine slm:<candidate>+lora:<served name> --source <B0 calib> --on modal
bench judge j8 --loadtest <run> --loadtest <run> --slo-from <B0 calib>
```

- On RunPod (the section after step 9): `--on runpod` instead of `--on modal`, with the candidate up.
- The first load test is the base (B3's engine); the second runs once per adapter.
- Each `bench loadtest` writes one run per concurrency level. J8 takes every run of every engine,
  each as a `--loadtest`.
- J3 prices an arm only with engines J8 measured, and J8 refuses any `--slo-from` other than
  `cost.slo_from`.
- J8 stops if one engine has no load level within the SLO. That is a result: the SLM does not meet
  the latency of the API it replaces.

### 11. S6: the allocation

```sh
bench replay <B0 calib> --arm B4
bench eval <the B4 replay> --per-call
bench replay <B0 calib> --engine cheap_alt
bench eval <the cheap replay> --per-call
bench judge j7 --centroids <fact> --adapters <fact> --cheap-alt <replay>=<eval> --slm <B4 replay>=<eval> \
      --teacher-eval <B0 per-call eval> --j8 <J8 result> --j6 <J6 result> --teacher-self-replay <the self-replay of step 4>
```

J7 prints the allocation fact. Write its path into `arms.B5.allocation`.

### 12. Pre-registration: before the first test question

`config.yaml` is now final. Against the Stage 0 commit it may differ only in:

- what steps 2 and 3 corrected so that every engine answers (endpoints, `params`, `reasoning`);
- the arms' facts (`choice`, `centroids`, `adapters`, `allocation`);
- `arms.B1.few_shot` (`k`, `source_run`) and `cost.slo_from`;
- `curation.max_per_question_call_site`, if step 9 lowered it;
- `configs/runpod.patch.yaml`, if inference moved to RunPod (its line in `prereg/DEVIATIONS.md`, the
  section after step 9).

Check it, keep the output for the report, and register:

```sh
mkdir -p reports
printf 'j8: %s\n' "<J8 result>" > reports/before-registration.yaml
bench verify --plan reports/before-registration.yaml
git diff "$(cat prereg/STAGE0)" -- config.yaml SPEC.md
git diff --stat "$(cat prereg/STAGE0)" -- bench/judge bench/evaluate.py bench/data.py bench/report.py bench/contracts bench/paths.py
git add config.yaml judgments/
git commit -m "The final configuration and the judgments it names"
git status --porcelain
bench prereg
ots stamp prereg/HASH
git add prereg/HASH.ots
git commit -m "Pre-registration: timestamp"
git push origin main
```

- `bench verify` must report 0 divergences: every judgment the arms' facts rest on (J5, J6, J7, J8) is
  what the final configuration and code compute. A value of `config.yaml` changed after one of them
  ran (a price, a threshold) shows here, for free. Found by the report after the test, it would need
  a new registration and the test again.
- **The second `git diff` must print nothing:** it is the analysis code, the files the registration
  freezes. If a defect forced a change to one of them after Stage 0, add a line to
  `prereg/DEVIATIONS.md` before `bench prereg`, with the date, the files and the defect
  (`- 2026-10-03 · analysis code changed after Stage 0: bench/judge/j2.py: <the defect>`), and commit
  it. The report prints that file. Both commits are public and timestamped, so anyone can run the
  same diff.
- `git status --porcelain` must print nothing: `bench prereg` refuses any uncommitted or untracked
  file. `judgments/` holds numbers and ids only, no prompt and no model output.
- `bench prereg` refuses a configuration with no `cost.slo_from`, and a `arms.B1.few_shot.k` that
  is not the stored choice of step 7.
- The registration hashes the whole configuration. Registering earlier would force
  `bench prereg --replace` here, and every test run already made would be superseded.

### 13. K2 and K4: the test split

**Before the first test run, check every item** (nothing here is checked by the harness except the
keys' length and the registry):

- **Balances.** Anthropic (the production LLM: B0, B2-production, B5): about US$ 255 in the worst case,
  B5 with every cluster on the LLM (step 5's table). OpenRouter (the cheap alternative): B1 and
  B2-cheap. OpenAI: the embeddings of retrieval (every agent arm). A balance that runs out mid-run fails that run.
- **Free disk:** the test adds about 5 GB under `runs/`. Leave more than that free.
- **A fresh serving pod with enough `--hours`.** A server terminates itself after `runpod.max_pod_hours`
  (12 h), counted from its creation, not from the test's start. Run `down` and `up --hours <h>` just
  before, with `<h>` covering B3, B4, B5 and the B4 replay with a margin (step 5's time projection), and
  check that it serves every adapter (the RunPod section after step 9).
- **The machine stays awake** for the whole loop: `caffeinate -dims` in another terminal (macOS). A
  sleeping laptop drops the SSH tunnels; `bench serve --on runpod tunnel` reopens one.
- **No merges, pulls, checkouts or other git activity in this checkout while the runs go.** Each run
  commits and pushes to `main` from here. A registry commit waits up to about 30 s for another git's
  lock (`index.lock`), then fails the run. Work on code in another worktree, and merge nothing to `main`
  until the loop is over (a push to origin's main from elsewhere makes the next intent's push fail).
- **The environments:** the runs and `bench eval` in `.venv-agent`; `bench judge`, `bench verify` and
  `bench report` in `.venv-analysis` (the notes at the top).

```sh
set -o pipefail
START=B0   # after a failure: the configuration that failed (B0 B1 B3 B4 B5 B2-production B2-cheap B4-replay)
STEPS=(B0 B1 B3 B4 B5 B2-production B2-cheap B4-replay)
go=
for step in "${STEPS[@]}"; do
  [ "$step" = "$START" ] && go=1
  [ -n "$go" ] || continue
  run=$(case "$step" in
    (B2-production) bench run --arm B2 --split test --engine production_llm ;;
    (B2-cheap)      bench run --arm B2 --split test --engine cheap_alt ;;
    (B4-replay)     bench replay <B0 test> --arm B4 ;;
    (*)             bench run --arm "$step" --split test --workers 8 ;;
  esac | tail -n 1) || { echo "stopped at $step: no run ended (see the error)"; break; }
  git push origin main || { echo "stopped at $step: the push failed"; break; }
  status=$(python -c 'import json, sys; print(json.load(open(sys.argv[1] + "/manifest.json"))["status"])' "$run")
  [ "$status" = done ] || { echo "stopped at $step: $run is $status"; break; }
done
```

- `set -o pipefail` first, or the `| tail` hides a refusal. A run that ends `failed` or `interrupted`
  still exits 0 and prints its directory: the loop reads its manifest's `status`, pushes its manifest
  either way, and stops unless it is `done`.
- The loop stops at the first configuration that does not end `done`, or cannot be pushed, and prints
  where. Read why. If the run raised before its push (exit code not 0), publish what it committed with
  `git push origin main` (the next test run refuses to start while a registry commit is unpushed). Fix
  the cause **outside this checkout** (the code is frozen here: see below), then finish the configuration
  that stopped by hand, push, and run the block again with `START` set to the configuration after it. An
  attempt that did not end is **resumed**; it is run again from its first question only when resuming is
  impossible or a defect affects questions it already completed. The configurations before it are done and are **not** run again.
  - **Resume** (an infrastructure failure: a network error, a dead pod, a Ctrl-C, a killed process): add
    `--resume-from <run_id>` to its command. Allowed only for the latest attempt of that configuration,
    under the same pre-registration and configuration, with the source and this session both on a clean
    tree at the registered code, and with the source's `predictions.json` and `calls.jsonl` still
    hashing to what its manifest recorded when it ended. Anything else is refused before the intent.
    A session killed outright (SIGKILL, a lost machine) never closed its manifest: it stays `running`,
    in `runs/` and in the registry (each test execution commits its manifest as it starts, with its
    intent). It is resumable once its process is gone (its `session.lock` is free): from the questions
    it journaled as they ended (`answers*.jsonl`), with their C1 lines. Nothing recorded the hashes of
    those files, so they are computed first and recorded in the new intent and manifest
    (`unregistered_source_hashes`).
  - **Run again from the first question**, only when the latest attempt cannot be resumed (its outputs
    do not match their hash, its run directory is gone, an intent that was never pushed: the refusal of
    `--resume-from` says why), or a defect affects questions it completed: first add a dated line to
    `prereg/DEVIATIONS.md` naming that attempt, the defect and its direction (SPEC D22), commit and push
    it; then add `--redo-reason "<the defect>"` to its command, and `--redo-affects-completed` when the
    attempt could be resumed but the defect reaches what it completed. A redo of a resumable attempt
    without that flag is refused and points to `--resume-from`. The reason goes into the intent, published
    before the first question, and into the manifest (`redo`: the attempt, whether the defect affects
    completed questions, why it could not be resumed); the earlier attempts stay in the registry and are
    listed in the new manifest (`earlier_attempts`).
- **One completed run per configuration** (type, arm, engine, pre-registration, as the report keys
  it). A second *completed* run of one configuration would leave that arm's comparisons without a
  verdict, so `bench run` and `bench replay` refuse to open it, before its intent and before any call.
  Rerunning the block from `B0` by mistake therefore stops at once at B0, with the run id that already
  completed it.
- **Only the registered code runs on test.** Every test execution refuses a working tree that `git -c
  core.excludesFile=/dev/null status --porcelain` shows anything in (an untracked `judgments/` file
  included; what the repository's `.gitignore` ignores, such as `runs/`, does not count; your own global
  excludes file does not count either way), a `.git/info/exclude` with any entry (empty it before the
  test: `git rev-parse --git-path info/exclude` finds it), a HEAD that differs from the commit
  `prereg/manifest.json` records outside `prereg/` and `registry/`, and execution code (every file of
  `bench/` outside the analysis code, `vendor/chess`, `modal_apps`, `runpod_apps`, `env/`,
  `pyproject.toml`) whose hashes differ from the registered ones. A fix to the code after the
  registration is therefore a new registration (`bench prereg --replace`).
- Every execution's manifest records, when it ends, `end_reason` (why it did not end `done`, or null),
  `predictions_sha256`, `n_predictions` and `calls_sha256` (what it had produced): an interrupted test
  execution is in the registry with all of them.
- Each run pushes its intent to `origin/main` before its first question; a run that cannot push
  does not start. When it ends it **commits** its manifest, and the `git push` after it publishes
  that.
- The last replay is the per-call-site evaluation.
- A defect that needs a new registration: `bench prereg --replace --reason "<the defect>"`. It
  becomes a line of `prereg/DEVIATIONS.md`, which the report prints.

### 14. Evaluation, cost, verification, report

In `.venv-agent` (the SQLite every eval has used):

```sh
bench eval <each test run>
bench eval <each abandoned test attempt that left predictions> --partial
bench eval <B0 test> --per-call
bench eval <the B4 test replay> --per-call
```

Then in `.venv-analysis`:

```sh
bench judge j3 --run <run> --eval <its eval> --j8 <J8 result>
bench judge j3 --run <B0 train>
bench judge j2 --run <B0 test>
bench judge j2 --run <B4 test>
bench judge j2 --replay <the B4 test replay> --replay-eval <its eval> --teacher-eval <B0 test per-call eval>
git push origin main
bench verify --plan plan.yaml
bench report --plan plan.yaml
```

- The first `judge j3` runs once per arm; `--j8` only for the arms with SLM calls (B3, B4, B5).
- `bench eval` and `bench report` need no clean tree (the judgments write `judgments/`), but they read the
  test only at the registered code: HEAD holds the registered commit's tree outside `prereg/`,
  `registry/` and `judgments/`, and the analysis code hashes to the registered one.
- The report lists every attempt of each configuration: how it ended, the defect stated to run it again,
  and n and EX. An attempt abandoned with predictions gets a partial, descriptive EX (on the questions it
  completed, the model's failures counted as wrong; what did not finish or met a harness error left
  out), from its `bench eval --partial`,
  which the plan names under `attempts: {<attempt>: <its partial eval>}`; a plan that leaves one out is
  refused.
- `plan.yaml` only names what the steps above printed; its keys are in the header of
  `bench/report.py`. A test plan with B1 names the `b1k` choice of step 7.
- `bench verify` must report 0 divergences.
- The report reads the registry as `origin/main` holds it, hence the push before it.

## What this runbook does not decide

- **How many `--workers` K1 can run on one machine** under `--batch`: step 6's small run measures it.

- **Parallel training on Modal was never exercised** (nothing paid ran): step 9's loop is the first
  time several training containers write to one volume. If a run fails registering the adapters,
  run that cluster's `bench train` again: it collects the stored result.
- **`pilot_spend` is arithmetic by hand** (step 5), not a check of `bench preflight`.
- **The b1k choice is not recomputed by `bench verify`.** It is stored by content, `bench prereg` and
  the report check its k, and anyone can run `bench judge b1k` again on the published replays.
