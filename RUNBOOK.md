# Runbook: the two days of the experiment

The order below is the one SPEC.md §7 fixes. Every step names the command, what it must show, and
what to do when it does not. Nothing here is paid until step 2.

- `<...>` is a value a previous step printed (a run id, a result path).
- **Run the blocks in `bash`**, from the repository root, in the agent environment unless a step
  says otherwise. They use bash arrays; zsh splits words differently.
- **Paths written into `config.yaml` are relative to the repository root** (`judgments/J6/<sha>/choice.json`).
  The commands print absolute paths: cut the prefix. The configuration is registered and published.

## Before Day 1 (no paid call)

1. **Accounts and credentials.**
   - In the shell: the variables `config.yaml` names under each role's `endpoint.api_key_env` and
     `endpoint.headers_env`, and the embeddings key. None is written to a file of this repository.
   - In Modal: a token, and the two secrets `modal.secrets` names. `slm-bench-vllm-api-key` holds
     `VLLM_API_KEY`; `slm-bench-proxy-auth` holds the proxy-auth variables of `headers_env`.
2. **A spending limit on each provider account:** a prepaid balance, or the provider's own limit
   where it has one. Together they sum to the API budget (US$ 150). The harness has no spend cap of
   its own; step 5 projects the spend by hand, and this limit is what stops a projection that was
   wrong.
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
bench replay <B0 calib> --engine slm:<candidate>
bench eval <each zero-shot replay> --per-call
bench judge j6 --zeroshot <replay>=<its eval> --zeroshot <replay>=<its eval> --teacher-eval <B0 per-call eval>
```

- `--workers` is the number of questions answered at once: choose it by the provider's rate limit.
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

- **Cost:** USD per question × questions, per engine. For B2 use its smoke run's cost per
  question. The total API spend must stay under **US$ 150**.
- **Time:** the pilot run's wall time per question (`finished_at` − `started_at`, ÷ 200) × the
  questions left, at the same `--workers`. K1 must end on the afternoon of Day 1.
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

### 6. K1: the teacher's logs on train

```sh
bench run --arm B0 --split train --workers 8
bench call-sites <B0 train>
git add registry/call_sites.json
git commit -m "The call sites observed on train and calib"
```

Accept K1 with at least 90% of the questions answered end to end (`predictions.json`).

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

## Day 2

### 10. The load test and the SLM's cost

```sh
bench loadtest --engine slm:<candidate> --source <B0 calib> --on modal
bench loadtest --engine slm:<candidate>+lora:<served name> --source <B0 calib> --on modal
bench judge j8 --loadtest <run> --loadtest <run> --slo-from <B0 calib>
```

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
- `curation.max_per_question_call_site`, if step 9 lowered it.

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

```sh
for arm in B0 B1 B3 B4 B5; do
  bench run --arm "$arm" --split test --workers 8 && git push origin main || break
done
bench run --arm B2 --split test --engine production_llm && git push origin main
bench run --arm B2 --split test --engine cheap_alt && git push origin main
bench replay <B0 test> --arm B4 && git push origin main
```

- The loop stops at the first run that fails or cannot be pushed: read why before going on.

- Each run pushes its intent to `origin/main` before its first question; a run that cannot push
  does not start. When it ends it **commits** its manifest, and the `git push` after it publishes
  that: the next test run refuses to start while a registry commit is unpushed.
- The last replay is the per-call-site evaluation.
- **One completed run per configuration.** A run that died is listed and may be run again; a second
  *completed* run of one configuration leaves that arm's comparisons without a verdict.
- A defect that needs a new registration: `bench prereg --replace --reason "<the defect>"`. It
  becomes a line of `prereg/DEVIATIONS.md`, which the report prints.

### 14. Evaluation, cost, verification, report

```sh
bench eval <each test run>
bench eval <B0 test> --per-call
bench eval <the B4 test replay> --per-call
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
- `plan.yaml` only names what the steps above printed; its keys are in the header of
  `bench/report.py`. A test plan with B1 names the `b1k` choice of step 7.
- `bench verify` must report 0 divergences.
- The report reads the registry as `origin/main` holds it, hence the push before it.

## What this runbook does not decide

- **Parallel training on Modal was never exercised** (nothing paid ran): step 9's loop is the first
  time several training containers write to one volume. If a run fails registering the adapters,
  run that cluster's `bench train` again: it collects the stored result.
- **`pilot_spend` is arithmetic by hand** (step 5), not a check of `bench preflight`.
- **The b1k choice is not recomputed by `bench verify`.** It is stored by content, `bench prereg` and
  the report check its k, and anyone can run `bench judge b1k` again on the published replays.
