# Runbook: the two days of the experiment

The order below is the one SPEC.md §7 fixes. Every step names the command, what it must show, and
what to do when it does not. Nothing here is paid until step 3.

`<...>` is a value a previous step printed (a run id, a result path). Commands run from the
repository root, in the agent environment unless a step says otherwise.

## Before Day 1 (no paid call)

1. **Accounts and credentials.** The variables `config.yaml` names under each role's
   `endpoint.api_key_env`, the embeddings key, a Modal token and a Hugging Face token. Export them
   in the shell; none is written to a file of this repository.
2. **Fill every `before runs` value of `config.yaml`:**
   - each API role's `endpoint.base_url`, `provider` (name, quantization, checkpoint) and, for the
     teacher, `terms` (licence, provider terms, date);
   - `reasoning` for each API role, and the provider's own switch for it in `params.extra_body`;
   - `prices.as_of` and one `prices.table` entry per API model, **with its `provider`**: J3 refuses
     a price that names no provider, and a call whose provider differs from its price's;
   - `modal.gpu_prices` (GPU, CPU core and GiB of memory per second, with `as_of`);
   - `selection.footprint_gb` and `selection.triage`.
3. **Data and preprocessing:** `bench data`, then `bench preprocess --db <id>` for every database.
4. **The two SLM candidates on Modal**, each once:
   ```sh
   BENCH_CANDIDATE=<name> modal run -m modal_apps.serve_vllm::download
   BENCH_MIN_CONTAINERS=1 BENCH_CANDIDATE=<name> modal deploy -m modal_apps.serve_vllm
   ```
   Set the candidate's `endpoint.base_url` to the printed URL + `/v1`.

## Day 1

### 1. Stage 0: the rules are public before the pilot is read

```sh
git add config.yaml SPEC.md && git commit -m "Stage 0: the configuration and the verdict rules" && git push origin main
git rev-parse HEAD > prereg/STAGE0 && ots stamp prereg/STAGE0      # opentimestamps-client
git add prereg/STAGE0 prereg/STAGE0.ots && git commit -m "Stage 0: timestamp" && git push origin main
gh api -X POST repos/{owner}/{repo}/rulesets --input - <<'EOF'
{"name": "main is append-only", "target": "branch", "enforcement": "active",
 "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
 "rules": [{"type": "non_fast_forward"}, {"type": "deletion"}]}
EOF
```

After this commit, `config.yaml` changes only where a rule already in it computes the value, or
where an arm's fact is written (step 12 lists them).

### 2. Preflight

```sh
bench preflight
```

`engines` must pass for every engine at every temperature: text, `usage`, `finish_reason: stop`,
reasoning in the declared state. A failed check prints its own action. `pilot_spend` stays pending:
step 5 decides it. `training_time` and P-4 are decided in step 9.

### 3. Smoke test: two questions per arm, configuration only

```sh
bench run --arm B0 --split train --limit 2
bench run --arm B1 --split train --limit 2 --config configs/b1-k0.yaml
bench run --arm B2 --split train --limit 2 --engine production_llm
bench run --arm B2 --split train --limit 2 --engine cheap_alt
bench replay <the B0 run> --engine slm:<candidate>          # once per candidate
```

In each run's `calls.jsonl`, every line has `usage.source: api`, `usage.cached_input` not null,
`finish_reason: stop` and `parsed_ok: true`. A line that does not is a configuration to fix, never
a result. This test says nothing about time or cost.

### 4. The pilot (calibration split)

```sh
PILOT=$(python -c "from bench.contracts.config import load_config; from bench.data import pilot_ids; print(*pilot_ids(load_config('config.yaml')))")
bench run --arm B0 --split calib --workers 8                 # the teacher on calib; workers by the provider's rate limit
bench eval <B0 calib> && bench eval <B0 calib> --per-call
bench call-sites <B0 calib>
bench replay <B0 calib> --engine slm:<candidate>             # zero-shot, once per candidate
bench eval <each zero-shot replay> --per-call
bench judge j6 --zeroshot <replay>=<its eval> --zeroshot <replay>=<its eval> --teacher-eval <B0 per-call eval>
#   -> write the choice fact's path into arms.B3.choice, arms.B4.choice, arms.B5.choice
bench run --arm B3 --split calib --ids $PILOT && bench eval <B3 pilot>
bench replay <B0 calib> --engine production_llm --ids $PILOT \
      --call-sites agent_ir agent_ss agent_cg extract_keywords filter_column select_tables select_columns
```

The last replay is the teacher against itself: it ties the agreement bar to the teacher (J7's
`--teacher-self-replay`). Its `--call-sites` are every call site except the two with gold,
`generate_candidate` and `revise`.

Set `cost.slo_from` to `<B0 calib>`: the SLO is the stricter of the cap and that run's p95 per call.

### 5. The projection: does the rest fit the budget and the two days?

From the pilot's B0 run:

```sh
bench judge j3 --run <B0 calib> --eval <its eval>            # per_question.standard = USD per question
```

| Remaining execution | Questions | Engine |
|---|---|---|
| K1, the teacher on train | 834 | production LLM |
| B0 and B2-production on test | 498 each | production LLM |
| B1 and B2-cheap on test, the calib replay, the two B1 pilot replays | 498, 498, 200, 50 + 50 | cheap alternative |

- **Cost:** USD per question × questions, per engine; B2 is one call per question (use its smoke
  run's cost per question). The total API spend must stay under **US$ 150**.
- **Time:** (`finished_at` − `started_at`) of the pilot run ÷ its questions × the questions
  left ÷ the workers. K1 must end on the afternoon of Day 1.
- **Cache:** the share of `usage.cached_input` in the input tokens. If it is zero, the provider is
  not caching: fix that before K1, since it changes every cost of the report.

Over the budget or the two days: apply the cuts of SPEC §7.3 **before** K1. Never cut K2, K5, B5 or
the per-call-site evaluation.

### 6. K1: the teacher's logs on train

```sh
bench run --arm B0 --split train --workers 8
bench call-sites <B0 train>
git add registry/call_sites.json && git commit -m "The call sites observed on train and calib"
```

Accept K1 with at least 90% of the questions answered end to end (`predictions.json`).

### 7. B1's few-shot k

Set `arms.B1.few_shot.source_run` to `<B0 train>`. Then, on the pilot questions:

```sh
bench replay <B0 calib> --engine cheap_alt --ids $PILOT --config configs/b1-k0.yaml
bench replay <B0 calib> --engine cheap_alt --ids $PILOT --config configs/b1-k3.yaml
bench eval <each of the two> --per-call
bench judge b1k --k0 <k0 replay>=<its eval> --k3 <k3 replay>=<its eval> --teacher-eval <B0 per-call eval>
```

Write the `k` of the printed `choice.json` into `arms.B1.few_shot.k`. The report refuses a k the
rule did not choose.

### 8. S2 and S3: curate, cluster, write the datasets

```sh
bench curate --source <B0 train>                 # the manifest's counts.cap shows what the cap left out
bench embed --source <curate run>
bench embed --source <B0 calib>
bench judge j5 --curated <curate run> --embed <its embed> --calib-embed <the calib embed>
#   -> write the centroids fact's path into arms.B4.centroids and arms.B5.centroids
bench datasets --curated <curate run> --j5 <J5 result>
```

More than 10 clusters do not train at once on Modal's Starter plan (10 GPUs): train in two waves.

### 9. S5: train, smallest cluster first (training environment)

```sh
bench train --cluster <the smallest> --on modal
bench preflight --parity <the smallest>          # P-4, and training_time projects the largest dataset
```

- **P-4 fails:** the action it prints (the serving, or the reserve candidate). Do not train the rest.
- **`training_time` fails** (above `preflight.schedule.train_hours_max`): lower
  `curation.max_per_question_call_site`, then repeat step 8 and this step.

Then the rest, in parallel, overnight:

```sh
for c in <the other clusters>; do bench train --cluster $c --on modal & done; wait
```

Write the adapters fact's path into `arms.B4.adapters` and `arms.B5.adapters`, and deploy the server
again (`modal deploy -m modal_apps.serve_vllm`): it serves the adapters that fact names.

## Day 2

### 10. The load test and the SLM's cost

```sh
bench loadtest --engine slm:<candidate> --source <B0 calib> --on modal                       # the base, B3's engine
bench loadtest --engine slm:<candidate>+lora:<served name> --source <B0 calib> --on modal   # once per adapter
bench judge j8 --loadtest <run> --loadtest <run> ... --slo-from <B0 calib>
```

Each `bench loadtest` writes one run per concurrency level; J8 takes them all. J3 prices an arm only
with engines J8 measured, and J8 refuses any `--slo-from` other than `cost.slo_from`.

### 11. S6: the allocation

```sh
bench replay <B0 calib> --arm B4 && bench eval <it> --per-call
bench replay <B0 calib> --engine cheap_alt && bench eval <it> --per-call
bench judge j7 --centroids <fact> --adapters <fact> --cheap-alt <replay>=<eval> --slm <B4 replay>=<eval> \
      --teacher-eval <B0 per-call eval> --j8 <J8 result> --j6 <J6 result> --teacher-self-replay <the self-replay of step 4>
#   -> write the allocation fact's path into arms.B5.allocation
```

### 12. Pre-registration: before the first test question

`config.yaml` is now final. Against the Stage 0 commit it may differ only in: the arms' facts
(`choice`, `centroids`, `adapters`, `allocation`), `arms.B1.few_shot` (`k`, `source_run`),
`cost.slo_from`, and the SLM endpoints. Check it, and keep the output for the report:

```sh
git diff $(cat prereg/STAGE0) -- config.yaml SPEC.md
git commit -am "The final configuration"
bench prereg && ots stamp prereg/HASH
git add prereg/HASH.ots && git commit -m "Pre-registration: timestamp" && git push origin main
```

The registration hashes the whole configuration. Registering earlier would force
`bench prereg --replace` here, and every test run already made would be superseded.

### 13. K2 and K4: the test split

```sh
for arm in B0 B1 B3 B4 B5; do bench run --arm $arm --split test --workers 8; done
bench run --arm B2 --split test --engine production_llm
bench run --arm B2 --split test --engine cheap_alt
bench replay <B0 test> --arm B4                   # the per-call-site evaluation
```

- Each run pushes its intent to `origin/main` before its first question, and its manifest when it
  ends. A run that cannot push does not start.
- **One completed run per configuration.** A run that died is listed and may be run again; a second
  *completed* run of one configuration leaves that arm's comparisons without a verdict.
- A defect that needs a new registration: `bench prereg --replace --reason "<the defect>"`. It
  becomes a line of `prereg/DEVIATIONS.md`, which the report prints.

### 14. Evaluation, cost, verification, report

```sh
bench eval <each test run>                        # and --per-call for <B0 test> and the B4 replay
bench judge j3 --run <run> --eval <its eval> [--j8 <J8 result>]     # once per arm, and for <B0 train> without --eval
bench judge j2 --run <B0 test> && bench judge j2 --run <B4 test>    # format validity
bench judge j2 --replay <B4 test replay> --replay-eval <its eval> --teacher-eval <B0 test per-call eval>
bench verify --plan plan.yaml                     # must report 0 divergences
bench report --plan plan.yaml
```

`plan.yaml` only names what the steps above printed; its keys are in the header of
`bench/report.py`. A test plan with B1 names the `b1k` choice of step 7.

## What this runbook does not decide

- **Parallel training on Modal was never exercised** (nothing paid ran): step 9's loop is the first
  time several training containers write to one volume. If a run fails registering the adapters,
  run that cluster's `bench train` again: it collects the stored result.
- **`pilot_spend` is arithmetic by hand** (step 5), not a check of `bench preflight`.
- **The b1k choice is not recomputed by `bench verify`.** It is stored by content, the report checks
  its k and its questions, and anyone can run `bench judge b1k` again on the published replays.
