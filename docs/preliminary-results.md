# Preliminary results (2026-11-12)

**Status: preliminary, not the registered report.** Five of the six registered arms have finished and
been evaluated on the 498 test questions. B1 has not finished. The registered verdicts come only from
`bench report`, which is run after the last test execution; nothing below is a verdict. Every interval
on this page is exploratory (a paired per-question bootstrap, unadjusted) and is not the registered
analysis. The page was revised after an internal review; section 5 lists what that review found.

## 1. End to end, on the test (n = 498)

| Arm | What it is | EX |
|---|---|---|
| B2, production LLM | one call to Claude Sonnet 5.5, no agent | 416 / 498 = 83.5% |
| B5 | the agent with clusters c0 and c3 on Qwen3-8B + LoRA, c1 and c2 on the production LLM | 411 / 498 = 82.5% |
| B0 | the CHESS agent on the production LLM (the reference) | 407 / 498 = 81.7% |
| B2, cheaper LLM | one call to gpt-oss-120b, no agent | 385 / 498 = 77.3% |
| B4 | the agent with all four clusters on Qwen3-8B + one LoRA adapter per cluster | 332 / 498 = 66.7% |
| B3 | the agent on base Qwen3-8B, no training | 265 / 498 = 53.2% |
| B1 | the agent on gpt-oss-120b with three few-shot examples | not finished |

Paired differences, in percentage points, with the 90% interval the registration uses (paired
bootstrap over questions) and the discordant pairs:

| Comparison | Right only in the first / only in the second | Difference | 90% interval |
|---|---|---|---|
| B5 − B0 | 14 / 10 | +0.8 | [−0.8, +2.4] |
| B4 − B0 | 9 / 84 | −15.1 | [−18.1, −12.1] |
| B4 − B5 | 8 / 87 | −15.9 | [−18.9, −12.9] |
| B4 − B3 | 92 / 25 | +13.5 | [+10.0, +16.9] |
| B2 production − B0 | 18 / 9 | +1.8 | [+0.2, +3.6] |
| B2 cheaper − B0 | 29 / 51 | −4.4 | [−7.4, −1.4] |
| B4 − B2 cheaper | 33 / 86 | −10.6 | not computed |

The registered non-inferiority margin is 5 points. No verdict is stated here.

What these numbers describe, and what limits each reading:

- **All calls on the fine-tuned small model (B4)** scores about 15 points below the agent on the
  production LLM, and about 11 points below a single call to the untrained cheaper LLM. The hard cluster
  (c1) was trained on 2,517 rows, below the paper's rule of thumb of ten thousand; under the registered
  reading (SPEC section 5, row V1/A1) a negative result at that volume does not refute the claim for
  volumes within the rule.
- **Two clusters on the small model (B5)** scores level with the agent on the production LLM.
- **A single call to the production LLM scores at least as high as the whole agent.** SPEC section 3
  states the consequence in advance: if the single call matches the agent, the per-call-site conversion
  loses its object. The calls B5 moves to the small model (the column filter and the agent's steps) may
  therefore have little bearing on the final answer with this LLM, and B5's result cannot by itself
  show that the small model does the work of the large one. No control isolates this yet (for example,
  B5 with the base model, or with a constant, in place of the column-filter adapter).
- **B4 − B3** is not the effect of training alone: the two arms also differ in the order of the
  pipeline's steps (SPEC section 8).
- **One execution per arm.** The engines are not deterministic and the intervals cover the sampling of
  questions only, not the variation between executions.
- **Within B4**, the 127 questions answered in its first session (8 workers) show a larger deficit to
  B0 (−26.0 points) than the 371 answered later with 3 workers (−11.3). The two groups are also
  different databases, so the cause is not identified.

## 2. Per task, on the teacher's inputs (exploratory)

The end-to-end number mixes tasks that were trained on very different amounts of data. This table
separates them. Each row replays the production LLM's own calls from the B0 test run on another engine,
so every engine sees exactly the same input and no error accumulates along the agent's path.

- "Agreement" is agreement with the production LLM's output on the same call; "EX" is execution
  accuracy of the generated SQL against the gold answer.
- "Ceiling" is the production LLM replayed against itself on a sample of 100 questions (for agreement
  rows), or its own EX on the same calls (for EX rows).
- "Training rows" is the number of examples the cluster's adapter saw for that task. Cluster c1 is one
  adapter trained on 2,517 rows across its tasks.
- The gpt-oss-120b column is not available on the test: that replay has not been run.

| Cluster | Task | Training rows | Calls | Base Qwen3-8B | Qwen3-8B + LoRA | Ceiling (production LLM) | Metric |
|---|---|---|---|---|---|---|---|
| c0 | column filter | 11,982 | 38,255 | 88.8% | 97.8% | 99.4% | agreement |
| c1 | table selection | 833 | 498 | 83.9% | 90.6% | 100% | agreement (set of tables) |
| c1 | column selection | 833 | 498 | 34.9% | 76.1% | 95.0% | agreement (exact set) |
| c1 | SQL generation | 826 | 498 | 55.8% | 67.3% | 81.3% | EX on gold |
| c1 | SQL revision | 6 | 2 | 1 / 2 | 0 / 2 | 2 / 2 | EX on gold (noise) |
| c2 | keyword extraction | 833 | 498 | 0.4% | 9.9% | 53.0% | agreement (exact set) |
| c3 | agent steps | 9,164 | 5,478 | 90.9% | 100% | 100% | agreement (same tool) |

Notes that change how the rows should be read:

- **c0** is the only cluster trained above ten thousand rows, and raw agreement overstates it. In the
  form the SPEC fixes for the column filter (same 38,318 calls; the production LLM repeated on 100
  questions):

  | Column filter | Recall of the kept columns | Agreement on the dropped | Cohen's kappa | Raw agreement |
  |---|---|---|---|---|
  | Trivial rule (always "No") | 0.000 | 1.000 | 0.000 | 90.8% |
  | Base Qwen3-8B | 0.826 | 0.902 | 0.536 | 88.7% |
  | Qwen3-8B + LoRA | 0.838 | 0.992 | 0.860 | 97.7% |
  | Production LLM, repeated | 0.967 | 0.997 | 0.966 | 99.4% |

  Training taught the adapter to drop the columns the production LLM drops; its recall of the columns
  the production LLM keeps barely moved and stays well below the production LLM's own repeatability.
- **c1** was trained on 2,517 rows. The adapter improves every task (column selection by 41 points, SQL
  generation by 11), and remains 14 points below the production LLM on SQL generation, the step that
  produces the final answer.
- **c2** has the least data. Exact-set agreement is a harsh metric here: the production LLM agrees with
  itself only 53% of the time. Mean Jaccard overlap with the production LLM's keywords is 0.49 for the
  base model and 0.74 with the adapter.
- **c3** is not evidence about training volume. Its 5,478 calls are 11 distinct prompts with a fixed
  expected output each; the base model already answers 10 of the 11 correctly. A rate over calls is
  not meaningful for this cluster.
- In c1, low data volume and task difficulty are confounded. This table alone cannot say whether more
  data would close the gap; the learning curves below are meant to address that.

The same table on the 200 calibration questions shows the same pattern (c0 97.2%, table selection
89.0%, column selection 73.0%, SQL generation 61.8%, keyword extraction 12.5%, agent steps 100%).

The full table in the form SPEC section 6.1 fixes (primary metric per call site, the trivial rule,
intervals that resample questions, latency and tokens per call) has been computed outside the registered
code; the gpt-oss-120b column is "not run", and two trivial rules (keyword extraction, SQL generation)
are not defined precisely enough in the SPEC to compute.

## 3. Cost and latency (exploratory, recomputed from the logged calls)

| Arm | EX | API cost per question (list price) | Seconds per question, median / p95 |
|---|---|---|---|
| B2, production LLM | 83.5% | US$ 0.023 | 5.0 / 6.9 |
| B2, cheaper LLM | 77.3% | US$ 0.0005 | 3.5 / 6.6 |
| B5 | 82.5% | US$ 0.026 + the small model's serving | 52.5 / 97.2 |
| B0 | 81.7% | US$ 0.277 | 55.6 / 83.3 |
| B4 | 66.7% | the small model's serving only | 59.7 / 136.6 |
| B3 | 53.2% | the small model's serving only | 53.8 / 98.5 |

- The column filter carries about 86% of B0's cost; B5 removes most of it. Priced with the load-test
  cost per request at full utilisation, B5's serving adds about US$ 0.005 per question.
- The single call to the production LLM costs less than B5's remaining API calls alone, at higher or
  equal accuracy.
- Wall time is not comparable across arms: the number of workers differed between and within arms,
  several arms were resumed, and the small model was served on one GPU. The small model's calls ran
  outside the latency limit used to price them (per-call p95 of about 22 s in B4 and 8 s in B5).
- Training cost is not recorded in the adapters' manifests and is not included.

The registered cost comparisons need B1 and are produced by `bench report`.
The full exploratory analysis of cost, latency and capacity is in [`preliminary-cost-latency.md`](preliminary-cost-latency.md).

## 4. What is still missing

- **B1** on the test. The provider serving gpt-oss-120b on the registered route has been returning
  rate-limit and timeout errors; the run is resumed whenever the provider is healthy.
- **The registered report** (`bench report`), with the registered verdicts, cost and latency, after B1.
- **Learning curves** at 25%, 50% and 100% of each cluster's training data, with every metric, for c0,
  c1 and c2. Only the c1 curve is in the registration (SPEC section 6.1); the c0 and c2 curves were
  added after test results were seen and are post hoc. The 25% and 50% adapters for c1 and c2 are trained and not yet measured; the c0 adapters
  are partly trained. The c3 curve is omitted: with 11 distinct prompts and the base model already
  right on 10 of them, a curve would not measure anything.
- **The gpt-oss-120b replay** on the test, for the missing column of the per-task table.
- The per-task table in its registered form, and latency, tokens and cost per task.
- The declaration of the test's execution conditions in `prereg/DEVIATIONS.md`.

## 5. Known problems with this preliminary reading

- The test was executed under constraints that are not yet declared in `prereg/DEVIATIONS.md`: arms
  resumed over several sessions (B0 in 5, B4 in 3, B5 in 11), different numbers of workers between and
  within arms, the small model's arms served from two different GPU instances, and one evaluation run
  twice with the same result. Every attempt is in `registry/test/`.
- B1 lost the answers of its earlier attempts through an operator error and restarted from its first
  questions, without the declaration the SPEC requires before a restart.
- The replays behind section 2 were run before the last registered execution (B1) had finished; SPEC
  section 6.1 places them after.
- Results of finished arms were read while other arms were still running.

## 6. Provenance

| Item | Run |
|---|---|
| B0 | `agent-B0-test-20261111T060814.371294Z` |
| B3 | `agent-B3-test-20261111T012223.392984Z` |
| B4 | `agent-B4-test-20261111T032707.693347Z` |
| B5 | `agent-B5-test-20261111T200809.693765Z` |
| B2, cheaper LLM | `agent-B2-cheap_alt-test-20261111T081959.861177Z` |
| B2, production LLM | `agent-B2-production_llm-test-20261111T144852.703285Z` |
| Replay, base Qwen3-8B | `replay-slm_qwen3-8b-test-20261111T225111.186203Z` |
| Replay, Qwen3-8B + LoRA | `replay-B4-test-20261111T225020.625646Z` |
| Replay, production LLM (100 questions) | `replay-production_llm-test-20261111T202527.888323Z` |

Every test execution, including the attempts that were interrupted and resumed, is listed in
`registry/test/`. The per-task table was computed with the repository's own comparison code
(`bench.judge.j2.compare`) on the per-call evaluations of the runs above, by a script outside the
registered code.
