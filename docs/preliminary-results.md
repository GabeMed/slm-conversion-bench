# Preliminary results (2026-11-12)

**Status: preliminary, not the registered report.** Five of the six registered arms have finished and
been evaluated on the 498 test questions. B1 has not finished. The registered verdicts come only from
`bench report`, which is run after the last test execution; nothing below is a verdict. Every interval
on this page is exploratory (a per-question bootstrap or a Wilson interval, unadjusted, 95%) and is not
the registered analysis.

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

Paired differences, in percentage points (the registered non-inferiority margin is 5):

| Comparison | Difference | Exploratory 95% interval |
|---|---|---|
| B5 − B0 | +0.8 | [−1.2, +2.8] |
| B4 − B0 | −15.1 | [−18.7, −11.6] |
| B4 − B5 | −15.9 | [−19.5, −12.4] |
| B4 − B3 | +13.5 | [+9.4, +17.7] |
| B2 production − B0 | +1.8 | [−0.2, +4.0] |
| B2 cheaper − B0 | −4.4 | [−7.8, −0.8] |

Preliminary reading:

- Replacing every call with the fine-tuned small model (B4) loses about 15 points against the
  production LLM.
- Replacing only clusters c0 and c3 (B5) stays within the margin of the production LLM.
- Fine-tuning adds about 13 points over the base small model, and that is not enough to close the gap.
- A single call to the production LLM is level with the whole agent on this test.

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

- **c0** is the only cluster trained above ten thousand rows on a non-trivial task, and the adapter
  comes within two points of the ceiling.
- **c1** was trained on 2,517 rows. The adapter improves every task (column selection by 41 points, SQL
  generation by 11), and remains 14 points below the production LLM on SQL generation, the step that
  produces the final answer.
- **c2** has the least data. Exact-set agreement is a harsh metric here: the production LLM agrees with
  itself only 53% of the time. Mean Jaccard overlap with the production LLM's keywords is 0.49 for the
  base model and 0.74 with the adapter.
- **c3** is not evidence about training volume. Its 5,478 calls are 11 distinct prompts with a fixed
  expected output each; the base model already answers 10 of the 11 correctly. Intervals over calls are
  meaningless for this cluster.
- In c1, low data volume and task difficulty are confounded. This table alone cannot say whether more
  data would close the gap; the learning curves below are meant to address that.

The same table on the 200 calibration questions shows the same pattern (c0 97.2%, table selection
89.0%, column selection 73.0%, SQL generation 61.8%, keyword extraction 12.5%, agent steps 100%).

## 3. What is still missing

- **B1** on the test. The provider serving gpt-oss-120b on the registered route has been returning
  rate-limit and timeout errors; the run is resumed whenever the provider is healthy.
- **The registered report** (`bench report`), with the registered verdicts, cost and latency, after B1.
- **Learning curves** at 25%, 50% and 100% of each cluster's training data, with every metric, for c0,
  c1 and c2. The 25% and 50% adapters for c1 and c2 are trained and not yet measured; the c0 adapters
  are partly trained. The c3 curve is omitted: with 11 distinct prompts and the base model already
  right on 10 of them, a curve would not measure anything.
- **The gpt-oss-120b replay** on the test, for the missing column of the per-task table.
- Latency, tokens and cost per task, alongside the accuracy figures above.

## 4. Provenance

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
