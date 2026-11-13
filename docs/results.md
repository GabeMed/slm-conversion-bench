# Results, axis by axis

One page per question a reader may ask: what was compared, how accurate each option is, what it costs to run,
what it costs to build, how fast it answers, how much training data matters, whether the calls it took over
matter, and what none of this shows. Each
section starts with its answer. The registered verdicts are in
[`results/report/report.md`](../results/report/report.md); a two-page version of this page is
[`summary.md`](summary.md).

Labels used below: **registered** (a verdict or number of the pre-registered report), **measured** (computed
from the logged calls or from a load test, outside the registered verdicts), **exploratory** (decided after
test results were seen).

## 1. What was compared

**One agent, one task, the same 498 test questions. Only the model that answers the agent's calls changes.**

The agent is CHESS, an open-source text-to-SQL agent; the task is BIRD. To answer one question the agent makes
about 93 LLM calls, of five kinds:

| Task inside the agent | Share of the calls | What it does |
|---|---|---|
| Column filter | 84% | for one column of the database, answers yes or no: is it relevant to the question? |
| Agent steps | 12% | picks the next tool; a handful of fixed prompts |
| Keyword extraction | 1% | lists the question's keywords |
| Table and column selection | 2% | chooses the tables and columns the SQL will use |
| SQL generation (and revision) | 1% | writes the SQL; this step decides the answer |

The configurations, in words (the protocol's codes in brackets):

| Configuration | Who answers the agent's calls |
|---|---|
| Agent on the large LLM [B0] | Claude Sonnet 5.5, every call. The reference. |
| Hybrid [B5] | Qwen3-8B with LoRA adapters on the column filter and the agent steps (95.5% of calls); Claude Sonnet 5.5 on the rest |
| Agent on the fine-tuned small model [B4] | Qwen3-8B with one LoRA adapter per group of tasks, every call |
| Agent on the small model, no training [B3] | Qwen3-8B as released, every call |
| Agent on a cheaper LLM [B1] | gpt-oss-120b, an open-weight LLM, every call, with three examples in the prompt |
| No agent, large LLM [B2] | one call to Claude Sonnet 5.5 per question |
| No agent, cheaper LLM [B2] | one call to gpt-oss-120b per question |

The two "no agent" rows answer a different question (is the agent needed at all?) and are kept apart from the
agent rows in every table.

## 2. Accuracy of the final answer (registered)

**The hybrid matches the agent on the large LLM. The small model on every call does not.**

Execution accuracy: the share of questions whose SQL returns the right result. Margin of non-inferiority: 5
points. Intervals: 90%, over the questions.

| Configuration | Accuracy | Against the agent on the large LLM | Registered outcome |
|---|---|---|---|
| Agent on the large LLM | 81.7% | — | reference |
| Hybrid | 82.5% | +0.8 [−0.8, +2.4] | non-inferior |
| Agent on a cheaper LLM | 76.7% | −5.0 [−8.0, −2.2] | inconclusive |
| Agent on the fine-tuned small model | 66.7% | −15.1 [−18.1, −12.0] | worse |
| Agent on the small model, no training | 53.2% | −28.5 [−32.3, −24.7] | worse |

| Without the agent | Accuracy | Against the agent on the large LLM | Registered outcome |
|---|---|---|---|
| One call to the large LLM | 83.5% | +1.8 [+0.2, +3.6] | non-inferior |
| One call to the cheaper LLM | 77.3% | −4.4 [−7.4, −1.4] | inconclusive |

- Fine-tuning adds 13.5 points to the small model [+10.0, +16.9].
- One execution per configuration: the intervals cover the choice of questions, not the variation between runs.

## 3. Accuracy per task (measured)

**The small model is close to the large one on the narrow tasks and far on SQL generation.**

Each task was replayed on the large LLM's own inputs from the test, so every model sees the same prompt and no
error accumulates along the agent.

| Task | Training examples | Metric | Small model, no training | Small model, fine-tuned | Large LLM |
|---|---|---|---|---|---|
| Column filter | 11,982 | agreement with the large LLM (Cohen's kappa) | 0.54 | 0.86 | 0.97 (repeated) |
| Column filter | | recall of the columns the large LLM keeps | 0.83 | 0.84 | 0.97 (repeated) |
| Agent steps | 9,164 | same tool chosen | 90.9% | 100% | 100% (repeated) |
| Table selection | 833 | same set of tables | 83.9% | 90.6% | 100% (repeated) |
| Column selection | 833 | same set of columns | 34.9% | 76.1% | 95.0% (repeated) |
| Keyword extraction | 833 | overlap of keywords (Jaccard) | 0.49 | 0.74 | — |
| SQL generation | 826 | execution accuracy | 55.8% | 67.3% | 81.3% |

- "Repeated" is the large LLM asked the same thing twice: the ceiling any model can reach on agreement.
- The hybrid gives the small model the two tasks where it is closest (column filter, agent steps) and keeps the
  rest on the large LLM. The allocation was decided by a registered rule on calibration data, before the test.
- Agent steps is a few fixed prompts; the untrained model already gets most of them right.

## 4. Cost of running (registered)

**Against the agent on the large LLM, the hybrid costs 81–89% less per correct answer. It is not cheaper than
calling the large LLM once with no agent.**

Cost per correct answer, API list prices; the small model priced from a load test of one rented A100 at
US$ 1.79 per hour, from a GPU in full use to one used 20% of the time.

| Configuration | Cost per correct answer |
|---|---|
| Agent on the large LLM | $0.338 |
| Hybrid | $0.038 (GPU in full use) to $0.063 (20% use) |
| Agent on a cheaper LLM | $0.033 |
| Agent on the fine-tuned small model | $0.015 to $0.074 |
| Agent on the small model, no training | $0.009 to $0.045 |

| Without the agent | Cost per correct answer |
|---|---|
| One call to the large LLM | $0.028 |
| One call to the cheaper LLM | $0.0006 |

- Where the saving comes from: the column filter is 86% of the cost of the agent on the large LLM. The hybrid
  moves it to the small model.
- The registered cost claim compares the hybrid with the cheapest option that needs no training and is
  non-inferior to the reference. That option is the single call to the large LLM, so the registered verdict
  on the hybrid's cost is "refuted". Against the agent it replaces, the saving stands.
- The small model's price assumes it answers within a latency limit that the test did not respect (section 6).

## 5. Cost of building the small model (measured)

**Converting the agent cost about US$ 281: US$ 226 to collect the large LLM's outputs and US$ 56 to train the
four adapters. At the hybrid's saving that is paid back in about 1,200 questions.**

| Step | What | Cost |
|---|---|---|
| Collect training data | the agent on the large LLM over the training questions: 82,403 calls | $225.80 (registered) |
| Train adapter: column filter | 11,982 examples, 2 epochs, 99.1 M tokens, 79 min | $43.62 |
| Train adapter: selection and SQL | 2,517 examples, 21.7 M tokens, 22 min | $9.55 |
| Train adapter: keyword extraction | 833 examples, 1.3 M tokens, 7 min | $0.56 |
| Train adapter: agent steps | 9,164 examples, 4.4 M tokens, 61 min | $1.95 |
| **Total** | | **$281.48** |

- Training cost is what the fine-tuning service billed for the four adapters the test used (US$ 0.44 per
  million training tokens). Adapters trained for other purposes (learning curves, repeats) are not included.
- Payback: the hybrid saves US$ 0.225 to 0.246 per question against the agent on the large LLM, so the
  US$ 281 is recovered after about 1,150 to 1,250 questions.
- Not included: the engineering time, the load tests, the GPU while idle.

## 6. Latency (measured)

**On the tasks it took over, the fine-tuned small model answers as fast as the large LLM or faster while its
GPU is within capacity. In the test one GPU was overloaded, and there it was slower.**

**6.1 Per call, with the load under control.** Load test of the served model on one A100, replaying real
prompts of each task at a fixed number of simultaneous calls (500 requests per level). 95th percentile:

| Simultaneous calls on one GPU | Column filter, fine-tuned small model | Agent steps, fine-tuned small model | SQL generation, fine-tuned small model |
|---|---|---|---|
| 1 | 1.6 s | 0.11 s | 9.2 s |
| 8 | 2.0 s | 0.14 s | 11.8 s |
| 16 | 2.6 s | 0.15 s | 16.8 s |
| 32 | 4.5 s | 0.18 s | 27.0 s |
| 64 | 7.9 s | 0.25 s | 47.2 s |
| **Large LLM, same task, in the test** | **2.9 s** | **2.3 s** | **6.9 s** |

- Column filter: faster than the large LLM up to 16 simultaneous calls per GPU, slower beyond.
- Agent steps: ten times faster at any load.
- SQL generation: slower at any load (long outputs). The hybrid leaves it on the large LLM.

**6.2 Per call, as the test ran.** Median / 95th percentile, calls that ended without error:

| Task | Large LLM | Small model in the hybrid | Small model on every call | Cheaper LLM |
|---|---|---|---|---|
| Column filter | 2.0 / 2.9 s | 3.5 / 8.2 s | 6.0 / 22.8 s | 4.1 / 9.2 s |
| Agent steps | 1.2 / 2.3 s | 0.4 / 0.8 s | 0.3 / 0.8 s | 1.9 / 5.3 s |
| SQL generation | 5.1 / 6.9 s | (large LLM) | 8.7 / 13.5 s | 4.8 / 9.8 s |

The agent fires up to 32 calls at once per step and several questions ran in parallel, so the one GPU served
14 to 33 calls at a time on average, with peaks of 96 to 256: past the 16 at which it keeps up with the large LLM.
This is a sizing result, not a property of the model: more GPUs would bring the latency of 6.1 and raise the
cost of section 4.

**6.3 Per question, as the test ran.** From the first call to the last, median / 95th percentile:

| Configuration | Seconds per question |
|---|---|
| Agent on the large LLM | 56 / 86 |
| Hybrid | 53 / 97 |
| Agent on the fine-tuned small model | 60 / 138 |
| Agent on the small model, no training | 54 / 99 |
| Agent on a cheaper LLM | 64 / 121 |
| One call to the large LLM, no agent | 5 / 7 |
| One call to the cheaper LLM, no agent | 4 / 7 |

Even overloaded, the hybrid's median time per question equals the agent's on the large LLM. The number of
parallel questions differed between and within configurations, so these times are indicative.

## 7. How much training data matters (exploratory)

**Every task is still improving at the amount of data we had.**

Adapters trained on 25%, 50% and 100% of each group's examples, measured on the 200 calibration questions.

| Task | Metric | No training | 25% | 50% | 100% | Large LLM |
|---|---|---|---|---|---|---|
| Column filter | Cohen's kappa | 0.59 | 0.82 | 0.85 | 0.87 | 0.97 |
| Column selection | recall | 0.81 | 0.91 | 0.94 | 0.95 | 0.99 |
| Keyword extraction | Jaccard | 0.49 | 0.64 | 0.69 | 0.73 | 0.90 |
| SQL generation | execution accuracy | 54.3% | 56.8% | 60.3% | 61.3% | 68.8% |

- SQL generation had 826 examples, about a quarter of them with a wrong answer from the large LLM. The paper's
  rule of thumb is ten thousand. The failure of the small model on every call is a result at this volume.
- One adapter per point; neighbouring points are within each other's noise. The curves say "not saturated",
  not "by how much".

## 8. Is the agent needed? (registered and exploratory)

**Not for the large models on this task. Yes for the small one.**

| Model | Inside the agent | One call, no agent | Difference |
|---|---|---|---|
| Claude Sonnet 5.5 (test) | 81.7% | 83.5% | the single call is 1.8 points higher |
| gpt-oss-120b (test) | 76.7% | 77.3% | level |
| Qwen3-8B, no training (calibration, exploratory) | 57.0% | 48.0% | the agent adds 9.0 points [4.0, 14.0] |

Breaking the task into narrow steps is what makes the small model usable. For a large LLM on this benchmark
the same steps are redundant, which is why the single call is the cheapest way to this accuracy.

## 9. Does the column filter matter to the answer? (exploratory control)

**Not measurably. With the column filter replaced by a constant, the agent scores the same. The hybrid's result
therefore does not show that the small model does the filter's job; it shows that this agent does not need the
filter with this LLM.**

The control fixed in [`post-hoc-analysis-plan.md`](post-hoc-analysis-plan.md) (P1), run on the 200 calibration
questions on 2026-11-12, one execution each, same GPU, two questions in parallel. In all three the agent steps
run on the small model and selection, keywords and SQL generation on the large LLM, as in the hybrid.

| Column filter answered by | Accuracy | Against the hybrid | Large-LLM cost per question | Seconds per question, median |
|---|---|---|---|---|
| Fine-tuned small model (the hybrid) | 141 / 200 = 70.5% | — | $0.026 | 45 |
| The constant "No" (drop every column) | 139 / 200 = 69.5% | −1.0 [−4.0, +2.0] | $0.026 | 35 |
| The constant "Yes" (keep every column) | 139 / 200 = 69.5% | −1.0 [−3.0, +1.0] | $0.038 | 34 |
| For reference: the agent on the large LLM, calibration | 139 / 200 = 69.5% | | | |

- Both intervals include 0. Under the reading fixed before the runs, the "No" control is uninformative at this
  n (a difference under about 5 points cannot be seen) and the "Yes" control says that removing the step is not
  seen to cost accuracy. Neither shows a drop.
- The filter is 84% of the agent's calls and 86% of its cost on the large LLM. Answering it with a constant
  "No" removes those calls altogether, with no model and no GPU for them, at the same accuracy and the same
  downstream cost. Keeping every column instead raises the large LLM's cost downstream by 46%.
- An earlier estimate from the logs said an always-"No" filter would drop a column the large LLM later used in
  31% of calibration questions. The live control shows no loss; we have not traced which later step restores
  what the answer needs.
- Limits: calibration, not the test (the test barrier accepts only registered code); the calibration gold is
  the uncorrected one; one execution per arm; 200 questions.

What this changes: "a fine-tuned 8B model took 95% of the agent's calls with no loss" is true, and so is "a
constant took 84% of the agent's calls with no loss". The evidence that a fine-tuned small model can do a
narrow task as well as the large LLM is the per-task comparison of section 3, not the end-to-end result.

## 10. What this does not show

- **That the small model's column filter matters to the answer.** The control of section 9 found no
  difference between it and a constant.
- **Anything about the paper's claim at the data volume it recommends**, or about other agents, domains, small
  models or databases unseen in training.
- **Run-to-run variation.** One execution per configuration.
- **A clean execution.** Sessions were resumed, the number of parallel questions changed, one configuration
  was restarted twice, and the report needed a small published patch of the analysis code. All of it is listed
  in [`prereg/DEVIATIONS.md`](../prereg/DEVIATIONS.md).

## 11. Where each number comes from

| Section | Source |
|---|---|
| 2, 4, 8 (test) | [`results/report/report.md`](../results/report/report.md) and `report.json` |
| 3 | per-call replays of the test; tables in [`results/tables/`](../results/tables) |
| 5 | the report (collection); the fine-tuning service's billing matched to each adapter's training manifest (training) |
| 6.1 | the load-test judgment, `judgments/J8/` |
| 6.2, 6.3 | the logged calls of each test execution (release `data-2026-11-12`); method in [`preliminary-cost-latency.md`](preliminary-cost-latency.md) |
| 7, 8 (calibration) | [`preliminary-results.md`](preliminary-results.md), section 4 |
| 9 | [`results/calibration/post-hoc/filter-control/`](../results/calibration/post-hoc/filter-control) (runs, evaluations, the patch) |
