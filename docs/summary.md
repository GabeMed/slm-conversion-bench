# Which LLM calls of a real agent can a fine-tuned SLM take over? A two-page summary

*Small Language Models are the Future of Agentic AI* (Belcak et al., arXiv 2506.02153) argues that most of an
agent's calls to a large model could go to small specialised ones, and outlines a conversion algorithm (S1–S6).
It reports no experiment. We ran the algorithm end to end on one real agent and pre-registered how the result
would be read. This page is the short version; the report is [`results/report/report.md`](../results/report/report.md).

## What we did

- **Agent and task.** CHESS, an open-source text-to-SQL agent, on BIRD. Each question costs about 93 LLM calls;
  84% of them are one narrow task, a yes/no filter over database columns.
- **Conversion.** We logged the agent on Claude Sonnet 5.5 over a training split, curated the logs, clustered
  the calls into four tasks, picked Qwen3-8B from two candidates by a zero-shot score, trained one LoRA adapter
  per cluster on the large model's outputs, load-tested the served model, and let a registered rule decide,
  per cluster, which engine keeps the calls.
- **Test.** 498 held-out questions, the same 11 databases as training. Seven configurations answered them once
  each. Verdict rules, the 5-point non-inferiority margin, seeds and thresholds were registered and
  timestamped before any test result.

## What came out

| Configuration | Execution accuracy | Cost per correct answer |
|---|---|---|
| Hybrid: Qwen3-8B + LoRA on 2 of 4 clusters (95.5% of calls), Sonnet on the rest | 82.5% | $0.038 – $0.063 |
| The agent on Claude Sonnet 5.5 (reference) | 81.7% | $0.338 |
| The agent on gpt-oss-120b, few-shot | 76.7% | $0.033 |
| Qwen3-8B + LoRA on every call | 66.7% | $0.015 – $0.074 |
| Qwen3-8B zero-shot on every call | 53.2% | $0.009 – $0.045 |
| One call to Claude Sonnet 5.5, no agent | 83.5% | $0.028 |
| One call to gpt-oss-120b, no agent | 77.3% | $0.0006 |

Small-model costs run from a fully used rented GPU to one used 20% of the time; API costs are list prices.

1. **A fine-tuned 8B model took 95.5% of the agent's calls with no measurable loss.** The hybrid is
   non-inferior to the agent on the large model (+0.8 points, 90% CI [−0.8, +2.4]) at 81–89% lower cost per
   correct answer.
2. **Converting every call failed.** With all four clusters on the small model the agent lost 15.1 points
   (90% CI [−18.1, −12.0]). Fine-tuning recovered 13.5 of the 28.5 points the zero-shot model lost. Replayed on
   the large model's own inputs, the adapter generates SQL at 67.3% against 81.3%.
3. **The agent itself was not needed for the large models.** One call to the same LLM, with no agent, scored at
   least as high as the agent for Sonnet (+1.8) and level for gpt-oss-120b (+0.6), at a fraction of the cost.
   Against that baseline the hybrid is not cheaper, and the registered cost claim is refuted. The agent did help
   the small model: on calibration, a single call to Qwen3-8B was 9 points below the agent on Qwen3-8B.
4. **Latency depends on sizing.** Within its GPU's capacity (up to 16 simultaneous calls on one A100) the
   fine-tuned small model answers the column filter at a p95 of 1.6 to 2.6 s, against 2.9 s for the hosted LLM,
   and the agent's steps ten times faster. In the test one GPU served 14 to 33 calls at a time and the p95 rose
   to 8 to 22 s. Time per question was the same as the agent on the LLM (median 53 s against 56 s).
5. **Building it cost about US$ 281**: US$ 226 to collect the large model's outputs and US$ 56 to train the four
   adapters, paid back in about 1,200 questions at the hybrid's saving.

## What this does and does not show

- **It does not test the paper at the data volume it recommends.** SQL generation, the step that decides the
  answer, was trained on 826 examples, about a quarter of them with a wrong label from the teacher; the paper's
  rule of thumb is ten thousand. Our learning curves (25%, 50%, 100% of the rows) are still rising at every
  task. The failure of full conversion is a result about this budget.
- **The hybrid's result says more about the task than about the small model.** The calls it took over are the
  column filter and the agent's fixed control steps. We ran the control afterwards, on calibration: with the
  column filter answered by a constant ("no" to every column, or "yes" to every column) the agent scores the
  same as with the fine-tuned model (69.5% against 70.5%, intervals including 0). So the end-to-end result does
  not show that the small model does the filter's job; this agent does not need the filter with this LLM. What
  shows the small model's ability on a narrow task is the per-call comparison (agreement of 0.86 with the large
  model against 0.97 for the large model with itself).
- **One run per configuration**, on engines that are not deterministic; intervals cover the sampling of
  questions only. One agent, one domain, one small model, the same databases in training and test.
- **The execution departed from the protocol** in ways listed in [`prereg/DEVIATIONS.md`](../prereg/DEVIATIONS.md):
  resumed sessions, changing worker counts, one arm restarted twice, and a report produced with a small
  published patch of the analysis code because the registered one could not handle two edge cases.

## What we would like to know next

- Does SQL generation on an 8B model close the gap at ten thousand clean examples, or is it a capacity limit?
- Which calls of an agent actually bear on its answer? A cheap control (replace a call site with a constant)
  seems necessary before any per-call conversion claim.
- When a single LLM call matches an agent, what is the right baseline for "replacing the LLM inside the agent"?

## Everything is public

Protocol and registration: [`SPEC.md`](../SPEC.md), [`prereg/`](../prereg). Report, tables and scripts:
[`results/`](../results/README.md). Per-task results, learning curves, cost and latency:
[`preliminary-results.md`](preliminary-results.md), [`preliminary-cost-latency.md`](preliminary-cost-latency.md).
Every prompt and response of every run, the adapters and the training sets: the `data-2026-11-12` release.
The results axis by axis, each with its source: [`results.md`](results.md).
