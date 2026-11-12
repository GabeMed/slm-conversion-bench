# Post hoc analysis plan (2026-11-12)

Everything on this page was decided on 2026-11-12, after the test results of B0, B2 (both engines), B3, B4 and
B5 were known and after an internal review of them. None of it enters a registered verdict; the registered
analysis is the one in `prereg/` and `SPEC.md`. This page fixes, before the runs it describes, which contrast is
primary and how it will be read, so that each reading is chosen before its numbers exist. The commit that adds
this page gives it its date.

Intervals: paired bootstrap over questions, the registered seed and number of resamples, two-sided 90%,
unadjusted for multiplicity. Six contrasts are named as primary below; a reader may apply Bonferroni at
0.10 / 6.

## State when this page was written

Already run, before this page existed (reported as exploratory, with that fact stated):
- the single call on the small model on calibration (P2), both variants;
- the learning-curve replays of c1 and c2 on calibration (P4).

Not yet run: P1 and P3. P5 is not funded and may not run.

## P1. Does the column filter matter to the answer? (a control for B5)

Why: a single call to the production LLM scores at least as high as the whole agent, so B5 matching B0 could
mean only that the calls B5 moves to the small model do not affect the answer.

- Split: calibration (n = 200). The test split is not available for this: the test barrier accepts only the
  registered code, and a constant in place of a model is new code. Calibration has a different mix of
  difficulty and databases from the test, which is a limit of this control and is reported with it.
- Arms, same workers, same GPU instance, run one after another in one session: B5 as registered; B5 with every
  column-filter call answered "No" without a model (P1-no); B5 with every column-filter call answered "Yes"
  (P1-yes). In all three, c3 runs on its adapter and c1 and c2 on the production LLM.
- Primary contrasts: P1-no − B5 and P1-yes − B5, execution accuracy, paired.
- Reading fixed in advance:
  - P1-no: an interval entirely below 0 means the column filter matters to the answer, so B5's result is
    informative about cluster c0. An interval that includes 0 is uninformative at this n (differences smaller
    than about 5 points cannot be seen); it does not show that the filter is irrelevant.
  - P1-yes: an interval that includes 0 means that removing the step is not seen to cost accuracy with this
    LLM; beside it, the cost of the downstream calls with the full schema is reported.
- Only if P1-no shows a drop: the same run with the base model at the column filter, same contrast.
- One execution of each; no repeat on seeing a result.

## P2. The single call on the small model

- Arm: one call per question with the registered single-call prompt, on base Qwen3-8B, calibration (n = 200).
- Primary contrast: P2 − B3 (the agent on the same base model), execution accuracy, paired. An interval
  entirely below 0 means the agent's decomposition helps the small model.
- Secondary, descriptive: the same call with the existing c1 adapter (trained on the agent's prompts) against
  B4, and against the single call on the cheaper LLM.

## P3. Run-to-run variation of the all-SLM arm

- B4 on calibration twice more, once at 3 workers and once at 8, same GPU instance type.
- Quantities, descriptive: the share of questions whose outcome differs between executions, and the difference
  in execution accuracy between 3 and 8 workers. Reported beside B4 − B0.
- Runs only if GPU budget remains after P1.

## P4. Learning curves

- Registered (SPEC 6.1): c1 at 25 / 50 / 100% of its rows by replay on calibration; the registered quantity
  (100% against 25% at SQL generation, execution accuracy) is reported first, under its registered reading rule.
- Post hoc: c2 and c0 at the same points, with the primary metric of each call site as SPEC 6.1 defines it.
  c3 is not run: its calls are 11 distinct prompts with a fixed expected output.

## P5. Training volume at SQL generation (not funded)

- Adapters for the single call trained on BIRD-train gold SQL at nested sizes (826, 2,500, 9,400 rows), the
  registered recipe, plus a label control: the teacher rows whose SQL is correct. Scored as single calls.
- Primary quantity: execution accuracy at each size and the contrast between the largest and the smallest.
- Limits stated in advance: gold labels are not the agent's logs; the training databases are not the test's;
  9,400 is still below ten thousand; one recipe and one seed.

## Replicates that already exist and are reported as such

- The production LLM's repetition of its own calls on 100 test questions.
- B1's abandoned attempt (192 predictions) beside the final B1 on the same questions.
