# Results data

The data behind the tables of this benchmark: what each execution answered, how it was evaluated, the tables
computed from them and the scripts that computed the tables. Everything here is derived from the runs listed in
`registry/test/` (test split) and from calibration runs.

The registered verdicts come only from `bench report`. Files under `tables/` and `analysis/` are exploratory
unless a registered judgment says otherwise; `docs/post-hoc-analysis-plan.md` says which runs were decided
after test results were seen.

## Layout

- `test/<run id>/`: for each finished test execution and replay, its `manifest.json` and, for agent runs,
  `predictions.json` (the SQL each question got).
- `test/eval-<run id>-<time>/`: the evaluation of that run: `manifest.json` and `results.jsonl`, one line per
  question (predicted SQL, gold SQL, whether the results match). `eval-per-call-…` scores each SQL-generation
  call of a run or replay.
- `calibration/`: the same for the calibration split (200 questions).
- `calibration/post-hoc/`: runs made on 2026-11-12 from a scratch checkout: the learning-curve replays
  (adapters trained on 25% and 50% of a cluster's rows, and the registered adapters), the base model replayed
  on the same server, and the single call on the small model (base, and with the c1 adapter).
- `tables/`: the per-call-site table in the form SPEC 6.1 fixes (test and calibration), the learning curves,
  the single call on the small model, cost and latency tables, and the list of every test attempt.
- `analysis/`: the scripts that produced the tables. They were run outside the registered code; paths in
  them are placeholders (`<repo>`, `<scratch-worktree>`).

## Not here

- The call logs (`calls.jsonl`, every prompt and response; tens of gigabytes) and the routing logs.
- The LoRA adapters and their training sets.
- B1 (the agent on the cheaper LLM): unfinished when this directory was written.

## Licences and sources

Questions, gold SQL and database content come from BIRD (CC BY-SA 4.0) and from the corrected Mini-Dev
variant named in `SPEC.md` section 2; files that embed them are shared under the same terms. The predicted SQL
was produced by the models named in `config.yaml`.
