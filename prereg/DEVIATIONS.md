# Deviations from the Stage 0 registration

Changes after Stage 0 that can change a result or its interpretation.

- 2026-11-09 · adapters trained on Tinker instead of Modal (the training stack differs; the SLM arms' accuracy
  rests on these adapters); serving on RunPod (same vLLM, weights and GPU, so the outputs are unaffected) at
  RunPod's L40S price instead of Modal's, which changes J8's cost: J8 is also reported at Modal's 2026-09-30
  price, as sensitivity. config.yaml: modal.gpu_prices, the candidates' endpoints, serving.unauthenticated, runpod
- 2026-11-09 · reference GPU changed from L40S to A100 80GB PCIe (RunPod Secure, $1.79/h, CPU and RAM bundled),
  before the zero-shot pilot reading, because L40S was unavailable on RunPod in data centers with network
  volumes; accuracy is unaffected, J8's cost and the SLM's p95 are measured on the A100; J8's sensitivity is
  Modal's price for the same GPU (A100-80GB, $0.000694 a second, 2026-11-09). config.yaml: serving.gpu,
  train.gpu, modal.gpu_prices, runpod.gpu_ids, the triage's footprints
- 2026-11-10 · patch 16: CHESS's agent tool-call parser accepts the tool name inside JSON ({"name": ...} or
  {"tool": ...}), followed by arguments, or closed by </think> (vendor/chess/src/workflow/agents/agent.py); the
  prompt and every other rule are unchanged, for every arm. Upstream accepted only <tool_call>name</tool_call>,
  which non-Claude models (Qwen3-8B, gpt-oss-120b) answered with the right tool in another format and the agent
  then stopped without SQL; plain text, empty tags and unknown tools still fail. B0's calibration pilot and K1 ran
  under the strict parser: 1 of the pilot's 2,198 answered agent calls (agent_cg) and 3 of K1's 9,167 (2 agent_cg, 1
  agent_ss), each a JSON {"name": ...}, would parse under patch 16; no call the strict parser accepted changes
  its tool
- 2026-11-10 · the load test shuffles its payload pool with seeds.loadtest_shuffle (20260935) before the warm-up
  and the concurrency levels are sliced (bench/loadtest.py), decided before any load test ran. In the source's
  file order the six levels (4 to 128, 500 requests each) read only the first 19% of the B0 calibration calls:
  6 to 8 questions and 3 to 4 of the 11 databases per level, 4 databases in all, a different mix at each level;
  shuffled, each level draws 165 to 176 questions from 10 to 11 databases. J8's rule is unchanged.
  config.yaml: seeds.loadtest_shuffle
- 2026-11-10 · B1's few-shot examples at the agent call sites (agent_ir, agent_ss, agent_cg) were sampled from
  any step of the teacher's invocations, and these prompts carry no state, so at its first step cheap_alt was
  shown the teacher's later steps: with k = 3 the agent never called extract_keywords and skipped the steps
  before SQL. Corrected (PR #19, bench/agent/few_shot.py): each agent step takes examples of the same step
  (invocation key), sampled with seeds.few_shot, a shortfall filled from the nearest steps; the tool call sites
  keep their examples and digests, and k = 0 is unchanged. B1's k = 3 pilot execution before the correction
  (agent-B1-calib-20261110T030824.551966Z, EX 7/50) is superseded; an execution's manifest records the sampling
  rule
- 2026-11-10 · analysis code changed after Stage 0: bench/judge/b1k.py. D23's rule for B1's few-shot k keeps its
  form (k = 3 only if the one-sided 95% lower bound of EX_k3 − EX_k0, paired bootstrap, seeds.bootstrap,
  stats.n_boot, is above 0; else k = 0) but its unit changes from EX per call on the gold call sites of two
  replays of the teacher's execution to EX end to end on the pilot questions, from B1's own agent executions
  with k = 0 and k = 3 (on cheap_alt). The change was decided after reading the pilot: with k = 0, 31 of the 50
  pilot questions ended without SQL and EX was 12/50; with the corrected k = 3, 0 without SQL and EX 25/50 (B0:
  28/50). The per-call rule had chosen k = 0 (EX per call 0.519 with k = 0 against 0.500 with k = 3, lower bound
  −0.077): a replay gives every call the teacher's context, so it cannot see the agent stopping before SQL. End
  to end, diff = +0.26 (k = 3 alone right on 14 questions, k = 0 alone on 1), lower bound +0.14 over n = 50:
  k = 3 (config.yaml: arms.B1.few_shot.k). The change strengthens B1, the untrained comparator arm, which makes
  the SLM's cost and quality verdicts against it harder, not easier. The cheap_alt replay of the calibration that J7 reads
  (RUNBOOK step 11) uses the same corrected k = 3; B1 with k = 0 is not run on test (the pilot is its record)
- 2026-11-10 · the training file of the CHESS agent's own call sites (agent_ir, agent_ss, agent_cg; J5's cluster
  c3) is no longer the deduplicated curated examples but a sample that keeps the teacher's frequency of (step,
  answer): step = the invocation key (ir:0..3, ss:0..3, cg:0..2), answer = the tool the agent called or DONE; from
  K1's final attempts that parsed (9,164 of 9,167), with no exact or near deduplication, the same 415 rows c3 had,
  quotas proportional to the teacher's cells by largest remainder, and in each cell the invocations first by the
  cap's seeded rank (seeds.curation_sample), masked by curation.mask (bench/curate.py, `bench datasets`; recorded
  under `agent_sites` in train/datasets/manifest.json; c0, c1 and c2 come out byte-identical). The agent's prompt
  does not depend on the question, so exact dedup kept one row per distinct (prompt, completion) and trained on
  the rare answers: DONE was 0.7% of c3's rows against 27.3% of the teacher's calls, prose answers 96.6% against
  4.6% (the rule: DONE 27.2%, prose 2.9%; 26 distinct rows). The rule only selects rows: curate's examples, the
  embed execution, J5's clusters and centroids and the other clusters' files are unchanged, and every c3 row
  repeats a (prompt, completion) J5 put in c3. The change touches only the trained arms (B4, B5) and tends to
  favour them (the previous file trained on the teacher's rare answers), which is why it is declared. Decided
  after the base model's calibration replay and the B3 pilot had been read (agent call sites included), before any
  adapter was evaluated; the only c3 training so far was a speed measurement on the previous file, outside the
  registry and never evaluated
- 2026-11-10 · the training rows of the column filter and of the CHESS agent's own call sites (agent_ir, agent_ss,
  agent_cg) are no longer the curated examples but drawn from K1's final attempts that parsed, masked by
  curation.mask, with no deduplication and no per-question cap: every agent-site invocation (9,164 of 9,167) and a
  uniform sample of N = 12,000 of the 64,136 column-filter invocations, by call (the smallest sha256 rank under
  seeds.curation_sample = 20260934); each row goes to the cluster of its prompt's nearest centroid (the router's
  assignment at run time; J5's centroids b98b66cd… unchanged), not to J5's cluster of a curated twin
  (bench/curate.py, `bench datasets`; N, the seed and the rule recorded under `routed_sites` in
  train/datasets/manifest.json; every other call site's rows are unchanged). Exact deduplication and the
  per-question cap (curation.max_per_question_call_site) were additions of this project, not the paper's method, and
  on these sites they trained on another distribution than the teacher's: DONE was 0.7% of the agent rows against
  27.3% of the teacher's calls, and the cap, 4 invocations per question whatever the database's width, weighed
  questions rather than calls ("yes" 14.9% of the column-filter rows against 9.4% of the teacher's;
  european_football_2 22.3% of the calls, 8.6% of the rows), with 3,332 column-filter rows, below the paper's
  10k–100k rule of thumb (now: "yes" 9.2%, DONE 27.3%). Curate, the embed execution and J5's clusters and centroids
  are unchanged; c0 (3,292 → 11,982 rows), c1 (2,539 → 2,517: its 40 column-filter rows replaced by the 18 the
  router sends it) and c3 (415 → 9,164) change, c2 is byte-identical (the file its adapter was trained on). This
  replaces the 415-row frequency rule of the line above before any official c3 training with it and before any
  dataset regeneration under it. Decided after the base model's calibration results and the B3 pilot had been read,
  before any c0 or c3 adapter was evaluated (c3 was trained once, as a speed measurement on the earlier file,
  outside the registry and never evaluated). It changes the trained arms' (B4, B5) training data and tends to
  favour them, which is why it is declared
- 2026-11-10 · analysis code changed after Stage 0: bench/judge/b1k.py, bench/judge/j7.py, bench/judge/base.py,
  bench/contracts/facts.py, bench/report.py, bench/prereg.py — refusal guards only, no stored result changes (J3
  37b6afbb…, J5 d6d05ccb…, J6 2a766ca7…, b1k 11dedaac… recompute to the same bytes). b1k refuses a k = 3 execution
  whose few-shot was not drawn by today's sampling rule (manifest few_shot.sampling, the per-step rule), and a
  k = 0 one whose prefix is not empty; J7 refuses a cheap_alt replay whose few-shot is not B1's (arms.B1.few_shot.k,
  now in J7's reads, and the same sampling rule); prereg refuses unless exactly one b1k choice exists, stored by
  its sha256, by today's rule, on today's pilot questions, with the configured k, and unless every fact arms.*
  names (J6's choice, J5's centroids, S5's adapters, J7's allocation; B4's and B5's adapters and B5's allocation
  included, which it only listed as null before) is set, present and tracked by git. A missing J8 or other judgment
  result, a missing fact file, and a report plan whose arm lacks its eval or cost are refused with a message
  instead of a traceback.
- 2026-11-10 · analysis code changed after Stage 0: bench/report.py — Appendix B gets the allocation.min_calls
  floor (100) that A5 and J7 already keep: with fewer repair calls (revise, with gold) than that, its verdict is
  "inconclusive by design", whatever the bootstrap. The calibration replay had 5 revise calls and K1 6, so on this
  workload Appendix B cannot be decided by the repair test. Decided after the revise outcomes were known, before
  any test run: J6 (judgments/J6, written 04:19 and 05:16 UTC, before the floor) and the teacher's per-call eval
  (eval-per-call-agent-B0-calib-20261109T151141.039667Z-20261109T153122.727095Z) give 0 of 5 for both SLM
  candidates replayed and 2 of 5 for the teacher (questions 519 and 689). Direction: the floor stops the repair
  test from deciding against the SLM on these 5 calls, which favours the SLM arms in Appendix B
