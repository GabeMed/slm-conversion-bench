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
- 2026-11-10 · on 2026-11-10 07:25 UTC the #29 rule's datasets were regenerated once by mistake in a working copy
  (the code path, not a decision), never read by any training or evaluation, and replaced by D1's
- 2026-11-09 · `bench run --resume-from` added after Stage 0 (f2f1630, 14:53 UTC): an agent execution that stopped
  on an infrastructure failure continues as a new execution that copies the questions the source completed and
  answers the others; the source's calls on unfinished questions are discarded, and the manifest records
  `resumed_from` (source run_id, commit, dirty, config_sha256, discarded calls) and `sessions`. Three executions
  that results read were resumed:
  - K1 (agent-B0-train-20261109T214533.609401Z, commit 805fad8, dirty): source
    agent-B0-train-20261109T214505.204680Z (failed after 18 s, dirty), 8 calls on 8 questions discarded, none
    billed (0 tokens); 550 of its 834 questions copied, through a chain of 7 sessions on commits 855890e and 805fad8
  - B3's calibration pilot (agent-B3-calib-20261110T051808.826362Z, commit 517f70d, dirty): source
    agent-B3-calib-20261110T050758.509113Z (failed, dirty), 29 calls on 1 question (1518) discarded; 49 of its
    50 questions copied
  - B0's calibration pilot, cost.slo_from (agent-B0-calib-20261109T151141.039667Z, commit 855890e, clean):
    source agent-B0-calib-20261109T141525.618762Z (failed, commit 5f98818, clean), 207 calls on 8 questions
    discarded; 67 of its 200 questions copied. The SLO's p95 (J8's `slo`: latency_ms of first attempts) is
    3,803 ms over 16,140 calls pooled; 4,209 ms over the source session's 5,071 calls; 3,348 ms over the
    resumed session's 11,069. The cap is 17,000 ms, so the SLO is the pooled p95: the pooled SLO is 455 ms
    looser than the resumed session's alone and 406 ms stricter than the source session's alone.
  Direction: neutral for accuracy (only failed sessions' unfinished questions are re-answered); J8's SLO is the
  pooled p95, between the two sessions' p95s
- 2026-11-09 · K1 (agent-B0-train-20261109T214533.609401Z) ran through the paid Anthropic API at the standard
  price (manifest batch false, via_claude_code null), not through Claude Code as SPEC D25 registers
  (SPEC.md:127, 437, 545; the Message Batches API was D25's fallback). The teacher's path is now the one
  calibration and test use: the same API, without Claude Code's fixed ~530-token context, so the training data
  are the teacher's own calls. K1's cost is now a real API cost, which §6.8's payback can count, and the calls
  ran under the API's terms, not the subscription's. Direction: neutral for the verdicts; it removes the
  context mismatch §11.3 was to measure
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
- 2026-11-09 · the model-rejection rule narrowed after Stage 0 (9f2a972, 17:33 UTC; b65a3d4, 17:43 UTC;
  bench/agent/rejections.py, bench/agent/hooks.py classify, patch 5b). At Stage 0 any HTTP 400 or 413 from the
  engine was the model's failure (scored, the question wrong), except Anthropic's "credit balance" 400. Now only a
  413, or a 400 whose message says the prompt is longer than the context, is the model's; a 400 naming max_tokens
  or max_completion_tokens, and every other 400 (workspace, key, permission, billing, invalid parameter or model),
  is the harness's: the run fails and nothing is scored. The rule is the same for every arm and engine (B0 to B5,
  API and vLLM). 49556b3 (2026-11-10 07:09 UTC, branch rejection-maxtokens) decides a message that gives the
  lengths in numbers by them: the model's only when the prompt alone is longer than the context. vLLM 0.30's
  "your prompt contains at least N input tokens", which the pattern matched even when the prompt fits and only
  prompt + max_tokens does not, is now the harness's, since with max_tokens set it never proves the prompt alone
  too long; vLLM's input-processor and Responses messages for the prompt alone are now the model's. Direction:
  up to 2026-11-10, 11 HTTP 400s and no 413 occurred, all Anthropic's "credit balance is too low", all in
  K1's sessions and classified as the harness's (the runs failed and were resumed); no 400 about the context or
  max_tokens occurred, so no recorded call changes class. The largest prompt in the Qwen replay of record
  (replay-slm_qwen3-8b-calib-20261110T025007.659032Z), B3's pilot (agent-B3-calib-20261110T051808.826362Z) and
  Granite's replay (replay-slm_granite-4.2-8b-calib-20261110T032255.812515Z) is 11,951 tokens (B3's pilot,
  select_tables), counted with the pinned tokenizer and equal to vLLM's usage.input; with max_tokens 4,096 and a
  context of 32,768, none of their calls would have triggered it. A prompt that fits but leaves no room for
  max_tokens, which has not occurred, fails the execution as a harness error; if it occurs on test, it is a
  documented defect
- 2026-11-10 · the load test, J8, J3, J7 and the report redesigned after Stage 0 (bench/loadtest.py,
  bench/judge/j3.py, j7.py, j8.py, bench/report.py, bench/agent/hooks.py; config.yaml: loadtest,
  cost.limit_min_calls, cost.limit_pool, modal.gpu_prices_sensitivity), before any load test ran (no J8 result
  exists). (1) One latency limit rule for every sweep: min(17,000 ms, the teacher's p95 in cost.slo_from reweighted
  to the sweep's own call-site mix), a weighted inverse-distribution quantile; a call site with fewer than 30
  teacher calls is pooled (revise, 5 calls, with generate_candidate). It replaces one limit for all engines (3,803
  ms); with B0's routed mixes as a proxy, the adapters' limits would be 3,566 ms (c0), 7,170 (c1), 4,491 (c2) and
  2,396 (c3): stricter for c0 and c3 (against the SLM), looser for c1 and c2 (for it). The teacher's own-mix limit
  is reported as a sensitivity. (2) Each engine replays its own arm's end-to-end calls on calib, not B0's: the base
  B3's, each adapter the calls B4 sent it; the SLM's traffic carries its own retries and repairs (direction unknown
  before the runs). (3) B4 is priced by an "as deployed" sweep (one server, the four adapters, B4's mix), no longer
  by the dearest adapter's (which was against the SLM); B5 and J7 price each cluster at its own adapter's sweep, no
  longer at the dearest. (4) Each level draws its own calls from the engine's pool, stratified by cluster, or by
  call site for the base, which is not routed (largest remainder), so a small cluster's calls come back in later
  levels, and every level of every sweep starts from an empty prefix cache, which lowers throughput (against the
  SLM); the levels start at concurrency 1 (for it). The cache is emptied with vLLM's /reset_prefix_cache, which
  needs the serving pod in vLLM's dev mode (VLLM_SERVER_DEV_MODE=1, an environment change of the server, checked by
  c0's P-4 in that mode). (5) The router's embedding time (the p95 over B4's calib run, whose routes.jsonl J8
  hashes) is added to the p95 of every sweep the router feeds (the adapters, as deployed) before the limit, against
  the SLM; the wait for the embedding model's lock, an artifact of one model per worker process, is not added (for
  it) and is reported; the cost without the addition is a sensitivity. The time is measured by
  bench/agent/hooks.py, which wraps clusters.embed and clusters._model at runtime (B4 and B5 runs only) solely to
  time the lock wait and the embedding; the frozen module is not edited. (6) An engine with no level within its
  limit is a result, not a stop: it has no cost, J7 never allocates its cluster to it, and a cost claim with no SLM
  arm priced is "not testable at the API's latency". (7) A V3 or A6 confirmation must also hold at Modal's price
  for the same GPU, CPU and memory billed apart ($3.13/h against RunPod's $1.79/h; against the SLM)
- 2026-11-10 · the rules for a failure during the test split, fixed before the registration (no execution has
  touched the test split); analysis code changed after Stage 0 for them: bench/evaluate.py, bench/report.py, and
  bench/prereg.py (frozen with prereg/). The registration (bench/prereg.py) now records `execution_code`: the
  sha256 of every tracked file of the code that runs, defined by exclusion so that no new file is left out:
  everything under bench/ outside the analysis code, vendor/chess, modal_apps, runpod_apps, env/ and
  pyproject.toml (and a uv.lock if one appears). An execution on test (`bench run`, `bench replay`) starts only
  on a clean working tree (`git status --porcelain` with no user-level excludes file, and an empty
  .git/info/exclude; what the repository's .gitignore ignores, such as runs/, does not count; .claude/ is
  added to it), whose HEAD holds the registered commit's tree outside prereg/ and registry/, and whose
  execution code hashes to the registered one. A reading of the test (bench eval, the report) needs no clean
  tree, since it writes judgments/, but HEAD must hold the registered tree outside prereg/, registry/ and
  judgments/, besides the analysis code's hashes it already checked. Every execution's manifest, and on test
  the registry, records `end_reason`, `n_predictions` and the sha256 of its predictions.json and calls.jsonl
  when it ends, and a test execution's manifest is committed as `running` with its intent, so a session killed
  outright stays listed. A test execution that did not end is resumed (`--resume-from`): from the latest
  attempt of its configuration, under the same registration and config_sha256, with the source and the
  resumed session both clean at the registered code and the source's outputs still hashing to its registered
  manifest; a killed session (still `running`, its lock free) is resumed from the questions it journaled as
  they ended, after the sha256 of what it left is computed and published in the new intent as
  `unregistered_source_hashes`, since no record vouched for them. It is run again from its first question
  (`--redo-reason`, the defect, published in the intent before the first question) only when resuming it is
  impossible (the code refuses a redo of an attempt it can resume) or a declared defect affects questions it
  completed (`--redo-affects-completed`), and only once this file, as published, has a line naming that
  attempt (D22); every earlier attempt stays in the registry. The report lists, per configuration, every
  attempt with how it ended, its redo reason, and its n and EX: for an attempt abandoned with predictions, a
  partial EX labelled descriptive (`bench eval --partial`: the questions it completed, the model's failures
  scored as wrong; the questions it did not finish and those with a harness error left out), which overlaps
  the questions a final run resumed from it copied, and is never read for a verdict. Before this, resuming on
  test was refused, an execution that did not end could be run again from scratch with no stated reason, and
  the report listed attempts by status only. Direction: none on any measured value; it narrows which test
  executions can be produced and makes every attempt, its reason and its partial accuracy public, so no
  configuration can be rerun until a preferred outcome appears
- 2026-11-10 · SPEC.md and SPEC-v1.md translated from Portuguese into English after Stage 0 (PR #45), with no change
  of content: every rule, number, threshold, identifier and section number is unchanged (checked line by line
  against the original by an independent review and by script), dates are written in ISO form and numbers with
  English separators. The Portuguese originals remain in the history (Stage 0 ae60d1b; 3a28b19 before the
  translation), so `git diff STAGE0 -- SPEC.md` shows the whole file as changed. Direction: none
- 2026-11-10 · analysis code changed after Stage 0 (bench/report.py, bench/evaluate.py, bench/judge/j4.py): the
  report prints "—" as the planned power of a comparison whose pilot d is 0 and of repair when it is inconclusive by
  design (the stored planned_power, J4's power, is unchanged); the S2 row states the training rows per cluster from
  train/datasets/manifest.json and the routed sites' rule, with curate's figures as labelled context; a table prints
  every J4 comparison already in report.json; the K4 heading names the report's split; J4's refusal on unpaired
  questions names the pair (message only); `bench eval --partial` also scores a test attempt killed outright (still
  `running`, its session lock free) on the questions its journal holds, under the same scoring rules, checked against
  the hashes a resume recorded, and the report asks for it like any abandoned attempt's. No stored result or verdict
  changes (bench verify 0 divergences). Direction: none
- 2026-11-10 · the load test's AIPerf client ran co-located on the serving pod instead of in a CPU pod of its own.
  This happened after Stage 0 and before any load test ran, because no CPU pod was available in the serving pod's
  data center. AIPerf ran in the serving pod's container (A100 80GB PCIe, listed with 16 vCPUs as config.yaml's
  price note records; the cgroup's quota read from cpu.max is about 30.1 vCPUs, 3013888/100000, and the 75%
  criterion uses it), in the same training environment from the volume, pinned with `taskset` to 4 cores that vLLM
  is not pinned to. It reached vLLM on the loopback, not through the SSH tunnel a CPU pod uses. Measured in the
  smoke test at concurrency 4 and 128: AIPerf used about 0.8 core, vLLM about 0.4 to 0.5 core, and the whole
  container about 1.25 cores. The loopback adds about 0 ms per request, against about 1 ms through the data
  center's SSH tunnel. All six sweeps (the base, the four adapters, and B4 as deployed) use the same client:
  `bench loadtest` refuses a sweep whose client kind differs from a done load test of the same candidate under the
  same configuration. Each level's manifest records the client (its pod, pinned cores, nproc, and the cgroup's
  cpuset and CPU quota), the loopback's calibration, and a CPU sample of AIPerf, of vLLM and of the container over
  the level (`cpu_sample`). Direction: negligible. If anything, it slightly favours the SLM, because its measured
  latencies leave out the network a separate client would add (about 1 ms). The per-level CPU samples show whether
  the client and the server competed for CPU. config.yaml is unchanged; the client is `bench loadtest --on runpod
  --client colocated` (opt-in; the default stays the CPU pod). Each level must pass a no-contention check
  (runpod_apps/colocated.py), or the load test exits 3: the cgroup did not throttle (cpu.stat's nr_throttled and
  throttled_usec unchanged), the container used under 75% of its allotted vCPUs (the cgroup's quota), AIPerf's own
  CPU time used under 90% of its 4 pinned cores (the cores' host-wide busy share from /proc/stat, which includes
  other tenants, is recorded apart as data), and no vLLM thread's affinity included a pinned core (vLLM's threads
  are kept off them before the sweep); a quantity that is not measured fails the level.
- 2026-11-10 · the cost verdict rules (bench/report.py; SPEC.md §1, §5, §6.4, §6.6), decided before any SLM load
  test result or test-split run was read; only the discarded smoke levels of the load test had been seen. (1) V3
  and AV2 are judged on B4, the all-SLM arm, alone. Before, V3 took the cheapest of B4 and B5 (B5 counted as an
  SLM arm if it made any SLM call), so a hybrid whose savings came from gpt-oss or the teacher could decide it;
  B5 now belongs to A6, whose row prints its share of calls, tokens and cost per engine. AV2 prints B5 beside B4,
  deciding nothing. If B4 is not non-inferior to B0, V3 reads "not testable: no all-SLM arm is non-inferior"
  (before: inconclusive when B4 was inconclusive), and B4's cost ratio is still printed, as descriptive. (2) The
  cost claims (V3, A6) read two comparators: the cheapest untrained arm non-inferior to B0, as before, and the
  cheapest not shown worse than B0 (non-inferior or inconclusive). A claim confirms only if it holds against
  both and refutes only if it fails against both; otherwise it is inconclusive, and both are printed. Before,
  an untrained arm that was only inconclusive was dropped, leaving a dearer comparator (at worst B0, the most
  expensive). A test report also refuses a plan that leaves out an arm with a done test run. (3) Labels only:
  J4's interval is printed as two-sided 90% (one-sided 95% bounds); §5 states the refutation rule (the upper
  bound below −Δ), which the code already applied; A5, AV2 and A6's cost part are labelled "by point estimate, no
  interval"; the map states that its six claims are not corrected for multiplicity; the planned power is stated
  at a true difference of 0. The "joint power" of §6.4, which the analysis code never computed, is removed.
  (4) Text only: costs are marginal, and the training cost and hours are reported descriptively. The fixed cost
  and payback of §1, §5 (AV2) and §6.6 were never computed by the analysis code. No stored judgment changes
  (`bench verify`: 0 divergences); on the calibration report only labels change. Direction: (1) against the SLM:
  it removes a hybrid's route to confirming V3 and makes AV2 easier to win against the SLM; (2) against
  confirming: the comparator is never dearer, and refuting takes both comparators as before; (3) none; (4) none
- 2026-11-10 · patch 16 extended (vendor/chess/src/workflow/agents/agent.py): the agent's tool is now the first
  `<tool_call>` block that names one of its tools, not the first block. Decided after reading B1's calibration
  pilot: in agent-B1-calib-20261110T035103.709962Z (k = 3), gpt-oss-120b wrote the tag inside a sentence before a
  valid call in 2 of 50 questions (874 and 1181, agent_ir ir:0: "We need to output a <tool_call> with the tool name
  ...", then the call); the old rule read "with", rejected it, and that agent's loop ended. Re-parsing every recorded
  agent answer offline with both rules, only these change: B1's k = 3 pilot 2 of 481 answered agent calls (874,
  1181; both still scored correct there); the granite-4.2-8b zero-shot replay 4 of 2,198 (182, 430, 515, 953;
  agent_ir and agent_ss), which moves granite's J6 mean by under 0.001 against qwen3-8b's lead of 0.21, so the
  choice stands. Zero change in B0's calibration pilot (2,198), K1 (9,167), B3's calibration run (2,200), B4's
  (184240: 1,760; its resume 191453: 2,200), the B4 replay (2,198), the Qwen base replays (2 x 2,198), the cheap_alt
  replays (40, 625, 2,198) and B1's k = 0 pilot and first k = 3 pilot (375, 402). No answer the old rule accepted
  changes its tool. Direction: favours B1, the comparator (only gpt-oss writes this form; the SLM arms had no such
  failure on calib), so against confirming the SLM's non-inferiority. B1's few-shot k (b1k, k = 3) and the pilot
  readings were made under the old rule; k = 0's pilot had no affected answer, k = 3's had the two above
- 2026-11-10 · analysis code changed after Stage 0, before any execution on test: bench/evaluate.py records the
  wall-clock seconds of each gold and each prediction (`gold_seconds`, `pred_seconds`), and bench/judge/j1.py
  refuses eval executions of one split whose golds failed (`gold_error`, a timeout included) on different
  questions. The evaluator's 60 s timeout is wall clock, and the gold of test question 701 (codebase_community)
  timed out on the machine that built the data (data/MANIFEST.json, gold_observed.timeouts; also train 1131), so it
  could time out in one arm's eval and not in another's. Rule, declared now: a gold that fails is wrong for the arm
  and stays in n; with the refusal, 701 is wrong for every arm or scored for every arm, and the test runs are
  evaluated on an idle machine, one after the other (RUNBOOK step 14). Direction: none between arms (it removes a
  timing asymmetry); if 701's gold times out, every arm's EX is lower by 1/498
- 2026-11-10 · SPEC §6.1 and §8 now declare exploratory readings before the test. They change no verdict, margin,
  arm or allocation. The new text covers the primary descriptive metric per call site, beside a trivial rule, the
  base, the adapter, the cheap alternative and the teacher's repetition. It also covers question-level intervals,
  the published family of cuts, and a reading rule for the c1 learning curve. Two readings may be run on the test:
  the teacher on its own test calls, and c0 alone swapped live (extension 1). The call sites are classified against
  the paper's conditional claim; the classification was written after the calibration and is shown beside the
  registered predictions. Three limits of the training data are added (207 of 826 SQL rows wrong against the
  gold; an 87.6% always-"No" floor at the column filter; 1,534 questions on the 11 databases). Together they bound
  the reading in both directions: a loss outside the claim's antecedent does not refute it, and agreement at the
  column filter does not support it without an end-to-end measure
