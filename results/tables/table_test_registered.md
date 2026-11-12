# Per-call-site table, test split — exploratory, on the teacher's inputs

> EXPLORATORY, no verdict. Per call site, on the teacher's inputs (test split): each engine answers the calls of the teacher's registered run `agent-B0-test-20261111T060814.371294Z`. Descriptive only; unadjusted 95% intervals; these numbers allocate nothing and enter no decision rule.

Inputs:

- teacher: `agent-B0-test-20261111T060814.371294Z` ; `eval-per-call-agent-B0-test-20261111T060814.371294Z-20261112T043501.196637Z`
- base: `replay-slm_qwen3-8b-test-20261111T225111.186203Z` ; `eval-per-call-replay-slm_qwen3-8b-test-20261111T225111.186203Z-20261112T042104.945806Z`
- adapter: `replay-B4-test-20261111T225020.625646Z` ; `eval-per-call-replay-B4-test-20261111T225020.625646Z-20261112T041858.585025Z`
- cheap: not run
- self: `replay-production_llm-test-20261111T202527.888323Z` ; `eval-per-call-replay-production_llm-test-20261111T202527.888323Z-20261112T043430.618816Z`

Intervals: 95%, unadjusted, percentile bootstrap over QUESTIONS (2000 resamples, seed 20261112); all of a question's calls stay together. The agent's steps get no interval: their effective sample is the number of distinct prompts.
The teacher's repetition covers 100 questions (a seeded sample in the test's difficulty mix); its cells are computed on those questions only.

## Call site × the five columns

| call site (cluster) | questions / calls (distinct prompts; teacher outputs that did not parse) | training rows: call site / its cluster | primary metric | trivial rule (no language model) | base SLM (Qwen3-8B, no adapter) | SLM + adapters (B4's replay) | cheaper LLM (gpt-oss) | teacher's own repetition |
|---|---|---|---|---|---|---|---|---|
| filter_column (c0; B4's replay: 38255 routed to c0, 63 routed to c1) | 498 / 38318 (38318; 0) | 12000 (11982 in c0 + 18 in c1) / 11982 | recall of the columns the teacher kept; agreement on the columns it dropped; Cohen's kappa | recall kept 0.000 [0.000, 0.000]; agreement dropped 1.000 [1.000, 1.000]; kappa 0.000 [0.000, 0.000]; raw agreement 0.908 [0.901, 0.914] (n = 498 questions, 38318 calls) | recall kept 0.826 [0.809, 0.842]; agreement dropped 0.902 [0.894, 0.910]; kappa 0.536 [0.517, 0.556]; raw agreement 0.887 [0.877, 0.897] (n = 498 questions, 38318 calls) | recall kept 0.838 [0.823, 0.854]; agreement dropped 0.992 [0.990, 0.993]; kappa 0.860 [0.849, 0.872]; raw agreement 0.977 [0.975, 0.980] (n = 498 questions, 38318 calls) | not run | recall kept 0.967 [0.951, 0.981]; agreement dropped 0.997 [0.996, 0.998]; kappa 0.966 [0.955, 0.977]; raw agreement 0.994 [0.992, 0.996] (n = 100 questions, 7592 calls) |
| select_tables (c1; B4's replay: 498 routed to c1) | 498 / 498 (498; 0) | 833 (833 in c1) / 2517 | recall of the teacher's set; exact-set agreement beside it | set recall 1.000 [1.000, 1.000]; exact-set agreement (= raw agreement 0.004 [0.000, 0.010]) (n = 498 questions, 498 calls) | set recall 0.970 [0.959, 0.979]; exact-set agreement (= raw agreement 0.839 [0.805, 0.871]) (n = 498 questions, 498 calls) | set recall 0.969 [0.958, 0.979]; exact-set agreement (= raw agreement 0.906 [0.880, 0.930]) (n = 498 questions, 498 calls) | not run | set recall 1.000 [1.000, 1.000]; exact-set agreement (= raw agreement 1.000 [1.000, 1.000]) (n = 100 questions, 100 calls) |
| select_columns (c1; B4's replay: 498 routed to c1) | 498 / 498 (498; 0) | 833 (833 in c1) / 2517 | recall of the teacher's set; exact-set agreement beside it | set recall 0.999 [0.998, 1.000]; exact-set agreement (= raw agreement 0.074 [0.052, 0.098]) (n = 498 questions, 498 calls) | set recall 0.791 [0.771, 0.808]; exact-set agreement (= raw agreement 0.349 [0.307, 0.392]) (n = 498 questions, 498 calls) | set recall 0.945 [0.932, 0.957]; exact-set agreement (= raw agreement 0.761 [0.723, 0.797]) (n = 498 questions, 498 calls) | not run | set recall 0.992 [0.981, 1.000]; exact-set agreement (= raw agreement 0.950 [0.900, 0.990]) (n = 100 questions, 100 calls) |
| generate_candidate (c1; B4's replay: 498 routed to c1) | 498 / 498 (498; 0) | 826 (826 in c1) / 2517 | EX against the gold | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 278/498 = 0.558 [0.514, 0.602]; raw agreement: not applicable (judged by execution) (n = 498 questions, 498 calls) | EX 335/498 = 0.673 [0.631, 0.713]; raw agreement: not applicable (judged by execution) (n = 498 questions, 498 calls) | not run | replay: EX 80/100 = 0.800 [0.720, 0.880]; the registered run on the same invocations: EX 79/100 = 0.790 [0.710, 0.870]; raw agreement: not applicable (judged by execution) (n = 100 questions, 100 calls) |
| revise (c1; B4's replay: 2 routed to c1) | 2 / 2 (2; 0) | 6 (6 in c1) / 2517 | EX against the gold | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 1/2 = 0.500 [0.000, 1.000]; raw agreement: not applicable (judged by execution) (n = 2 questions, 2 calls) | EX 0/2 = 0.000 [0.000, 0.000]; raw agreement: not applicable (judged by execution) (n = 2 questions, 2 calls) | not run | no invocation of this call site in this column's questions |
| extract_keywords (c2; B4's replay: 498 routed to c2) | 498 / 498 (498; 1) | 834 (1 in c1 + 833 in c2) / 833 | mean Jaccard against the teacher's set; exact-set agreement beside it | not defined in SPEC (the stop-word list and the tokenisation of "the question's words" are not fixed) | Jaccard 0.492 [0.478, 0.505]; exact-set agreement (= raw agreement 0.004 [0.000, 0.010]) (n = 498 questions, 498 calls) | Jaccard 0.737 [0.722, 0.751]; exact-set agreement (= raw agreement 0.099 [0.073, 0.127]) (n = 498 questions, 498 calls) | not run | Jaccard 0.918 [0.893, 0.940]; exact-set agreement (= raw agreement 0.530 [0.430, 0.620]) (n = 100 questions, 100 calls) |
| agent_ir (c3; B4's replay: 1992 routed to c3) | 498 / 1992 (4; 0) | 3336 (3336 in c3) / 9164 | agreement per distinct prompt (no rate over calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 1992 of 1992 calls (n = 498 questions, 1992 calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 1992 of 1992 calls (n = 498 questions, 1992 calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 1992 of 1992 calls (n = 498 questions, 1992 calls) | not run | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 400 of 400 calls (n = 100 questions, 400 calls) |
| agent_ss (c3; B4's replay: 1992 routed to c3) | 498 / 1992 (4; 0) | 3332 (3332 in c3) / 9164 | agreement per distinct prompt (no rate over calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 1992 of 1992 calls (n = 498 questions, 1992 calls) | agreement per distinct prompt 0.750 (k = 4 prompts, 3 with every call agreeing); raw agreement, as counts only: 1494 of 1992 calls (n = 498 questions, 1992 calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 1992 of 1992 calls (n = 498 questions, 1992 calls) | not run | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 400 of 400 calls (n = 100 questions, 400 calls) |
| agent_cg (c3; B4's replay: 1494 routed to c3) | 498 / 1494 (3; 0) | 2496 (2496 in c3) / 9164 | agreement per distinct prompt (no rate over calls) | agreement per distinct prompt 1.000 (k = 3 prompts, 3 with every call agreeing); raw agreement, as counts only: 1494 of 1494 calls (n = 498 questions, 1494 calls) | agreement per distinct prompt 1.000 (k = 3 prompts, 3 with every call agreeing); raw agreement, as counts only: 1494 of 1494 calls (n = 498 questions, 1494 calls) | agreement per distinct prompt 1.000 (k = 3 prompts, 3 with every call agreeing); raw agreement, as counts only: 1494 of 1494 calls (n = 498 questions, 1494 calls) | not run | agreement per distinct prompt 1.000 (k = 3 prompts, 3 with every call agreeing); raw agreement, as counts only: 300 of 300 calls (n = 100 questions, 300 calls) |

The teacher's EX in its registered run, on all invocations of the call sites with gold:

- generate_candidate: EX 405/498 = 0.813 [0.779, 0.845] (n = 498 questions, 498 calls)
- revise: EX 2/2 = 1.000 [1.000, 1.000] (n = 2 questions, 2 calls)

Clusters (a cluster's row lists its call sites; metrics that differ are not pooled):

- c0 (11982 training rows): filter_column
- c1 (2517 training rows): select_tables, select_columns, generate_candidate, revise
- c2 (833 training rows): extract_keywords
- c3 (9164 training rows): agent_ir, agent_ss, agent_cg

Trivial rules used: filter_column: always "No". select_tables / select_columns: every table (every column) of the CREATE TABLE blocks in the call's own prompt. Agent steps: a lookup by prompt, built from the c3 training rows (prompt → the action those rows give it; the SPEC does not name the lookup's source). Lookup: 5478 calls found their prompt, 0 did not, 0 hit a prompt whose training rows give more than one action.

Warnings published with the table (SPEC 6.1): exact-set agreement on keywords is rarely met even by the teacher, so a low value there says little about the decision. The agent's steps have about 11 distinct prompts (the prompt does not depend on the question), so their effective n is the number of steps, not of calls. Repair has a small n (the teacher hardly repairs) and 6 training rows, and is read as such. Raw agreement is not the primary metric where the teacher's answers are imbalanced or its self-agreement is below 95%.

## SQL generation and repair by difficulty (exploratory)

| difficulty | call site | teacher, registered run | trivial rule (no language model) | base SLM (Qwen3-8B, no adapter) | SLM + adapters (B4's replay) | cheaper LLM (gpt-oss) | teacher's own repetition |
|---|---|---|---|---|---|---|---|
| simple | generate_candidate | EX 133/148 = 0.899 [0.845, 0.946] (n = 148 questions, 148 calls) | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 112/148 = 0.757 [0.689, 0.824]; raw agreement: not applicable (judged by execution) (n = 148 questions, 148 calls) | EX 122/148 = 0.824 [0.764, 0.885]; raw agreement: not applicable (judged by execution) (n = 148 questions, 148 calls) | not run | replay: EX 25/30 = 0.833 [0.700, 0.967]; the registered run on the same invocations: EX 25/30 = 0.833 [0.700, 0.967]; raw agreement: not applicable (judged by execution) (n = 30 questions, 30 calls) |
| simple | revise | no invocation at this difficulty | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | no invocation at this difficulty in this column's questions | no invocation at this difficulty in this column's questions | not run | no invocation at this difficulty in this column's questions |
| moderate | generate_candidate | EX 201/248 = 0.810 [0.762, 0.859] (n = 248 questions, 248 calls) | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 130/248 = 0.524 [0.464, 0.585]; raw agreement: not applicable (judged by execution) (n = 248 questions, 248 calls) | EX 157/248 = 0.633 [0.577, 0.690]; raw agreement: not applicable (judged by execution) (n = 248 questions, 248 calls) | not run | replay: EX 42/50 = 0.840 [0.740, 0.920]; the registered run on the same invocations: EX 42/50 = 0.840 [0.720, 0.920]; raw agreement: not applicable (judged by execution) (n = 50 questions, 50 calls) |
| moderate | revise | EX 2/2 = 1.000 [1.000, 1.000] (n = 2 questions, 2 calls) | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 1/2 = 0.500 [0.000, 1.000]; raw agreement: not applicable (judged by execution) (n = 2 questions, 2 calls) | EX 0/2 = 0.000 [0.000, 0.000]; raw agreement: not applicable (judged by execution) (n = 2 questions, 2 calls) | not run | no invocation at this difficulty in this column's questions |
| challenging | generate_candidate | EX 71/102 = 0.696 [0.608, 0.784] (n = 102 questions, 102 calls) | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 36/102 = 0.353 [0.265, 0.451]; raw agreement: not applicable (judged by execution) (n = 102 questions, 102 calls) | EX 56/102 = 0.549 [0.451, 0.647]; raw agreement: not applicable (judged by execution) (n = 102 questions, 102 calls) | not run | replay: EX 13/20 = 0.650 [0.450, 0.850]; the registered run on the same invocations: EX 12/20 = 0.600 [0.400, 0.800]; raw agreement: not applicable (judged by execution) (n = 20 questions, 20 calls) |
| challenging | revise | no invocation at this difficulty | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | no invocation at this difficulty in this column's questions | no invocation at this difficulty in this column's questions | not run | no invocation at this difficulty in this column's questions |

## Latency and tokens per call, by engine (final attempt of each invocation)

| call site | engine | calls | latency p50 / p95 (ms) | mean tokens in / out per call | attempts per invocation |
|---|---|---|---|---|---|
| filter_column | teacher (registered B0 run) | 38318 | 2011 / 2937 | 5895 / 137 | 1.000 |
| filter_column | base SLM (Qwen3-8B, no adapter) | 38318 | 14794 / 30423 | 4061 / 84 | 1.022 |
| filter_column | SLM + adapters (B4's replay) | 38318 | 17297 / 33019 | 4061 / 92 | 1.011 |
| filter_column | cheaper LLM (gpt-oss) | not run | not run | not run | not run |
| filter_column | teacher's own repetition | 7592 | 1946 / 2821 | 5891 / 139 | 1.000 |
| filter_column | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| select_tables | teacher (registered B0 run) | 498 | 2517 / 3606 | 1933 / 215 | 1.000 |
| select_tables | base SLM (Qwen3-8B, no adapter) | 498 | 16802 / 42594 | 1269 / 155 | 1.006 |
| select_tables | SLM + adapters (B4's replay) | 498 | 19014 / 43434 | 1269 / 137 | 1.002 |
| select_tables | cheaper LLM (gpt-oss) | not run | not run | not run | not run |
| select_tables | teacher's own repetition | 100 | 2502 / 3376 | 1981 / 207 | 1.000 |
| select_tables | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| select_columns | teacher (registered B0 run) | 498 | 2366 / 3220 | 1660 / 218 | 1.000 |
| select_columns | base SLM (Qwen3-8B, no adapter) | 498 | 18026 / 45515 | 1133 / 167 | 1.010 |
| select_columns | SLM + adapters (B4's replay) | 498 | 19427 / 43219 | 1133 / 142 | 1.002 |
| select_columns | cheaper LLM (gpt-oss) | not run | not run | not run | not run |
| select_columns | teacher's own repetition | 100 | 2427 / 3403 | 1666 / 221 | 1.000 |
| select_columns | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| generate_candidate | teacher (registered B0 run) | 498 | 5109 / 6865 | 15559 / 786 | 1.000 |
| generate_candidate | base SLM (Qwen3-8B, no adapter) | 498 | 64904 / 163030 | 9881 / 639 | 1.002 |
| generate_candidate | SLM + adapters (B4's replay) | 498 | 56819 / 110104 | 9881 / 472 | 1.000 |
| generate_candidate | cheaper LLM (gpt-oss) | not run | not run | not run | not run |
| generate_candidate | teacher's own repetition | 100 | 4921 / 6511 | 15559 / 786 | 1.000 |
| generate_candidate | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| revise | teacher (registered B0 run) | 2 | 4376 / 9215 | 5648 / 890 | 1.000 |
| revise | base SLM (Qwen3-8B, no adapter) | 2 | 52181 / 56838 | 3508 / 605 | 1.000 |
| revise | SLM + adapters (B4's replay) | 2 | 27034 / 56306 | 3508 / 282 | 1.000 |
| revise | cheaper LLM (gpt-oss) | not run | not run | not run | not run |
| revise | teacher's own repetition | no invocation | — | — | — |
| revise | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| extract_keywords | teacher (registered B0 run) | 498 | 1748 / 5435 | 1085 / 99 | 1.002 |
| extract_keywords | base SLM (Qwen3-8B, no adapter) | 498 | 6384 / 18315 | 716 / 45 | 1.000 |
| extract_keywords | SLM + adapters (B4's replay) | 498 | 10934 / 29296 | 716 / 71 | 1.008 |
| extract_keywords | cheaper LLM (gpt-oss) | not run | not run | not run | not run |
| extract_keywords | teacher's own repetition | 100 | 1532 / 5259 | 1083 / 95 | 1.000 |
| extract_keywords | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| agent_ir | teacher (registered B0 run) | 1992 | 1152 / 1692 | 377 / 17 | 1.000 |
| agent_ir | base SLM (Qwen3-8B, no adapter) | 1992 | 1737 / 3663 | 246 / 5 | 1.001 |
| agent_ir | SLM + adapters (B4's replay) | 1992 | 1629 / 3240 | 246 / 4 | 1.000 |
| agent_ir | cheaper LLM (gpt-oss) | not run | not run | not run | not run |
| agent_ir | teacher's own repetition | 400 | 1271 / 1827 | 377 / 17 | 1.002 |
| agent_ir | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| agent_ss | teacher (registered B0 run) | 1992 | 1244 / 2155 | 345 / 48 | 1.000 |
| agent_ss | base SLM (Qwen3-8B, no adapter) | 1992 | 1500 / 3732 | 232 / 5 | 1.007 |
| agent_ss | SLM + adapters (B4's replay) | 1992 | 1792 / 4334 | 232 / 5 | 1.004 |
| agent_ss | cheaper LLM (gpt-oss) | not run | not run | not run | not run |
| agent_ss | teacher's own repetition | 400 | 1233 / 2209 | 345 / 64 | 1.000 |
| agent_ss | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| agent_cg | teacher (registered B0 run) | 1494 | 1469 / 2446 | 337 / 65 | 1.000 |
| agent_cg | base SLM (Qwen3-8B, no adapter) | 1494 | 1551 / 13726 | 230 / 23 | 1.009 |
| agent_cg | SLM + adapters (B4's replay) | 1494 | 1546 / 4566 | 230 / 5 | 1.001 |
| agent_cg | cheaper LLM (gpt-oss) | not run | not run | not run | not run |
| agent_cg | teacher's own repetition | 300 | 1335 / 2408 | 337 / 66 | 1.000 |
| agent_cg | trivial rule (no language model) | no model call | no model call | no model call | no model call |

The teacher's repetition rows cover its sampled questions only. Latency is as each run logged it, under that run's own concurrency and serving; it is not a load test.

