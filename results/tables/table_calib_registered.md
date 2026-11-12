# Per-call-site table, calib split — exploratory, on the teacher's inputs

> EXPLORATORY, no verdict. Per call site, on the teacher's inputs (calib split): each engine answers the calls of the teacher's registered run `agent-B0-calib-20261109T151141.039667Z`. Descriptive only; unadjusted 95% intervals; these numbers allocate nothing and enter no decision rule.

Inputs:

- teacher: `agent-B0-calib-20261109T151141.039667Z` ; `eval-per-call-agent-B0-calib-20261109T151141.039667Z-20261109T153122.727095Z`
- base: `replay-slm_qwen3-8b-calib-20261110T025007.659032Z` ; `eval-per-call-replay-slm_qwen3-8b-calib-20261110T025007.659032Z-20261110T031423.931711Z`
- adapter: `replay-B4-calib-20261110T174825.799047Z` ; `eval-per-call-replay-B4-calib-20261110T174825.799047Z-20261110T181832.677536Z`
- cheap: `replay-cheap_alt-calib-20261110T160847.505204Z` ; `eval-per-call-replay-cheap_alt-calib-20261110T160847.505204Z-20261110T162517.048684Z`
- self: `replay-production_llm-calib-20261110T021304.240338Z`

Intervals: 95%, unadjusted, percentile bootstrap over QUESTIONS (2000 resamples, seed 20261112); all of a question's calls stay together. The agent's steps get no interval: their effective sample is the number of distinct prompts.
The teacher's repetition covers 50 questions (a seeded sample in the test's difficulty mix); its cells are computed on those questions only.

## Call site × the five columns

| call site (cluster) | questions / calls (distinct prompts; teacher outputs that did not parse) | training rows: call site / its cluster | primary metric | trivial rule (no language model) | base SLM (Qwen3-8B, no adapter) | SLM + adapters (B4's replay) | cheaper LLM (gpt-oss) | teacher's own repetition |
|---|---|---|---|---|---|---|---|---|
| filter_column (c0; B4's replay: 13128 routed to c0, 10 routed to c1) | 200 / 13138 (13138; 0) | 12000 (11982 in c0 + 18 in c1) / 11982 | recall of the columns the teacher kept; agreement on the columns it dropped; Cohen's kappa | recall kept 0.000 [0.000, 0.000]; agreement dropped 1.000 [1.000, 1.000]; kappa 0.000 [0.000, 0.000]; raw agreement 0.876 [0.858, 0.892] (n = 200 questions, 13138 calls) | recall kept 0.859 [0.836, 0.880]; agreement dropped 0.893 [0.879, 0.906]; kappa 0.594 [0.564, 0.623]; raw agreement 0.888 [0.877, 0.900] (n = 200 questions, 13138 calls) | recall kept 0.869 [0.847, 0.891]; agreement dropped 0.987 [0.983, 0.990]; kappa 0.869 [0.852, 0.887]; raw agreement 0.972 [0.967, 0.977] (n = 200 questions, 13138 calls) | recall kept 0.893 [0.873, 0.912]; agreement dropped 0.948 [0.940, 0.956]; kappa 0.757 [0.731, 0.784]; raw agreement 0.941 [0.933, 0.949] (n = 200 questions, 13138 calls) | recall kept 0.972 [0.958, 0.985]; agreement dropped 0.994 [0.991, 0.997]; kappa 0.966 [0.955, 0.976]; raw agreement 0.991 [0.986, 0.994] (n = 50 questions, 3311 calls) |
| select_tables (c1; B4's replay: 200 routed to c1) | 200 / 200 (200; 0) | 833 (833 in c1) / 2517 | recall of the teacher's set; exact-set agreement beside it | set recall 1.000 [1.000, 1.000]; exact-set agreement (= raw agreement 0.030 [0.010, 0.055]) (n = 200 questions, 200 calls) | set recall 0.969 [0.954, 0.983]; exact-set agreement (= raw agreement 0.820 [0.770, 0.870]) (n = 200 questions, 200 calls) | set recall 0.973 [0.957, 0.987]; exact-set agreement (= raw agreement 0.890 [0.845, 0.930]) (n = 200 questions, 200 calls) | set recall 0.991 [0.983, 0.998]; exact-set agreement (= raw agreement 0.935 [0.900, 0.970]) (n = 200 questions, 200 calls) | set recall 0.997 [0.990, 1.000]; exact-set agreement (= raw agreement 0.980 [0.940, 1.000]) (n = 50 questions, 50 calls) |
| select_columns (c1; B4's replay: 200 routed to c1) | 200 / 200 (200; 0) | 833 (833 in c1) / 2517 | recall of the teacher's set; exact-set agreement beside it | set recall 1.000 [1.000, 1.000]; exact-set agreement (= raw agreement 0.085 [0.050, 0.125]) (n = 200 questions, 200 calls) | set recall 0.813 [0.786, 0.839]; exact-set agreement (= raw agreement 0.395 [0.330, 0.460]) (n = 200 questions, 200 calls) | set recall 0.946 [0.927, 0.963]; exact-set agreement (= raw agreement 0.730 [0.670, 0.795]) (n = 200 questions, 200 calls) | set recall 0.945 [0.928, 0.962]; exact-set agreement (= raw agreement 0.745 [0.685, 0.800]) (n = 200 questions, 200 calls) | set recall 0.994 [0.986, 1.000]; exact-set agreement (= raw agreement 0.860 [0.760, 0.940]) (n = 50 questions, 50 calls) |
| generate_candidate (c1; B4's replay: 199 routed to c1) | 199 / 199 (199; 0) | 826 (826 in c1) / 2517 | EX against the gold | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 108/199 = 0.543 [0.472, 0.613]; raw agreement: not applicable (judged by execution) (n = 199 questions, 199 calls) | EX 123/199 = 0.618 [0.553, 0.683]; raw agreement: not applicable (judged by execution) (n = 199 questions, 199 calls) | EX 130/199 = 0.653 [0.588, 0.719]; raw agreement: not applicable (judged by execution) (n = 199 questions, 199 calls) | not run at this call site (the replay covered agent_cg, agent_ir, agent_ss, extract_keywords, filter_column, select_columns, select_tables) |
| revise (c1; B4's replay: 5 routed to c1) | 5 / 5 (5; 0) | 6 (6 in c1) / 2517 | EX against the gold | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 0/5 = 0.000 [0.000, 0.000]; raw agreement: not applicable (judged by execution) (n = 5 questions, 5 calls) | EX 2/5 = 0.400 [0.000, 0.800]; raw agreement: not applicable (judged by execution) (n = 5 questions, 5 calls) | EX 0/5 = 0.000 [0.000, 0.000]; raw agreement: not applicable (judged by execution) (n = 5 questions, 5 calls) | not run at this call site (the replay covered agent_cg, agent_ir, agent_ss, extract_keywords, filter_column, select_columns, select_tables) |
| extract_keywords (c2; B4's replay: 200 routed to c2) | 200 / 200 (200; 0) | 834 (1 in c1 + 833 in c2) / 833 | mean Jaccard against the teacher's set; exact-set agreement beside it | not defined in SPEC (the stop-word list and the tokenisation of "the question's words" are not fixed) | Jaccard 0.487 [0.464, 0.510]; exact-set agreement (= raw agreement 0.000 [0.000, 0.000]) (n = 200 questions, 200 calls) | Jaccard 0.731 [0.707, 0.752]; exact-set agreement (= raw agreement 0.125 [0.080, 0.170]) (n = 200 questions, 200 calls) | Jaccard 0.605 [0.583, 0.627]; exact-set agreement (= raw agreement 0.020 [0.005, 0.040]) (n = 200 questions, 200 calls) | Jaccard 0.899 [0.864, 0.931]; exact-set agreement (= raw agreement 0.440 [0.300, 0.580]) (n = 50 questions, 50 calls) |
| agent_ir (c3; B4's replay: 800 routed to c3) | 200 / 800 (4; 0) | 3336 (3336 in c3) / 9164 | agreement per distinct prompt (no rate over calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 800 of 800 calls (n = 200 questions, 800 calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 800 of 800 calls (n = 200 questions, 800 calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 800 of 800 calls (n = 200 questions, 800 calls) | agreement per distinct prompt 0.878 (k = 4 prompts, 0 with every call agreeing); raw agreement, as counts only: 702 of 800 calls (n = 200 questions, 800 calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 200 of 200 calls (n = 50 questions, 200 calls) |
| agent_ss (c3; B4's replay: 800 routed to c3) | 200 / 800 (4; 0) | 3332 (3332 in c3) / 9164 | agreement per distinct prompt (no rate over calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 800 of 800 calls (n = 200 questions, 800 calls) | agreement per distinct prompt 0.750 (k = 4 prompts, 3 with every call agreeing); raw agreement, as counts only: 600 of 800 calls (n = 200 questions, 800 calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 800 of 800 calls (n = 200 questions, 800 calls) | agreement per distinct prompt 0.834 (k = 4 prompts, 2 with every call agreeing); raw agreement, as counts only: 667 of 800 calls (n = 200 questions, 800 calls) | agreement per distinct prompt 1.000 (k = 4 prompts, 4 with every call agreeing); raw agreement, as counts only: 200 of 200 calls (n = 50 questions, 200 calls) |
| agent_cg (c3; B4's replay: 598 routed to c3) | 200 / 598 (3; 1) | 2496 (2496 in c3) / 9164 | agreement per distinct prompt (no rate over calls) | agreement per distinct prompt 1.000 (k = 3 prompts, 3 with every call agreeing); raw agreement, as counts only: 597 of 597 calls (n = 200 questions, 598 calls) | agreement per distinct prompt 1.000 (k = 3 prompts, 3 with every call agreeing); raw agreement, as counts only: 597 of 597 calls (n = 200 questions, 598 calls) | agreement per distinct prompt 1.000 (k = 3 prompts, 3 with every call agreeing); raw agreement, as counts only: 597 of 597 calls (n = 200 questions, 598 calls) | agreement per distinct prompt 0.988 (k = 3 prompts, 0 with every call agreeing); raw agreement, as counts only: 590 of 597 calls (n = 200 questions, 598 calls) | agreement per distinct prompt 0.993 (k = 3 prompts, 2 with every call agreeing); raw agreement, as counts only: 149 of 150 calls (n = 50 questions, 150 calls) |

The teacher's EX in its registered run, on all invocations of the call sites with gold:

- generate_candidate: EX 137/199 = 0.688 [0.623, 0.749] (n = 199 questions, 199 calls)
- revise: EX 2/5 = 0.400 [0.000, 0.800] (n = 5 questions, 5 calls)

Clusters (a cluster's row lists its call sites; metrics that differ are not pooled):

- c0 (11982 training rows): filter_column
- c1 (2517 training rows): select_tables, select_columns, generate_candidate, revise
- c2 (833 training rows): extract_keywords
- c3 (9164 training rows): agent_ir, agent_ss, agent_cg

Trivial rules used: filter_column: always "No". select_tables / select_columns: every table (every column) of the CREATE TABLE blocks in the call's own prompt. Agent steps: a lookup by prompt, built from the c3 training rows (prompt → the action those rows give it; the SPEC does not name the lookup's source). Lookup: 2198 calls found their prompt, 0 did not, 0 hit a prompt whose training rows give more than one action.

Warnings published with the table (SPEC 6.1): exact-set agreement on keywords is rarely met even by the teacher, so a low value there says little about the decision. The agent's steps have about 11 distinct prompts (the prompt does not depend on the question), so their effective n is the number of steps, not of calls. Repair has a small n (the teacher hardly repairs) and 6 training rows, and is read as such. Raw agreement is not the primary metric where the teacher's answers are imbalanced or its self-agreement is below 95%.

## SQL generation and repair by difficulty (exploratory)

| difficulty | call site | teacher, registered run | trivial rule (no language model) | base SLM (Qwen3-8B, no adapter) | SLM + adapters (B4's replay) | cheaper LLM (gpt-oss) | teacher's own repetition |
|---|---|---|---|---|---|---|---|
| simple | generate_candidate | EX 108/133 = 0.812 [0.744, 0.880] (n = 133 questions, 133 calls) | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 95/133 = 0.714 [0.639, 0.789]; raw agreement: not applicable (judged by execution) (n = 133 questions, 133 calls) | EX 101/133 = 0.759 [0.684, 0.835]; raw agreement: not applicable (judged by execution) (n = 133 questions, 133 calls) | EX 102/133 = 0.767 [0.692, 0.842]; raw agreement: not applicable (judged by execution) (n = 133 questions, 133 calls) | not run at this call site (the replay covered agent_cg, agent_ir, agent_ss, extract_keywords, filter_column, select_columns, select_tables) |
| simple | revise | EX 2/3 = 0.667 [0.000, 1.000] (n = 3 questions, 3 calls) | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 0/3 = 0.000 [0.000, 0.000]; raw agreement: not applicable (judged by execution) (n = 3 questions, 3 calls) | EX 1/3 = 0.333 [0.000, 1.000]; raw agreement: not applicable (judged by execution) (n = 3 questions, 3 calls) | EX 0/3 = 0.000 [0.000, 0.000]; raw agreement: not applicable (judged by execution) (n = 3 questions, 3 calls) | not run at this call site (the replay covered agent_cg, agent_ir, agent_ss, extract_keywords, filter_column, select_columns, select_tables) |
| moderate | generate_candidate | EX 22/34 = 0.647 [0.471, 0.794] (n = 34 questions, 34 calls) | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 10/34 = 0.294 [0.147, 0.441]; raw agreement: not applicable (judged by execution) (n = 34 questions, 34 calls) | EX 16/34 = 0.471 [0.294, 0.647]; raw agreement: not applicable (judged by execution) (n = 34 questions, 34 calls) | EX 21/34 = 0.618 [0.441, 0.765]; raw agreement: not applicable (judged by execution) (n = 34 questions, 34 calls) | not run at this call site (the replay covered agent_cg, agent_ir, agent_ss, extract_keywords, filter_column, select_columns, select_tables) |
| moderate | revise | EX 0/1 = 0.000 [0.000, 0.000] (n = 1 questions, 1 calls) | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 0/1 = 0.000 [0.000, 0.000]; raw agreement: not applicable (judged by execution) (n = 1 questions, 1 calls) | EX 1/1 = 1.000 [1.000, 1.000]; raw agreement: not applicable (judged by execution) (n = 1 questions, 1 calls) | EX 0/1 = 0.000 [0.000, 0.000]; raw agreement: not applicable (judged by execution) (n = 1 questions, 1 calls) | not run at this call site (the replay covered agent_cg, agent_ir, agent_ss, extract_keywords, filter_column, select_columns, select_tables) |
| challenging | generate_candidate | EX 7/32 = 0.219 [0.094, 0.375] (n = 32 questions, 32 calls) | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 3/32 = 0.094 [0.000, 0.188]; raw agreement: not applicable (judged by execution) (n = 32 questions, 32 calls) | EX 6/32 = 0.188 [0.062, 0.312]; raw agreement: not applicable (judged by execution) (n = 32 questions, 32 calls) | EX 7/32 = 0.219 [0.094, 0.375]; raw agreement: not applicable (judged by execution) (n = 32 questions, 32 calls) | not run at this call site (the replay covered agent_cg, agent_ir, agent_ss, extract_keywords, filter_column, select_columns, select_tables) |
| challenging | revise | EX 0/1 = 0.000 [0.000, 0.000] (n = 1 questions, 1 calls) | not defined in SPEC (the distance behind "nearest training question" is not fixed; its EX would also need a gold evaluation that does not exist) | EX 0/1 = 0.000 [0.000, 0.000]; raw agreement: not applicable (judged by execution) (n = 1 questions, 1 calls) | EX 0/1 = 0.000 [0.000, 0.000]; raw agreement: not applicable (judged by execution) (n = 1 questions, 1 calls) | EX 0/1 = 0.000 [0.000, 0.000]; raw agreement: not applicable (judged by execution) (n = 1 questions, 1 calls) | not run at this call site (the replay covered agent_cg, agent_ir, agent_ss, extract_keywords, filter_column, select_columns, select_tables) |

## Latency and tokens per call, by engine (final attempt of each invocation)

| call site | engine | calls | latency p50 / p95 (ms) | mean tokens in / out per call | attempts per invocation |
|---|---|---|---|---|---|
| filter_column | teacher (registered B0 run) | 13138 | 2008 / 3566 | 5879 / 136 | 1.000 |
| filter_column | base SLM (Qwen3-8B, no adapter) | 13138 | 7465 / 18325 | 4052 / 84 | 1.000 |
| filter_column | SLM + adapters (B4's replay) | 13138 | 12186 / 24217 | 4052 / 91 | 1.000 |
| filter_column | cheaper LLM (gpt-oss) | 13138 | 3066 / 6772 | 16142 / 206 | 1.152 |
| filter_column | teacher's own repetition | 3311 | 1957 / 3060 | 5891 / 139 | 1.000 |
| filter_column | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| select_tables | teacher (registered B0 run) | 200 | 2439 / 3979 | 1872 / 211 | 1.000 |
| select_tables | base SLM (Qwen3-8B, no adapter) | 200 | 10115 / 19782 | 1241 / 154 | 1.000 |
| select_tables | SLM + adapters (B4's replay) | 200 | 12495 / 26964 | 1241 / 136 | 1.000 |
| select_tables | cheaper LLM (gpt-oss) | 200 | 2822 / 6561 | 4535 / 321 | 1.140 |
| select_tables | teacher's own repetition | 50 | 2510 / 4105 | 2010 / 239 | 1.000 |
| select_tables | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| select_columns | teacher (registered B0 run) | 200 | 2219 / 3681 | 1637 / 221 | 1.000 |
| select_columns | base SLM (Qwen3-8B, no adapter) | 200 | 10387 / 21140 | 1125 / 165 | 1.000 |
| select_columns | SLM + adapters (B4's replay) | 200 | 12331 / 28648 | 1125 / 147 | 1.000 |
| select_columns | cheaper LLM (gpt-oss) | 200 | 2928 / 6752 | 3913 / 343 | 1.150 |
| select_columns | teacher's own repetition | 50 | 2221 / 3620 | 1797 / 247 | 1.000 |
| select_columns | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| generate_candidate | teacher (registered B0 run) | 199 | 4822 / 12794 | 15621 / 833 | 1.000 |
| generate_candidate | base SLM (Qwen3-8B, no adapter) | 199 | 32295 / 94110 | 9929 / 685 | 1.000 |
| generate_candidate | SLM + adapters (B4's replay) | 199 | 34424 / 71127 | 9929 / 476 | 1.000 |
| generate_candidate | cheaper LLM (gpt-oss) | 199 | 3874 / 17444 | 39997 / 414 | 1.126 |
| generate_candidate | teacher's own repetition | no invocation | — | — | — |
| generate_candidate | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| revise | teacher (registered B0 run) | 5 | 11469 / 12519 | 6097 / 1448 | 1.000 |
| revise | base SLM (Qwen3-8B, no adapter) | 5 | 58217 / 66739 | 3901 / 802 | 1.000 |
| revise | SLM + adapters (B4's replay) | 5 | 27174 / 37663 | 3901 / 374 | 1.000 |
| revise | cheaper LLM (gpt-oss) | 5 | 6951 / 14735 | 15052 / 981 | 1.000 |
| revise | teacher's own repetition | no invocation | — | — | — |
| revise | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| extract_keywords | teacher (registered B0 run) | 200 | 1765 / 4491 | 1071 / 83 | 1.005 |
| extract_keywords | base SLM (Qwen3-8B, no adapter) | 200 | 3494 / 11008 | 707 / 41 | 1.005 |
| extract_keywords | SLM + adapters (B4's replay) | 200 | 5259 / 15432 | 707 / 61 | 1.000 |
| extract_keywords | cheaper LLM (gpt-oss) | 200 | 3817 / 8755 | 2959 / 420 | 1.095 |
| extract_keywords | teacher's own repetition | 50 | 1741 / 5114 | 1082 / 98 | 1.000 |
| extract_keywords | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| agent_ir | teacher (registered B0 run) | 800 | 1312 / 1804 | 377 / 17 | 1.000 |
| agent_ir | base SLM (Qwen3-8B, no adapter) | 800 | 1276 / 2435 | 246 / 5 | 1.002 |
| agent_ir | SLM + adapters (B4's replay) | 800 | 935 / 2099 | 246 / 4 | 1.000 |
| agent_ir | cheaper LLM (gpt-oss) | 800 | 1774 / 3969 | 1080 / 161 | 1.103 |
| agent_ir | teacher's own repetition | 200 | 1245 / 1549 | 377 / 17 | 1.000 |
| agent_ir | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| agent_ss | teacher (registered B0 run) | 800 | 1449 / 2393 | 345 / 42 | 1.000 |
| agent_ss | base SLM (Qwen3-8B, no adapter) | 800 | 1095 / 2633 | 232 / 5 | 1.000 |
| agent_ss | SLM + adapters (B4's replay) | 800 | 1261 / 2492 | 232 / 5 | 1.006 |
| agent_ss | cheaper LLM (gpt-oss) | 800 | 1699 / 4894 | 1033 / 190 | 1.144 |
| agent_ss | teacher's own repetition | 200 | 1246 / 2076 | 345 / 47 | 1.000 |
| agent_ss | trivial rule (no language model) | no model call | no model call | no model call | no model call |
| agent_cg | teacher (registered B0 run) | 598 | 1947 / 2595 | 337 / 75 | 1.000 |
| agent_cg | base SLM (Qwen3-8B, no adapter) | 598 | 1093 / 7812 | 230 / 23 | 1.000 |
| agent_cg | SLM + adapters (B4's replay) | 598 | 1187 / 2625 | 230 / 5 | 1.002 |
| agent_cg | cheaper LLM (gpt-oss) | 598 | 1543 / 2975 | 1013 / 128 | 1.162 |
| agent_cg | teacher's own repetition | 150 | 1343 / 2428 | 337 / 67 | 1.000 |
| agent_cg | trivial rule (no language model) | no model call | no model call | no model call | no model call |

The teacher's repetition rows cover its sampled questions only. Latency is as each run logged it, under that run's own concurrency and serving; it is not a load test.

