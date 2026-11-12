# Reviewer 3 tables (all numbers recomputed from runs/<id>/calls.jsonl, manifests, eval results.jsonl, J8 result, config.yaml)

## T1. Per arm: calls, tokens, cost (498 test questions). API $ = usage x config.yaml prices (J3 `standard`); SLM $ J8 = calls x J8 cost per request at 20/50/100% utilisation (B4 as-deployed sweep, B3 base sweep, B5 per-adapter sweeps); SLM $ GPU-time = sum of the run's session wall hours x $1.79/h
| arm | EX | calls | calls/q | SLM calls | in Mtok | cached | out Mtok | API $ | API $ no cache | SLM $ J8 20/50/100% | SLM $ GPU-time | error calls | unbilled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 407/498 (81.7%) | 45801 | 92.0 | 0 | 237.92 | 93% | 6.149 | 137.68 | 537.33 | - | - | 12 | 10 |
| B2-prod | 416/498 (83.5%) | 500 | 1.0 | 0 | 10.40 | 71% | 0.395 | 11.44 | 24.75 | - | - | 2 | 2 |
| B2-cheap | 385/498 (77.3%) | 500 | 1.0 | 0 | 6.52 | 97% | 0.240 | 0.24 | 0.24 | - | - | 2 | 0 |
| B3 | 265/498 (53.2%) | 24115 | 48.4 | 24093 | 77.23 | 93% | 2.025 | 0.00 | 0.00 | 11.92/4.77/2.38 | 2.01 | 484 | 22 |
| B4 | 332/498 (66.7%) | 46244 | 92.9 | 45840 | 163.54 | 96% | 3.984 | 0.00 | 0.00 | 24.73/9.89/4.95 | 4.79 | 424 | 404 |
| B5 | 411/498 (82.5%) | 46310 | 93.0 | 43737 | 167.13 | 95% | 4.226 | 12.81 | 27.66 | 13.14/5.26/2.63 | 5.75 | 523 | 514 |

## T2. Latency (s). Question wall = last call end - first call start within the attempt that completed the question (includes tool/router time between calls; questions run `workers` at a time, calls within a question up to 32 at a time)
| arm | workers by session | sessions | session wall min | q/h achieved | q wall p50 | q wall p95 | q sum-of-call-latency p50 | call p50 | call p95 | SLM call p50 | SLM call p95 | API call p50 | API call p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 3/5/8 | 5 | 151.5 | 197 | 55.6 | 83.3 | 166.9 | 1.98 | 3.18 | - | - | 1.98 | 3.17 |
| B2-prod | 1 | 1 | 58.6 | 510 | 5.0 | 6.9 | 4.9 | 4.94 | 6.86 | - | - | 4.94 | 6.85 |
| B2-cheap | 1 | 1 | 32.9 | 908 | 3.5 | 6.6 | 3.5 | 3.54 | 6.54 | - | - | 3.54 | 6.54 |
| B3 | 8 | 1 | 67.3 | 444 | 53.8 | 98.5 | 122.7 | 3.83 | 11.31 | 3.83 | 11.31 | - | - |
| B4 | 3/8 | 3 | 160.7 | 186 | 59.7 | 136.6 | 380.7 | 5.19 | 21.62 | 5.23 | 21.66 | - | - |
| B5 | 3 | 11 | 192.6 | 155 | 52.5 | 97.2 | 268.1 | 3.26 | 7.93 | 3.28 | 8.00 | 2.65 | 6.01 |

## T3. Cost per question / per correct answer (marginal). SLM arms under four views
| arm | EX % | $/q (API + SLM@100%) | $/correct @100% | $/correct @50% | $/correct @20% | $/q GPU-time view | $/correct GPU-time view |
|---|---|---|---|---|---|---|---|
| B0 | 81.7 | 0.2765 | 0.3383 | - | - | - | - |
| B2-prod | 83.5 | 0.0230 | 0.0275 | - | - | - | - |
| B2-cheap | 77.3 | 0.0005 | 0.0006 | - | - | - | - |
| B3 | 53.2 | 0.0048 | 0.0090 | 0.0180 | 0.0450 | 0.0040 | 0.0076 |
| B4 | 66.7 | 0.0099 | 0.0149 | 0.0298 | 0.0745 | 0.0096 | 0.0144 |
| B5 | 82.5 | 0.0310 | 0.0376 | 0.0440 | 0.0631 | 0.0373 | 0.0452 |

## T4. Per call site (per question averages over 498). cost = API $ + SLM $ at J8 100% (20% in brackets); share of the arm's total at 100%
| arm | cluster | call site | calls/q | on SLM | in tok/q | cached | out tok/q | $/q @100% (@20%) | cost share | lat p50 s | lat p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | c0 | filter_column | 76.96 | 0% | 453,591 | 95% | 10,573 | 0.23865 (0.23865) | 86.3% | 2.01 | 2.94 |
| B0 | c1 | select_tables | 1.00 | 0% | 1,933 | 0% | 215 | 0.00601 (0.00601) | 2.2% | 2.52 | 3.60 |
| B0 | c1 | select_columns | 1.00 | 0% | 1,660 | 0% | 218 | 0.00550 (0.00550) | 2.0% | 2.37 | 3.22 |
| B0 | c1 | generate_candidate | 1.00 | 0% | 15,559 | 95% | 786 | 0.01237 (0.01237) | 4.5% | 5.11 | 6.86 |
| B0 | c1 | revise | 0.00 | 0% | 23 | 42% | 4 | 0.00008 (0.00008) | 0.0% | 6.80 | 8.97 |
| B0 | c2 | extract_keywords | 1.00 | 0% | 1,088 | 85% | 99 | 0.00152 (0.00152) | 0.6% | 1.75 | 5.43 |
| B0 | c3 | agent_ir | 4.00 | 0% | 1,508 | 0% | 66 | 0.00368 (0.00368) | 1.3% | 1.15 | 1.69 |
| B0 | c3 | agent_ss | 4.00 | 0% | 1,380 | 0% | 192 | 0.00468 (0.00468) | 1.7% | 1.24 | 2.15 |
| B0 | c3 | agent_cg | 3.00 | 0% | 1,012 | 0% | 196 | 0.00399 (0.00399) | 1.4% | 1.47 | 2.44 |

| arm | cluster (by call site) | calls/q | call share | on SLM | token share | $/q @100% | cost share | $/q @20% |
|---|---|---|---|---|---|---|---|---|
| B0 | c0 | 76.96 | 83.7% | 0% | 94.7% | 0.23865 | 86.3% | 0.23865 |
| B0 | c1 | 3.00 | 3.3% | 0% | 4.2% | 0.02396 | 8.7% | 0.02396 |
| B0 | c2 | 1.00 | 1.1% | 0% | 0.2% | 0.00152 | 0.6% | 0.00152 |
| B0 | c3 | 11.00 | 12.0% | 0% | 0.9% | 0.01234 | 4.5% | 0.01234 |

| arm | cluster | call site | calls/q | on SLM | in tok/q | cached | out tok/q | $/q @100% (@20%) | cost share | lat p50 s | lat p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B3 | c0 | filter_column | 33.17 | 100% | 134,920 | 96% | 2,824 | 0.00328 (0.01642) | 68.6% | 4.51 | 11.22 |
| B3 | c1 | select_tables | 1.00 | 100% | 5,035 | 15% | 151 | 0.00010 (0.00049) | 2.1% | 3.62 | 11.24 |
| B3 | c1 | select_columns | 1.00 | 100% | 1,198 | 8% | 169 | 0.00010 (0.00050) | 2.1% | 3.79 | 10.66 |
| B3 | c1 | generate_candidate | 1.00 | 100% | 9,854 | 96% | 638 | 0.00010 (0.00049) | 2.1% | 12.42 | 28.05 |
| B3 | c1 | revise | 0.21 | 100% | 768 | 80% | 132 | 0.00002 (0.00010) | 0.4% | 11.09 | 20.08 |
| B3 | c2 | extract_keywords | 1.00 | 100% | 716 | 85% | 45 | 0.00010 (0.00049) | 2.1% | 1.40 | 3.12 |
| B3 | c3 | agent_ir | 4.04 | 99% | 982 | 96% | 19 | 0.00040 (0.00198) | 8.3% | 0.45 | 0.93 |
| B3 | c3 | agent_ss | 4.00 | 100% | 926 | 97% | 19 | 0.00040 (0.00198) | 8.3% | 0.40 | 0.91 |
| B3 | c3 | agent_cg | 3.00 | 100% | 689 | 95% | 69 | 0.00030 (0.00148) | 6.2% | 0.75 | 2.61 |

| arm | cluster (by call site) | calls/q | call share | on SLM | token share | $/q @100% | cost share | $/q @20% |
|---|---|---|---|---|---|---|---|---|
| B3 | c0 | 33.17 | 68.5% | 100% | 86.5% | 0.00328 | 68.6% | 0.01642 |
| B3 | c1 | 3.21 | 6.6% | 100% | 11.3% | 0.00032 | 6.6% | 0.00159 |
| B3 | c2 | 1.00 | 2.1% | 100% | 0.5% | 0.00010 | 2.1% | 0.00049 |
| B3 | c3 | 11.04 | 22.8% | 100% | 1.7% | 0.00109 | 22.7% | 0.00544 |

| arm | cluster | call site | calls/q | on SLM | in tok/q | cached | out tok/q | $/q @100% (@20%) | cost share | lat p50 s | lat p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B4 | c0 | filter_column | 77.69 | 99% | 312,544 | 96% | 7,102 | 0.00830 (0.04151) | 83.6% | 5.94 | 22.75 |
| B4 | c1 | select_tables | 1.00 | 100% | 1,249 | 17% | 137 | 0.00011 (0.00054) | 1.1% | 3.02 | 7.74 |
| B4 | c1 | select_columns | 1.00 | 100% | 1,105 | 10% | 143 | 0.00011 (0.00054) | 1.1% | 2.67 | 5.06 |
| B4 | c1 | generate_candidate | 1.00 | 100% | 9,880 | 96% | 474 | 0.00011 (0.00054) | 1.1% | 8.67 | 13.31 |
| B4 | c1 | revise | 0.08 | 100% | 301 | 64% | 24 | 0.00001 (0.00004) | 0.1% | 4.96 | 9.75 |
| B4 | c2 | extract_keywords | 1.01 | 100% | 721 | 85% | 70 | 0.00011 (0.00054) | 1.1% | 1.56 | 2.65 |
| B4 | c3 | agent_ir | 4.01 | 100% | 982 | 96% | 17 | 0.00043 (0.00216) | 4.3% | 0.31 | 0.72 |
| B4 | c3 | agent_ss | 4.05 | 99% | 926 | 97% | 19 | 0.00043 (0.00216) | 4.3% | 0.37 | 0.97 |
| B4 | c3 | agent_cg | 3.01 | 100% | 689 | 95% | 14 | 0.00032 (0.00162) | 3.3% | 0.37 | 0.69 |

| arm | cluster (by call site) | calls/q | call share | on SLM | token share | $/q @100% | cost share | $/q @20% |
|---|---|---|---|---|---|---|---|---|
| B4 | c0 | 77.69 | 83.7% | 99% | 95.0% | 0.00830 | 83.6% | 0.04151 |
| B4 | c1 | 3.09 | 3.3% | 100% | 4.0% | 0.00033 | 3.4% | 0.00166 |
| B4 | c2 | 1.01 | 1.1% | 100% | 0.2% | 0.00011 | 1.1% | 0.00054 |
| B4 | c3 | 11.07 | 11.9% | 99% | 0.8% | 0.00119 | 12.0% | 0.00593 |

| arm | cluster | call site | calls/q | on SLM | in tok/q | cached | out tok/q | $/q @100% (@20%) | cost share | lat p50 s | lat p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B5 | c0 | filter_column | 77.82 | 99% | 312,774 | 96% | 7,104 | 0.00580 (0.02685) | 18.7% | 3.47 | 8.23 |
| B5 | c1 | select_tables | 1.00 | 0% | 1,911 | 0% | 212 | 0.00594 (0.00594) | 19.2% | 2.74 | 3.96 |
| B5 | c1 | select_columns | 1.00 | 0% | 1,638 | 0% | 222 | 0.00550 (0.00550) | 17.7% | 2.42 | 3.50 |
| B5 | c1 | generate_candidate | 1.00 | 0% | 15,556 | 96% | 794 | 0.01221 (0.01221) | 39.4% | 5.10 | 6.89 |
| B5 | c1 | revise | 0.01 | 0% | 34 | 85% | 5 | 0.00006 (0.00006) | 0.2% | 4.50 | 10.99 |
| B5 | c2 | extract_keywords | 1.00 | 0% | 1,088 | 87% | 99 | 0.00147 (0.00147) | 4.7% | 1.78 | 5.00 |
| B5 | c3 | agent_ir | 4.04 | 99% | 982 | 96% | 17 | 0.00001 (0.00003) | 0.0% | 0.39 | 0.78 |
| B5 | c3 | agent_ss | 4.03 | 99% | 926 | 97% | 19 | 0.00001 (0.00003) | 0.0% | 0.38 | 0.83 |
| B5 | c3 | agent_cg | 3.10 | 97% | 689 | 95% | 14 | 0.00000 (0.00002) | 0.0% | 0.35 | 0.80 |

| arm | cluster (by call site) | calls/q | call share | on SLM | token share | $/q @100% | cost share | $/q @20% |
|---|---|---|---|---|---|---|---|---|
| B5 | c0 | 77.82 | 83.7% | 99% | 93.0% | 0.00580 | 18.7% | 0.02685 |
| B5 | c1 | 3.01 | 3.2% | 0% | 5.9% | 0.02372 | 76.5% | 0.02372 |
| B5 | c2 | 1.00 | 1.1% | 0% | 0.3% | 0.00147 | 4.7% | 0.00147 |
| B5 | c3 | 11.16 | 12.0% | 99% | 0.8% | 0.00002 | 0.0% | 0.00008 |

### B4/B5 recorded router cluster vs call-site cluster (calls, SLM calls)
- B4: c0: 38629 calls (38261 SLM), c1: 1600 calls (1600 SLM), c2: 501 calls (501 SLM), c3: 5514 calls (5478 SLM)
- B5: c0: 38690 calls (38259 SLM), c1: 1562 calls (0 SLM), c2: 499 calls (0 SLM), c3: 5559 calls (5478 SLM)

## T5. Allocation counterfactuals: B0's measured calls, clusters in S moved to the SLM at J8's per-adapter cost per request (c1 adapter is outside its latency limit in J8: no registered price; priced here, labelled ESTIMATE, at its best load level regardless of the limit: 2.69 req/s at concurrency 64 -> $1.79/3600/2.69 per request at 100%)
| clusters on SLM | $/q @100% | saving vs B0 | $/q @20% | saving vs B0 |
|---|---|---|---|---|
| {} | 0.2765 | -0.0% | 0.2765 | -0.0% |
| {c0} | 0.0431 | 84.4% | 0.0642 | 76.8% |
| {c3} | 0.2642 | 4.5% | 0.2642 | 4.4% |
| {c0,c3} | 0.0308 | 88.9% | 0.0519 | 81.2% |
| {c0,c2,c3} | 0.0293 | 89.4% | 0.0505 | 81.7% |
| {c0,c1,c2,c3} | 0.0059 | 97.9% | 0.0293 | 89.4% |

Measured B5: API $12.81 + SLM J8 $2.63 (100%) / $13.14 (20%) vs B0 API $137.68: saving 88.8% @100%, 81.2% @20%, 86.5% in the GPU-time view ($5.75 of pod time over all 11 sessions).

## T6. Break-even of a dedicated $1.79/h GPU ($42.96/day) against the API
| comparison | API $/q before | API $/q after | API saving $/q | break-even q/h | break-even q/day (24h) | GPU capacity q/h within latency limit | note |
|---|---|---|---|---|---|---|---|
| B5 vs B0 (standard price) | 0.2765 | 0.0257 | 0.2508 | 7 | 171 | 340 | capacity = c0 sweep req/h / c0 calls per q |
| B5 vs B0 (B0 and B5 API at batch -50%) | 0.1382 | 0.0129 | 0.1254 | 14 | 343 | 340 | batch is not latency-comparable |
| B4 vs B0 (ignores B4's -15 pp EX) | 0.2765 | 0.0000 | 0.2765 | 6 | 155 | 179 | as-deployed sweep |
| B5 vs B2-prod single call | 0.0230 | 0.0257 | -0.0028 | never | never | 340 | B5's API residue alone vs B2-prod |
| B4 vs B2-prod (ignores -17 pp EX) | 0.0230 | 0.0000 | 0.0230 | 78 | 1,870 | 179 |  |

## T7. SLM server throughput actually reached in the test runs (one A100 80GB, vLLM 0.30, prefix cache on)
| arm | SLM calls | session wall h | calls/s | input tok/s | output tok/s | prefix-cached input | mean in-flight (sum latency / wall) | peak in-flight | calls/h achieved | implied $/Mtok (GPU-time / all tokens) |
|---|---|---|---|---|---|---|---|---|---|---|
| B3 | 24093 | 1.12 | 5.97 | 19,135 | 501.7 | 93% | 26.31 | 203 | 21,489 | 0.025 |
| B4 | 45840 | 2.68 | 4.75 | 16,963 | 413.3 | 96% | 32.78 | 256 | 17,117 | 0.029 |
| B5 | 43737 | 3.21 | 3.78 | 13,553 | 307.5 | 96% | 14.08 | 96 | 13,622 | 0.036 |

| J8 sweep (calibration traffic) | p95 limit ms | sustained concurrency | sustained req/s | req/h | $ per 1000 req @100% | max req/s at any level |
|---|---|---|---|---|---|---|
| base | 3,135 | 8 | 5.02 | 18,076 | 0.0990 | 9.07 @c128 (p95 18.2s) |
| base+as-deployed | 2,933 | 8 | 4.61 | 16,585 | 0.1079 | 8.72 @c128 (p95 18.2s) |
| base+lora:c0-842db723c47e | 2,801 | 16 | 7.26 | 26,121 | 0.0685 | 9.54 @c64 (p95 7.9s) |
| base+lora:c1-fa3a060fdb1c | 7,364 | none | n/a | n/a | n/a | 2.69 @c64 (p95 47.2s) |
| base+lora:c2-64efb1484f8c | 4,491 | 32 | 20.75 | 74,694 | 0.0240 | 25.67 @c64 (p95 3.1s) |
| base+lora:c3-22febb722a91 | 2,396 | 128 | 359.53 | 1,294,323 | 0.0014 | 359.53 @c128 (p95 0.4s) |

## T8. Ledger of logged model calls (all runs/ directories with calls.jsonl except the *-B1-* ones, which were not read; de-duplicated by call_id in directory-name order). `paid est.` = standard list price, x0.5 for Sonnet calls of runs whose manifest says batch, 0 for runs sent through a subscription. Embedding calls (OpenAI) are not in calls.jsonl and are not priced.
| run group | dirs | call lines | unique new calls | SLM calls | in Mtok | out Mtok | API $ list | API $ paid est. | unpriced models |
|---|---|---|---|---|---|---|---|---|---|
| _invalid/agent-B0-train | 1 | 29064 | 29064 | 0 | 140.9 | 3.73 | 82.17 | 82.17 | 0 |
| agent-B0-calib | 2 | 21419 | 16348 | 0 | 83.1 | 2.18 | 48.38 | 48.38 | 0 |
| agent-B0-test | 5 | 62272 | 45961 | 0 | 238.6 | 6.17 | 138.11 | 138.11 | 0 |
| agent-B0-train | 19 | 285344 | 60965 | 0 | 279.5 | 7.04 | 156.11 | 156.11 | 616 |
| agent-B0-train [batch] | 1 | 525 | 525 | 0 | 2.7 | 0.07 | 1.69 | 0.84 | 0 |
| agent-B0-train [subscription] | 3 | 2595 | 2595 | 0 | 13.8 | 0.35 | 57.26 | 0.00 | 0 |
| agent-B2-cheap_alt-calib | 1 | 200 | 200 | 0 | 2.5 | 0.11 | 0.09 | 0.09 | 0 |
| agent-B2-cheap_alt-test | 1 | 500 | 500 | 0 | 6.5 | 0.24 | 0.24 | 0.24 | 0 |
| agent-B2-cheap_alt-train | 2 | 4 | 4 | 0 | 0.1 | 0.00 | 0.00 | 0.00 | 0 |
| agent-B2-production_llm-test | 1 | 500 | 500 | 0 | 10.4 | 0.40 | 11.44 | 11.44 | 0 |
| agent-B2-production_llm-train | 2 | 4 | 4 | 0 | 0.1 | 0.00 | 0.11 | 0.11 | 0 |
| agent-B3-calib | 5 | 16205 | 13591 | 13578 | 43.7 | 1.19 | 0.00 | 0.00 | 0 |
| agent-B3-test | 1 | 24115 | 24115 | 24093 | 77.2 | 2.02 | 0.00 | 0.00 | 0 |
| agent-B4-calib | 3 | 29807 | 16422 | 16412 | 57.3 | 1.40 | 0.00 | 0.00 | 0 |
| agent-B4-test | 2 | 56215 | 46244 | 45840 | 163.5 | 3.98 | 0.00 | 0.00 | 0 |
| agent-B5-test | 11 | 176880 | 51475 | 48104 | 183.6 | 4.65 | 14.10 | 14.10 | 0 |
| replay-B4-calib | 1 | 16152 | 16152 | 16140 | 56.4 | 1.37 | 0.00 | 0.00 | 0 |
| replay-B4-test | 2 | 89859 | 46240 | 45800 | 163.4 | 3.97 | 0.00 | 0.00 | 0 |
| replay-cheap_alt-calib | 6 | 33215 | 27526 | 0 | 302.9 | 5.41 | 10.00 | 10.00 | 0 |
| replay-production_llm-calib | 1 | 4011 | 4011 | 0 | 19.9 | 0.51 | 11.71 | 11.71 | 0 |
| replay-production_llm-test | 1 | 9093 | 9093 | 0 | 47.1 | 1.24 | 27.39 | 27.39 | 0 |
| replay-slm_granite-4.2-8b-calib | 1 | 16228 | 16228 | 16224 | 61.7 | 3.31 | 0.00 | 0.00 | 0 |
| replay-slm_qwen3-8b-calib | 2 | 32477 | 32477 | 32285 | 112.8 | 2.67 | 0.00 | 0.00 | 0 |
| replay-slm_qwen3-8b-test | 2 | 84536 | 46683 | 46099 | 164.6 | 3.81 | 0.00 | 0.00 | 0 |
| TOTAL (excl. B1) |  |  | 506923 | 304575 |  |  | 558.81 | 500.71 |  |

Skipped (not read): 21 *-B1-* directories with calls.jsonl.

## T9. GPU pods (.runpod/state.json: created_at -> terminated_at x cost_per_hour; excludes the network volume's storage fee and any CPU pods not listed)
| pod | role | candidate | created (UTC) | terminated (UTC) | hours | $/h | $ | adapters |
|---|---|---|---|---|---|---|---|---|
| ffy86d4gfktfpv | download | qwen3-8b | 2026-11-09T23:48 | 2026-11-10T00:07 | 0.31 | 1.79 | 0.56 | 0 |
| 6n8mywpcubaivr | serve | granite-4.2-8b | 2026-11-10T02:09 | 2026-11-10T02:39 | 0.50 | 1.79 | 0.90 | 0 |
| 05qx1swv1oq2sn | serve | granite-4.2-8b | 2026-11-10T02:39 | 2026-11-10T03:09 | 0.50 | 1.79 | 0.90 | 0 |
| 9mg4iongeb1lxi | serve | qwen3-8b | 2026-11-10T00:14 | 2026-11-10T05:24 | 5.17 | 1.79 | 9.25 | 0 |
| kw329b06mou4yh | serve | qwen3-8b | 2026-11-10T07:05 | 2026-11-10T07:36 | 0.51 | 1.79 | 0.92 | 1 |
| 86ovr3x89eh3bx | serve | qwen3-8b | 2026-11-10T16:12 | 2026-11-11T05:33 | 13.35 | 1.79 | 23.90 | 4 |
| sg5e07uhpcomz6 | serve | qwen3-8b | 2026-11-11T08:05 | 2026-11-11T08:19 | 0.24 | 1.79 | 0.42 | 4 |
| ejuv1ctuz0z2lw | serve | qwen3-8b | 2026-11-11T15:30 | 2026-11-11T16:05 | 0.59 | 1.79 | 1.05 | 4 |
| 92e3cqaubrafmj | serve | qwen3-8b | 2026-11-11T16:43 | 2026-11-11T20:20 | 3.62 | 1.79 | 6.48 | 4 |
| 5z556nr7hs2ql2 | serve | qwen3-8b | 2026-11-11T20:27 | 2026-11-11T23:08 | 2.68 | 1.79 | 4.80 | 4 |
| TOTAL |  |  |  |  | 27.48 |  | 49.19 |  |

## T10. Training (train/adapters/c*/manifest.json): one LoRA r=16 per cluster, 2 epochs, hosted fine-tuning service
| cluster | rows | dataset Mtok | Mtok seen (2 epochs) | completion Mtok | steps | train min | tok/s | longest row tok | cost_usd | backend |
|---|---|---|---|---|---|---|---|---|---|---|
| c0 | 11982 | 49.58 | 99.14 | 1.111 | 1498 | 79.3 | 20,832 | 4522 | None | tinker |
| c1 | 2517 | 10.85 | 21.70 | 0.608 | 316 | 21.6 | 16,743 | 21150 | None | tinker |
| c2 | 833 | 0.63 | 1.27 | 0.048 | 106 | 6.7 | 3,132 | 1047 | None | tinker |
| c3 | 9164 | 2.23 | 4.44 | 0.064 | 1146 | 61.0 | 1,212 | 286 | None | tinker |
| TOTAL | 24496 |  | 126.54 |  |  | 168.7 |  |  | not recorded |  |

## T11. Router time per SLM-routed call (runs/<B4|B5 test>/routes.jsonl): the local embedding that picks the adapter
| arm | routes | embed p50 ms | embed p95 ms | embed wait p50 ms | embed wait p95 ms | route p50 ms | route p95 ms | route s per question (sum) | of which wait s/q |
|---|---|---|---|---|---|---|---|---|---|
| B4 | 45829 | 51 | 97 | 44 | 1458 | 102 | 1536 | 42.8 | 34.6 |
| B5 | 45791 | 60 | 291 | 286 | 5480 | 352 | 5748 | 121.0 | 111.4 |

## T12. Teacher-trace collection (K1) and reconstructed totals

K1 final run (agent-B0-train-20261109T214533): calls 82403, questions 834, in 396.7 Mtok, out 10.15 Mtok, list cost $225.78, $0.2707/q, sessions 7, session hours 3.49, n_predictions None

- All B0-on-train spend at list price, unique calls (T8): $82.17 (a completed 834-question run later moved to runs/_invalid; part of its calls are reused by the final run's resume chain) + $156.11 + $0.84 (batch probe) = $239.12 paid est.; $57.26 more at list price went through a subscription (not billed per token).
- Reconstructed total, excluding everything not logged or not read: API $500.71 (T8, list price, B1 excluded) + GPU pods $49.19 (T9) = $549.90. Not included: every *-B1-* run (not read), OpenAI embedding calls (not in calls.jsonl), the hosted fine-tuning charges (cost_usd null in all four adapter manifests), network-volume storage, CPU pods of the load-test client.
- Test phase only, including failed attempts and discarded calls: B0 $138.11, B2-prod $11.44, B2-cheap $0.24, B5 API $14.10; SLM pods on 2026-11-11 serving B3/B4/B5 tests share pods 86ovr (13.35 h, also calibration and load tests), sg5e, ejuv, 92e3 (4.45 h = $7.97 for the B5 attempts).
- Pod utilisation: test-arm sessions used 1.12 h (B3) + 2.68 h (B4) + 3.21 h (B5) = 7.01 h of 27.48 pod-hours (26%).
- B5 at the benchmark's actual volume: API $14.10 (with discarded calls) + $7.97 pods = $22.07 against B0 $138.11 (-84%) and B2-prod $11.44 (+93%).
