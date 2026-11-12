# Cost, latency and capacity of the finished arms (preliminary, 2026-11-12)

**Status: exploratory, not the registered report.** Everything on this page is recomputed from data
already logged by the five finished arms of the test (B0, B2 on two engines, B3, B4, B5; 498 questions).
No execution was made for it. B1 finished after this page was computed and is not in it (382 / 498 =
76.7%; computed with a one-line patch of the cost judgment, which otherwise refuses three empty calls, its
API cost is about US$ 0.025 per question, a lower bound; its latency, and the load-test latency per task, are in
[`results.md`](results.md), section 6). The registered cost comparisons come only from
`bench report`; nothing below is a verdict, and every table
is labelled exploratory. Accuracy and its intervals are in
[`preliminary-results.md`](preliminary-results.md); the limits listed there (one execution per arm,
resumed sessions, the single call scoring at least as high as the agent) apply to every number here.

Arms: **B0** the CHESS agent on the production LLM (Claude Sonnet 5.5); **B2** one call, no agent, on the
production LLM and on the cheaper LLM (gpt-oss-120b); **B3** the agent on base Qwen3-8B; **B4** the agent
on Qwen3-8B with one LoRA adapter per cluster; **B5** clusters c0 (column filter) and c3 (agent steps) on
the small model (SLM), c1 (table and column selection, SQL generation, revision) and c2 (keyword
extraction) on the production LLM.

## 0. Sources and definitions

| What | Source |
|---|---|
| Calls, tokens, latency | `runs/<run id>/calls.jsonl` of the six runs in section 10, every line (retries and failed calls included) |
| Correct answers | `results.jsonl` of each run's end-to-end `eval` execution |
| API prices | `config.yaml`, `prices.table`, dated 2026-10-22: production LLM US$ 2.00 per million input tokens, 0.20 cached, 4.00 written to the cache, 10.00 output; cheaper LLM 0.03 input (cached or not), 0.17 output |
| SLM price, load-test view | judgment J8, `judgments/J8/de3754a7…/result.json`: cost per request of each sweep at 20, 50 and 100% utilisation, one A100 80GB at US$ 1.79 per hour |
| SLM price, as-run view | the run manifest's sessions (start to finish of each) at US$ 1.79 per hour |
| Router time | `runs/<run id>/routes.jsonl` (B4, B5) |
| Pod hours | `.runpod/state.json` (not in the public repository) |
| Training | `train/adapters/c*/manifest.json` |

- **API cost** is each call's logged `usage` times the list price, by the registered function
  (`bench.judge.j3.judge`, SPEC section 6.6), called as a library on the logged calls by a script outside
  the registered code. "No caching" prices the cached tokens at the input price, from the same `usage`.
- **SLM cost, load-test view** is the registered one: each SLM call costs its sweep's cost per request
  from J8. B3 is priced at the base model's sweep, B4 at the sweep of the server as deployed (four
  adapters, B4's own mix), B5 call by call at the sweep of the adapter that served it (an extrapolation:
  the adapters were load-tested one at a time, B5 serves two on one server). Utilisation is the share of
  the hour the GPU is busy at the throughput it sustains within the latency limit; the price at 20% is
  five times the price at 100%.
- **SLM cost, as-run view** is not registered: the wall time of the run's sessions times the hourly
  price, whatever the server was doing. It includes the idle time inside the sessions and leaves out the
  time the pod was up between them.
- **A failed call with no response has no usage and is not priced** (the registered rule; the totals
  are lower bounds). Counts are in the first table.
- Section 9 lists where a definition on this page departs from a registered one.

## 1. Per arm (exploratory)

Calls and tokens, whole run (498 questions):

| Arm | EX | Calls | Calls per question | Of which on the SLM (priced) | Input tokens, millions | Of which read from cache | Output tokens, millions | Input tokens per question | Output tokens per question | Failed calls, not priced |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 407 / 498 = 81.7% | 45,801 | 92.0 | 0 | 237.92 | 93% | 6.149 | 477,753 | 12,348 | 10 |
| B2, production LLM | 416 / 498 = 83.5% | 500 | 1.0 | 0 | 10.40 | 71% | 0.395 | 20,879 | 793 | 2 |
| B2, cheaper LLM | 385 / 498 = 77.3% | 500 | 1.0 | 0 | 6.52 | 97% | 0.240 | 13,102 | 482 | 0 |
| B3 | 265 / 498 = 53.2% | 24,115 | 48.4 | 24,093 | 77.23 | 93% | 2.025 | 155,088 | 4,066 | 22 |
| B4 | 332 / 498 = 66.7% | 46,244 | 92.9 | 45,840 | 163.54 | 96% | 3.984 | 328,397 | 8,001 | 404 |
| B5 | 411 / 498 = 82.5% | 46,310 | 93.0 | 43,737 | 167.13 | 95% | 4.226 | 335,597 | 8,485 | 514 |

Totals over the 498 questions, in US$:

| Arm | API, list price, with caching | API, list price, no caching | SLM serving, load-test price at 20% / 50% / 100% utilisation | SLM serving, GPU time as run (session wall time) | SLM serving, GPU time with a call in flight |
|---|---|---|---|---|---|
| B0 | 137.68 | 537.33 | – | – | – |
| B2, production LLM | 11.44 | 24.75 | – | – | – |
| B2, cheaper LLM | 0.24 | 0.24 | – | – | – |
| B3 | 0.00 | 0.00 | 11.92 / 4.77 / 2.38 | 2.01 | 1.87 |
| B4 | 0.00 | 0.00 | 24.73 / 9.89 / 4.95 | 4.79 | 3.88 |
| B5 | 12.81 | 27.66 | 13.14 / 5.26 / 2.63 | 5.75 | 3.27 |

Per question and per correct answer, in US$ (API at list price with caching, plus the SLM's serving):

| Arm | EX | Cost per question (SLM arms: at 20% / 50% / 100% utilisation) | Cost per question, GPU time as run | Cost per correct answer (SLM arms: at 20% / 50% / 100%) | Cost per correct answer, GPU time as run | Cost per question without caching (SLM at 100%) |
|---|---|---|---|---|---|---|
| B0 | 81.7% | 0.2765 | – | 0.3383 | – | 1.0790 |
| B2, production LLM | 83.5% | 0.0230 | – | 0.0275 | – | 0.0497 |
| B2, cheaper LLM | 77.3% | 0.0005 | – | 0.0006 | – | 0.0005 |
| B3 | 53.2% | 0.0239 / 0.0096 / 0.0048 | 0.0040 | 0.0450 / 0.0180 / 0.0090 | 0.0076 | 0.0048 |
| B4 | 66.7% | 0.0496 / 0.0199 / 0.0099 | 0.0096 | 0.0745 / 0.0298 / 0.0149 | 0.0144 | 0.0099 |
| B5 | 82.5% | 0.0521 / 0.0363 / 0.0310 | 0.0373 | 0.0631 / 0.0440 / 0.0376 | 0.0452 | 0.0608 |

- B0 is US$ 0.277 per question; 93% of its input tokens were read from the cache, and without caching
  the same calls would be US$ 1.08 per question.
- B5 is US$ 0.031 per question with the SLM at full utilisation and US$ 0.052 at 20%; US$ 0.0257 of it
  is the production LLM's remaining calls, at any utilisation.
- B4 and B3 have no API cost. At full utilisation the load-test view and the as-run view are close for
  them (B4: US$ 4.95 and 4.79 for the run). For B5 the as-run view is higher (US$ 5.75 against 2.63)
  because its eleven sessions held the server for 3.21 hours and had a call in flight for 1.83.
- At the sensitivity price of the registration (another provider's price for the same GPU with CPU and
  memory charged apart, US$ 3.13 per hour) every SLM serving figure above is multiplied by 1.75.

## 2. The cost–accuracy frontier (exploratory)

Ordered by cost per question (SLM at full utilisation, its best case):

| Arm | Cost per question (US$) | EX | Dominated by (lower cost and higher EX, point values) |
|---|---|---|---|
| B2, cheaper LLM | 0.0005 | 77.3% | – (on the frontier) |
| B3 | 0.0048 | 53.2% | B2 cheaper LLM |
| B4 | 0.0099 | 66.7% | B2 cheaper LLM |
| B2, production LLM | 0.0230 | 83.5% | – (on the frontier) |
| B5 | 0.0310 | 82.5% | B2 production LLM |
| B0 | 0.2765 | 81.7% | B2 production LLM, B5 |

- On point values, the frontier has two arms, and both are single calls with no agent: the cheaper LLM
  (lowest cost) and the production LLM (highest EX).
- Every agent arm is dominated by a single call. B3 and B4 cost about ten and twenty-one times the single call on
  the cheaper LLM and score 24 and 11 points below it. B5 costs 1.35 times the single call on the
  production LLM (1.6 times at 50% utilisation, 2.3 at 20%) and scores 1.0 point below it; the production
  LLM's calls that B5 keeps cost, alone, more than the whole single call (US$ 0.0257 against 0.0230).
- B0 is dominated by B5 as well: B5 costs 11% of B0 at full utilisation and 19% at 20%, and scores
  0.8 point above it.
- The same dominance relations hold per correct answer, at every utilisation, in the as-run view and
  without caching. At 20% utilisation B4 (US$ 0.0496) is also dominated by the single call on the production
  LLM.
- These are point values of one execution per arm. The EX differences of 1 point or less (B5, B0 and the
  single call on the production LLM) are inside the sampling intervals of `preliminary-results.md`,
  which has no interval for B5 against the single call. The cost differences are not: B0 costs nine
  times B5 and twelve times the single call.
- Latency points the same way: a single call answers in about 5 seconds, the agent arms in 50 to 60
  (section 4).

## 3. Where the cost is: clusters and call sites (exploratory)

Clusters are taken by call site here (the column filter is c0, and so on). In B4 and B5 the router
decides per call: it sent 63 (B4) and 64 (B5) of about 38,700 column-filter calls to cluster c1, which
is why B5 shows US$ 0.27 of API cost under c0. "Served by the SLM" counts the SLM calls that returned
a response; in B4 the remainder are SLM calls that failed, not calls sent elsewhere.

| Arm | Cluster (by call site) | Calls per question | Share of calls | Served by the SLM | Share of tokens | Cost per question (SLM at 100%) | Share of cost (SLM at 100%) | Cost per question (SLM at 20%) | Share of cost (SLM at 20%) |
|---|---|---|---|---|---|---|---|---|---|
| B0 | c0 | 76.96 | 83.7% | 0% | 94.7% | 0.23865 | 86.3% | – | – |
| B0 | c1 | 3.00 | 3.3% | 0% | 4.2% | 0.02396 | 8.7% | – | – |
| B0 | c2 | 1.00 | 1.1% | 0% | 0.2% | 0.00152 | 0.6% | – | – |
| B0 | c3 | 11.00 | 12.0% | 0% | 0.9% | 0.01234 | 4.5% | – | – |
| B4 | c0 | 77.69 | 83.7% | 99% | 95.0% | 0.00830 | 83.6% | 0.04151 | 83.6% |
| B4 | c1 | 3.09 | 3.3% | 100% | 4.0% | 0.00033 | 3.4% | 0.00166 | 3.4% |
| B4 | c2 | 1.01 | 1.1% | 100% | 0.2% | 0.00011 | 1.1% | 0.00054 | 1.1% |
| B4 | c3 | 11.07 | 11.9% | 99% | 0.8% | 0.00119 | 12.0% | 0.00593 | 12.0% |
| B5 | c0 | 77.82 | 83.7% | 99% | 93.0% | 0.00580 | 18.7% | 0.02685 | 51.5% |
| B5 | c1 | 3.01 | 3.2% | 0% | 5.9% | 0.02372 | 76.5% | 0.02372 | 45.5% |
| B5 | c2 | 1.00 | 1.1% | 0% | 0.3% | 0.00147 | 4.7% | 0.00147 | 2.8% |
| B5 | c3 | 11.16 | 12.0% | 99% | 0.8% | 0.00002 | 0.0% | 0.00008 | 0.1% |

Per call site:

| Arm | Cluster | Call site | Calls per question | Share of calls | Served by the SLM | Input tokens per question | Read from cache | Output tokens per question | Share of tokens | Cost per question (SLM at 100%) | Share of cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | c0 | `filter_column` | 76.96 | 83.7% | 0% | 453,591 | 95% | 10,573 | 94.7% | 0.23865 | 86.3% |
| B0 | c1 | `select_tables` | 1.00 | 1.1% | 0% | 1,933 | 0% | 215 | 0.4% | 0.00601 | 2.2% |
| B0 | c1 | `select_columns` | 1.00 | 1.1% | 0% | 1,660 | 0% | 218 | 0.4% | 0.00550 | 2.0% |
| B0 | c1 | `generate_candidate` | 1.00 | 1.1% | 0% | 15,559 | 95% | 786 | 3.3% | 0.01237 | 4.5% |
| B0 | c1 | `revise` | 0.00 | 0.0% | 0% | 23 | 42% | 4 | 0.0% | 0.00008 | 0.0% |
| B0 | c2 | `extract_keywords` | 1.00 | 1.1% | 0% | 1,088 | 85% | 99 | 0.2% | 0.00152 | 0.6% |
| B0 | c3 | `agent_ir` | 4.00 | 4.3% | 0% | 1,508 | 0% | 66 | 0.3% | 0.00368 | 1.3% |
| B0 | c3 | `agent_ss` | 4.00 | 4.3% | 0% | 1,380 | 0% | 192 | 0.3% | 0.00468 | 1.7% |
| B0 | c3 | `agent_cg` | 3.00 | 3.3% | 0% | 1,012 | 0% | 196 | 0.2% | 0.00399 | 1.4% |
| B4 | c0 | `filter_column` | 77.69 | 83.7% | 99% | 312,544 | 96% | 7,102 | 95.0% | 0.00830 | 83.6% |
| B4 | c1 | `select_tables` | 1.00 | 1.1% | 100% | 1,249 | 17% | 137 | 0.4% | 0.00011 | 1.1% |
| B4 | c1 | `select_columns` | 1.00 | 1.1% | 100% | 1,105 | 10% | 143 | 0.4% | 0.00011 | 1.1% |
| B4 | c1 | `generate_candidate` | 1.00 | 1.1% | 100% | 9,880 | 96% | 474 | 3.1% | 0.00011 | 1.1% |
| B4 | c1 | `revise` | 0.08 | 0.1% | 100% | 301 | 64% | 24 | 0.1% | 0.00001 | 0.1% |
| B4 | c2 | `extract_keywords` | 1.01 | 1.1% | 100% | 721 | 85% | 70 | 0.2% | 0.00011 | 1.1% |
| B4 | c3 | `agent_ir` | 4.01 | 4.3% | 100% | 982 | 96% | 17 | 0.3% | 0.00043 | 4.3% |
| B4 | c3 | `agent_ss` | 4.05 | 4.4% | 99% | 926 | 97% | 19 | 0.3% | 0.00043 | 4.3% |
| B4 | c3 | `agent_cg` | 3.01 | 3.2% | 100% | 689 | 95% | 14 | 0.2% | 0.00032 | 3.3% |
| B5 | c0 | `filter_column` | 77.82 | 83.7% | 99% | 312,774 | 96% | 7,104 | 93.0% | 0.00580 | 18.7% |
| B5 | c1 | `select_tables` | 1.00 | 1.1% | 0% | 1,911 | 0% | 212 | 0.6% | 0.00594 | 19.2% |
| B5 | c1 | `select_columns` | 1.00 | 1.1% | 0% | 1,638 | 0% | 222 | 0.5% | 0.00550 | 17.7% |
| B5 | c1 | `generate_candidate` | 1.00 | 1.1% | 0% | 15,556 | 96% | 794 | 4.8% | 0.01221 | 39.4% |
| B5 | c1 | `revise` | 0.01 | 0.0% | 0% | 34 | 85% | 5 | 0.0% | 0.00006 | 0.2% |
| B5 | c2 | `extract_keywords` | 1.00 | 1.1% | 0% | 1,088 | 87% | 99 | 0.3% | 0.00147 | 4.7% |
| B5 | c3 | `agent_ir` | 4.04 | 4.3% | 99% | 982 | 96% | 17 | 0.3% | 0.00001 | 0.0% |
| B5 | c3 | `agent_ss` | 4.03 | 4.3% | 99% | 926 | 97% | 19 | 0.3% | 0.00001 | 0.0% |
| B5 | c3 | `agent_cg` | 3.10 | 3.3% | 97% | 689 | 95% | 14 | 0.2% | 0.00000 | 0.0% |

What B5 takes out of B0's cost, by cluster, over the 498 questions:

| Cluster (by call site) | B0, API (US$) | B5, API (US$) | B5, SLM serving at 100% / 20% (US$) | B0 minus B5 at 100% / 20% (US$) |
|---|---|---|---|---|
| c0 | 118.85 | 0.27 | 2.62 / 13.10 | 115.96 / 105.47 |
| c1 | 11.93 | 11.81 | 0.00 / 0.00 | 0.12 / 0.12 |
| c2 | 0.76 | 0.73 | 0.00 / 0.00 | 0.03 / 0.03 |
| c3 | 6.15 | 0.00 | 0.01 / 0.04 | 6.14 / 6.11 |
| Total | 137.68 | 12.81 | 2.63 / 13.14 | 122.25 / 111.73 |

- **The column filter is 84% of B0's calls, 95% of its tokens and 86% of its cost.** The agent's own
  steps (c3) are 12% of the calls and 4.5% of the cost. The three tasks that decide the SQL (c1) are
  3.3% of the calls and 8.7% of the cost.
- **B5 removes 88.8% of B0's cost with the SLM at full utilisation, 86.9% at 50% and 81.2% at 20%**
  (86.5% in the as-run view; 74.0% at 20% and the sensitivity price). Counting the API alone, B5's
  production-LLM bill is 9.3% of B0's.
- Almost all of it is the column filter: US$ 116 of the US$ 122 removed at full utilisation; the agent
  steps account for US$ 6. What B5 keeps on the production LLM (c1 and c2) costs about the same as in B0.
- In the registered form of this measure (the replaceable fraction, SPEC section 5), the SLM served
  95.5% of B5's calls and 93.5% of its tokens; priced at the production LLM's price, those calls are
  85.8% of the execution's cost. For B4 the three shares are 100%.
- In B4, cost follows the number of calls, because the load-test price is per request: the column
  filter is 83.6% of it.

## 4. Latency (exploratory, descriptive)

Per question. "Seconds per question" is the time from the start of the question's first model call to
the end of its last one, so it includes the tools, the router and any queueing between calls:

| Arm | Sessions | Workers (questions at once), by session | Session wall time, minutes | Seconds per question, p50 / p95 | Same, by the worker count of the session that answered the question | Sum of the question's call latencies, p50, seconds |
|---|---|---|---|---|---|---|
| B0 | 5 | 8 / 5 / 3 | 152 | 55.6 / 83.3 | 8 workers: 57.9 / 342.2 (n = 90); 5 workers: 56.4 / 69.6 (n = 10); 3 workers: 55.3 / 71.2 (n = 398) | 167 |
| B2, production LLM | 1 | 1 | 59 | 5.0 / 6.9 | – | 5 |
| B2, cheaper LLM | 1 | 1 | 33 | 3.5 / 6.6 | – | 4 |
| B3 | 1 | 8 | 67 | 53.8 / 98.5 | – | 123 |
| B4 | 3 | 8 / 3 | 161 | 59.7 / 136.6 | 8 workers: 95.4 / 439.0 (n = 127); 3 workers: 56.8 / 95.8 (n = 371) | 381 |
| B5 | 11 | 3 | 193 | 52.5 / 97.2 | – | 268 |

Per call, as the client measured it (`latency_ms`; the router's time is logged apart and is not
inside it):

| Arm | Calls | All calls, p50 / p95 (s) | SLM calls, p50 / p95 (s) | API calls, p50 / p95 (s) | Calls that ended in an error (included) |
|---|---|---|---|---|---|
| B0 | 45,801 | 1.98 / 3.18 | – | 1.98 / 3.18 | 12 |
| B2, production LLM | 500 | 4.94 / 6.86 | – | 4.94 / 6.86 | 2 |
| B2, cheaper LLM | 500 | 3.54 / 6.54 | – | 3.54 / 6.54 | 2 |
| B3 | 24,115 | 3.83 / 11.31 | 3.83 / 11.31 | – | 484 |
| B4 | 46,244 | 5.19 / 21.62 | 5.19 / 21.62 | – | 424 |
| B5 | 46,310 | 3.26 / 7.93 | 3.28 / 8.01 | 2.65 / 6.02 | 523 |

**These wall times are not comparable across arms, and some not within an arm:**

- **Workers.** The number of questions in flight differed: 8, 5 and 3 in B0; 8 and 3 in B4; 8 in B3; 3 in
  B5; 1 in the single calls. Within a question the agent sends up to 32 calls at once. In B4 the
  questions answered with 8 workers took a median of 95 s (p95 439 s) and those with 3 workers 57 s
  (p95 96 s); its SLM calls had a p95 of 28.7 s with 8 workers and 9.6 s with 3.
- **Resumed sessions.** B0 ran in 5 sessions, B4 in 3, B5 in 11. A question's time is measured inside
  the session that completed it; the calls of interrupted questions were discarded and their time is in
  no row.
- **Shared client.** B0's and B4's 8-worker sessions overlapped in time (02:47 to 03:08 UTC on
  2026-11-11) on one client machine; B0's 8-worker questions, which have its long tail (p95 342 s
  against 71 s with 3 workers), were all answered in that window.
- **One GPU.** Every SLM arm was served by a single A100 80GB at a time, so SLM latency is latency
  under whatever load the client put on that one server (section 6): a median of 12 to 23 calls in
  flight, peaks of 96 to 256.
- **Router queueing.** In B4 and B5 each routed call first waits for one local embedding model per
  client process (table below). The embedding itself is 51 to 60 ms at the median. The wait is a
  harness artefact (SPEC section 6.6 leaves it out of the priced latency) and it is large in B5: summed
  over a question's calls, 111 s, spread over calls that run in parallel.
- **Errors.** Calls that ended in an error are in the percentiles with the time they took; two API calls
  ran into the 900 s client timeout (one in B2 on the production LLM, one in B5), which sets the
  maximum, not the p95.

Router time per routed call (B4, B5):

| Arm | Routed invocations | Embedding, p50 / p95 (ms) | Wait for the embedding model, p50 / p95 (ms) | Whole route, p50 / p95 (ms) | Route time summed per question (s) | Of which waiting (s) |
|---|---|---|---|---|---|---|
| B4 | 45,829 | 51 / 97 | 44 / 1,458 | 102 / 1,536 | 43 | 35 |
| B5 | 45,791 | 60 / 291 | 286 / 5,480 | 352 / 5,748 | 121 | 111 |

**The SLM's latency as run, against the limit used to price it.** The load-test price is the cost per
request at the highest load whose p95 stays within a limit: the production LLM's own p95 on calibration,
reweighted to the call-site mix of the sweep (SPEC section 6.6). One limit applies to each sweep's
traffic as a whole:

| Arm | Traffic priced by one sweep | SLM calls | As run, p50 (s) | As run, p95 (s) | p95 limit of the sweep (s) | As-run p95 over the limit | Calls slower than the limit |
|---|---|---|---|---|---|---|---|
| B3 | all SLM calls (base sweep) | 24,115 | 3.83 | 11.31 | 3.13 | 3.6x | 62% |
| B4 | all SLM calls (as-deployed sweep) | 46,244 | 5.19 | 21.62 | 2.93 | 7.4x | 80% |
| B5 | adapter c0 (its own sweep) | 38,690 | 3.47 | 8.24 | 2.80 | 2.9x | 76% |
| B5 | adapter c3 (its own sweep) | 5,559 | 0.37 | 0.81 | 2.40 | 0.3x | 0.4% |

Per call site, beside the production LLM's p95 on the same group of call sites in the calibration run
that sets the limit (a sensitivity in J8, not a limit):

| Arm | Call site | SLM calls | As run, p50 (s) | As run, p95 (s) | Production LLM's p95 on the same group of call sites, calibration (s) | As-run p95 against it |
|---|---|---|---|---|---|---|
| B3 | `filter_column` | 16,518 | 4.51 | 11.22 | 2.80 | over |
| B3 | `select_tables` | 498 | 3.62 | 11.24 | 3.81 | over |
| B3 | `select_columns` | 499 | 3.79 | 10.66 | 3.81 | over |
| B3 | `generate_candidate` | 498 | 12.42 | 28.05 | 12.49 | over |
| B3 | `revise` | 104 | 11.09 | 20.08 | 12.49 | over |
| B3 | `extract_keywords` | 498 | 1.40 | 3.12 | 4.51 | under |
| B3 | `agent_ir` | 2,011 | 0.45 | 0.93 | 2.39 | under |
| B3 | `agent_ss` | 1,994 | 0.40 | 0.91 | 2.39 | under |
| B3 | `agent_cg` | 1,495 | 0.75 | 2.61 | 2.39 | over |
| B4 | `filter_column` | 38,692 | 5.94 | 22.75 | 2.80 | over |
| B4 | `select_tables` | 499 | 3.02 | 7.74 | 3.81 | over |
| B4 | `select_columns` | 499 | 2.67 | 5.06 | 3.81 | over |
| B4 | `generate_candidate` | 498 | 8.67 | 13.31 | 12.49 | over |
| B4 | `revise` | 41 | 4.96 | 9.75 | 12.49 | under |
| B4 | `extract_keywords` | 501 | 1.56 | 2.65 | 4.51 | under |
| B4 | `agent_ir` | 1,997 | 0.31 | 0.72 | 2.39 | under |
| B4 | `agent_ss` | 2,018 | 0.37 | 0.97 | 2.39 | under |
| B4 | `agent_cg` | 1,499 | 0.37 | 0.69 | 2.39 | under |
| B5 | `filter_column` | 38,690 | 3.47 | 8.24 | 2.80 | over |
| B5 | `agent_ir` | 2,010 | 0.39 | 0.78 | 2.39 | under |
| B5 | `agent_ss` | 2,005 | 0.38 | 0.83 | 2.39 | under |
| B5 | `agent_cg` | 1,544 | 0.35 | 0.80 | 2.39 | under |

- **B3, B4 and the column filter of B5 ran outside the limit they are priced at.** The as-run p95 is 3.6
  times the limit in B3, 7.4 times in B4 and 2.9 times for B5's column filter; 62%, 80% and 76% of
  those calls were slower than the limit. With 3 workers only, B4's p95 is still 9.6 s against 2.93 s.
- **B5's agent steps (c3) ran inside it**: p95 0.81 s against 2.40 s.
- The runs sent more load than the level the price assumes: the limit holds up to 8 calls in flight
  (base model, B4 as deployed) or 16 (c0 adapter) in the load test, and the test runs kept a median of
  12 to 23 and a mean of 26 to 41 in flight. So the load-test price at 100% describes a server run
  within the limit, which is not how these runs were served; at the load of these runs the server
  answered more slowly than the API it replaces on the column filter.
- The registered rule adds the router's embedding time (155.6 ms, p95 on calibration) to the load
  test's latency; the as-run figures above do not include the router, so the distance to the limit is
  understated by that much.
- The production LLM itself, in B0 on the test, had a per-call p95 of 3.18 s overall and 2.94 s on the
  column filter.

## 5. Break-even of a dedicated GPU (exploratory)

A dedicated GPU costs US$ 1.79 per hour whether it is used or not (US$ 42.96 per day). It beats the API
when the API cost it removes per hour is larger: break-even questions per hour = 1.79 ÷ (API cost per
question before − API cost per question after). Capacity is the load test's sustained requests per hour
within the latency limit divided by the arm's SLM calls per question (for B5, the column-filter
adapter's sweep; its agent-step calls take about 0.3% of the GPU's time).

| Comparison | EX difference, points | API cost per question before (US$) | API cost per question after (US$) | API saving per question (US$) | Break-even, questions per hour | Break-even, questions per day (24 h) | Questions per hour one GPU sustains within the latency limit | Break-even as a share of that capacity | Break-even at the sensitivity price (US$ 3.13/h), questions per hour |
|---|---|---|---|---|---|---|---|---|---|
| B5 in place of B0 | +0.8 | 0.2765 | 0.0257 | 0.2508 | 7.1 | 171 | 340 | 2.1% | 12.5 |
| B5 in place of B2, production LLM | −1.0 | 0.0230 | 0.0257 | −0.0028 | never (the SLM arm's API calls alone cost more) | – | 340 | – | – |
| B5 in place of B2, cheaper LLM | +5.2 | 0.0005 | 0.0257 | −0.0252 | never (the SLM arm's API calls alone cost more) | – | 340 | – | – |
| B4 in place of B0 | −15.1 | 0.2765 | 0.0000 | 0.2765 | 6.5 | 155 | 179 | 3.6% | 11.3 |
| B4 in place of B2, production LLM | −16.9 | 0.0230 | 0.0000 | 0.0230 | 77.9 | 1,870 | 179 | 43.6% | 136 |
| B4 in place of B2, cheaper LLM | −10.6 | 0.0005 | 0.0000 | 0.0005 | 3,768 | 90,439 | 179 | 2,110%: above the capacity of one GPU | 6,592 |

- **Against B0:** a dedicated GPU serving B5's two clusters pays for itself from about 7 questions per
  hour (171 per day), 2% of what one GPU sustains within the limit; at the sensitivity price, 12.5 per
  hour.
- **Against the single call on the production LLM:** B5 never breaks even, at any volume: the
  production-LLM calls it keeps cost more than the single call, before any GPU. B4 breaks even at 78
  questions per hour (44% of one GPU's capacity), with an EX 17 points lower.
- **Against the single call on the cheaper LLM:** no SLM arm breaks even within the capacity of one
  GPU; B4 would need 3,768 questions per hour and one GPU sustains 179.
- This is not the registered tipping point (`bench report` gives the utilisation at which an SLM arm's
  cost per correct answer meets a comparator's). It compares costs only; the EX differences are in the
  second column and are not priced.
- The capacities assume the load test's conditions (within the limit, calibration traffic). As section 4
  shows, the test runs were not served within the limit.

## 6. Capacity the SLM server actually reached (exploratory)

One A100 80GB, prefix caching on, as the test runs loaded it. "In flight" is counted from the client's
call start to its end.

| Arm | SLM calls | Session wall time (h) | Time with a call in flight (h) | Calls/s over session wall time | Calls/s while a call was in flight | Best 5 minutes, calls/s | Best minute, calls/s | Input tokens/s while in flight | Output tokens/s while in flight | Best 5 minutes, input tokens/s | Best 5 minutes, output tokens/s | Input read from the prefix cache | Calls in flight, mean | Calls in flight, p50 / p95 over time | Calls in flight, peak | Questions per hour over session wall time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B3 | 24,115 | 1.12 | 1.04 | 5.97 | 6.41 | 9.50 | 11.67 | 20,550 | 539 | 34,868 | 779 | 93% | 28.3 | 12 / 99 | 203 | 444 |
| B4 | 46,244 | 2.68 | 2.17 | 4.75 | 5.87 | 9.05 | 13.18 | 20,947 | 510 | 34,158 | 779 | 96% | 40.7 | 18 / 179 | 256 | 186 |
| B5 | 44,249 | 3.21 | 1.83 | 3.78 | 6.65 | 7.91 | 9.15 | 23,824 | 540 | 30,558 | 682 | 96% | 25.6 | 23 / 64 | 96 | 155 |

For reference, the load test (J8), on calibration traffic, each level starting with an empty prefix
cache:

| Sweep (calibration traffic) | p95 limit (s) | Highest concurrency within the limit | Requests/s there | Requests per hour there | US$ per 1,000 requests at 20% / 50% / 100% utilisation | Highest requests/s at any level |
|---|---|---|---|---|---|---|
| base model, B3's calls | 3.13 | 8 | 5.02 | 18,076 | 0.4949 / 0.1980 / 0.0990 | 9.07 at concurrency 128 (p95 18.2 s) |
| four adapters on one server, B4's calls | 2.93 | 8 | 4.61 | 16,585 | 0.5394 / 0.2158 / 0.1079 | 8.72 at concurrency 128 (p95 18.2 s) |
| adapter c0 alone | 2.80 | 16 | 7.26 | 26,121 | 0.3425 / 0.1370 / 0.0685 | 9.54 at concurrency 64 (p95 7.9 s) |
| adapter c1 alone | 7.36 | none | – | – | no cost (outside the limit at every level) | 2.69 at concurrency 64 (p95 47.2 s) |
| adapter c2 alone | 4.49 | 32 | 20.75 | 74,694 | 0.1198 / 0.0479 / 0.0240 | 25.67 at concurrency 64 (p95 3.1 s) |
| adapter c3 alone | 2.40 | 128 | 359.53 | 1,294,323 | 0.0069 / 0.0028 / 0.0014 | 359.53 at concurrency 128 (p95 0.4 s) |

- While it had work, the server completed 5.9 to 6.7 calls per second (about 20,500 to 23,800 input and
  510 to 540 output tokens per second, 93 to 96% of the input read from the prefix cache); its best five
  minutes reached 7.9 to 9.5 calls per second and about 30,600 to 34,900 input tokens per second.
- The best five minutes are close to the highest throughput the load test saw at any concurrency (8.7
  to 9.5 requests per second). The average while busy is above what the load test sustains within the
  limit for the base model and for B4 as deployed (5.0 and 4.6), and below it for the c0 adapter alone
  (7.3). At these rates the latency was outside the limit (section 4).
- Concurrency was set by the client, not by the server: up to 32 calls per question times 3 or 8
  workers gives the peaks of 96 (B5), 203 (B3) and 256 (B4).
- B5 used 57% of its session time (1.83 of 3.21 hours had a call in flight); the rest is the
  production LLM's turns, tools, routing and the restarts of eleven sessions.
- Per hour of session, the arms completed 444 (B3), 186 (B4) and 155 (B5) questions. The first two used
  8 workers for all or part of the run.
- The c1 adapter (SQL generation and selection) has no load level within its limit in the load test,
  so it has no registered price on its own; B4 is priced by the sweep of the server as deployed, where
  those calls are 4% of the mix.

## 7. What the benchmark cost to run (exploratory ledger)

**API calls, at list price, from the logs.** Every `calls.jsonl` under `runs/` outside the B1
directories (22 directories, not read), de-duplicated by call id because a resumed run copies the
calls it keeps. Computed on 2026-11-12. List price of the logged usage is not an invoice: it prices
every logged call as a metered, non-batch call.

| Category | Run directories | Distinct calls | Of which on the SLM | Input tokens, millions | Output tokens, millions | API cost at list price (US$) |
|---|---|---|---|---|---|---|
| Production-LLM traces on the training questions (the final collection run, its interrupted attempts and the earlier probes) | 24 | 93,149 | 0 | 437.0 | 11.19 | 297.22 |
| Single-call probes on training questions | 4 | 8 | 0 | 0.1 | 0.01 | 0.11 |
| Calibration: agent on the production LLM | 2 | 16,348 | 0 | 83.1 | 2.18 | 48.38 |
| Calibration: single call, cheaper LLM | 1 | 200 | 0 | 2.5 | 0.11 | 0.09 |
| Calibration: replays on the cheaper LLM | 6 | 27,526 | 0 | 302.9 | 5.41 | 10.00 |
| Calibration: replay on the production LLM | 1 | 4,011 | 0 | 19.9 | 0.51 | 11.71 |
| Calibration: SLM agents and replays (B3, B4, two SLM candidates) | 12 | 94,870 | 94,639 | 331.7 | 9.94 | 0.00 |
| Test: B0, every attempt | 5 | 45,961 | 0 | 238.6 | 6.17 | 138.11 |
| Test: B2, production LLM | 1 | 500 | 0 | 10.4 | 0.40 | 11.44 |
| Test: B2, cheaper LLM | 1 | 500 | 0 | 6.5 | 0.24 | 0.24 |
| Test: B3 and B4 (SLM only) | 3 | 70,359 | 69,933 | 240.8 | 6.01 | 0.00 |
| Test: B5, every attempt | 11 | 51,475 | 48,104 | 183.6 | 4.65 | 14.10 |
| Test: replay on the production LLM (100 questions) | 1 | 9,093 | 0 | 47.1 | 1.24 | 27.39 |
| Test: replays on the SLM | 4 | 92,923 | 91,899 | 328.1 | 7.78 | 0.00 |
| **Total, B1 excluded** | 76 | 506,923 | 304,575 |  |  | **558.81** |

- By phase: training questions US$ 297.34, calibration US$ 70.19, test US$ 191.28.
- The run that produced the training traces (`agent-B0-train-20261109T214533.609401Z`, 834 questions,
  82,403 calls, 7 sessions) is US$ 225.78 at list price, US$ 0.271 per question. The rest of the first
  row is interrupted attempts, the probes of 2026-09-30 to 2026-10-23, and one run later set aside as
  invalid; that run's calls (US$ 82.17) are partly reused by the final run, so the two figures do not
  add.
- On the test, the calls that were discarded when a session was resumed cost US$ 0.43 for B0
  (138.11 against 137.68 kept) and US$ 1.29 for B5 (14.10 against 12.81).

**GPU pods**, from creation to termination at US$ 1.79 per hour (pods terminated by 2026-11-12 05:00
UTC; a pod started after that for work outside this page is not counted):

| Created (UTC) | Terminated (UTC) | Hours | US$ at 1.79/h | What ran on it |
|---|---|---|---|---|
| 2026-11-09 23:48 | 2026-11-10 00:07 | 0.31 | 0.56 | download of the weights |
| 2026-11-10 02:09 | 2026-11-10 02:39 | 0.50 | 0.90 | the second SLM candidate |
| 2026-11-10 02:39 | 2026-11-10 03:09 | 0.50 | 0.90 | the second SLM candidate |
| 2026-11-10 00:14 | 2026-11-10 05:24 | 5.17 | 9.25 | base-model replays and the B3 pilot on calibration |
| 2026-11-10 07:05 | 2026-11-10 07:36 | 0.51 | 0.92 | a check with one adapter loaded |
| 2026-11-10 16:12 | 2026-11-11 05:33 | 13.35 | 23.90 | B3 and B4 on calibration, the load tests behind J8, B3 and B4 on the test |
| 2026-11-11 08:05 | 2026-11-11 08:19 | 0.24 | 0.42 | B5 on the test, session 1 |
| 2026-11-11 15:30 | 2026-11-11 16:05 | 0.59 | 1.05 | B5 on the test, sessions 2 to 4 |
| 2026-11-11 16:43 | 2026-11-11 20:20 | 3.62 | 6.48 | B5 on the test, sessions 5 to 11 |
| 2026-11-11 20:27 | 2026-11-11 23:08 | 2.68 | 4.80 | replays on the test |
| **Total** |  | **27.48** | **49.19** |  |

- The three test runs on the SLM held the server for 7.0 of those 27.5 hours (1.12 h B3, 2.68 h B4,
  3.21 h B5). The pods that served B5's sessions were up for 4.45 hours (US$ 7.96).
- B5 at the volume of this benchmark, everything included: US$ 14.10 of API calls plus US$ 7.96 of pods
  = US$ 22.06, against US$ 138.11 for B0 and US$ 11.44 for the single call on the production LLM.

**Training**, from the adapters' manifests:

| Adapter | Training rows | Dataset tokens, millions | Tokens seen in 2 epochs, millions | Steps | Training time, minutes | `cost_usd` in the manifest |
|---|---|---|---|---|---|---|
| c0 | 11,982 | 49.58 | 99.14 | 1498 | 79.3 | not recorded (null) |
| c1 | 2,517 | 10.85 | 21.70 | 316 | 21.6 | not recorded (null) |
| c2 | 833 | 0.63 | 1.27 | 106 | 6.7 | not recorded (null) |
| c3 | 9,164 | 2.23 | 4.44 | 1146 | 61.0 | not recorded (null) |
| **Total** | 24,496 | 63.30 | 126.54 | 3066 | 168.7 | not recorded |

**Total recorded: US$ 558.81 of API calls at list price + US$ 49.19 of GPU pods = US$ 608.00.**

**Not recorded, and not in that total:**

- **Fine-tuning charges.** The four adapters were trained on a hosted fine-tuning service; `cost_usd`
  and `gpu_seconds` are null in all four manifests. Only the volume is known: 126.5 million tokens
  seen, 169 minutes. The partial-data adapters trained for the learning curves are outside this page.
- **Embeddings.** The retrieval inside CHESS uses an embedding model, configured as an external API;
  those calls are not in `calls.jsonl` and are not priced.
- **B1.** None of its runs (calibration, training probes, the unfinished test) was read.
- **Calls with no price in `config.yaml`:** 561 calls of the October probes to the provider's
  top-of-line model (2.89 million input and 0.07 million output tokens), and 55 calls to small local
  models in the first probes.
- **Failed calls with no usage** (a provider may bill some of them), the network volume's storage
  (100 GB since 2026-11-09), the client machine, and the hours of work.
- Whether a logged call was charged at list price: one probe run was sent as a batch and three probe
  runs are marked in their manifests as sent outside the metered API (about US$ 59 of the first row at
  list price).

## 8. What this does not measure

- **Energy and FLOPs.** The paper's 10 to 30 times is stated for "latency, energy consumption, and
  FLOPs". Nothing here measures energy or FLOPs; cost in US$ at one provider's price and latency under
  one load are the only quantities. The registration reads the claim as cost per correct answer (SPEC
  section 5, row V3).
- **Repeat executions.** One execution per arm. The engines are not deterministic, and neither cost,
  latency nor EX has a between-execution spread. Resumed sessions and different worker counts add
  variation that a single run cannot separate.
- **The batch price.** No arm was run through a batch interface. Repricing the same usage with the
  production LLM's batch discount (50%) gives US$ 68.84 for B0 and US$ 5.72 for the single call, and
  doubles the break-even volumes of section 5 against B0; it is arithmetic on the list price, a batch
  call is not comparable in latency, and the cheaper LLM has no batch price on the registered route.
- **B1**, the agent on the cheaper LLM with few-shot examples, which the registered cost comparisons
  read.
- **The SLM within its latency limit.** The accuracy of B3, B4 and B5 was measured on runs served
  outside the limit at which their serving is priced. No run pairs the registered price with the
  latency it assumes.
- **More than one GPU, another GPU, another server configuration, or autoscaling.** One A100 80GB, one
  serving stack; utilisation is a parameter (20, 50, 100%), not a measurement.
- **Training cost and payback.** Not recorded (section 7); no payback is computed, as registered.
- **Discounts, committed-use prices, taxes**, and any price after the dates of the two price tables
  (API 2026-10-22, GPU 2026-11-09).

## 9. Where this page departs from the registered definitions

- **Not a registered result.** The API and SLM costs use the registered functions (`bench/judge/j3.py`)
  and the registered J8 result, but were computed by a script outside the registered code, with no
  judgment hash, before B1 finished. `bench report` is the registered source.
- **The as-run GPU view, the break-even in questions per hour, the capacity tables and the ledger** are
  not in the registration.
- **Latency.** SPEC section 6.7 asks for p50 and p95 per arm as measured, with no verdict depending on
  them; outside the load test this page found no registered function for it, so "seconds per question"
  and the per-call percentiles (linear interpolation, failed calls included) are this page's
  definitions.
- **Cost shares.** Sections 1 to 3 give each cluster's share of the arm's own cost. The registered
  replaceable fraction is different (the share, at the production LLM's price, of the tokens that fell
  on SLM calls); both are given in section 3.
- **Clusters by call site**, not by the router's per-call decision (they differ on about 0.2% of the
  column-filter calls).
- **Which utilisation and price decide.** The registration reads 20% for one direction and 100% for the
  other, and requires the sensitivity price as well; this page shows all three utilisations at the
  serving provider's price and gives the sensitivity factor (1.75).
- **GPU price.** J8 uses US$ 0.000497 per second (US$ 1.7892 per hour); the as-run view, the break-even
  and the pod ledger use the pod's US$ 1.79.
- **B5's SLM price is an extrapolation** from load tests of one adapter at a time; that is the
  registered rule for B5, and J3 labels it so.

## 10. Provenance

| Arm | Run | Evaluation |
|---|---|---|
| B0 | `agent-B0-test-20261111T060814.371294Z` | `eval-agent-B0-test-20261111T060814.371294Z-20261111T080311.709677Z` |
| B2, production LLM | `agent-B2-production_llm-test-20261111T144852.703285Z` | `eval-…-20261111T202239.433284Z` (and `…-20261111T154903.696240Z`, same result) |
| B2, cheaper LLM | `agent-B2-cheap_alt-test-20261111T081959.861177Z` | `eval-…-20261111T085323.154993Z` |
| B3 | `agent-B3-test-20261111T012223.392984Z` | `eval-…-20261111T054533.117283Z` |
| B4 | `agent-B4-test-20261111T032707.693347Z` | `eval-…-20261111T054821.932671Z` |
| B5 | `agent-B5-test-20261111T200809.693765Z` | `eval-…-20261111T202046.064721Z` |

Load test: J8 result `de3754a76358b70edb33ff0b29f88616d84755923e8becf79cad333750ab5e66` (sweeps of
2026-11-10 and 2026-11-11 on calibration traffic; latency limit from
`agent-B0-calib-20261109T151141.039667Z`). The final run of each arm holds the kept calls of its earlier
sessions; every attempt is listed in `registry/test/`.

The arm-level numbers were derived twice, independently: once with the registered pricing function and
once by a separate script that reimplements the price arithmetic; calls, tokens, API cost, SLM cost at
the three utilisations and the correct counts agree to the cent. The raw `runs/` directories are not in the
public repository.
