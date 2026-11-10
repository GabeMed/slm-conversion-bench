# SPEC v2 — slm-conversion-bench

**Empirical execution of "Small Language Models are the Future of Agentic AI"**
(Belcak et al., NVIDIA Research, arXiv 2506.02153, v3 of 2026-09-22)

Version 2.4, of 2026-11-10. v2 is of 2026-09-30 and replaces v1, which remains preserved in `SPEC-v1.md`. v2.1 applies the corrections from a validity audit done before any paid execution (section 11.1). v2.2 swaps the open-weight teacher for a production-grade closed LLM, for cost, before Stage 0 (section 11.2). v2.3 sends the K1 calls through the author's subscription, with the provider's verbal authorization for this project only, also before Stage 0 (section 11.3). v2.4, after Stage 0 (2026-10-23) and before the pre-registration, aligns the spec with the changes declared in `prereg/DEVIATIONS.md`: K1 through the paid API, the training rows of the column filter and of the agent's call sites, the unit of B1's k rule, the Appendix B call floor, the rule for model rejections and the tool-choice parser (section 11.4). Section 11 says what changed and why.

## How to read this document

**The paper.** It is a *position paper*. It claims that SLMs are the future of agentic AI, supports the position with arguments and proposes an LLM→SLM conversion algorithm (S1–S6), but contains no experiment. The "case studies" in Appendix B are estimates.

**What this POC does.** It executes the paper's algorithm end to end on a real open-source agent, with the protocol fixed before any result. The central question is v1's: **find out whether it improves, by how much, and which step of the algorithm delivers the gain**. In operational terms:
- which of the agent's calls a specialized SLM takes over **without losing quality** against the LLM the agent uses;
- whether this also holds against the **cheapest alternative that requires no training**;
- at what **cost per correct task**.

**Scope of the question "which step delivers the gain".** In the core, the contribution is **measured** for S5 (specialization: B3 × B4) and for S6 (allocation: B4 × B5). S1–S4 are **executed and described**, but their contribution is not measured in the core (table in section 5).

**Labels.** V (views), A (arguments), AV (alternative views), CA (counter-arguments), B (barriers), S (algorithm steps) and WD (definitions) are the paper's. D are our decisions (section 10). Every citation of the paper gives the page of the v3 PDF.

**Level rule.** This spec does not name models, providers or libraries. That goes in the configuration and in the development sessions. Here, "production LLM", "teacher", "cheap alternative" and "SLMs" are roles.
- **Declared exception:** the agent (CHESS) and the benchmark (BIRD and Arcwise-Plat-SQL) are named, because they are the **object** of the test, not an implementation choice.

**Format of each item.** The document answers three questions: **what the paper says**, **what we execute** and **what this reproduces**. When an item is not testable in this POC, that is said.

---

## 1. Position (Section 2 of the paper)

### Definitions

**Paper (p.2).**
- **WD1:** an SLM is a model that fits on a consumer electronic device and performs inference with practical latency to serve one user's agentic requests.
- **WD2:** an LLM is a model that is not an SLM.
- In 2025, the authors say they would be comfortable considering most models below 10 billion parameters SLMs.

**We execute.** We adopt WD1 and WD2.
- The candidate SLMs come out of the S4 triage (section 4), within ~10B.
- A round on the author's consumer device is optional, and enters as data to honor WD1 literally.

### The three views

**Paper (p.2–3).**
- SLMs are (V1) sufficiently powerful, (V2) inherently more suitable and (V3) necessarily more economical for the vast majority of LM invocations in agentic systems.
- The authors say they make no recommendation nor impose an obligation: *"We do not make a recommendation or try to impose an obligation"* (p.2).
- They attribute the reading as "Humean moral ought" to others: *"represents to many… a Humean moral ought"* (p.3).

**This POC** does not test the value. It tests the empirical premises:

| View | What the paper claims | The measure that settles it in this POC |
|---|---|---|
| **V1** Sufficiency | SLMs handle the language "errands" of agentic applications | The agent's execution accuracy with the specialized SLM × with the production LLM, and the contrast with the zero-shot SLM (section 5) |
| **V2** Operational suitability | SLMs are more suitable for agentic systems | Not a direct measure. The paper supports it with A3–A7 (p.5–6); here A3 (adaptation time and cost), A5 (format), A6 (heterogeneous allocation) and A7 (logs as data) enter. A4 enters through the distribution of calls. |
| **V3** Economy | SLMs are more economical by virtue of size | Cost per correct query, with measured throughput, utilization of 20/50/100% and fixed cost (section 6.6) |

---

## 2. The workload: the agent and the benchmark

**Paper.**
- Fig. 1 (p.4) distinguishes *language model agency* (the LM orchestrates the tools) from *code agency* (a controller calls LMs).
- S1 says to log the agent's **non-HCI** calls (p.9).
- S3 cites as typical tasks *"intent recognition, data extraction, summarization of specific document types, or code generation with respect to tools available to the agent"* (p.9).
- Appendix B reserves for the LLM, among other things, the resolution of unstructured errors (p.16–17).

**We execute.** The open-source agent **CHESS**, in the IR → SS → CG flow, running on the questions of the **BIRD** benchmark.
- It is *code agency*: a controller calls LMs in narrow calls, and none of them converses with a user.
- It has several call types, including **one the paper reserves for the LLM**: repairing the SQL after an error.

**Call sites** (the agent's LM call types):

| Call site | What it does | The paper's prediction, registered beforehand |
|---|---|---|
| choice of the next tool | the agent decides the next step | uncertain |
| keyword extraction | extraction | SLM |
| column filter | binary classification per column (high volume) | SLM |
| table selection | structured selection | SLM |
| column selection | structured selection | SLM |
| candidate SQL generation | code generation for the tool (the database) | uncertain |
| SQL repair after an error or empty result | *unstructured error resolution* | **LLM** |

**Not LM calls:** context and entity retrieval. They do not enter the conversion.

**Data:**
- **Logs and training:** the 1,034 BIRD dev questions that are not in Mini-Dev, on the 11 dev databases.
  - ~200 of them are reserved for **calibration**: pilot, selection and thresholds.
  - The remaining ~834 generate the logs.
  - Questions, evidence and gold come from the 2025-11-06 revision of BIRD dev, on the 2024 databases. Two caveats, measured in the 2026-10-01 audit:
    - this revision **changed 85 questions** relative to the previous one, 21 of them in the calibration;
    - its gold is **not** the one corrected by experts: on the 254 test items with identical question and evidence, it disagrees with Plat-SQL's gold on **18.9%** (48 of 254; 95% CI from 14.6% to 24.2%). That is why every verdict is read on the test, and the calibration serves only the pilot, the selection and the thresholds.
- **Test:** **Arcwise-Plat-SQL**: **498 questions** from BIRD's Mini-Dev, used **as they are in the file** (question, evidence and corrected gold SQL).
  - In ~81 questions and ~68 evidences, the text was rewritten relative to Mini-Dev, and the gold corresponds to the rewritten text. The schema is the original.
  - Ids 119 and 120 (absent from Plat-SQL) stay out of training and test.
  - The numbers are **not comparable** with EX published on Mini-Dev.
  - It is touched **once** per frozen configuration.
- **Training × test correspondence:**
  - **same:** the agent, the 11 databases and the source of the questions;
  - **different:** the questions are disjoint by id, and the test uses Plat-SQL's text and gold answers. One pair repeats across the split: test question 875 has the database, question text and gold SQL of training question 863 (`formula_1`; the test's evidence adds "the season page refers to url;"). It is the only test question whose (database, normalized question) appears in training or calibration. It is kept and declared; it can favour the trained arms (B4, B5) by at most 1 of 498. A sensitivity reading without it (n = 497), descriptive and without a verdict, is computed from the stored predictions after the test by a script outside the registered code;
  - **known difference:** Mini-Dev was assembled with a 30/50/20 difficulty mix (simple/moderate/challenging), so the results are **stratified by difficulty**.
- **Ground truth:** execution accuracy (EX) of the final SQL against the corrected gold, measured by **our own evaluator**:
  - pairs by `question_id`;
  - has a timeout;
  - executes prediction and gold at the same moment, which resolves the golds that depend on the date.

  Fidelity to the teacher is reported separately, because imitating the teacher is not being right.

**Correction of the agent, by the author's decision (2026-09-30).** In the published CHESS, SQL generation and repair use the **full** schema, and the schema selector's output does not reach them (issues #31 and #34 of the CHESS repository). The POC applies the fix suggested in issue #34: generation and repair now use the **selected schema**. With this, every call site is consequential, as in the design of the CHESS paper. The correction is declared in the report.

**Adjustments to the agent's harness,** so as not to contaminate the comparison:
- no automatic fallback to another model;
- retries on parse errors limited and **counted in the cost**;
- the choice of the next tool **logged as its own call site**;
- generation parameters adapted to what each model accepts, **recorded in the configuration**;
- the tool-choice parser also accepts the tool name in JSON, followed by arguments or closed by the reasoning tag (patch 16), with the same prompt, in every arm; free text, empty tags and unknown tools still fail;
- **a request refused by the engine is a model failure only** (scored, the question wrong) when the engine says the prompt is too large, or that it alone does not fit in the context; any other refusal, including that of a prompt that fits but leaves no room for `max_tokens`, is the harness's: the execution fails and nothing is scored. The rule is the same in every arm and engine.

**Reproduces.** The test Appendix B did not do: on a real agent, which calls a specialized SLM takes over.

---

## 3. Roles and arms

**Paper.** In S1 and in S5 (p.9), what is logged and what the SLM learns to imitate is the LLM the agent already uses. The teacher **is** the LLM to be replaced.

**Roles:**
- **Production LLM = teacher.** The generalist LLM the agent uses.
  - It is a **production-grade closed LLM**, accessed through the provider's own paid API: the intermediate model of the line, which teams usually put in production for cost, not the frontier top.
  - The provider's terms and the permission under which the author trains on the outputs are recorded in the configuration, with link and date. Responsibility for them lies with whoever runs the experiment.
  - K1, the calibration and the test use the paid API (D25). The provider's verbal authorization to run K1 through the author's subscription (v2.3) was not used.
  - Selection criterion: the cost of the whole core fits the budget with the intermediate model and not with the top of the line, measured in paid probes before Stage 0 (section 11.2). The configuration records which it was.
  - **What this changes in the reading.** The question becomes whether an SLM replaces the model a team actually puts in production. A weaker teacher lowers the ceiling of what the SLM learns and makes the quality comparison easier; a cheaper LLM makes the cost one harder. The report says both things.
- **Cheap alternative without training.** An open-weight generalist cheaper than the production LLM, with the **same optimization budget**: prompt with examples (few-shot) and cache.
- **Single-call version.** Text-to-SQL in a single call, without the agent, with the production LLM and with the cheap alternative. It is "what one would do without an agent".
- **Candidate SLMs.** Two models out of the S4 desk triage, within ~10B (WD1), with a license that permits commercial use and fine-tuning (section 4).

**Core arms:**

| Arm | What it answers |
|---|---|
| **B0** Production LLM in the agent | quality ceiling and reference cost |
| **B1** Cheap alternative in the agent | the trivial alternative: switching to a cheaper model without training |
| **B2** Single call (production and cheap) | is the agent worth it? It stays in the core because it is cheap (one call per question) and because, if the single call matches the agent, the per-call-site conversion loses its object |
| **B3** Zero-shot SLM in the agent | the gain from specialization (contrast with B4) |
| **B4** Specialized SLM on all call sites | V1 in the strongest form of this POC |
| **B5** Allocation per cluster | each cluster served by the cheapest engine that passes non-inferiority (sections 4, S6, and 6) |

**Extensions, in this order, each one only after the core is closed:**
1. swapping one cluster at a time;
2. at the SQL generation call site, the teacher × gold pair (BIRD train has gold SQL), which separates the label's source from capability;
3. dynamic escalation to the production LLM when the SQL fails on execution;
4. format-constrained decoding and a second open generalist without training;
5. same databases × new databases (train on BIRD train and test on the same test), which measures the value of "knowing the databases";
6. a multi-turn probe, inference only, on a conversational benchmark;
7. an off-the-shelf text-to-SQL SLM as an untrained reference;
8. a second open-weight teacher;
9. an arm with the frontier closed LLM (the top of the teacher's line or of another provider), open to contributors (section 8);
10. a broad zero-shot round on the calibration (6–8 candidates from the S4 triage), to test B2.

**Compliance rule.** Outputs of any model whose terms forbid training on them go to a separate **do-not-train** area, and the training pipeline refuses that area.

---

## 4. The conversion algorithm (Section 6 of the paper): S1–S6 executed literally

**The rule.** All steps run in the main arm, in the paper's order, and **the result of each one feeds the next**. The only part that can be cut is the **S6 retraining loop**, and the cut is declared (section 7.3). The S6 router is never cut. A step that does not apply to public data is still **executed** and has its effect recorded; the deviation is declared. Executing a step is not ablating it: each step's contribution is measured against named alternatives (section 5), or declared untested.

### S1 · Secure usage collection

**Paper (p.9).** *"deploying instrumentation to log all non-HCI agent calls, capturing input prompts, output responses, contents of individual tool calls, and optionally latency metrics… encrypted logging pipelines with role-based access controls and anonymize all data with respect to its origins before storage."*

**We execute.**
- A logger at the call interface, the *listener* of Fig. 1, records per call:
  - prompt;
  - response;
  - executed SQL and execution result;
  - latency;
  - input, cache and output tokens;
  - model and version;
  - call site id, **hidden from S3**.
- The schema is provider-neutral.
- There is encryption at rest, restricted access and origin anonymization.
- The teacher's internal reasoning, when exposed, stays out of the log (D5).
- What is logged: the trajectories of the teacher (= production LLM) on the training questions.

**Deviation.** Public data, no real users. The security controls run, but have nothing to protect.

### S2 · Curation and filtering

**Paper (p.9).** *"10k-100k examples being sufficient for fine-tuning of small models as a rule of thumb… remove any PII, PHI… can be often automatically paraphrased to obfuscate named entities and numerical details."*

**We execute:**
- **volume per cluster,** reported against the 10k–100k **rule of thumb** (it is not a floor). The low-volume clusters fall below it, and that is data about the rule;
- **masking of sensitive data,** executed, with the number of detections reported;
- **success filter** with a signal available in production, never the gold: the SQL executes without error, the result is not empty and the format is valid;
- **deduplication**, exact and by near-duplicate;
- **volume that fits in training:** after the filter and the deduplication, each (question, call site) pair keeps at most N invocations. It is a seeded **uniform** sample, which preserves the distribution of the teacher's decisions. N is fixed in the configuration before registration, and the preflight checks that the largest adapter trains within the overnight window. Coverage before and after the cap is reported: rows per call site and, in the column filter, tables and columns covered per database and the yes/no proportion;
- **the column filter and the agent's call sites** (choice of the next tool) **go through neither the deduplication nor the per-question cap.** On them, both took the training away from the teacher's distribution: the agent's prompt does not depend on the question, and exact deduplication left one row per distinct (prompt, response) pair, which trained on the rare responses; the cap weighed questions, not calls, at a call site that runs once per database column. The rows come from K1's final attempts that passed the parse, masked:
  - all invocations of the agent's call sites;
  - in the column filter, a **uniform per-call** sample of 12,000 rows, a number fixed in the registered code (not in the configuration), within the rule of thumb, with the seed `seeds.curation_sample`, recorded with the rule in the datasets manifest. Without a cap, the column filter alone would take from 55 to 113 GPU-hours;
  - each row goes to the cluster of the centroid nearest to its prompt, the assignment the router makes at run time (S3).

**Deviation.** The paraphrasing of entities and numbers is **not applied**. In text-to-SQL, the values and names are part of the correct answer, and paraphrasing them would break the task. Recorded as a finding about S2 in structured tasks.

### S3 · Task clustering

**Paper (p.9).** *"Employ unsupervised clustering techniques on the collected prompts and agent actions to identify recurring patterns."*

**We execute.**
- Unsupervised clustering over prompt + action, with an automatic number of clusters and **without seeing the call site id**.
- **The discovered clusters define the units of specialization** of S4 and S5.
- Validation: agreement (ARI) between clusters and call sites.
  - **ARI ≈ 1:** S3 is trivial in *code agency*, and that is reported as such.
  - **Otherwise:** the pipeline proceeds with the discovered clusters, and the cost of the clustering error shows up in the final quality.
- **Assignment of a new call to a cluster, at run time.** Uses **only what exists before the response**: the prompt, which in the repair already carries the failed SQL and the error message.
  - Rule: each call goes to the nearest centroid, computed over the prompt only.
  - The rate of correct assignment, against the S3 clusters, is measured on the calibration and reported.
  - The teacher's action helps **form** the clusters, but never enters the **assignment**.

### S4 · SLM selection

**Paper (p.9).** *"For each identified task, select one or more candidate SLMs. Criteria for selection include the SLM's inherent capabilities (e.g., instruction following, reasoning, context window size), its performance on relevant benchmarks for the task type, its licensing, and its deployment footprint (memory, computational requirements). Models of Section 3.2 serve as good starting candidates."* The text refers to section 3.2, but the list of models is in 3.1 (A1, p.3–4).

**We execute.**
- **(a) Documented desk triage.**
  - Starts from the paper's own list (A1), in their current versions, plus the current candidates within ~10B (WD1).
  - Applies the four criteria, with each candidate's result in a table published in the report:
    1. capabilities: instruction following, reasoning, and context window ≥ the largest prompt observed in S1 for the cluster;
    2. public benchmarks relevant to the task type;
    3. license that permits commercial use and fine-tuning;
    4. footprint: fits on one reference GPU.
- **Starting from the paper's list gives it no preference.** The choice is by the criteria only.
- **(b)** **2 candidates** come out for the zero-shot check on the calibration, **never on the test**.
  - First, the mandatory filters: license, window, footprint and support in the inference server.
  - Then, the order is by **independent evidence on the task type** (tool calling, structured output, SQL). A self-reported number only breaks ties.
- **(c)** The best one goes on to S5. 2 go on if the budget allows, as S5 admits (*"fine-tune the chosen SLMs"*).
- **Pre-registered tie-break:** highest accuracy in the per-call-site evaluation on the calibration; if tied, smallest footprint.

**Deviation.** The paper asks for "one or more" candidates per task; the POC uses the same 2 candidates for all clusters. B2 (section 5) is not testable with 2 candidates and is left for extension 10.

### S5 · Specialized fine-tuning

**Paper (p.9).** *"fine-tune the chosen SLMs… PEFT techniques such as LoRA or QLoRA… knowledge distillation, where the specialist SLM is trained to mimic the outputs of the more powerful generalist LLM."*

**We execute.**
- One parameter-efficient adapter per cluster, on the base chosen in S4, all served together.
- **Sequence-level distillation:** SFT on the teacher's curated outputs.
- Logit distillation only enters if teacher and student share the tokenizer and the teacher exposes logits. Otherwise, it is declared as not applied.
- The time and cost of each adapter are recorded. It is A2's *"overnight rather than over weeks"* (p.5).

### S6 · Iteration and refinement

**Paper (p.9).** *"One may retrain the SLMs and the router model periodically with new data… returning to step S2 or step S4 as appropriate."* The paper presents the iteration as optional ("One may") and **presupposes a router without defining it**.

**We execute.**
- **Router (never cut):** it is **a decision of this POC**, needed to test A6 (heterogeneous system). It is not an obligation of the S6 text. The POC uses the simplest router compatible with the paper, a **per-cluster policy** (arm B5).
  - The call is assigned to the cluster by the S3 rule (the prompt only).
  - **Clusters with per-call gold** (SQL generation and repair): the cheapest engine whose per-call-site evaluation passes non-inferiority on the calibration, with a selection margin of 2.5 p.p. and sufficient n (6.4).
  - **Clusters without per-call gold** (keywords, column filter, table and column selection, tool choice): the cheapest engine whose **agreement with the teacher** on the calibration reaches the pre-registered bar (6.4). The bar never exceeds the teacher's agreement with itself. It is a **proxy**, declared as such: it only allocates, it does not support a per-cluster claim.
  - **A cluster without sufficient evidence stays with the production LLM.**
  - A trained router (the paper's "router model") is an extension.
- **One pass of the loop (the only cuttable part of S6, declared if cut):**
  1. error analysis on the calibration;
  2. back to S2 (adjust the filter) or to S4 (swap the base) on the weakest cluster;
  3. one retraining;
  4. new evaluation on the calibration.

  The test is only touched at the end, with everything frozen.

**Deviation.** The "periodic, with new data over time" retraining is not testable in 2 days.

---

## 5. Map of the claims: what confirms, what refutes, the limit

Each row was fixed before any result. "Confirms" and "refutes" use the margin Δ of section 6. A result without power is "inconclusive", not confirmation.

**Scope of every row: this workload.** One agent (CHESS), one domain (SQL), *code agency*, and the same 11 databases in training, calibration and test. The report repeats this line at the top of the map and writes "(on this workload)" on each claim.

| Claim (paper) | Comparison | Confirms if | Refutes if | Limit of what is concluded |
|---|---|---|---|---|
| **V1 / A1:** SLMs suffice for agents' calls (p.3–4) | **end to end** (n = 498): **B4 × B0**, and the contrast B3 × B4; **per call site** (6.2) only in the clusters with gold (generation and repair). B5 × B0 goes to the A6 row | B4 non-inferior to B0 end to end | B4 below −Δ end to end | one agent, one domain (SQL), *code agency*, the same databases in training and test; in the clusters without gold there is no per-cluster claim, only agreement with the teacher as a proxy. V1 is judged on B4 because B5's selection on the calibration has little power: a B5 that sent almost everything to the LLM would pass without any SLM. **Training volume, a limit in both directions:** rows per cluster c0 11,982 (column filter); c1 2,517 (table selection 833, column selection 833, candidate SQL generation 826, plus 18 column-filter, 6 repair and 1 keyword-extraction rows); c2 833 (keyword extraction); c3 9,164 (the agent's steps), against the paper's rule of thumb of 10k–100k examples (p.9): only c0 is within it. A negative result of B4, end to end or in a cluster below the rule, does not refute the claim for log volumes within the rule; a positive one holds only for this volume and these 11 databases. V1 is still judged on B4 end to end, with the same arm, margin and decision rule |
| **A4 / A11:** the calls are narrow and the subtasks simple (p.5, p.7) | replaceable fraction by **call, token and cost** | high fraction also by token and cost | high fraction only by call (the cheap ones) and low by cost | same |
| **Appendix B:** the LLM keeps "unstructured error resolution" (p.16) | **per-call-site evaluation** on the repair cluster: the EX of the SQL repaired by the SLM × by the LLM, with the same context | the SLM loses on repair | the SLM is non-inferior on repair (the partition is conservative) | one repair call site, **decided by the repair only**; the routine shows up as agreement with the teacher, a proxy with no effect on the verdict (D15); without power, "inconclusive, with the power"; with fewer than `min_calls` repair calls, "inconclusive" by design (D17) |
| **A5:** a single format with a trained SLM is preferable (p.6) | format validity per call site: B4 × B0 | B4 ≥ B0 − 1 p.p. on every call site, with no decoding trick | B4 more than 1 p.p. below B0 on a call site with sufficient calls (`min_calls`) in both arms | this agent's formats; below `min_calls`, descriptive |
| **A6:** heterogeneous systems (p.6) | B5 × B4 × B0 × B1 | B5 non-inferior to B0 and cheaper than the best untrained one, **at the lowest utilization** (the SLM's worst case) | B5 does not beat the best untrained arm **even at the highest utilization** | per-cluster policy designed by us; the paper says "can be" and admits an SLM-only system (Fig. 1). Outside the two edges, "depends on utilization", with the tipping point. If B5 allocated nothing to an SLM, "not testable" |
| **A7:** the agent's logs become data (p.6) | B4 trained on the teacher's curated logs (+ extension 2) | logs ≈ gold | gold ≫ logs | open teacher |
| **V3 / A2:** a 7B SLM is 10–30× cheaper "in latency, energy consumption, and FLOPs" than a 70–175B LLM (p.4) | ratio of cost per correct query (best untrained arm that passes V1 ÷ SLM arm that passes V1), with a 95% CI by paired bootstrap over the questions | **reaches the paper's bar** if the lower bound ≥ 3; **cheaper, below the bar** if the lower bound is between 1 and 3 | the upper bound ≤ 1 (the SLM is not cheaper) | the measure is dollar cost per correct query, **not** the 10–30× in FLOPs; market price of the whole container. Confirmation read at the lowest utilization, refutation at the highest; between the two, "depends on utilization", with the tipping point; the rest, "inconclusive" |
| **AV2 / CA3–CA4:** centralized scale may be cheaper (p.7–8) | the same, against B1 with cache, at 20/50/100% utilization and with fixed cost | — | B1 with cache ≤ SLM per correct query **also at the highest utilization** (AV2 wins) | same; between the edges, "depends on utilization" |
| **A2 (fine-tuning agility) / A3:** adapting is fast and cheap (p.5) | time and cost per adapter; the effort log | — | — | descriptive measure |
| **B2:** generalist benchmarks guide selection poorly (p.8) | **not testable in the core**: with 2 candidates, comparing rankings is not informative. In extension 10, a broad zero-shot round (6–8 candidates) | (extension 10) the rankings diverge | (extension 10) the rankings coincide | only with extension 10 |
| **S3:** clustering discovers the tasks (p.9) | ARI clusters × call sites | — | — | in *code agency* it tends to be trivial |
| **AV1:** a same-generation LLM always wins (disputes V2, p.7) | B0 × B4 | — | — | our models are **not** "of the same generation" in the paper's sense; it is not a direct test of AV1 |

**Replaceable fraction.**
- It is operational and oracle-free: the largest fraction of calls, tokens and cost served by an SLM in B5 with non-inferiority satisfied.
- The oracle ceiling appears separately, only as a reference.
- It is **not comparable** to Appendix B's 60/40/70% estimates. Those are fractions per role in whole agents, and one of them, Cradle, is a screenshot agent.
- It counts **only the clusters B5 assigned to the SLM**, which makes it conservative.

**Contribution of each step (D8).** What is measured in the core, and what is declared untested:

| Step | Comparison in the core | If there is none |
|---|---|---|
| S1 (collection) | — | **untested** as a contribution: without logs there is no specialization; volume and cost reported |
| S2 (curation) | effect of masking and the filter's pass rate, descriptive | **untested** as a contribution; filtered × unfiltered is an extension |
| S3 (clustering) | ARI and assignment rate | contribution **untested**; if ARI ≈ 1, trivial |
| S4 (selection) | triage documented by the 4 criteria; zero-shot of the 2 candidates on the calibration | contribution **untested** in the core (B2 is left for extension 10) |
| S5 (specialization) | **B3 × B4** (zero-shot × specialized), end to end and per call site | — |
| S6 (router and iteration) | **B4 × B5** (all on SLM × allocation); the retraining loop: calibration before × after | if the loop is cut: router measured, iteration **untested** |

---

## 6. Measurement protocol

Everything here is fixed **before** any result on the test.

### 6.1 Pre-registration and test registry
- **Before touching the test,** the hash of the following is published:
  - this spec;
  - the configuration (models per role, providers, parameters, seeds);
  - the lists of calibration and test IDs;
  - the margin Δ and the verdict rules (the analysis code that applies them enters the hash);
  - the code that executes, file by file, defined by exclusion so that no new file is left out: every file in `bench/` that is not analysis code, plus CHESS, the serving and training apps and the environments that install them; and the registered commit.
- **Stage 0, before the pilot.** The configuration with all of this spec's verdict rules is published **before the pilot is read**. After the pilot, only values that an already-fixed rule computes enter: the agreement bar, the SLO, B1's k. The analysis code that applies the rules is in the same commit as Stage 0 and does not change until the pre-registration; a change to it is valid only for a defect or a change declared in the deviation table before the pre-registration, and it enters that table, which the report prints.
- **Independent timestamp.** The git date is chosen by whoever commits. That is why the Stage 0 hash and the pre-registration hash receive a third-party timestamp, and the repository's `main` does not accept force-push.
- **Test registry:** each configuration is evaluated once, and all evaluations made are reported.
  - **The intent is published before the first question.** An execution on the test that cannot publish its intent does not start.
  - **Only the registered code runs on the test.** An execution on the test does not start with a dirty tree (a modified or untracked file; only the repository's `.gitignore` decides what is ignored, and nothing may be in `.git/info/exclude`) nor on a commit whose content differs from the registered one outside the registry itself (`prereg/` and `registry/`). The reading of the test (evaluation, judgments, report) writes to `judgments/` and therefore does not require a clean tree, but requires the same code: the registered content outside `prereg/`, `registry/` and `judgments/`. Fixing the code after the pre-registration is registering again.
  - **Only completed executions count.** Two completed executions of the same configuration, under the same registration, leave that arm's comparisons without a verdict.
  - **An execution that did not finish** (network drop, lost pod, interruption, killed process) remains listed. Its manifest enters the registry right at the start, with the intent, and at the end carries the reason and the hash of the predictions and of the calls it had up to that point.
    - **It is resumed**, from the most recent attempt: only under the same registration, the same configuration (hash) and the registered code, with a clean tree both in the source execution and in the resumption, and with the copied predictions and calls still matching the registered hash. A killed execution that did not close its manifest is resumed from what it recorded question by question; since no hash vouched for it, the hash of what it left is computed before anything else and published in the resumption's intent, declared as unregistered.
    - **Redoing from the start is not an alternative to resuming.** It is valid only when resuming is impossible (outputs that do not match the hash, the execution without a local directory, an intent never published) or when a declared defect affects questions already completed. The defect is published in the intent, before the first question, and enters the manifest; before that, the deviation table gains a line that names the abandoned attempt (D22). The code refuses to redo an attempt that can still be resumed, except when the defect is declared to affect what it completed. The previous attempts remain in the registry.
  - **The report lists all attempts** of each configuration: how each one ended, the defect that led to redoing it, and the n and the EX. An abandoned attempt with predictions appears with the partial EX, descriptive, beside the final one: on the questions it completed, with model failures counted as errors, and without those that did not finish or had a harness error. It overlaps the questions the final execution copied when resuming it, and never enters a verdict.
  - **Exploratory readings of the test, without a verdict.** After the registered test executions, B0's test calls are replayed with the base SLM (the chosen candidate, no adapter) and with the cheap alternative LLM, each once, under the registered configuration; their per-call evaluations feed a per-call-site table beside B4's replay, computed after the test by a script outside the registered code (`bench report` does not print it). They are listed in the test registry like any execution, never enter a verdict, and are reported as exploratory.
    - **The per-call-site table** (descriptive, no verdict). One row per call site, and one per cluster: the base SLM, the adapter (B4's replay), the cheap alternative (gpt-oss) and the teacher; the metric of the row (EX against the gold in generation and repair, where the teacher's cell is its own EX on the same invocations; agreement with the teacher elsewhere, where the teacher's cell is its self-agreement A of 6.4, measured on the pilot, labelled as such); the effective n; and the training rows of the cluster (section 5, row V1/A1). Warnings published with the table: keyword extraction is scored by exact-set agreement, which even the teacher rarely meets, so a low value there says little about the decision; the agent's steps have about 11 distinct prompts (the prompt does not depend on the question), so their effective n is the number of steps, not of calls; repair has a small n (the teacher hardly repairs) and 6 training rows, and is read as such.
    - **Projection-tolerant EX** (exploratory, no verdict). BIRD's EX compares sets of row tuples, so a prediction that returns the gold's columns plus others is wrong. Beside each arm's EX, the share of questions that are right once the prediction's extra columns are dropped: some of its columns, kept in their order, give the gold's set of rows (the gold is an ordered projection of the prediction). Declared before the test; computed from the stored predictions after the test by a script outside the registered code; it never enters a verdict.
    - **The c1 learning curve** (exploratory, no verdict). c1 adapters trained on 25%, 50% and 100% of its rows (nested subsets, by call site), with the registered recipe and the same number of epochs, one seed each, evaluated per call site by replaying B0's calibration calls at c1's call sites. Declared before it runs, reported after the test verdicts and beside them, and it feeds no registered decision: not the arms, not B5's allocation, not any verdict.
  - **Registering again** is valid only for a documented defect. The defect enters a deviation table, which the report prints, and the executions of the previous registration appear as superseded.

### 6.2 Units
- **Question:** for accuracy (EX) and cost.
- **Call:** for format and replaceable fraction.
- **Call with the teacher's context (per-call-site evaluation).** Each call site is evaluated with the same inputs the production LLM received in the B0 execution, and the SLM answers in its place.
  - The output is compared with the gold when it exists: the EX of the generated or repaired SQL.
  - When there is no gold, it is compared with the teacher's output, and that is reported as **fidelity**, not as correctness.
  - It is the basis of the per-cluster conclusions. The end-to-end conclusions come from the complete executions.
- **Cluster:** for the allocation.

### 6.3 Metrics, per arm and per difficulty
- execution accuracy (EX);
- format validity per call site;
- **cost per correct query**;
- p50 and p95 latency;
- replaceable fraction (call, token, cost);
- time and cost per adapter;
- effort log (6.8).

### 6.4 Statistics
- **Margin: Δ = 5 p.p., fixed.** It is the EX loss this POC accepts calling "without losing quality", decided before any data. A margin cannot depend on what the study is able to detect.
- **Pilot:** 50 calibration questions, drawn with a seed **in the test's difficulty mix** (15 simple, 25 moderate, 10 challenging). The calibration has 67% simple questions (134 of 200, by the labels of the revision the splits use) and the test, 30%.
  - The pilot measures d, the paired discordance between the zero-shot SLM and the production LLM.
  - d serves only the **planned power** at n = 498, reported before the test.
  - The pilot also gives the projection of time and spend (7.1).
- **Non-inferiority:** one-sided CI by paired bootstrap, with the question as the unit.
- **Conjunctive claims** ("replaces in all routine clusters") require all tests to pass, without Holm correction, and the joint power is reported.
- **"Inconclusive"** counts as not confirmed, and is reported **with the power**, so as not to be read as a refutation. The report shows the CI of each comparison: the lower bound is the largest loss the data still admit.
- **Selection on the calibration** (B5's allocation) uses a 2.5 p.p. margin, half of Δ, so as not to choose engines that are at the limit. A cluster needs at least `min_calls` calls on the calibration to count as evidence.
- **Agreement bar** (clusters without gold, in B5): per call site, **min(95%, A − 2 p.p.)**, where A is the teacher's agreement with itself.
  - A is measured by re-running the teacher on the pilot's calls. A model via API does not repeat its own output even at temperature 0, and a bar above A would be unreachable for any engine.
  - Agreement is defined per call site (same list, same JSON, same decision). The rule is fixed in the pre-registration, and A is reported per call site and per difficulty.
  - The bar measures reproducibility, not correctness: it only allocates. Whether swapping these calls preserves correctness is judged end to end (B5 × B0 on the test).
- **B5's selection runs on a calibration easier than the test.** It is reported per difficulty. A wrong selection shows up on the test as B5 failing against B0: it costs a confirmation of A6, and does not produce a false confirmation.
- **Per-cluster conclusions** exist only in the clusters with gold (generation and repair). In the others, the claim is **end to end only** (B5 × B0 on the test).
  - The margin is the same, Δ = 5 p.p., and the power of each conclusion is reported.
  - In B5, a cluster without sufficient evidence on the calibration **stays with the production LLM**.
  - The resampling unit remains the question, grouping its calls.

### 6.5 Router and replaceable fraction
- The router uses only information available at run time: the cluster assigned by the **prompt** (S3 rule) and the policy chosen on the calibration. The teacher's response never enters the decision.
- Any score-based router (extension) calibrates the threshold on the calibration, with binary loss.

### 6.6 Cost
- **Models via API:** each call's `usage` (input, cache, output) × the official price table, dated in the configuration.
  - The price variants (without cache, with batch discount) are **recomputed from the same `usage`**, without running again.
  - Retries are included.
- **SLM:**
  - cost per question = hourly price **of the whole container** (GPU, CPU and memory, as the provider charges) ÷ (questions per hour sustained within the p95 × utilization), at **20, 50 and 100%**;
  - **the p95 limit is a single rule, for every sweep:** the stricter of an external reference and the production LLM's per-call p95 measured in the pilot, **reweighted to the sweep's own call-site mix** (each call site weighs in the teacher's latency what it weighs in the calls the sweep replays). A call site with fewer than 30 teacher calls is grouped with the one of the same task (repair with SQL generation). The SLM has to answer at least as fast as the API it replaces, on the traffic it replaces. The p95 without reweighting (the teacher's own mix) is reported as sensitivity. The rule is fixed before any load test;
  - **the router's time enters the latency** of the sweeps that go through it (each adapter and B4 as deployed; B5 uses the same adapters and the same router): the p95 of the embedding time measured in B4's calibration execution is added to each level's p95. The wait for the embedding model, a harness artifact (one model per process), stays out and is declared. The base model (B3) does not go through a router. The cost without this addition is reported as sensitivity;
  - **the utilization that decides:** a confirmation is valid only if it holds at 20% (the SLM's worst case), and a refutation is valid only if it holds at 100% (the best case). Between the two, the verdict is "depends on utilization", with the tipping point;
  - **the price that decides:** a confirmation (V3, A6) is valid only if it also holds at Modal's price for the same GPU, with CPU and memory charged separately (the provider sensitivity);
  - the cost-per-correct-query ratio carries a 95% CI by paired bootstrap over the questions (section 5, V3 row);
  - throughput is measured by a **load test that replays the agent's real calls**, on a rented GPU billed per second and with prefix cache on, **each engine on its own arm's traffic on the calibration**: the base model on B3's calls; each adapter on the calls B4 sent it, the per-engine sweep with which S6 and B5 are priced; and **B4 as deployed**, one server with the four adapters and all of B4's calls, each one to the adapter the router chose, which gives B4's cost. Each level draws its calls from all of the engine's traffic, stratified by cluster (by call site on the base model, which does not go through a router) and without repetition within the level, and **every level of every sweep starts with an empty prefix cache**. The concurrency levels start at 1;
  - **an engine with no level within its limit is a result, not a failure**: the SLM does not keep up with the API it replaces on that traffic. That engine is left without a cost, S6 does not give it the cluster, and a cost claim without any SLM arm priced because of this is "not testable at the API's latency";
  - the fixed cost (engineering hours and retrainings) enters the payback at three reference volumes.
- **Optional:** a measurement on the author's consumer device (WD1), reported separately.
- **Extension:** the estimate of the cost on closed LLMs, repricing the token counts, labeled as an estimate and without a quality claim.

### 6.7 Latency (descriptive)
- p50 and p95 per arm, as measured. No verdict depends on them.
- The API arms and those served on our own GPU do not run under the same load profile, and that is said in the report.

### 6.8 Effort log, per step
- human hours;
- GPU hours;
- API spend;
- what ran on its own and what required judgment;
- what would change in a second task.

The time to build the harness is kept **separate** from the conversion time.

---

## 7. 2-day core, preconditions, cuts and fallback

### 7.1 Preconditions (morning of Day 1)

Each one with the corresponding cut action:

| Precondition | If it fails |
|---|---|
| The agent runs one question end to end in ≤3 h | use an equivalent controller: the same call sites, with the same prompts, the same order and the same data flow, and the repair loop. The report declares which was used. If not even that runs by the end of the morning, go to the **fallback** (7.4). |
| The set of call sites observed in the training logs and in the calibration rounds is recorded (id written on each call) | fix the harness before proceeding |
| The Mini-Dev IDs match the dev, and the corrected gold is accessible | fix the mapping; without gold, there is no test |
| The teacher provider's terms and the author's permission to train on the outputs are recorded, with link and date (D6) | do not start K1: without this record, no teacher output becomes training data |
| Each engine answers a real call with the configured parameters, at every call-site temperature: HTTP 200, text, `usage`, normal stop (not by token limit) and reasoning in the configured state | fix the configuration; no experiment spend before that |
| A 2-question smoke per arm checks `usage`, cached tokens, stop reason and format | fix the configuration |
| The provider, the quantization and the checkpoint of each model via API are pinned and recorded | pin before any execution |
| The pilot's spend and time, projected to the core (seconds per question, cache rate, cost), fit within the ceiling and the phases of 7.3 | cut extensions and reduce the pilot before touching the core; then, the cuts of 7.3 |
| The measured throughput and training time confirm the schedule | apply the cuts of 7.3 |

**Check during the test (not a precondition).** In Day 2's test rounds, an automatic assertion compares each call's id with the recorded set. If a call site outside the set appears, the configuration is interrupted, flagged, and the case goes to the report.

### 7.2 Core (everything has to run for the POC to be valid)

| # | Item | Acceptance criterion |
|---|---|---|
| K1 | Teacher logs on the training questions, on all call sites | ≥90% of the questions with a complete trajectory; volume per call site recorded |
| K2 | B0, B1, B2 and B3 on the test | the 498 questions evaluated; cost and p95 measured |
| K3 | S2 → S3 → S4 → S5: the trained adapters | training converges; valid format ≥95% on the calibration |
| K4 | B4 and B5 on the test, with the S6 router (after the retraining loop, or without it if cut, declared), and the per-call-site evaluation | the 498 questions; cost per correct query; n and power per cluster |
| K5 | The report (section 8), including the negatives, the power and the test registry | all items of section 8 |

### 7.3 Schedule and cuts

| | Morning | Afternoon / night |
|---|---|---|
| **Day 1** | Stage 0 (6.1), preconditions (7.1), pilot (with the S4 choice, which the pilot's B3 uses), projection of time and spend | K1 (logs), the choice of B1's k, S2 and S3; launch the S5 trainings to run overnight |
| **Day 2** | S6 (one loop), the pre-registration, and K2 and K4 on the test | cost, statistics, report |

- **The pre-registration comes after S6 and before any test question.**
  - It registers the final configuration, and that names what the arms use: the S4 choice, the centroids, the adapters and the allocation. All of that only exists after training.
  - B1's examples come from the training logs, so the choice of its k (6.4) also comes after K1.
  - The verdict rules have been published and timestamped since Stage 0. Between Stage 0 and the pre-registration only these facts change, the values that an already-fixed rule computes and the changes declared in `prereg/DEVIATIONS.md` (section 11.4), and the difference between the two commits is public.
- **The days need not be consecutive** (D26). The project may pause between phases while the next phase's budget is not secured: after the pilot and the projection, before K1, and before the test. The order of this table and the pre-registration before the first test question do not change.
- **K1 runs through the paid API** (D25), like the calibration and the test: the same teacher calls, with the same parameters, and each one's log with the API's `usage`. No verdict reads its cost or latency.
  - An execution outside the test that stops because of an infrastructure failure (such as the account balance running out) continues in a new execution, which copies the questions the source one completed and answers the others; the source's calls on unfinished questions are discarded, and the manifest records the source and each session. It is a single execution, however many sessions it takes. On the test, resuming and redoing follow 6.1.
  - The registered one ran like this: 834 training questions, 82,403 recorded calls, seven sessions on 2026-11-09, without the batch API.
- **If time or budget get tight,** the extensions are cut in reverse order of the list, and then the S6 retraining loop (declared; K4 runs without it).
- **Never cut:**
  - K2 (the comparison with the untrained path);
  - K5 (the report);
  - the S6 router (B5);
  - the per-call-site evaluation.

### 7.4 Fallback: a narrower test, declared as such

If the agent does not run, the POC becomes a **single-call** text-to-SQL test: production LLM × cheap alternative × specialized SLM. It does **not** claim to test the conversion of an agent.

| Remains testable | Stops being testable |
|---|---|
| V1 (on a single task) | A4/A11 (the distribution of calls in an agent) |
| A5 (format) | the Appendix B partition (the repair) |
| A7 (single-call logs as data) | A6 (heterogeneous allocation per call site) |
| V3/AV2 (cost per correct query) | S3 (clustering) |
| | the replaceable fraction per call type |

---

## 8. Deliverables

1. **Public repository:**
   - code;
   - configuration (where the models and providers are named);
   - the pre-registration hash;
   - the test registry;
   - the aggregated logs.
2. **Report:**
   - the main chart: EX × cost per correct query, with B0–B5;
   - the **section 5 map filled in**: confirms, refutes or inconclusive, with the power;
   - the S1–S6 table: what each step did, cost and changed;
   - the **S4 triage table**: candidates × the paper's 4 criteria, with each one's result;
   - the replaceable fraction by call, token and cost;
   - the analysis of the SLM's errors;
   - the limitations. They are a declared limit in both directions, not an excuse for either result:
     - **scope, first:** one agent (CHESS), one domain (text-to-SQL, BIRD), teacher Claude Sonnet 5.5, one 8B SLM (Qwen3-8B), one LoRA recipe with no hyperparameter search;
     - **training volume:** the rows per cluster beside the paper's 10k–100k rule of thumb (p.9), with the reading of section 5, row V1/A1: below the rule, a negative result does not refute the claim for volumes within it, and a positive one holds only for this volume and these 11 databases. V1's verdict is B4's end to end, unchanged;
     - **the exploratory readings of 6.1** (the per-call-site table, projection-tolerant EX and the c1 learning curve), with their warnings and without a verdict;
     - **limits of the measurement found on the calibration** (text, no verdict; each one bounds a reading, in either direction):
       - **pipeline order:** the base SLM (B3) runs CHESS's schema selection in another order: in 200 of 200 calibration questions it calls `select_tables` before `filter_column`, where B0 and B4 call `filter_column` first. B4 − B3 therefore mixes the fine-tuning with a change of pipeline order, and is not the effect of the training alone;
       - **the SLM's reasoning is off:** the candidates are served and trained with thinking off (`chat_template_kwargs: {enable_thinking: false}` in `config.yaml`), while the teacher and gpt-oss reason. A gap between them includes that difference;
       - **one run per arm, on an engine that is not deterministic:** the temperatures are CHESS's (0 at most call sites, 0.2 at keyword extraction, 0.01 at generation), vLLM's output is not batch-invariant even at 0, and no seed is sent. Two B3 executions of the same 50 pilot questions gave 15 and 18 right (the same final SQL on 22 of 50). Each arm runs once on the test, so a difference of that size between two arms can be noise of a single run;
       - **the calibration is not stratified by database:** it is a seeded draw (the smallest sha256 under `seeds.calib_split`) of the 1,034 questions outside Mini-Dev, so its mix differs from the test's (`european_football_2` is 5 of 200 calibration questions and 51 of 498 test questions; `debit_card_specializing` 4 and 30). Thresholds and choices made on the calibration weigh the databases differently from the test;
       - **per-call EX in generation is a first candidate:** in a replay, `generate_candidate` is scored on the engine's candidate before any repair, and `revise` repairs the teacher's candidate, not the engine's own; end to end, the agent repairs its own SQL;
       - **format validity at the SQL call sites is 100% by construction:** CHESS's parsers for `generate_candidate` and `revise` accept any text as SQL, so a malformed answer shows up as a wrong or failing query, not as a format failure;
       - **extra columns are wrong:** BIRD's EX compares sets of row tuples, so a prediction that returns the gold's columns plus others is wrong. On the calibration this costs B3 8 of 200 questions, B4 2 and B0 1 (the questions where the gold is an ordered projection of the prediction); the projection-tolerant EX of 6.1 reports it on the test;
       - **one test question repeats a training question** (section 2: test 875, train 863), with its sensitivity reading;
   - a section of **non-comparable external references**: published results of models on the same questions or on the same benchmark, obtained with another harness, another version or another gold, always labeled as such.
3. **Contributor package** (produced **after the core**, outside the 2 days):
   - one command per extra arm (extensions 8 and 9);
   - IDs, commit and parameters pinned;
   - the required artifacts: raw logs with the request identifiers and each call's `usage`;
   - attribution to whoever ran it.

   The compliance rule (section 3) applies to any contribution.
4. **Submission to the authors** through the channel the paper itself opened for contributions and critiques (p.1 and p.9), which they commit to publishing.

The final texts (report, posts) are written by the author. The development sessions deliver structure, code and review.

---

## 9. Out of scope

| Item | Reason |
|---|---|
| A2 (parameter and embedding space utilization) | requires internal instrumentation of the models |
| A3 (democratization) | not testable in a POC |
| A8 (architecture by size) and A10 (inference compute) | separate experiments |
| A12, A13, AV3, B1, B3 | market or adoption trends |
| Whole conversational agents | the paper leaves that ground to the LLM (Appendix B); only the inference probe as an extension |
| Periodic retraining over time | does not fit in 2 days |
| Real data, real PII, real drift, speed of a second conversion | outside a public POC with benchmark data |
| Vision-language models, environmental claims | no instrumentation to measure |
| Other domains and languages | the same harness serves a second task, as future work |

---

## 10. Rules and decisions

### Rules
- **Data:** only public data and code, with a license that permits this use. No private data.
- **The test** does not enter training, selection or calibration, in any arm.
- **Outputs forbidden for training:** those of any model whose terms forbid training on them never become a training target, nor a filter, nor a label (do-not-train area, section 3).
- **Project:** public, prior to any contract and single-authored.
- **Level:** this spec does not name models, providers or libraries; that goes in the configuration (exception: the agent and the benchmark, which are the object).

### Fixed decisions

| # | Decision |
|---|---|
| D1 | Workload: the CHESS agent on the BIRD questions. Logs on the dev outside Mini-Dev; test on Arcwise-Plat-SQL (498, used as it is in the file) |
| D2 | One workload and one benchmark; no multi-task suite |
| D3 | Protocol fixed beforehand, **with** a pre-registered non-inferiority margin (6.4 and D16). Curves and frontiers are still reported |
| D4 | Abstract roles, with the concrete choice in the configuration |
| D5 | The teacher's internal reasoning stays out of the logs |
| D6 | Production LLM = teacher: a production-grade closed LLM, through the provider's paid API, K1 included (D25); the provider's terms and the author's permission to train on the outputs are recorded in the configuration, with link and date (author, 2026-10-14; until v2.1, an open-weight one) |
| D7 | Provider-neutral log schema, with model and version in each record |
| D8 | Each step's contribution is measured against named alternatives (section 5 and extensions), never by "removing the step" |
| D9 | Results stratified by difficulty, with all categories reported |
| D10 | In the core, the only closed LLM is the teacher (D6); the cheap alternative and the SLMs are open-weight. The arm with the frontier closed one is an extension open to contributors (author, 2026-10-14; until v2.1, no closed one in the core) |
| D11 | The single-call fallback is declared as a narrower test (7.4) |
| D12 | S4: desk triage documented by the paper's 4 criteria, starting from its list (with no preference for it); 2 candidates in the zero-shot; 1 (or 2) in the fine-tuning; pre-registered tie-break |
| D13 | CHESS is corrected so that generation and repair use the selected schema (CHESS issue #34), declared in the report (author, 2026-09-30) |
| D14 | The test is Arcwise-Plat-SQL with 498 questions, used as it is in the file; ids 119 and 120 stay out of training and test; not comparable with the EX published on Mini-Dev (author, 2026-09-30) |
| D15 | In the clusters without per-call gold, B5 chooses by agreement with the teacher (D17's bar), declared as a proxy; the claims in those clusters are end to end only (author, 2026-09-30) |
| D16 | Fixed non-inferiority margin, Δ = 5 p.p. (2.5 p.p. in the selection); the pilot only gives the planned power (author, 2026-10-01) |
| D17 | The agreement bar is min(95%, the teacher's agreement with itself − 2 p.p.), per call site; Appendix B is decided by the repair only, and with fewer than `min_calls` repair calls it is "inconclusive" by design (author, 2026-10-01; the floor, 2026-11-10) |
| D18 | V1 is judged on B4; B5 belongs to A6 (author, 2026-10-01) |
| D19 | Cost: ratio with CI, three readings (reaches 3×, cheaper below 3×, not cheaper), confirmation at 20% utilization and refutation at 100%, price of the whole container, and the p95 limit by the rule of 6.6 (author, 2026-10-01) |
| D20 | Each LLM via API runs as the provider delivers it, with the reasoning explicit in the configuration; provider, quantization and checkpoint pinned; a response cut by the token limit is recorded as such (author, 2026-10-01) |
| D21 | Training volume limited per (question, call site), by a seeded uniform sample, with the coverage reported; in the column filter and the agent's call sites, no deduplication nor per-question cap: all of the agent's invocations and a uniform per-call sample of the column filter, each row in the cluster of the nearest centroid (author, 2026-10-01; these call sites, 2026-11-10) |
| D22 | Test registry: only completed executions count; registering again requires a documented defect; Stage 0 and an independent timestamp (author, 2026-10-01) |
| D23 | B1's additional few-shot (k ∈ {0, 3}) is chosen on the pilot by a rule fixed beforehand: k = 3 only if the one-sided 95% lower bound of EX_k3 − EX_k0 (paired bootstrap, the question as the unit) is above zero, with the end-to-end EX on the pilot questions, from two executions of B1 itself in the agent, one with k = 0 and the other with k = 3 (author, 2026-10-01; the unit, 2026-11-10: before, the EX per call on the call sites with gold from two replays of the teacher's execution) |
| D24 | Objective and constraint: the objective is to validate the paper's thesis; the constraint is the budget. Every correction is the smallest one that closes the finding (author, 2026-10-01) |
| D25 | K1 runs through the paid API, like the calibration, which gives the SLO, and the test, in a single execution, resumed in sessions if an infrastructure failure stops it; the manifest records each session. No verdict reads K1's cost or latency. The provider's verbal authorization for K1 through the author's subscription (v2.3) was not used (author, 2026-10-14; the subscription, 2026-10-21; the paid API, 2026-11-09) |
| D26 | The core runs in the phases of 7.3, in its order, on days not necessarily consecutive: the project pauses between phases while the next one's budget is not secured (author, 2026-10-14) |

### Decisions in configuration, closed before the pilot's zero-shot reading
- models per role and the operational reserve;
- pinned provider;
- reference GPU;
- spend ceiling;
- generation and reasoning parameters;
- seeds.

---

## 11. What changed from v1 and why

| v1 problem | Where the correction is in v2 |
|---|---|
| Training and test came from different agents (logs from an open dataset, test on BFCL) | section 2 and D1 |
| There was no untrained alternative | B1 and B2 (section 3) |
| Router by "invalid call" and replaceable fraction dependent on an oracle | S6, 5 and 6.5 |
| "Tie within the CI" used as equivalence | 6.4 and D3 |
| The "remove one at a time" ablation decided by construction | D8 and section 5 |
| The scope did not fit the deadline | section 7 |

**Corrections of fidelity to the paper applied:**
- the paper claims SLM > LLM at several points (p.4, p.8), not only on format;
- 10k–100k is a "rule of thumb", not a floor (p.9);
- the 10–30× is latency, energy and FLOPs of 7B against 70–175B (p.4);
- *"overnight rather than over weeks"* is in A2 (p.5);
- V2 is supported by A3–A7 (p.5–6), and AV1 disputes V2 with models "of the same generation" (p.7);
- the "ought" is attributed "to many", and the paper makes no recommendation (p.2–3);
- A6 says "can be", with the LLM at the root, and admits an SLM-only system (Fig. 1);
- the replaceable fraction is not comparable to Appendix B;
- SmolLM2 starts at 125M parameters (p.3);
- the data rule was rewritten without the contradiction with the use of a public benchmark;
- the configuration decisions close before the zero-shot reading.

### 11.1 What changed in v2.1 (2026-10-01)

Before any paid execution, the design went through a validity audit: each step against the published failure modes and the gold standard of each one. The corrections below are those that prevent a wrong or empty verdict. The rest became a limitation line in the report.

| v2 problem | Where the correction is in v2.1 |
|---|---|
| The margin Δ came from the pilot's discordance: it measured what the study detects, not the acceptable loss, and exceeded the ceiling with d > 0.20 | 6.4 and D16 |
| The 95% agreement bar could sit above the teacher's agreement with itself | 6.4, S6 and D17 |
| V1 confirmed with "B4 or B5", and B5 can be almost all LLM | section 5 and D18 |
| V3 refuted with the SLM 2× cheaper; no utilization decided; the price counted only the GPU; the p95 limit was open | section 5, 6.6 and D19 |
| A5 refuted with any format difference | section 5 |
| The LLMs' reasoning was not configured, and a cut response counted as a model error; provider not pinned | 7.1 and D20 |
| Training on all of the column filter's calls did not fit in 2 days | S2 and D21 |
| An interrupted execution already counted as a second execution; the registration date depended only on git | 6.1 and D22 |
| B1's few-shot would be fixed without data | D23 |
| The pre-registration was on the morning of Day 1, before what the final configuration names existed (the adapters, the allocation, B1's k) | 7.3 |
| The pilot was easier than the test | 6.4 |
| It was missing that the 2025 revision of BIRD dev changed 85 questions and that its gold disagrees with the corrected one on 18.9% | section 2 |
| The map did not state the scope | section 5 |
| "Fair latency" promised an equal load profile, which API and own GPU do not have | 6.7 |

### 11.2 What changed in v2.2 (2026-10-14)

Before Stage 0, paid probes measured the teacher's cost in the agent. They were 5 training questions, none from the calibration or the test. These executions are not reused: K1 answers the same questions again, as it answers the others. v2.1's open-weight teacher had given way to a frontier closed one, and that one did not fit the budget. The intermediate model of the same line got the same 5 questions right for about 60% of the cost, and K1 in batch costs about half of that.

| v2.1 problem | Where the correction is in v2.2 |
|---|---|
| The chosen closed teacher did not fit the budget; the open-weight one had been discarded by the author | section 3 and D6: the production-grade closed one |
| D10 forbade a closed LLM in the core | D10: only the teacher |
| K1 alone was the core's largest spend | 7.3 and D25: K1 in batch, at half price |
| The core's budget is not entirely secured before the first spend | 7.3 and D26: phases with a pause between them, in the same order |

### 11.3 What changed in v2.3 (2026-10-21)

The provider denied API credits and verbally authorized, for this project only, K1 to run through the author's subscription. Before Stage 0, the author measured what this changes, again only on training questions. These executions are not reused either: K1 answers the same questions again.

- **What reaches the model.** A local proxy recorded Claude Code's request. The CHESS prompt arrives identical, byte for byte. With everything that can be switched off switched off, what remains is the line "You are a Claude agent, built on Anthropic's Claude Agent SDK." as system prompt, an environment block with the date and a reminder with the account's e-mail: 518 to 541 extra input tokens per call. `max_tokens`, the effort and adaptive reasoning are those of the configuration.
- **The effect on the responses.** On the 5 questions of the v2.2 probes, Claude Code got all 5 right, like the two executions through the API. On the calls with the same prompt, each one's decision agrees with the API as much as the API agrees with itself: 97.6% to 98.8% on `filter_column`, against 98.7% between the two executions through the API, and 100% on `agent_ir` and on `agent_cg`.
- **One difference.** On 94 `agent_ss` calls, one answered outside CHESS's format, with a tool call in JSON; the question proceeded without the schema selection and got it right anyway. Through the API it was 0 in 40 `agent_ss` calls and in 1,050 in total. The report counts these failures in K1.
- **No cache.** Claude Code puts the cache point at the end of the message, and each CHESS prompt is unique: each call wrote the whole prompt into the cache and almost never read it. Without cache, K1 spends about half of the usage limit per call.

| v2.2 problem | Where the correction is in v2.3 |
|---|---|
| The budget did not cover K1 even at half price | 7.3 and D25: K1 through the author's subscription |
| Claude Code adds context to each call | 7.3 and this section: measured, and the log keeps only the CHESS prompt |
| The subscription's usage limit interrupts long executions | 7.3: K1 pauses and continues on its own |

### 11.4 What changed after Stage 0 (v2.4, 2026-11-10)

Stage 0 (2026-10-23) published the configuration and the verdict rules. After it, each change that can alter a result or its reading is declared in `prereg/DEVIATIONS.md`, with the date, the reason, the moment of the decision and the direction of the effect. That is where the science of each change is recorded. This version only aligns the spec's text with those lines: it adds no rule. Everything comes before the pre-registration, and the difference against the Stage 0 commit remains public.

**Changes to the text in this version:**
- **K1 through the paid API** (section 3, 7.3, D6 and D25; deviation of 2026-11-09). K1 ran through the paid API, not through the subscription: 834 training questions and 82,403 calls, in a single execution resumed in seven sessions. v2.3's authorization was not used. The logs the SLMs learn from, therefore, do not have the ~530-token context that Claude Code added (section 11.3), and reached the model by the same path as the calibration and the test.
- **Resuming an execution** (7.3 and D25; deviation of 2026-11-09). An execution stopped by an infrastructure failure continues in a new execution, which copies the completed questions and discards the calls of those that did not finish. Five executions that the results read were resumed this way: K1, B3's pilot on the calibration, B0's pilot on the calibration, which gives the SLO (its p95 pools the two sessions), B4's calibration execution and the cheap alternative's replay of the calibration (the deviation's correction line of 2026-11-10 lists them, with the attempts abandoned without a resumption). On the test, resuming and redoing are those of 6.1, which D3's PR rewrites.
- **The training rows of the column filter and of the agent's call sites** (S2 and D21; deviation of 2026-11-10). No deduplication nor per-question cap: all of the agent's invocations and a uniform per-call sample of the column filter, each row in the cluster of the nearest centroid. Deduplication and the cap were additions of this project, not the paper's method, and on these call sites they trained on a distribution different from the teacher's. Curation, clustering and the centroids do not change. Decided after reading the base model's calibration and B3's pilot, before evaluating any adapter of these clusters; it changes the training data of the trained arms (B4, B5) and tends to favor them. It replaces the 415-row rule by (step, answer) frequency, declared earlier the same day and never used in an official training.
- **The unit of B1's k rule** (D23; deviation of 2026-11-10). The rule keeps its form, but moves from the EX per call on two replays of the teacher's execution to the end-to-end EX on the pilot questions, from two executions of B1 itself. Decided after reading the pilot: a replay gives each call the teacher's context and does not see the agent stop before the SQL. The change strengthens B1, the untrained arm, and makes the SLM's verdicts against it harder.
- **The Appendix B call floor** (section 5 and D17; deviation of 2026-11-10). With fewer than `min_calls` repair calls, the verdict is "inconclusive" by design, the same floor A5 and the selection already use. The teacher hardly repairs on this workload (5 calls on the calibration, 6 in K1). Decided after knowing the result of those 5 calls, before any execution on the test; the floor prevents them from deciding against the SLM and therefore favors the SLM arms in Appendix B.
- **Model rejections** (section 2; deviation of 2026-11-09). At Stage 0, every 400 or 413 error from the engine was a model failure, except the balance one. Now only the 413, or the 400 that says the prompt alone does not fit in the context, is the model's; the others, including that of a prompt that fits but leaves no room for `max_tokens`, are the harness's. No call of an execution the results read changes class; the 1,515 "not scoped to a workspace" 400s of one K1 session, which the Stage 0 rule scored as the model's, are why the rule changed, and that session was set aside.
- **The tool-choice parser** (section 2; deviation of 2026-11-10, patch 16). The prompt and the other rules do not change in any arm. B0's pilot and K1 ran with the strict parser: 1 of the pilot's 2,198 tool-choice calls and 3 of K1's 9,167 would pass with the patch.
- **The declared limits of V1 and two exploratory readings** (section 5, 6.1 and 8; to be declared in `prereg/DEVIATIONS.md`). The training rows per cluster stand beside the paper's 10k–100k rule of thumb in V1's limit and in the report's limitations, read in both directions; the per-call-site table and the c1 learning curve are declared as exploratory, without a verdict; and the limitations open with the scope. Written after the calibration had been read. It changes what the report says about V1, not its verdict: no number, margin, arm or decision rule changes.
- **Corrections to the record** (section 2, 6.1, 8 and this section; to be declared in `prereg/DEVIATIONS.md`). The training and test questions are disjoint by id, and one test question repeats a training one (875 and 863), with a reading without it; five resumed executions, not three; the model-rejection rule changed after one K1 session's 1,515 workspace 400s; limits of the measurement found on the calibration, as text; projection-tolerant EX declared as an exploratory reading, and the per-call-site table computed by a script outside the registered code. No rule, margin, arm or verdict changes.

**Changes declared in `prereg/DEVIATIONS.md` that are already in the text or do not change it:**
- **The load test, the latency limit and the SLM's price** (6.6, rewritten together with the change; deviation of 2026-11-10). One p95 limit per sweep, reweighted to its call-site mix; each engine on its own arm's traffic; B4 as deployed; each level with its own calls, stratified, and the prefix cache empty; the router's time added; an engine with no level within the limit is a result; and the condition of the registered provider's price. Before it, the load test already shuffled the calls with `seeds.loadtest_shuffle` before slicing the levels (deviation of 2026-11-10).
- **Training and serving provider, and the reference GPU** (2026-11-09). The adapters train on a managed fine-tuning service, instead of the registered GPU provider, and serving runs on another provider, with the same inference server, the same weights and the same GPU. The reference GPU became an A100 80GB before the pilot's zero-shot reading, for availability. The accuracy of the SLM arms depends on these adapters; the SLM's cost and p95 are measured on the A100. The spec does not name the GPU or the providers (level rule): the configuration records them.
- **The sample of B1's examples** (2026-11-10). Each agent step receives examples of the same step, drawn with `seeds.few_shot`. The pilot execution with k = 3 prior to the correction was superseded.
- **The analysis code's refusals** (2026-11-10). The analysis code and the pre-registration code refuse a choice of B1's k outside today's rule, a replay of the cheap alternative without B1's few-shot, and a pre-registration in which some fact the configuration names (the S4 choice, the centroids, the adapters, the allocation) is missing or not in git. No stored result changes.
- **Failures on the test** (6.1). The rules for resuming and redoing an execution on the test, and the code that may run on it, enter 6.1 with D3's PR and its deviation line; this version does not change 6.1.

**Note of 2026-11-10 on the first line of `prereg/DEVIATIONS.md`.** The line of 2026-11-09 says serving runs at the L40S price on the new provider. The following line, of the same day, changed the reference GPU to the A100 80GB before any cost measurement: the SLM's cost uses the serving provider's price for the A100, and the registered provider's price for the same GPU enters through 6.6's price rule, not the L40S one. Both lines stay as they were written; this note corrects the reading of the first.

| Problem after Stage 0 | Where the correction is in v2.4 |
|---|---|
| The spec sent K1 through the subscription, and it ran through the paid API | section 3, 7.3, D6, D25 and this section |
| The spec did not say how a stopped execution continues | 7.3, D25 and this section (on the test, 6.1) |
| Deduplication and the per-question cap took the column filter's and the agent's training away from the teacher's distribution | S2, D21 and this section |
| B1's k rule, per call over replays, did not see the agent stop before the SQL | D23 and this section |
| The Appendix B verdict had no floor of repair calls, and the teacher hardly repairs | section 5, D17 and this section |
| The spec did not say when an engine refusal is the model's, nor that the parser accepts the tool in JSON | section 2 and this section |
| The header and section 11 did not cite the rewrite of 6.6 | this section |
| The first line of the deviations speaks of the L40S price | this section (note of 2026-11-10) |
| V1's limit did not say that most clusters train below the paper's rule of thumb, nor how that bounds a result in either direction | section 5, 6.1, 8 and this section |

---

## Glossary of the paper's labels

- **WD1, WD2** — definitions of SLM and LLM (section 2.1, p.2)
- **V1–V3** — the three views (section 2.2, p.2)
- **A1–A7** — supporting arguments (section 3, p.3–6)
- **AV1–AV3** — alternative views (section 4, p.6–8)
- **CA1–CA4** — counter-arguments that support the alternative views (section 4, p.7–8)
- **A8–A13** — the paper's rebuttals to the alternative views (section 4, p.7–8)
- **B1–B3** — barriers to adoption (section 5, p.8)
- **S1–S6** — steps of the conversion algorithm (section 6, p.9)
- **D1–D26** — decisions of this project (section 10)
