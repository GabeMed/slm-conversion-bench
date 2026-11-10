# SPEC — slm-conversion-bench

**Empirical execution of "Small Language Models are the Future of Agentic AI"**
(Belcak et al., NVIDIA Research, arXiv 2506.02153)

## How to read this document

The paper is a _position paper_: it claims that SLMs are the future of agentic AI, supports the position with arguments and proposes an LLM→SLM conversion algorithm, but contains no experiment. The "case studies" in the appendix are estimates.

This proof of concept runs the paper's algorithm end to end on a real agentic task and turns each argument into a measured quantity, with a single goal: **find out whether it improves, by how much, and which step of the algorithm delivers the gain**.

Each section of this document mirrors a section of the paper. The labels **V** (views), **A** (arguments), **AV** (alternative views), **CA** (counter-arguments), **B** (barriers), **S** (algorithm steps) and **WD** (definitions) are the paper's. The **D** labels are our decisions, listed in Section 9.

Level rule: this spec names no models, providers or libraries. Those go in configuration files and in implementation decisions, made in the development sessions. Here, "the production LLM", "the teacher" and "the SLMs" are roles.

For each item of the paper, the document answers three questions: **what the paper says**, **what we execute** and **what that reproduces**. When an item is not testable in this PoC, that is stated explicitly.

---

## 1. Position (Section 2 of the paper)

### Definitions

**Paper.** WD1: an SLM is a model that fits on a consumer electronic device and performs inference with latency practical for serving one user's agentic requests. WD2: an LLM is a model that is not an SLM. In 2025, the paper considers most models below 10 billion parameters to be SLMs.

**We execute.** We adopt WD1 and WD2. The candidate SLMs are organized in three size bands (up to 1B, 1B to 4B, 7B to 9B), and the question implicit in every measurement is "how far down can one go". An inference run on a consumer machine is optional, to honor WD1 literally (D17).

### The three views

The paper claims that SLMs are (V1) sufficiently powerful, (V2) inherently more suitable and (V3) necessarily more economical for most invocations in agentic systems. It presents the position as a necessary consequence of these three views, and goes as far as calling it a moral "ought". This PoC does not test the value; it tests the empirical premises.

| View                         | What the paper claims                                    | The measure that settles it                                                                                                     |
| ---------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **V1** Sufficiency           | SLMs handle agents' language "errands"                   | SLM quality against the production LLM on the benchmark, zero-shot and after specialization, by difficulty category           |
| **V2** Operational suitability | SLMs are more suitable to agentic systems than LLMs    | Not a direct measure: it is the sum of A3 (adaptation time), A5 (format adherence) and A6 (mixed system)                        |
| **V3** Economy               | SLMs are more economical by virtue of size               | Cost per thousand calls with measured throughput, plus utilization and fixed cost (the points the paper admits leaving out in AV2) |

---

## 2. Arguments (Section 3 of the paper)

### A1 — SLMs are already sufficiently powerful

**Paper.** Cites models of 1 to 9 billion parameters that match much larger models in commonsense reasoning, tool calling, code generation and instruction following. All the evidence comes from third-party benchmarks; none comes from inside an agentic pipeline. The paper names the benchmarks it considers relevant to the model→tool and code→model interface.

**We execute.** The PoC's task is the one A1 itself puts at the center: **tool calling**, measured on the function-calling benchmark the paper cites (BFCL). Since that benchmark's single-turn categories are close to saturation, the difference between SLM and LLM is measured in the live and multi-turn categories, where it still shows (D9). The first data point is the zero-shot of candidates from the three size bands, under the same production prompt used by the LLM.

**Reproduces.** Checks whether A1's claims survive on the benchmark the paper chose, at the difficulty level where there is still something to measure.

**Out.** Commonsense reasoning and code generation not aimed at tools: the paper uses them as indicators, not as agent tasks.

### A2 — SLMs are more economical

**Paper.** Four sub-arguments: (i) inference 10–30× cheaper than an LLM's, citing third parties; (ii) fine-tuning in GPU-hours; (iii) deployment on edge devices; (iv) more efficient use of parameters, because LLMs activate only a fraction of them per input.

**We execute.** We measure (i) and (ii): inference cost from the throughput measured on a reference GPU, and the time and cost of each specialization. Item (iii) is optional (D17).

**Reproduces.** Replaces the cited "10–30×" with a number from the task, and states explicitly which part of A2 goes untested.

**Out.** Item (iv). It requires internal instrumentation of the models and is not a consequence of the pipeline.

### A3 — SLMs are more flexible

**Paper.** Being cheap to train and adapt, they allow multiple experts per routine and fast adaptation to new requirements, "overnight rather than weeks". Cites democratization as a consequence.

**We execute.** We record the time and cost of each trained expert, and the effort needed to switch tasks (what changes when only the task specification changes).

**Reproduces.** Measures the "overnight".

**Out.** Democratization is not testable in a PoC.

### A4 — Agents expose only a narrow slice of the model

**Paper.** An agent is a generalist LLM bound by prompts and context management to a narrow function; an SLM tuned for those prompts would suffice. The paper defers the natural objection (the generalist understands the world better) to AV1.

**We execute.** It is the workload's premise: a realistic production prompt, fixed and versioned before any result, the same for the LLM and for the SLMs. S3 measures the narrowness, counting how many distinct call types actually exist in the logs.

**Reproduces.** A4 is true if a narrow specialization closes the gap. That is exactly the comparison of specialized SLM against LLM under the same prompt.

### A5 — Agentic interactions require format alignment

**Paper.** The agent needs a single tool-call and output format; the LLM that alternates formats makes occasional format errors; an SLM trained on a single format is preferable.

**We execute.** Rate of calls invalid against the tool schema in all arms: production LLM, zero-shot SLM, specialized SLM; each with and without decoding constrained to the schema.

**Reproduces.** Separates the format gain that comes from training from the one that comes from a decoding trick. It is the only point where the paper bets the SLM beats the LLM.

### A6 — Agentic systems are naturally heterogeneous

**Paper.** One model can be another's tool; the natural architecture has an LLM at the root (conversation and orchestration) and SLMs in the subordinate calls; models of different sizes for different complexities.

**We execute.** This is where the PoC's roles are defined:

- **The production LLM**: the model the client uses today. It is the quality ceiling and the cost reference. It only evaluates; none of its outputs becomes a training target.
- **The teacher**: the model whose calls are logged and become specialization data. It needs a license that allows that use.
- **The SLMs**: the candidates from the three bands.

A6's mixed system is the SLM serving by default and the production LLM receiving the calls on which the SLM fails. We sweep the fallback rate from 0% to 100% and trace the quality×cost Pareto frontier.

**Reproduces.** Tests the paper's central architectural recommendation: does the mixed system dominate both pure systems?

**Note.** The quality difference between the teacher and the production LLM is reported as **compliance cost**: the price, in quality, of using as data source a model whose license allows training.

### A7 — Agentic interactions are a data source

**Paper.** A "listener" at the call interface generates specialized instruction data; the data can be post-filtered by the overall success of the workflow; producing expert SLMs is a natural step of deployment, not an auxiliary effort.

**We execute.** The S1 logs, the "real logs against synthetic data" ablation (S1) and the S2 filter.

**Reproduces.** It is the data-moat question: if synthetic data generated only from the task description ties with real logs, the client does not need to hand over logs.

---

## 3. Alternative views (Section 4 of the paper)

### AV1 — The generalist always wins on the same task

**Paper.** Supported by CA1 (the scaling laws: larger models win on any language task) and CA2 (the "semantic hub": LLMs abstract and integrate meaning in a way small models cannot). The paper answers with A8 (the scaling laws assume constant architecture; SLMs benefit from their own architectures), A9 (the SLM's flexibility allows tuning it up to the desired reliability), A10 (inference-time compute is cheap for SLMs) and A11 (agents decompose problems into sub-tasks too simple for the hub to matter).

**We execute.**

- **A9** is the S5 learning curve: does specialization reach the production LLM's level, and with how many examples?
- **A11** is the gap by the benchmark's difficulty category. If the difference between SLM and LLM concentrates in the multi-turn categories, the hub argument has force; if it is uniform, it does not.
- **A10** is an optional arm (D15): the SLM with reasoning or self-consistency at inference, measured also in latency and cost, because that is where A10 charges its price.

**Reproduces.** The paper rebuts AV1 with arguments; we rebut it, or confirm it, with the measured gap.

**Out.** **A8** requires comparing SLMs of different architectures at similar size; it is a separate experiment. A cheap data point is possible if the candidate list includes a hybrid-architecture model (D16).

### AV2 — Centralization makes the LLM cheaper

**Paper.** Supported by CA3 (it is hard to keep a specialist endpoint well utilized and balanced) and CA4 (infrastructure and talent cost to operate it, omitted from the calculations). The paper acknowledges AV2 as valid and says that "the jury is still out", pointing to A12 (advances in inference scheduling) and A13 (falling infra cost).

**We execute.** The cost model of Section 7 answers CA3 and CA4 directly: cost recomputed at 20%, 50% and 100% utilization; fixed engineering and retraining cost included in the payback; payback at three reference volumes.

**Reproduces.** It is the paper's weakest point, by its own admission. The PoC produces the first numbers there.

**Out.** A12 and A13 are market trends, not testable here. Our cost uses rented infrastructure, which amounts to assuming zero setup cost; this is recorded.

### AV3 — Equally possible worlds

**Paper.** Both worlds are possible, but inertia and the investment already made favor the LLM world. The paper acknowledges the possibility.

**We execute.** Not testable. The effort log (Section 7) is the only data produced on the cost of switching worlds.

---

## 4. Barriers (Section 5 of the paper)

**B1 — Investment in centralized infrastructure.** Not testable.

**B2 — SLMs are designed and evaluated with generalist benchmarks.** The paper says that, looking only at agentic utility, SLMs beat larger models. We execute: the S4 ablation, selection by public benchmark scores against selection by the logs themselves. Reproduces: tests whether the generalist ranking predicts performance on the task.

**B3 — Lack of public attention.** Not testable. Publishing the results is the answer the paper asks for.

---

## 5. Conversion algorithm (Section 6 of the paper)

The paper's algorithm presupposes two things it does not state: that there is an agent in production generating calls, and that there is a way to evaluate and decide. The paper does not define evaluation, acceptance criterion or the "router" that first appears in S6. Section 7 of this document covers what is missing.

**The production agent.** It is a workload: a set of tool-calling calls, with a fixed production prompt, that the teacher answers. The inputs come from an open function-calling dataset, distinct from the benchmark, whose labels are not used (D13). The benchmark stays untouched as the test set, so that the SLM and LLM numbers are comparable to the published ones.

**Ablation design.** The complete pipeline is the main arm. For each step there is an arm that replaces it with what a team would do without the paper, keeping everything else equal; the difference between the two is the step's contribution. The form is "remove one at a time", not cumulative, because the steps interact (S5 depends on S1). Each arm reports the quality difference, the cost difference and the effort to execute the step.

### S1 — Log collection

**Paper.** Instrument all of the agent's non-HCI calls, capturing input prompt, response, tool-call content and, optionally, latency; encrypted pipeline, with access control and anonymization.

**We execute.** A logger around the teacher's calls on the workload's training inputs. Each record keeps structured input, output, latency, tokens and the model's identification and version, in a provider-neutral schema (D7). The teacher's reasoning, when exposed, stays out of the log (D5): it is what the paper describes and it is what a client with a closed model would have.

**Ablation.** Real logs against synthetic data generated only from the task description.

**Reproduces.** A7.

### S2 — Curation and filtering

**Paper.** Accumulate 10k–100k examples ("rule of thumb" for fine-tuning small models); remove PII, PHI and sensitive application data; paraphrase specific inputs to obfuscate entities.

**We execute.** Masking of sensitive data is executed as a real step, even with near-zero effect on benchmark data; the effect is recorded. Filter by call valid against the schema (the only "success" signal available in production; using the gold answers would be cheating). Deduplication. Training and validation split by scenario. The S5 data curve (500 / 2k / 5k / all) tests the 10k floor.

**Ablation.** Raw logs against curated logs.

**Reproduces.** Says whether curation matters and whether the paper's data floor is real.

### S3 — Task clustering

**Paper.** Unsupervised clustering over the agent's prompts and actions to identify recurring patterns, which define the candidate tasks for specialization.

**We execute.** Embedding of the whole prompts (what a client would have), number of clusters chosen automatically (the client does not know how many tasks it has), and validation by purity against the benchmark's categories, which are recorded in the logs but hidden from the step. Two pipelines run: the "pure" one, which uses the discovered clusters, and the "oracle" one, which uses the real categories.

**Ablation.** A single SLM trained on all the logs mixed, without clustering; and the pure–oracle difference.

**Reproduces.** First test of whether clustering recovers the task structure and of how much its errors cost further down the line.

### S4 — SLM selection

**Paper.** For each task, choose candidates by inherent capabilities, performance on relevant benchmarks, license and deployment footprint.

**We execute.** Candidates in the three size bands, with license, context and footprint criteria (D12). Three selection arms: (a) the most popular model of the band, without evaluation; (b) selection by public benchmark scores, which is what the paper prescribes; (c) selection by evaluation on the logs themselves. One candidate per band advances to S5, to reveal the size×quality curve after specialization.

**Ablation.** (a) against (b) against (c).

**Reproduces.** Measures whether the selector changes the result, and tests B2.

### S5 — Specialization

**Paper.** Prepare a dataset per task from the curated data and fine-tune the SLM; parameter-efficient adaptation to reduce cost; full fine-tuning if resources allow; distillation from the LLM when useful.

**We execute.** Parameter-efficient adaptation per cluster on the selected SLM, with a data curve (500 / 2k / 5k / all) and a format arm (with and without constrained decoding). Logit distillation only if teacher and student share the same tokenizer (D10). The ablation runs on the best SLM and at a single data size; the complete curve stays only in the main arm.

**Ablation.** The selected SLM zero-shot against the same SLM specialized.

**Reproduces.** It is where V1 and A9 are decided. It is probably the step with the largest gain; the PoC exists to confirm or deny that.

### S6 — Iteration and refinement

**Paper.** Periodically retrain the SLMs and the router model with new data, going back to S2 or S4 as the case may be.

**We execute.** We define what the paper only names. The **router** is the fallback: the SLM serves; if the call comes out invalid against the schema, the turn goes to the production LLM. The fallback rate is swept from 0% to 100%. **Iteration** is simulated: one benchmark category is left out of training as "new distribution"; we measure the drop, then 200 and 500 logs of it enter and we measure the recovery.

**Ablation.** SLM alone against SLM with fallback.

**Reproduces.** The loop the paper only names, and A6's mixed system.

---

## 6. Case studies (Appendix B of the paper)

**Paper.** Estimates, without experiment, that 60% of MetaGPT's calls, 40% of Open Operator's and 70% of Cradle's could be served by specialized SLMs. The pattern is the same in all three: the SLM serves for routine code, templated responses, parsing and routing of commands and repetitive flows; the LLM keeps architectural reasoning, adaptive planning and unstructured error resolution.

**We execute.** The measured analogue of these estimates: the **replaceable fraction**, defined as the largest fraction of calls served by the SLM, without fallback, such that the mixed system's quality matches the production LLM's within the confidence interval. It comes out of the S6 fallback sweep.

**Reproduces.** The PoC's headline number, comparable to the appendix's estimates.

---

## 7. Measurement protocol

This is the only section with no counterpart in the paper, because the paper has no evaluation. Everything here is fixed **before** any result. What is fixed is how to measure; the effect size is not fixed (D3): there are no "good enough" thresholds, and every result is reported as a curve or frontier.

**Reference truth.** The benchmark's gold answers, as it defines them (structural comparison or execution). Fidelity to the teacher is reported separately: imitating the teacher is not being right.

**Metrics, for each arm and each difficulty category:**

- accuracy, as the benchmark defines it;
- rate of calls invalid against the schema;
- cost per thousand calls;
- p50 and p95 latency.

**Splits and contamination.** The training logs come from inputs distinct from the benchmark. The benchmark does not enter training in any arm. The benchmark's live categories, created to resist contamination, are treated with that care.

**Noise.** Each configuration runs with enough repetitions for a confidence interval; differences within the interval are reported as a tie.

**Cost model.**

- Model via API: public price per token.
- Local model: hourly cost of the reference GPU divided by the measured throughput.
- Utilization: cost recomputed at 20%, 50% and 100% (CA3).
- Fixed cost: engineering hours and periodic retrainings enter the payback (CA4).
- Payback at three reference volumes: small, medium and large (D18).

**Effort log, per algorithm step.** Human hours; GPU hours; API spend; what was done automatically and what required human judgment; what would need to change for a second task. This log separates product (what runs on its own when the task is switched) from service (what requires people each time).

**Deliverables.**

1. The main chart: benchmark accuracy against cost per thousand calls, with the production LLM, the zero-shot SLM, the specialized SLM and the mixed system's curve.
2. The ablation table: contribution and effort of each step S1–S6.
3. The measured replaceable fraction (Section 6).
4. The effort log.
5. The report, with the analysis of the SLM's errors.

**The three possible results, all valid.**

- The SLM matches the LLM at a fraction of the cost: V1–V3 confirmed for tool calling.
- The gap closes only with specialization and persists in the multi-turn categories: the position holds in the narrow form the paper defends (A6, A11).
- The SLM does not get there: the position does not hold on this task, and the reason is documented.

---

## 8. Out of scope

These remain the paper's opinion, untested in this PoC:

| Item                                 | Reason                                                                                                          |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| A2 (iv), efficient use of parameters | Requires internal instrumentation of the models                                                                 |
| A3, democratization                  | Not testable in a PoC                                                                                           |
| A8, architecture by size             | Separate experiment; only an optional data point (D16)                                                          |
| A10, inference-time compute          | Optional (D15), because of the cost in latency                                                                  |
| A12, A13                             | Market trends                                                                                                   |
| AV3                                  | Not testable; the effort log is the only data                                                                   |
| B1, B3                               | Not testable                                                                                                    |
| Edge as default                      | Optional (D17)                                                                                                  |
| Vision-language models               | The paper mentions them only as an extension                                                                    |
| Environmental claims                 | No instrumentation to measure them                                                                              |
| Domain, language and long context    | The task is tool calling in English; a "second client" in another domain is future work, with the same harness  |
| Production                           | It is a benchmark, not a system at client scale                                                                 |

---

## 9. Rules and decision log

### Rules

- Public data only. No third-party data or material enters the project.
- The test benchmark does not enter training in any arm.
- Outputs of the production LLM never become a training target. The teacher is always a model whose license or terms allow that use.
- The project is public, is born before any contract and has a single author. The final texts (report, memo) are written by the author; the development sessions deliver structure, code and review.
- This spec names no models, providers or libraries; those go in configuration.

### Fixed decisions

| #   | Decision                                                                                           |
| --- | -------------------------------------------------------------------------------------------------- |
| D1  | Single task: tool calling, measured on BFCL                                                        |
| D2  | One workload, one benchmark; no multi-task suite                                                   |
| D3  | No thresholds; protocol fixed beforehand, effect size not                                          |
| D4  | Three abstract roles (production LLM, teacher, SLMs), chosen in configuration                      |
| D5  | Teacher's reasoning out of the logs                                                                |
| D6  | Production LLM closed and evaluation only; teacher with a license that allows training             |
| D7  | Provider-neutral log schema, with the model version in each record                                 |
| D8  | "Remove one at a time" ablation over the complete pipeline; on the best SLM, at a single data size |
| D9  | Difficulty measured in the benchmark's live and multi-turn categories                              |

### Open decisions

| #   | Decision                                         | What decides it                                                                             |
| --- | ------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| D10 | The concrete teacher                             | License, position on the benchmark's leaderboard, tokenizer shared with some candidate      |
| D11 | The concrete production LLM                      | What clients actually use                                                                   |
| D12 | The list of SLM candidates                       | License, context, footprint, three bands; recheck versions on the day                       |
| D13 | The input dataset for the logs                   | License and distance from the benchmark                                                     |
| D14 | The reference GPU                                | Represent realistic inference, not the most expensive                                       |
| D15 | Include the A10 arm                              | Time available                                                                              |
| D16 | Include a hybrid-architecture candidate (A8)     | Availability in the band                                                                    |
| D17 | Run on a consumer machine (WD1, A2 iii)          | Time available                                                                              |
| D18 | The three reference volumes of the payback       | Plausible client scenarios                                                                  |

### Schedule

Three phases in about two weeks:

1. **Workload, logs, curation and zero-shot** — S1, S2 and the first reading of A1 with three candidates.
2. **Clustering, selection and specialization** — S3, S4, S5 with all candidates, curves and measurement of cost and latency.
3. **Router, ablation and report** — S6, the ablation arms, the complete cost model, the effort log and the report.

### Glossary of the paper's labels

- **WD1, WD2** — definitions of SLM and LLM (Section 2.1)
- **V1–V3** — the three views (Section 2.2)
- **A1–A7** — supporting arguments (Section 3)
- **AV1–AV3** — alternative views (Section 4)
- **CA1–CA4** — counter-arguments that support the alternative views (Section 4)
- **A8–A13** — the paper's rebuttals to the alternative views (Section 4)
- **B1–B3** — barriers to adoption (Section 5)
- **S1–S6** — steps of the conversion algorithm (Section 6)
- **D1–D18** — decisions of this project (this section)
