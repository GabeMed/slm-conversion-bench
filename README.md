<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/hero-dark.svg">
  <img src=".github/assets/hero-light.svg" width="100%" align="top" alt="Nest. Which LLM calls of a real agent can a fine-tuned SLM take over? NVIDIA's LLM-to-SLM conversion, run end to end on a real agent: CHESS answering BIRD text-to-SQL, with the protocol fixed before any result. The picture: the archive room of a lab in the Nest city, where a small white robot pulls an index card from a wall of card catalogs while a large machine sleeps behind it.">
</picture>

<br>
<br>

<p align="center">
  <a href="https://github.com/nest-ai-org/slm-conversion-bench/actions/workflows/ci.yml"><img alt="tests" src="https://img.shields.io/github/actions/workflow/status/nest-ai-org/slm-conversion-bench/ci.yml?branch=main&style=flat-square&label=tests&labelColor=262323"></a>
  <a href="https://arxiv.org/abs/2506.02153"><img alt="arXiv 2506.02153" src="https://img.shields.io/badge/arXiv-2506.02153-506959?style=flat-square&labelColor=262323"></a>
  <a href="prereg/STAGE0"><img alt="pre-registered: Stage 0" src="https://img.shields.io/badge/pre--registered-Stage%200-506959?style=flat-square&labelColor=262323"></a>
  <a href="#results"><img alt="results: reported" src="https://img.shields.io/badge/results-reported-506959?style=flat-square&labelColor=262323"></a>
  <a href="LICENSE"><img alt="license: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-506959?style=flat-square&labelColor=262323"></a>
</p>

## The question

*Small Language Models are the Future of Agentic AI* (Belcak et al., NVIDIA Research,
[arXiv 2506.02153](https://arxiv.org/abs/2506.02153)) argues that most of an agent's calls to a large model could go to
a small, specialised one, and proposes an algorithm (S1–S6) to make the switch. It runs no experiment. This repository
runs that algorithm end to end on a real open-source agent, [CHESS](https://github.com/ShayanTalaei/CHESS), answering
[BIRD](https://bird-bench.github.io/) text-to-SQL questions, tested on
[Arcwise-Plat-SQL](https://github.com/uiuc-kang-lab/text_to_sql_benchmarks), and asks:

- which of the agent's LLM calls a fine-tuned SLM takes over **without losing quality**, against the LLM the agent uses;
- whether that still holds against **the cheapest alternative that needs no training**;
- at what **cost per correct answer**.

The protocol is [`SPEC.md`](SPEC.md). Its verdict rules were fixed at Stage 0, before the pilot was read, and the whole
protocol is registered before any test result; every change after Stage 0 is a line of
[`prereg/DEVIATIONS.md`](prereg/DEVIATIONS.md).

## Status

- ✓ **Harness:** built and tested against a fake server, a small local model and CPU training.
- ✓ **Stage 0:** the configuration and the verdict rules committed and timestamped ([`prereg/STAGE0`](prereg/STAGE0), OpenTimestamps).
- ✓ **Day 1:** the pilot, the teacher's logs, curation, clusters, the SLM's choice, LoRA training.
- ✓ **Day 2:** the load test, the allocation, the pre-registration, the test split (498 questions, every arm), the
  report. What departed from the protocol is in [`prereg/DEVIATIONS.md`](prereg/DEVIATIONS.md).

[`RUNBOOK.md`](RUNBOOK.md) is the order of the two days, command by command.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/how-dark.svg">
  <img src=".github/assets/how-light.svg" width="100%" alt="How it works: S1 log every LLM call (bench run); S2 curate the teacher's logs (bench curate); S3 cluster the calls into tasks (bench embed, j5); S4 pick the small model (bench judge j6); S5 fine-tune one LoRA adapter per cluster (bench train); S6 route each cluster to the cheapest engine not worse than the LLM (bench judge j7). Then measure seven arms on 498 held-out questions.">
</picture>

Seven arms answer the same questions; each is the whole agent with its calls sent to a different engine:

| Arm | The agent's calls go to | It answers |
|---|---|---|
| **B0** | the production LLM, Claude Sonnet 5.5 | the quality ceiling and the reference cost |
| **B1** | a cheaper open-weight generalist, gpt-oss-120b, few-shot | the trivial alternative: a cheaper model, no training |
| **B2** | one call per question, no agent (production and cheap) | is the agent worth it? |
| **B3** | the chosen SLM, zero-shot | what specialising adds (against B4) |
| **B4** | the SLM fine-tuned with LoRA, one adapter per cluster, in every call site | the paper's V1 in its strongest form |
| **B5** | per cluster, the cheapest engine not worse than B0 | the paper's S6 |

The SLM is **Qwen3-8B** ([`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B), revision `b968826`), chosen by
S4's desk triage (J6) from the two candidates, Qwen3-8B and Granite-4.2-8B (Nemotron-3-Nano-4B in reserve). The LoRA adapters (r 16) are
trained with TRL and PEFT on Modal's GPUs (the registered configuration; the reference GPU is an A100 80GB) or on
[Tinker](https://thinkingmachines.ai/tinker/), Thinking Machines' fine-tuning API (`bench train --on tinker`, Qwen3-8B
only), and served with vLLM on Modal or on RunPod (`bench serve --on runpod up`, behind an SSH tunnel). Both backends write the same adapter and a manifest that pairs runs of the same plan, so
a Tinker adapter can be compared against a Modal one ([`RUNBOOK.md`](RUNBOOK.md), step 9).

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/results-dark.svg">
  <img src=".github/assets/results-light.svg" width="100%" alt="Results on 498 test questions, execution accuracy: the agent on Claude Sonnet 5.5 81.7%; the agent on gpt-oss-120b 76.7%; one call to Claude Sonnet 5.5 83.5%; one call to gpt-oss-120b 77.3%; the agent on Qwen3-8B zero-shot 53.2%, fine-tuned 66.7%, and fine-tuned on two of four task clusters with the LLM on the rest 82.5%.">
</picture>

**A fine-tuned 8B model took 95% of the agent's calls with no loss of accuracy, at 81–89% lower cost per correct
answer. Handing it every call lost 15 points.** Small models for the narrow tasks, the large model where it is still
needed.

One agent, one task, the same 498 test questions in every row. The only thing that changes is which model answers
the agent's calls:

| Who answers the agent's calls | Accuracy | Cost per correct answer |
|---|---|---|
| Claude Sonnet 5.5, every call | 81.7% | $0.338 |
| **Fine-tuned Qwen3-8B on the narrow tasks (95% of calls), Claude Sonnet 5.5 on the rest** | **82.5%** | **$0.038 – $0.063** |
| Fine-tuned Qwen3-8B, every call | 66.7% | $0.015 – $0.074 |

The small model's cost runs from a fully used GPU to one used 20% of the time.

Three things a reader should know before quoting this:

- **Full replacement failed at our data volume.** The step that writes the SQL was trained on 826 examples, far below
  the paper's rule of thumb of ten thousand; this says nothing about larger training sets.
- **For this task the agent itself is optional with a large model.** One call to Claude Sonnet 5.5, with no agent,
  scores 83.5% at $0.028 per correct answer, less than the hybrid. The 81–89% saving is against the agent it replaces.
- **Latency depends on sizing.** Within its GPU's capacity the fine-tuned small model answers its tasks as fast as
  the large LLM or faster; in the test one GPU was overloaded and it was slower. Time per question was the same.

<details>
<summary>Every configuration tested, and the registered verdicts</summary>

<br>

The protocol calls the configurations B0 to B5; the report uses those codes.

| Code | Configuration | Accuracy | Cost per correct answer |
|---|---|---|---|
| B0 | the agent, every call on Claude Sonnet 5.5 | 81.7% | $0.338 |
| B1 | the agent, every call on gpt-oss-120b (a cheaper open-weight LLM), few-shot | 76.7% | $0.033 |
| B2 | no agent: one call to Claude Sonnet 5.5 per question | 83.5% | $0.028 |
| B2 | no agent: one call to gpt-oss-120b per question | 77.3% | $0.0006 |
| B3 | the agent, every call on Qwen3-8B without training | 53.2% | $0.009 – $0.045 |
| B4 | the agent, every call on fine-tuned Qwen3-8B | 66.7% | $0.015 – $0.074 |
| B5 | the agent, fine-tuned Qwen3-8B on two of four task clusters (95.5% of calls), Claude Sonnet 5.5 on the rest | 82.5% | $0.038 – $0.063 |

The registered verdicts, for this workload only (margin 5 points, 90% intervals):

- B5 is non-inferior to B0: +0.8 points [−0.8, +2.4]. The small model takes 95.5% of the calls, 93.5% of the tokens
  and 85.8% of the cost at the LLM's price.
- B4 is worse than B0: −15.1 points [−18.1, −12.0]. The paper's claim in its strongest form is refuted here.
  Fine-tuning adds 13.5 points over the same model without training.
- B1 against B0 is inconclusive: −5.0 points [−8.0, −2.2].
- The registered cost claim for the hybrid is refuted: B5 is not cheaper than the cheapest configuration that needs
  no training and is non-inferior to B0, which is the single call to Claude Sonnet 5.5.

One execution per configuration.

</details>

- A two-page summary: [`docs/summary.md`](docs/summary.md).
- The full report, with every test and the SPEC §5 map: [`results/report/report.md`](results/report/report.md). It was
  produced with a small declared patch of the analysis code ([`results/report/analysis-patch.diff`](results/report/analysis-patch.diff)),
  explained in [`prereg/DEVIATIONS.md`](prereg/DEVIATIONS.md); no verdict rule changed.
- **Results, axis by axis** (accuracy, per task, cost of running, cost of building, latency, data volume, limits):
  [`docs/results.md`](docs/results.md).
- Working notes behind it: [`docs/preliminary-results.md`](docs/preliminary-results.md),
  [`docs/preliminary-cost-latency.md`](docs/preliminary-cost-latency.md).
- The data behind every table: [`results/`](results/README.md). Raw call logs, adapters and training sets: the
  [`data-2026-11-12`](https://github.com/nest-ai-org/slm-conversion-bench/releases/tag/data-2026-11-12) release.

The card is drawn from the report's `report.json` and changes no result:

```sh
python scripts/brand/results_card.py results/report --out .github/assets --title "<the finding>"
```

## Run it locally (no paid calls)

<details>
<summary>The agent environment, a small local model, one question</summary>

```sh
uv venv .venv-agent --python 3.11
uv pip install --python .venv-agent/bin/python -r env/agent/requirements.lock -e ".[test]"
source .venv-agent/bin/activate

bench data                                   # download, check hashes, write splits
./scripts/serve-local.sh &                   # a small model on llama.cpp, OpenAI-compatible
bench preprocess --config configs/smoke-local.yaml --db debit_card_specializing
bench run --config configs/smoke-local.yaml --arm B0 --split train --ids 1470
bench eval <run_id printed above>
pytest
```

Every execution writes `runs/<run_id>/` with a `manifest.json` (commit, configuration hash, data hashes),
`calls.jsonl` and `predictions.json`. The test split is refused until the pre-registration hash (`prereg/HASH`) is
committed and pushed.
</details>

## Layout

<details>
<summary>What is where</summary>

| Path | What |
|---|---|
| `SPEC.md` | the protocol (v2.3); `SPEC-v1.md` is kept as the record of what changed |
| `RUNBOOK.md` | the two days of the experiment, command by command |
| `config.yaml` | C2: every parameter, including the models by role; `configs/` extend it |
| `bench/contracts/` | the contracts between the parts: C1 `calls.jsonl` (one line per LLM call), C2 configuration, C3 agreement between outputs, C4 the router |
| `bench/` | the harness: `data`, `run` and `replay` (agent executions), `curate` and `embed` (training examples and their clusters), `train`, `loadtest`, `eval` (execution accuracy), `prereg`, `verify`, `report`, the test barrier |
| `bench/judge/` | the judgments (J1–J8 and the choice of B1's few-shot k): pure functions over what the executions wrote |
| `modal_apps/` | training and vLLM serving on Modal |
| `runpod_apps/` | inference on RunPod: the vLLM server behind an SSH tunnel (`bench serve --on runpod up`), the load test's client and P-4's reference; `configs/runpod.patch.yaml` switches the configuration to it |
| `vendor/chess/` | CHESS @ `3d6e835`, patched; every change is in [`vendor/chess/PATCHES.md`](vendor/chess/PATCHES.md) |
| `data/MANIFEST.json`, `data/splits.json` | hashes of every input and the train / calibration / test splits |
| `env/` | the three environments: `agent` (Python 3.11, CHESS's pinned LangChain 0.2), `analysis` (the judgments), `train` (training and serving) |
| `scripts/brand/` | the README's images in Nest's look: the hero, "How it works", the results card and the brand (standard library only) |
</details>

## Using a closed model as the teacher

The harness can call any engine with an OpenAI-compatible API and, through `endpoint.protocol:
anthropic`, Anthropic's Messages API. The registered configuration ([`config.yaml`](config.yaml)) makes
Claude Sonnet 5.5 the production LLM, the teacher whose answers the small models are trained on;
[`configs/teacher-claude.yaml`](configs/teacher-claude.yaml) swaps in Claude Opus 5.5.

**Whoever runs it is responsible for having the right to do so.** The pipeline does not only score the
teacher: steps S2–S5 fine-tune small models on its outputs. Most providers of closed models restrict, in
their terms, using outputs to train other models, and a subscription to a chat product is usually not
licensed for automated use as an API backend. Before training on any provider's outputs, read that
provider's current terms and obtain written permission where they require it. Record it in
`roles.production_llm.terms`: until it is filled, `bench preflight` fails and `bench curate` and
`bench train` refuse to turn the teacher's outputs into training data. The authors of this repository grant no such
right and take no responsibility for how it is used with any provider.

SPEC v2.2 (D6, D10) amended the design for this before Stage 0: up to v2.1 the teacher was an open-weight
model whose licence allows training on its outputs, and no closed LLM was in the core.

The registered K1, the teacher's logs on train, ran through the paid API, like calibration and test
(SPEC.md, D25). `bench run --via-claude-code` can send K1 through Claude Code instead of the paid API;
calibration and test always go through the API. Whoever runs it needs the provider's permission for that
use.

## Cite

GitHub's *Cite this repository* reads [`CITATION.cff`](CITATION.cff). The paper this repository runs:

```bibtex
@misc{belcak2025slm,
  title         = {Small Language Models are the Future of Agentic AI},
  author        = {Peter Belcak and Greg Heinrich and Shizhe Diao and Yonggan Fu and Xin Dong and
                   Saurav Muralidharan and Yingyan Celine Lin and Pavlo Molchanov},
  year          = {2025},
  eprint        = {2506.02153},
  archivePrefix = {arXiv}
}
```

## Licenses

Code: Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). CHESS is Apache-2.0. BIRD and
Arcwise-Plat-SQL are CC BY-SA 4.0 and are not redistributed here: `bench data` downloads them from
their publishers. Geist and Geist Mono, embedded in the README's images, are OFL-1.1
([`scripts/brand/fonts/OFL.txt`](scripts/brand/fonts/OFL.txt)); the model makers' logos in the results card are their
owners' trademarks, used only to name the models.

<p align="center"><sub>A <a href="https://nest.technology/?utm_source=github&utm_medium=readme&utm_campaign=slm-conversion-bench">Nest</a> research project · <a href="https://nest.technology/benchmarks/?utm_source=github&utm_medium=readme&utm_campaign=slm-conversion-bench">nest.technology/benchmarks</a></sub></p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/footer-dark.svg">
  <img src=".github/assets/footer-light.svg" width="100%" align="top" alt="Pixel art of the Nest city by day: white towers and domes among green hills, a river with sailboats and a bridge, snowy mountains, and an airship towing a NEST banner.">
</picture>
