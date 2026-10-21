# slm-conversion-bench

An empirical, pre-registered run of the LLM-to-SLM conversion algorithm (steps S1–S6) proposed in
*Small Language Models are the Future of Agentic AI* (Belcak et al., NVIDIA Research,
[arXiv 2506.02153](https://arxiv.org/abs/2506.02153)), on a real open-source agent:
[CHESS](https://github.com/ShayanTalaei/CHESS) answering [BIRD](https://bird-bench.github.io/)
text-to-SQL questions, tested on [Arcwise-Plat-SQL](https://github.com/uiuc-kang-lab/text_to_sql_benchmarks).

The question: which of the agent's LLM calls a specialised small model takes over without losing
quality, against the LLM the agent uses and against the cheapest alternative that needs no training,
and at what cost per correct answer. The protocol is [`SPEC.md`](SPEC.md) (in Portuguese), fixed
before any result.

**Status:** the harness is built and tested against a fake server, a small local model and CPU
training. No paid run has happened yet, so there are no results. [`RUNBOOK.md`](RUNBOOK.md) is the
order of the two days of the experiment.

## Layout

| Path | What |
|---|---|
| `SPEC.md` | the protocol (v2.1); `SPEC-v1.md` is kept as the record of what changed |
| `RUNBOOK.md` | the two days of the experiment, command by command |
| `config.yaml` | C2: every parameter, including the models by role; `configs/` extend it |
| `bench/contracts/` | the contracts between the parts: C1 `calls.jsonl` (one line per LLM call), C2 configuration, C3 agreement between outputs, C4 the router |
| `bench/` | the harness: `data`, `run` and `replay` (agent executions), `curate` and `embed` (training examples and their clusters), `train`, `loadtest`, `eval` (execution accuracy), `prereg`, `verify`, `report`, the test barrier |
| `bench/judge/` | the judgments (J1–J8 and the choice of B1's few-shot k): pure functions over what the executions wrote |
| `modal_apps/` | training and vLLM serving on Modal |
| `vendor/chess/` | CHESS @ `3d6e835`, patched; every change is in [`vendor/chess/PATCHES.md`](vendor/chess/PATCHES.md) |
| `data/MANIFEST.json`, `data/splits.json` | hashes of every input and the train / calibration / test splits |
| `env/` | the three environments: `agent` (Python 3.11, CHESS's pinned LangChain 0.2), `analysis` (the judgments), `train` (training and serving) |

## Running locally (no paid calls)

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

Every execution writes `runs/<run_id>/` with a `manifest.json` (commit, configuration hash, data
hashes), `calls.jsonl` and `predictions.json`. The test split is refused until the
pre-registration hash (`prereg/HASH`) is committed and pushed.

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
`bench train` refuse to turn the teacher's outputs into training data. The authors of this repository grant no such right and take no responsibility for how it is
used with any provider.

SPEC v2.2 (D6, D10) amended the design for this before Stage 0: up to v2.1 the teacher was an open-weight
model whose licence allows training on its outputs, and no closed LLM was in the core.

SPEC v2.3 (D25) runs K1, the teacher's logs on train, through Claude Code on the author's own Claude
subscription (`bench run --via-claude-code`), which Anthropic authorized verbally for this project only.
Calibration and test go through the paid API. Do not run `--via-claude-code` on your own subscription
without the provider's permission for that use.

## Licenses

Code: Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). CHESS is Apache-2.0. BIRD and
Arcwise-Plat-SQL are CC BY-SA 4.0 and are not redistributed here: `bench data` downloads them from
their publishers.
