# Patches to CHESS

Upstream: https://github.com/ShayanTalaei/CHESS @ `3d6e835f858d26885d21d4bc0215aeecf855efbe` (Apache-2.0).
The unmodified copy is the commit that added `vendor/chess/`; `git diff <that commit> -- vendor/chess`
shows every change. **This file is the single authority on how the agent under test differs from
the published one**, and the report cites it.

Why patch at all: the benchmark measures which LLM calls of a real agent a small model can take
over. Anything in the agent that hides calls, changes models behind the harness's back, or makes a
run irreproducible would bias that measurement. Each patch below removes one such mechanism and
nothing else. The agent's prompts, tools, order of steps and final-SQL rule are unchanged, except
for patch 1, a correctness fix decided by the author.

## Applied

| # | What | Where | Why |
|---|---|---|---|
| 1 | **SQL generation and repair use the selected schema** (`"complete"` → `"tentative"`), the fix proposed in CHESS issue #34 (decision D13) | `generate_candidate.py`, `revise.py` | In the published code, the schema selector's output never reaches generation (issues #31, #34), so every selector call is inert |
| 2 | **Every LLM call is routed by the harness**: `get_llm_chain` builds no model; each call asks the router (`bench/contracts/router.py`, C4) for an engine, built from `config.yaml` (C2) as an OpenAI-compatible client with **no hidden client retry** (`max_retries=0`). `ENGINE_CONFIGS` is no longer used | `llm/models.py`, `bench/agent/hooks.py` | One authority over which model answers; every attempt visible |
| 3 | **Temperature per call site, from the configuration**, 0 included, never written into shared state | `bench/agent/hooks.py` | Published: `if temperature:` ignores 0 and mutates the global engine dict, leaking between call sites |
| 4 | **No fallback to another model** on empty output: it is a parse failure, retried on the same engine | `llm/models.py`, `bench/agent/hooks.py` | Published: switches to Gemini silently |
| 5 | **Parse retries bounded** (`retries.parse_max_attempts`, 2) and **each attempt logged** with `retry_of` | `bench/agent/hooks.py` | Published: 12 identical retries at temperature 0, unlogged. Non-parser exceptions and HTTP errors are logged and raised, as published |
| 6 | **One C1 line per LLM invocation** (`calls.jsonl`), including **the agents' own choice of the next tool** (call sites `agent_ir`, `agent_ss`, `agent_cg`) and a stable `invocation_key` per invocation (`table.column` for the column filter; `<template>:<i>` for generation; `<revise round>:<i>` for repair; `<agent>:<iteration>` for the agents; `single` otherwise; the suffix `@<n>` when the agent repeats the same invocation within a question). The agent's action is parsed with the agent's own rules (`Agent.parse_action`). CHESS's per-conversation log (`Logger.log_conversation`) is no longer written from the call layer: C1 replaces it; the rest of CHESS's logger is unchanged | `llm/models.py`, `agent.py`, `filter_column.py`, `generate_candidate.py`, `revise.py` | Published: agent calls are not logged, and parallel or repeated calls of one call site cannot be told apart. The generation step name no longer carries the engine name |
| 10 | **Retrieval embeddings from the configuration** (`embeddings.provider`: `openai`, as published, or `fake` for tests), built when used rather than at import; the column-description vector DB lives in `context_vector_db_<provider>/`, and `bench preprocess` stamps each output with the settings it was built with (last, so a stamp means a complete build); a run refuses to start unless the stamps match its configuration. A missing DB raises a harness error, and any failure of the embeddings themselves (a missing or rejected key, the network) is a harness error too; either fails the run | `db_catalog/preprocess.py`, `database_manager.py`, `retrieve_entity.py` | Published: OpenAI embeddings are required at import time |
| 12 | **CHESS's own execution accuracy has no authority**: its evaluation node still runs, unchanged; predictions are taken by the published final-SQL rule (first SQL of the last key of `SQL_meta_infos`, `bench/agent/runner.py:final_sql`) and scored only by `bench eval` | `bench/agent/runner.py`, `bench/evaluate.py` | The internal EX compares against the gold inside the agent run and has known bugs |
| 15 | **The agent's SQL runs on read-only connections** (`file:…?mode=ro`) | `database_utils/execution.py` | Published: read-write connections that commit, so a model-written `DELETE` or `DROP` would change the pinned database for every later question |

Run-time settings made by the harness, not patches: `DB_ROOT_PATH` and `INDEX_SERVER_PORT` are set
before import (this is why patch 11 of the design, a default for `INDEX_SERVER_PORT`, is not
needed), the templates path and the results directory are pointed inside the run, and Chroma
telemetry is off (`bench/agent/runner.py`).

Harness failures inside the agent (routing, configuration, a missing API key, the retrieval
embeddings) cannot be swallowed by CHESS's catch-all handlers: a `HarnessError` records itself in
the run the moment it is created (`bench/agent/hooks.py`), and the run is marked `failed`. The agents
run in the explicit order `config.yaml › agent.team_order`, never in whatever order a mapping has.

## Not yet applied (front F1)

| # | What | Where |
|---|---|---|
| 5b | Retry with backoff on HTTP errors, each attempt logged | `bench/agent/hooks.py` |
| 7 | `eval` of model output → `ast.literal_eval` | `llm/parsers.py:32` |
| 8 | Seeded schema shuffling | `database_utils/schema_generator.py` |
| 9 | Bounded concurrency | `threading_utils.py:39` |
| 10b | `local` embeddings provider | `bench/agent/hooks.py` |
| 13 | The gold SQL never enters the agent's state (today it does, as published; only logging and the evaluation node read it) | `bench/agent/runner.py`, `system_state.py:check_schema_status` |
| 14 | `load_dotenv(override=True)` lets a `.env` override the harness's environment | `database_manager.py`, `db_catalog/preprocess.py`, `preprocess.py` |

## Observed, unchanged (agent behaviour, reported rather than patched)

- `revise` called when no SQL needs fixing builds a thread pool of size 0; CHESS catches the error and keeps the SQL. No LLM call happens.
- The agent's tool list is numbered (`1. extract_keywords`), and a model that answers with the number is rejected by the agent's own parser, which ends that agent's loop.
