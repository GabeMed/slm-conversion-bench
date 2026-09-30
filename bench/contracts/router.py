"""C4 · the router: the single authority over which engine serves each LLM call.

    route(arm, call_site, prompt_messages, config) -> engine

The patched CHESS asks this function on every invocation (vendor/chess/PATCHES.md); nothing else
in the code chooses a model. The engine name resolves to a model and endpoint through
`bench.contracts.config.engine_spec`.

| Arm | Rule |
|---|---|
| B0 | always `production_llm` |
| B1 | always `cheap_alt` (F1: with the same few-shot and cache budget) |
| B3 | always the SLM base chosen in S4, no adapter (F1) |
| B4 | the SLM plus the adapter of the cluster `assign(prompt)` returns (F1) |
| B5 | `allocation[assign(prompt)]` (F1) |

B2 (single call) does not run the agent and never reaches the router.
"""
from typing import Any, Dict, List

from bench.contracts.calls import CALL_SITES

AGENT_ARMS = ("B0", "B1", "B3", "B4", "B5")


def route(arm: str, call_site: str, prompt_messages: List[Dict[str, str]],
          config: Dict[str, Any]) -> str:
    if call_site not in CALL_SITES:
        raise ValueError(f"unknown call site {call_site!r}")
    if arm == "B0":
        return "production_llm"
    if arm in AGENT_ARMS:
        raise NotImplementedError(f"arm {arm} is built in F1")
    raise ValueError(f"arm {arm!r} does not run the agent")
