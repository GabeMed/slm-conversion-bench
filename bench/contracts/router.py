"""C4 · the router: the single authority over which engine serves each LLM call.

    route(arm, call_site, prompt_messages, config) -> Route(engine, cluster)

The patched CHESS asks this function on every invocation (vendor/chess/PATCHES.md); nothing else
in the code chooses a model. `engine` resolves to a model and endpoint through
`bench.contracts.config.engine_spec`; `cluster` is what `assign` returned, or None, and goes to C1.

| Arm | Rule |
|---|---|
| B0 | always `production_llm` |
| B1 | always `cheap_alt` (its few-shot and cache budget are prompt-side, F1) |
| B3 | always `slm:<choice.slm>`, the base chosen in S4, no adapter |
| B4 | `slm:<adapters.slm>+lora:<served name of the adapter of assign(prompt)>` |
| B5 | `allocation[assign(prompt)]`; `slm` means B4's engine for that cluster; a cluster missing from the allocation stays with `production_llm` (SPEC S6) |

B2 (single call) does not run the agent and never reaches the router. The facts an arm uses are
set in `config.yaml` under `arms` (bench/contracts/facts.py has their shapes).
"""
from typing import Any, Dict, List, NamedTuple, Optional

from bench import paths
from bench.contracts.calls import CALL_SITES
from bench.contracts.clusters import assign
from bench.contracts.facts import FactError, read_fact

AGENT_ARMS = ("B0", "B1", "B3", "B4", "B5")
ARM_FACTS = {"B3": ("choice",), "B4": ("centroids", "adapters"), "B5": ("centroids", "adapters", "allocation")}


class Route(NamedTuple):
    engine: str
    cluster: Optional[str]


def arm_facts(arm: str, config: Dict[str, Any]) -> Dict[str, tuple]:
    """{fact name: (payload, sha256)} for every fact the arm uses, checked for consistency."""
    settings = (config.get("arms") or {}).get(arm) or {}
    facts = {}
    for name in ARM_FACTS.get(arm, ()):
        if not settings.get(name):
            raise FactError(f"arms.{arm}.{name} is not set in the configuration")
        facts[name] = read_fact(str(paths.ROOT / settings[name]))
    if "adapters" in facts and facts["adapters"][0]["centroids"] != facts["centroids"][1]:
        raise FactError(f"arms.{arm}: the adapters were trained on other centroids")
    if "allocation" in facts:
        decided_on = facts["allocation"][0]
        if (decided_on["centroids"], decided_on["adapters"]) != (facts["centroids"][1], facts["adapters"][1]):
            raise FactError(f"arms.{arm}: the allocation was decided on other centroids or adapters")
    return facts


def _lora(adapters: Dict[str, Any], cluster: str) -> str:
    adapter = adapters["adapters"].get(cluster)
    if adapter is None:
        raise FactError(f"no adapter for cluster {cluster}")
    return f"slm:{adapters['slm']}+lora:{adapter['served_name']}"


def route(arm: str, call_site: str, prompt_messages: List[Dict[str, str]], config: Dict[str, Any]) -> Route:
    if call_site not in CALL_SITES:
        raise ValueError(f"unknown call site {call_site!r}")
    if arm not in AGENT_ARMS:
        raise ValueError(f"arm {arm!r} does not run the agent")
    if arm == "B0":
        return Route("production_llm", None)
    if arm == "B1":
        return Route("cheap_alt", None)
    facts = arm_facts(arm, config)
    if arm == "B3":
        return Route(f"slm:{facts['choice'][0]['slm']}", None)
    cluster = assign(prompt_messages, facts["centroids"][0])
    if arm == "B4":
        return Route(_lora(facts["adapters"][0], cluster), cluster)
    engine = facts["allocation"][0]["allocation"].get(cluster, "production_llm")
    return Route(_lora(facts["adapters"][0], cluster) if engine == "slm" else engine, cluster)


def possible_engines(arm: str, config: Dict[str, Any]) -> List[str]:
    """Every engine the arm can route to, so a run can check them all before it starts."""
    if arm not in AGENT_ARMS:
        raise ValueError(f"arm {arm!r} does not run the agent")
    if arm == "B0":
        return ["production_llm"]
    if arm == "B1":
        return ["cheap_alt"]
    facts = arm_facts(arm, config)
    if arm == "B3":
        return [f"slm:{facts['choice'][0]['slm']}"]
    adapters = facts["adapters"][0]
    loras = [_lora(adapters, cluster) for cluster in sorted(facts["centroids"][0]["clusters"])]
    if arm == "B4":
        return loras
    allocation = facts["allocation"][0]["allocation"]
    engines = {"production_llm"} | {e for e in allocation.values() if e != "slm"}
    engines |= {_lora(adapters, c) for c, e in allocation.items() if e == "slm"}
    return sorted(engines)
