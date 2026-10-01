"""C2 · config.yaml: the single source of every parameter (roles, call sites, seeds, data pins).

A file may declare `extends: <path>` (relative to itself); it is deep-merged over the file it
extends, with mappings merged key by key and every other value replaced. The identity of a
configuration is the sha256 of its merged, canonical JSON, which is what manifests record.
"""
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from bench.contracts.calls import CALL_SITES

ENDPOINT_KINDS = ("api", "vllm", "llamacpp", "fake")
EMBEDDING_PROVIDERS = ("openai", "local", "fake")
SINGLE_MODEL_ROLES = ("production_llm", "cheap_alt")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
REQUIRED = ("version", "roles", "agent", "call_sites", "retries", "seeds", "splits",
            "thresholds", "prices", "embeddings", "preprocess", "data", "eval", "arms")


class ConfigError(ValueError):
    pass


def _merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _merge(base.get(key), value) if key in base else value
        return merged
    return override


def _load(path: Path, chain: List[Path]) -> Dict[str, Any]:
    path = path.resolve()
    if path in chain:
        raise ConfigError(f"extends cycle: {' -> '.join(map(str, chain + [path]))}")
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _merge(_load(path.parent / parent, chain + [path]), raw)


def load_config(path) -> Dict[str, Any]:
    config = _load(Path(path), [])
    errors = validate_config(config)
    if errors:
        raise ConfigError(f"{path}: " + "; ".join(errors))
    return config


def config_sha256(config: Dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _check_endpoint(where: str, endpoint: Any) -> List[str]:
    if not isinstance(endpoint, dict):
        return [f"{where}.endpoint must be a mapping"]
    errors = []
    if endpoint.get("kind") not in ENDPOINT_KINDS:
        errors.append(f"{where}.endpoint.kind must be one of {ENDPOINT_KINDS}")
    for key in ("base_url", "api_key_env"):
        if endpoint.get(key) is not None and not isinstance(endpoint.get(key), str):
            errors.append(f"{where}.endpoint.{key} must be a string or null")
    headers = endpoint.get("headers_env")  # {header: environment variable holding its value}, e.g. Modal proxy auth
    if headers is not None and not (isinstance(headers, dict) and all(
            isinstance(h, str) and h and isinstance(v, str) and v for h, v in headers.items())):
        errors.append(f"{where}.endpoint.headers_env must map header names to environment variable names")
    return errors


def _check_params(where: str, params: Any) -> List[str]:
    timeout = (params or {}).get("timeout_s")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        return [f"{where}.params.timeout_s must be a number > 0"]
    return []


def _agent_engines(node: Any, where: str = "agent") -> List[str]:
    """Every engine in the CHESS team configuration must be `routed`, with no temperature:
    the router (C4) and `call_sites` are the only authorities, and a stale model name here
    would look like it chooses a model while being ignored."""
    errors = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("engine", "engine_name") and value != "routed":
                errors.append(f"{where}.{key} must be 'routed', not {value!r}")
            elif key == "temperature":
                errors.append(f"{where}.temperature: temperatures live in call_sites")
            else:
                errors += _agent_engines(value, f"{where}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            errors += _agent_engines(value, f"{where}[{i}]")
    return errors


# ---------------------------------------------------------------- the API roles' reasoning and provider pin

PROVIDER_KEYS = ("name", "quantization", "checkpoint", "reserve", "routing")


def _text_or_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and bool(value))


def _routing_errors(where: str, routing: Any) -> List[str]:
    """An aggregator's routing pin: one that could fall back, or drop a parameter, would pin nothing."""
    def names(value):
        return isinstance(value, list) and bool(value) and all(isinstance(v, str) and v for v in value)
    if not isinstance(routing, dict) or set(routing) != {"only", "allow_fallbacks", "require_parameters", "quantizations"}:
        return [f"{where}.routing must be null or {{only, allow_fallbacks, require_parameters, quantizations}}"]
    errors = [f"{where}.routing.{key} must be a non-empty list of names" for key in ("only", "quantizations")
              if not names(routing[key])]
    if routing["allow_fallbacks"] is not False or routing["require_parameters"] is not True:
        errors.append(f"{where}.routing must set allow_fallbacks: false and require_parameters: true")
    return errors


def api_role_errors(config: Dict[str, Any]) -> List[str]:
    """What each API role declares besides its endpoint: its reasoning state (`reasoning.enabled`, asserted by
    the preflight probe; `reasoning.forced_temperature`, the value C1 records when the provider forces one)
    and the provider that serves it (`provider`, whose `name` C1 records)."""
    errors = []
    for role in SINGLE_MODEL_ROLES:
        spec, where = config["roles"][role], f"roles.{role}"
        reasoning = spec.get("reasoning")
        if not isinstance(reasoning, dict) or not isinstance(reasoning.get("enabled"), bool):
            errors.append(f"{where}.reasoning.enabled must be true or false")
        else:
            forced = reasoning.get("forced_temperature")
            if forced is not None and (isinstance(forced, bool) or not isinstance(forced, (int, float)) or forced < 0):
                errors.append(f"{where}.reasoning.forced_temperature must be null or a number >= 0")
        extra_body = (spec.get("params") or {}).get("extra_body")
        if extra_body is not None and not isinstance(extra_body, dict):
            errors.append(f"{where}.params.extra_body must be a mapping")
        provider = spec.get("provider")
        if not isinstance(provider, dict) or set(provider) != set(PROVIDER_KEYS):
            errors.append(f"{where}.provider must have exactly {PROVIDER_KEYS} (null until decided)")
            continue
        errors += [f"{where}.provider.{key} must be a string or null" for key in ("name", "quantization")
                   if not _text_or_null(provider[key])]
        checkpoint = provider["checkpoint"]
        if checkpoint is not None and not (isinstance(checkpoint, dict) and set(checkpoint) == {"repo", "revision"}
                                           and isinstance(checkpoint["repo"], str) and checkpoint["repo"]
                                           and isinstance(checkpoint["revision"], str) and _COMMIT.match(checkpoint["revision"])):
            errors.append(f"{where}.provider.checkpoint must be null or {{repo, revision: a 40-hex commit}}")
        reserve = provider["reserve"]
        if reserve is not None and not (isinstance(reserve, dict) and set(reserve) == {"name", "base_url", "quantization"}
                                        and all(isinstance(reserve[k], str) and reserve[k] for k in ("name", "base_url"))
                                        and _text_or_null(reserve["quantization"])):
            errors.append(f"{where}.provider.reserve must be null or {{name, base_url, quantization}}")
        if provider["routing"] is not None:
            errors += _routing_errors(f"{where}.provider", provider["routing"])
            if isinstance(extra_body, dict) and "provider" in extra_body:  # one place decides the routing
                errors.append(f"{where}.params.extra_body.provider is set by {where}.provider.routing")
    return errors

# ---------------------------------------------------------------- end of the API roles' block


def agent_settings_errors(config: Dict[str, Any]) -> List[str]:
    """The agent's own settings: retries, concurrency, the B1 few-shot, local embeddings (F1), and the API
    roles' reasoning and provider pin."""
    errors = api_role_errors(config)

    def number(value, minimum, integer=False):
        kinds = (int,) if integer else (int, float)
        return isinstance(value, kinds) and not isinstance(value, bool) and value >= minimum
    retries = config["retries"]
    if not number(retries.get("http_max_attempts"), 1, integer=True):
        errors.append("retries.http_max_attempts must be an integer >= 1")
    backoff = retries.get("http_backoff_s")
    if not (isinstance(backoff, dict) and number(backoff.get("base"), 0) and number(backoff.get("max"), 0)):
        errors.append("retries.http_backoff_s needs base and max, numbers >= 0")
    if not number(config["agent"].get("max_workers"), 1, integer=True):
        errors.append("agent.max_workers must be an integer >= 1")
    if "few_shot" not in config["seeds"]:
        errors.append("seeds.few_shot is required")
    b1 = (config.get("arms") or {}).get("B1")
    few_shot = b1.get("few_shot") if isinstance(b1, dict) else None
    few_shot = few_shot if isinstance(few_shot, dict) else {}
    if not number(few_shot.get("k"), 0, integer=True):
        errors.append("arms.B1.few_shot.k must be an integer >= 0")
    if few_shot.get("source_run") is not None and not isinstance(few_shot["source_run"], str):
        errors.append("arms.B1.few_shot.source_run must be a run id or null")
    if config["embeddings"]["provider"] == "local":
        local = config["embeddings"].get("local") or {}
        if not isinstance(local.get("model"), str) or not local["model"]:
            errors.append("embeddings.local.model is required with provider local")
        if not isinstance(local.get("revision"), str) or not _COMMIT.match(local["revision"]):
            errors.append("embeddings.local.revision must be a 40-hex commit")
    return errors


def cost_settings_errors(config: Dict[str, Any]) -> List[str]:
    """What J8 prices the SLM with: the SLO's cap, the container's CPU and memory prices, and the memory
    it is priced at. A key that is given must be a usable number; J8 refuses to run without one."""
    def section(*keys: str) -> Dict[str, Any]:
        node: Any = config
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        return node if isinstance(node, dict) else {}
    errors = []
    for where, key, positive in ((("cost",), "p95_slo_cap_ms", True), (("serving",), "memory_gib", True),
                                 (("modal", "gpu_prices"), "cpu_usd_per_core_s", False),
                                 (("modal", "gpu_prices"), "memory_usd_per_gib_s", False)):
        if key not in section(*where):
            continue
        value = section(*where)[key]
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not number or value < 0 or (positive and value == 0):
            errors.append(f"{'.'.join(where)}.{key} must be a number {'> 0' if positive else '>= 0'}")
    if not isinstance(section("cost").get("slo_from"), (str, type(None))):
        errors.append("cost.slo_from must be a run id or null")
    return errors


def verdict_settings_errors(config: Dict[str, Any]) -> List[str]:
    """Front R's keys: what every verdict reads (design §6.3), fixed before the pilot is read."""
    errors = []

    def number(value, low, high):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high
    thresholds = config["thresholds"]
    for key, high in (("delta_pp", 100), ("selection_delta_pp", 100), ("concordance_min", 1),
                      ("concordance_slack_pp", 100), ("format_tolerance_pp", 100)):
        if not number(thresholds.get(key), 0, high):
            errors.append(f"thresholds.{key} must be a number in [0, {high}]")
    min_calls = (config.get("allocation") or {}).get("min_calls")
    if not (isinstance(min_calls, int) and not isinstance(min_calls, bool) and min_calls >= 1):
        errors.append("allocation.min_calls must be an integer >= 1")
    if not number((config.get("claims") or {}).get("v3_min_ratio"), 1, float("inf")):
        errors.append("claims.v3_min_ratio must be a number >= 1")
    return errors


def validate_config(config: Dict[str, Any]) -> List[str]:
    errors = [f"missing key: {k}" for k in REQUIRED if k not in config]
    if errors:
        return errors
    roles = config["roles"]
    for role in SINGLE_MODEL_ROLES:
        spec = roles.get(role)
        if not isinstance(spec, dict) or not isinstance(spec.get("model"), str):
            errors.append(f"roles.{role}.model must be a string")
            continue
        errors += _check_endpoint(f"roles.{role}", spec.get("endpoint"))
        errors += _check_params(f"roles.{role}", spec.get("params"))
    for i, candidate in enumerate(roles.get("slm_candidates") or []):
        if not isinstance(candidate.get("name"), str) or not isinstance(candidate.get("base"), str):
            errors.append(f"roles.slm_candidates[{i}] needs name and base")
        errors += _check_endpoint(f"roles.slm_candidates[{i}]", candidate.get("endpoint"))
        errors += _check_params(f"roles.slm_candidates[{i}]", candidate.get("params"))
    errors += _agent_engines(config["agent"])
    order, team = config["agent"].get("team_order"), config["agent"].get("team_agents") or {}
    if not isinstance(order, list) or sorted(order) != sorted(team) or len(set(order)) != len(order):
        errors.append("agent.team_order must list every agent of agent.team_agents once, in running order")
    call_sites = config["call_sites"]
    if set(call_sites) != set(CALL_SITES):
        errors.append(f"call_sites must list exactly {sorted(CALL_SITES)}")
    for site, spec in call_sites.items():
        temperature = (spec or {}).get("temperature")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            errors.append(f"call_sites.{site}.temperature must be a number >= 0")
    attempts = config["retries"].get("parse_max_attempts")
    if not isinstance(attempts, int) or attempts < 1:
        errors.append("retries.parse_max_attempts must be an integer >= 1")
    for name, seed in config["seeds"].items():
        if not isinstance(seed, int):
            errors.append(f"seeds.{name} must be an integer")
    if config["embeddings"].get("provider") not in EMBEDDING_PROVIDERS:
        errors.append(f"embeddings.provider must be one of {EMBEDDING_PROVIDERS}")
    for name, pin in config["data"].items():
        if not isinstance(pin, dict) or not pin.get("url") or not pin.get("sha256"):
            errors.append(f"data.{name} needs url and sha256")
    if not errors:  # the agent's settings read the keys checked above
        errors += agent_settings_errors(config)
    errors += verdict_settings_errors(config)
    return errors + data_training_errors(config) + cost_settings_errors(config)


def data_training_errors(config: Dict[str, Any]) -> List[str]:
    """The stratified pilot (`stats.pilot_mix`) and the cap on the training examples
    (`curation.max_per_question_call_site`; its seed, `seeds.curation_sample`, is checked with the seeds),
    each where it is set: a configuration without them is one that neither draws the pilot nor curates.
    That the mix names the difficulties and sums to `stats.pilot_size` is checked where the pilot is
    drawn (bench.data.pilot_ids)."""
    def count(value: Any, minimum: int) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= minimum
    errors = []
    stats, curation = config.get("stats") or {}, config.get("curation") or {}
    if "pilot_mix" in stats:
        mix = stats["pilot_mix"]
        if not (isinstance(mix, dict) and mix and all(count(n, 0) for n in mix.values())):
            errors.append("stats.pilot_mix must map each difficulty to an integer >= 0")
    if "max_per_question_call_site" in curation and not count(curation["max_per_question_call_site"], 1):
        errors.append("curation.max_per_question_call_site must be an integer >= 1")
    return errors


def chess_team_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """The configuration CHESS receives: its team, with the agents in `team_order`."""
    agent = config["agent"]
    return {"setting_name": agent["setting_name"],
            "team_agents": {name: agent["team_agents"][name] for name in agent["team_order"]}}


def engine_spec(config: Dict[str, Any], engine: str) -> Dict[str, Any]:
    """The model and endpoint behind an engine name returned by the router (C4):
    `production_llm`, `cheap_alt`, `slm:<candidate>` (the base, served under the candidate's
    name) or `slm:<candidate>+lora:<served_name>` (an adapter, served under its own name)."""
    if engine in SINGLE_MODEL_ROLES:
        return {"model_role": engine, **config["roles"][engine]}
    if engine.startswith("slm:"):
        name, lora, served_name = engine[len("slm:"):].partition("+lora:")
        if lora and not served_name:
            raise ConfigError(f"engine {engine!r}: empty adapter name")
        for candidate in config["roles"].get("slm_candidates") or []:
            if candidate["name"] == name:
                return {"model_role": "slm", "model": served_name if lora else name,
                        "endpoint": candidate["endpoint"], "params": candidate.get("params") or {}}
        raise ConfigError(f"engine {engine!r}: no slm candidate named {name!r}")
    raise ConfigError(f"unknown engine {engine!r}")


def endpoint_credential_envs(endpoint: Dict[str, Any]) -> List[str]:
    """The environment variables whose values an endpoint sends as credentials: its API key and the value
    of every header in `headers_env`."""
    names = [endpoint.get("api_key_env"), *(endpoint.get("headers_env") or {}).values()]
    return [name for name in names if name]


def credential_envs(config: Dict[str, Any]) -> List[str]:
    """Every credential variable of the configuration: those of each engine's endpoint, and OPENAI_API_KEY
    (the retrieval embeddings'). The one list whatever must know the credentials reads (the redaction of
    the public record, the key-length check), so a credential field added to C2 is added here."""
    roles = config["roles"]
    specs = [roles.get(role) for role in SINGLE_MODEL_ROLES] + list(roles.get("slm_candidates") or [])
    names = {name for spec in specs for name in endpoint_credential_envs((spec or {}).get("endpoint") or {})}
    return sorted(names | {"OPENAI_API_KEY"})
