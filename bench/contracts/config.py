"""C2 · config.yaml: the single source of every parameter (roles, call sites, seeds, data pins).

A file may declare `extends: <path>` (relative to itself); it is deep-merged over the file it
extends, with mappings merged key by key and every other value replaced. The identity of a
configuration is the sha256 of its merged, canonical JSON, which is what manifests record.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from bench.contracts.calls import CALL_SITES

ENDPOINT_KINDS = ("api", "vllm", "llamacpp", "fake")
EMBEDDING_PROVIDERS = ("openai", "local", "fake")
SINGLE_MODEL_ROLES = ("production_llm", "cheap_alt")
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
    for i, candidate in enumerate(roles.get("slm_candidates") or []):
        if not isinstance(candidate.get("name"), str) or not isinstance(candidate.get("base"), str):
            errors.append(f"roles.slm_candidates[{i}] needs name and base")
        errors += _check_endpoint(f"roles.slm_candidates[{i}]", candidate.get("endpoint"))
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
    return errors


def engine_spec(config: Dict[str, Any], engine: str) -> Dict[str, Any]:
    """The model and endpoint behind an engine name returned by the router (C4)."""
    if engine in SINGLE_MODEL_ROLES:
        return {"model_role": engine, **config["roles"][engine]}
    raise ConfigError(f"unknown engine {engine!r}")
