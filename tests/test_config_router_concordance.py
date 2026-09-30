"""C2 (configuration), C4 (router, facts, assign) and C3 (agreement)."""
import copy

import pytest
import yaml

from bench import paths
from bench.contracts import clusters, facts, router
from bench.contracts.calls import CALL_SITES
from bench.contracts.concordance import agree
from bench.contracts.config import ConfigError, config_sha256, engine_spec, load_config, validate_config
from bench.contracts.router import Route, possible_engines, route

BASE = paths.ROOT / "config.yaml"
SMOKE = paths.ROOT / "configs" / "smoke-local.yaml"
MSG = [{"role": "user", "content": "q"}]


# ---------------------------------------------------------------- C2

def test_shipped_configs_are_valid():
    base, smoke = load_config(BASE), load_config(SMOKE)
    assert smoke["roles"]["production_llm"]["endpoint"]["kind"] == "llamacpp"
    assert smoke["data"] == base["data"]  # inherited, never duplicated
    assert smoke["roles"]["slm_candidates"] == base["roles"]["slm_candidates"]
    assert config_sha256(base) != config_sha256(smoke)
    assert config_sha256(base) == config_sha256(load_config(BASE))


def test_extends_merges_mappings_and_replaces_the_rest(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(yaml.safe_load(BASE.read_text())))
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nsplits: {calib_size: 10}\n")
    child = load_config(tmp_path / "child.yaml")
    assert child["splits"] == {"calib_size": 10, "excluded": ["119", "120"]}


def test_extends_cycle(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ConfigError, match="cycle"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("mutate", [
    lambda c: c["call_sites"].pop("revise"),
    lambda c: c["call_sites"]["revise"].update(temperature=-1),
    lambda c: c["embeddings"].update(provider="cohere"),
    lambda c: c["roles"]["production_llm"]["endpoint"].update(kind="ollama"),
    lambda c: c["roles"]["production_llm"]["params"].pop("timeout_s"),
    lambda c: c["retries"].update(parse_max_attempts=0),
    lambda c: c["data"]["plat_sql_test"].pop("sha256"),
    lambda c: c.pop("seeds"),
    # a model name in the CHESS team config would look authoritative while being ignored
    lambda c: c["agent"]["team_agents"]["schema_selector"].update(engine="gpt-4o"),
    lambda c: c["agent"]["team_agents"]["schema_selector"]["tools"]["filter_column"]["engine_config"].update(temperature=0.5),
])
def test_invalid_configs(mutate):
    config = copy.deepcopy(load_config(BASE))
    mutate(config)
    assert validate_config(config)


def test_engine_spec_resolves_every_engine_name():
    config = load_config(BASE)
    assert engine_spec(config, "production_llm")["model"] == config["roles"]["production_llm"]["model"]
    base = engine_spec(config, "slm:qwen3-8b")
    assert (base["model_role"], base["model"]) == ("slm", "qwen3-8b")
    assert engine_spec(config, "slm:qwen3-8b+lora:c3-ab12")["model"] == "c3-ab12"
    with pytest.raises(ConfigError):
        engine_spec(config, "slm:unknown")
    with pytest.raises(ConfigError):
        engine_spec(config, "gpt-4o")


# ---------------------------------------------------------------- facts and C4

@pytest.fixture
def fact_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    facts.read_fact.cache_clear()
    yield tmp_path
    facts.read_fact.cache_clear()


def _write(root, judgment, name, payload):
    return str(facts.write_fact(judgment, name, payload, root=root).relative_to(root))


@pytest.fixture
def arms(fact_root, monkeypatch):
    """Facts for B3–B5: two clusters; assign() sends prompts that mention 'filter' to c1."""
    centroids = {"embedding": {"model": "m", "revision": "r"}, "clusters": {"c0": [1.0, 0.0], "c1": [0.0, 1.0]}}
    c_path = _write(fact_root, "J5", "centroids", centroids)
    c_sha = facts.read_fact(str(fact_root / c_path))[1]
    adapters = {"slm": "qwen3-8b", "centroids": c_sha,
                "adapters": {"c0": {"served_name": "c0-aa", "sha256": "x"}, "c1": {"served_name": "c1-bb", "sha256": "y"}}}
    a_path = _write(fact_root, "S5", "adapters", adapters)
    a_sha = facts.read_fact(str(fact_root / a_path))[1]
    allocation = {"centroids": c_sha, "adapters": a_sha, "allocation": {"c0": "cheap_alt", "c1": "slm"}}
    config = copy.deepcopy(load_config(BASE))
    config["arms"] = {
        "B3": {"choice": _write(fact_root, "J6", "choice", {"slm": "granite-4.2-8b"})},
        "B4": {"centroids": c_path, "adapters": a_path},
        "B5": {"centroids": c_path, "adapters": a_path, "allocation": _write(fact_root, "J7", "allocation", allocation)},
    }
    monkeypatch.setattr(clusters, "embed",
                        lambda texts, model, revision: [[0.0, 1.0] if "filter" in t else [1.0, 0.0] for t in texts])
    return config


def test_routes_of_every_arm(arms):
    filter_prompt = [{"role": "user", "content": "filter this column"}]
    for site in CALL_SITES:
        assert route("B0", site, MSG, arms) == Route("production_llm", None)
        assert route("B1", site, MSG, arms) == Route("cheap_alt", None)
    assert route("B3", "filter_column", MSG, arms) == Route("slm:granite-4.2-8b", None)
    assert route("B4", "filter_column", filter_prompt, arms) == Route("slm:qwen3-8b+lora:c1-bb", "c1")
    assert route("B4", "select_tables", MSG, arms) == Route("slm:qwen3-8b+lora:c0-aa", "c0")
    assert route("B5", "filter_column", filter_prompt, arms) == Route("slm:qwen3-8b+lora:c1-bb", "c1")
    assert route("B5", "select_tables", MSG, arms) == Route("cheap_alt", "c0")
    assert possible_engines("B5", arms) == ["cheap_alt", "production_llm", "slm:qwen3-8b+lora:c1-bb"]
    assert possible_engines("B4", arms) == ["slm:qwen3-8b+lora:c0-aa", "slm:qwen3-8b+lora:c1-bb"]


def test_a_cluster_without_allocation_stays_with_the_production_llm(arms, fact_root):
    payload, _ = facts.read_fact(str(fact_root / arms["arms"]["B5"]["allocation"]))
    arms["arms"]["B5"]["allocation"] = _write(fact_root, "J7", "allocation", {**payload, "allocation": {"c1": "slm"}})
    assert route("B5", "select_tables", MSG, arms) == Route("production_llm", "c0")


def test_facts_must_be_set_consistent_and_untampered(arms, fact_root):
    with pytest.raises(facts.FactError, match="not set"):
        route("B3", "filter_column", MSG, {**arms, "arms": {}})
    other = _write(fact_root, "J5", "centroids", {"embedding": {"model": "m", "revision": "r2"}, "clusters": {"c0": [1.0]}})
    with pytest.raises(facts.FactError, match="other centroids"):
        route("B4", "filter_column", MSG, {**arms, "arms": {"B4": {**arms["arms"]["B4"], "centroids": other}}})
    path = fact_root / arms["arms"]["B3"]["choice"]
    path.write_text('{"slm": "qwen3-8b"}')
    facts.read_fact.cache_clear()
    with pytest.raises(facts.FactError, match="not its directory"):
        route("B3", "filter_column", MSG, arms)


def test_router_refuses_unknown_arms_and_call_sites():
    config = load_config(BASE)
    with pytest.raises(ValueError):
        route("B2", "filter_column", MSG, config)
    with pytest.raises(ValueError):
        route("B0", "unit_test", MSG, config)


def test_fact_shapes_are_checked(fact_root):
    with pytest.raises(facts.FactError):
        facts.write_fact("J7", "allocation", {"centroids": "a", "adapters": "b", "allocation": {"c0": "gpt"}}, root=fact_root)
    with pytest.raises(facts.FactError):
        facts.write_fact("J5", "centroids", {"embedding": {"model": "m", "revision": "r"}, "clusters": {"a": [1.0], "b": [1.0, 2.0]}}, root=fact_root)
    with pytest.raises(facts.FactError):
        facts.write_fact("J6", "unknown", {}, root=fact_root)


def test_assign_uses_the_prompt_only_and_breaks_ties_by_id(monkeypatch):
    assert clusters.prompt_text([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]) == "user: a\n\nassistant: b"
    assert clusters.nearest([0.6, 0.8], {"b": [0.0, 1.0], "a": [1.0, 0.0]}) == "b"
    assert clusters.nearest([1.0, 1.0], {"b": [1.0, 0.0], "a": [0.0, 1.0]}) == "a"
    seen = []
    monkeypatch.setattr(clusters, "embed", lambda texts, model, revision: seen.extend(texts) or [[1.0, 0.0]])
    clusters.assign(MSG, {"embedding": {"model": "m", "revision": "r"}, "clusters": {"c0": [1.0, 0.0]}})
    assert seen == ["user: q"]


# ---------------------------------------------------------------- C3

def test_agreement_follows_the_decision_chess_takes():
    assert agree("extract_keywords", ["a", "b"], ["b", "a"])
    assert not agree("extract_keywords", ["a"], ["a", "b"])
    yes, no = {"is_column_information_relevant": "Yes"}, {"is_column_information_relevant": "No"}
    assert agree("filter_column", yes, {"is_column_information_relevant": "yes"})
    assert not agree("filter_column", yes, {"is_column_information_relevant": " yes"})  # CHESS drops " yes"
    assert agree("filter_column", no, {"is_column_information_relevant": "Not relevant"})  # both dropped
    assert agree("select_tables", {"table_names": ["frpm", "schools"]}, {"table_names": ["schools", "frpm"]})
    assert not agree("select_tables", {"table_names": ["Schools"]}, {"table_names": ["schools"]})  # raw names
    assert agree("select_columns",
                 {"chain_of_thought_reasoning": "x", "schools": ["`County`", "cds"]},
                 {"chain_of_thought_reasoning": "y", "`schools`": ["cds", "County"]})
    assert not agree("select_columns", {"schools": ["cds"]}, {"schools": ["cds", "County"]})
    assert agree("agent_ss", {"tool": "select_tables"}, {"tool": "select_tables"})
    assert not agree("agent_ss", {"tool": "select_tables"}, {"done": True})


def test_unparsed_never_agrees_and_gold_sites_refuse():
    assert not agree("extract_keywords", None, None)
    for site in ("generate_candidate", "revise"):
        with pytest.raises(ValueError):
            agree(site, {"SQL": "x"}, {"SQL": "x"})
