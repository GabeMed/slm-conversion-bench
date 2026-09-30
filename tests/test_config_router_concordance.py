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
    lambda c: c["agent"].pop("team_order"),
    lambda c: c["agent"].update(team_order=["schema_selector", "candidate_generator"]),
])
def test_invalid_configs(mutate):
    config = copy.deepcopy(load_config(BASE))
    mutate(config)
    assert validate_config(config)


def test_chess_receives_the_agents_in_team_order_whatever_the_mapping_order():
    from bench.contracts.config import chess_team_config
    config = copy.deepcopy(load_config(BASE))
    config["agent"]["team_agents"] = dict(sorted(config["agent"]["team_agents"].items()))  # as a sorting serializer leaves it
    assert list(chess_team_config(config)["team_agents"]) == ["information_retriever", "schema_selector", "candidate_generator"]
    reordered = copy.deepcopy(config)
    reordered["agent"]["team_order"] = ["schema_selector", "information_retriever", "candidate_generator"]
    assert config_sha256(reordered) != config_sha256(config)  # the order is part of the identity


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

EMBEDDING = {"model": "m", "revision": "a" * 40, "max_seq_length": 512, "truncation": "tail", "text": "prompt"}


@pytest.fixture
def fact_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    facts.read_fact.cache_clear()
    yield tmp_path
    facts.read_fact.cache_clear()


def _write(judgment, name, payload):
    path = facts.write_fact(judgment, name, payload)  # the default root follows paths.ROOT
    return str(path.relative_to(paths.ROOT)), facts.read_fact(str(path), name)[1]


@pytest.fixture
def arms(fact_root, monkeypatch):
    """Facts for B3–B5: two clusters; assign() sends prompts that mention 'filter' to c1."""
    choice_path, choice_sha = _write("J6", "choice", {"slm": "qwen3-8b"})
    c_path, c_sha = _write("J5", "centroids", {"embedding": EMBEDDING, "clusters": {"c0": [1.0, 0.0], "c1": [0.0, 1.0]}})
    adapters = {"slm": "qwen3-8b", "choice": choice_sha, "centroids": c_sha,
                "adapters": {"c0": {"served_name": "c0-aa", "sha256": "1" * 64}, "c1": {"served_name": "c1-bb", "sha256": "2" * 64}}}
    a_path, a_sha = _write("S5", "adapters", adapters)
    alloc_path, _ = _write("J7", "allocation", {"centroids": c_sha, "adapters": a_sha, "allocation": {"c0": "cheap_alt", "c1": "slm"}})
    config = copy.deepcopy(load_config(BASE))
    config["arms"] = {
        "B3": {"choice": choice_path},
        "B4": {"choice": choice_path, "centroids": c_path, "adapters": a_path},
        "B5": {"choice": choice_path, "centroids": c_path, "adapters": a_path, "allocation": alloc_path},
    }
    monkeypatch.setattr(clusters, "embed",
                        lambda texts, embedding: [[0.0, 1.0] if "filter" in t else [1.0, 0.0] for t in texts])
    return config, adapters


def test_routes_of_every_arm(arms):
    config, _ = arms
    filter_prompt = [{"role": "user", "content": "filter this column"}]
    for site in CALL_SITES:
        assert route("B0", site, MSG, config) == Route("production_llm", None)
        assert route("B1", site, MSG, config) == Route("cheap_alt", None)
    assert route("B3", "filter_column", MSG, config) == Route("slm:qwen3-8b", None)
    assert route("B4", "filter_column", filter_prompt, config) == Route("slm:qwen3-8b+lora:c1-bb", "c1")
    assert route("B4", "select_tables", MSG, config) == Route("slm:qwen3-8b+lora:c0-aa", "c0")
    assert route("B5", "filter_column", filter_prompt, config) == Route("slm:qwen3-8b+lora:c1-bb", "c1")
    assert route("B5", "select_tables", MSG, config) == Route("cheap_alt", "c0")
    assert possible_engines("B5", config) == ["cheap_alt", "production_llm", "slm:qwen3-8b+lora:c1-bb"]
    assert possible_engines("B4", config) == ["slm:qwen3-8b+lora:c0-aa", "slm:qwen3-8b+lora:c1-bb"]


def test_a_cluster_without_allocation_stays_with_the_production_llm(arms, fact_root):
    config, _ = arms
    payload, _ = facts.read_fact(str(fact_root / config["arms"]["B5"]["allocation"]), "allocation")
    config["arms"]["B5"]["allocation"] = _write("J7", "allocation", {**payload, "allocation": {"c1": "slm"}})[0]
    assert route("B5", "select_tables", MSG, config) == Route("production_llm", "c0")


def _with(config, arm, **facts_paths):
    return {**config, "arms": {**config["arms"], arm: {**config["arms"][arm], **facts_paths}}}


def test_facts_must_have_been_decided_on_each_other(arms, fact_root):
    config, adapters = arms
    with pytest.raises(facts.FactError, match="not set"):
        route("B3", "filter_column", MSG, {**config, "arms": {}})
    other_choice = _write("J6", "choice", {"slm": "granite-4.2-8b"})[0]
    with pytest.raises(facts.FactError, match="another choice"):
        route("B4", "filter_column", MSG, _with(config, "B4", choice=other_choice))
    other_centroids = _write("J5", "centroids", {"embedding": EMBEDDING, "clusters": {"c0": [1.0, 0.0], "c1": [0.0, 1.0], "c2": [0.6, 0.8]}})[0]
    with pytest.raises(facts.FactError, match="other centroids"):
        route("B4", "filter_column", MSG, _with(config, "B4", centroids=other_centroids))
    c_payload, c_sha = facts.read_fact(str(fact_root / config["arms"]["B4"]["centroids"]), "centroids")
    one_short = _write("S5", "adapters", {**adapters, "adapters": {"c0": adapters["adapters"]["c0"]}})[0]
    with pytest.raises(facts.FactError, match="clusters are not"):
        route("B4", "filter_column", MSG, _with(config, "B4", adapters=one_short))
    a_sha = facts.read_fact(str(fact_root / config["arms"]["B5"]["adapters"]), "adapters")[1]
    renamed = _write("J7", "allocation", {"centroids": c_sha, "adapters": a_sha, "allocation": {"0": "slm"}})[0]
    with pytest.raises(facts.FactError, match="do not have"):
        route("B5", "filter_column", MSG, _with(config, "B5", allocation=renamed))
    stale = _write("J7", "allocation", {"centroids": c_sha, "adapters": "0" * 64, "allocation": {"c0": "slm"}})[0]
    with pytest.raises(facts.FactError, match="other centroids or adapters"):
        route("B5", "filter_column", MSG, _with(config, "B5", allocation=stale))


def test_facts_are_read_by_the_name_the_arm_expects_and_untampered(arms, fact_root):
    config, _ = arms
    with pytest.raises(facts.FactError, match="not a choice fact"):
        route("B3", "filter_column", MSG, _with(config, "B3", choice=config["arms"]["B4"]["centroids"]))
    path = fact_root / config["arms"]["B3"]["choice"]
    path.write_text('{"slm": "granite-4.2-8b"}')
    facts.read_fact.cache_clear()
    with pytest.raises(facts.FactError, match="not its directory"):
        route("B3", "filter_column", MSG, config)


def test_router_refuses_unknown_arms_and_call_sites():
    config = load_config(BASE)
    with pytest.raises(ValueError):
        route("B2", "filter_column", MSG, config)
    with pytest.raises(ValueError):
        route("B0", "unit_test", MSG, config)


@pytest.mark.parametrize("name,payload", [
    ("allocation", {"centroids": "a", "adapters": "b", "allocation": {"c0": "gpt"}}),
    ("centroids", {"embedding": EMBEDDING, "clusters": {"a": [1.0], "b": [0.6, 0.8]}}),
    ("centroids", {"embedding": EMBEDDING, "clusters": {"a": [2.0, 0.0]}}),  # not unit-norm
    ("centroids", {"embedding": {**EMBEDDING, "revision": "main"}, "clusters": {"a": [1.0]}}),
    ("centroids", {"embedding": {**EMBEDDING, "truncation": "middle"}, "clusters": {"a": [1.0]}}),
    ("adapters", {"slm": "s", "choice": "c", "centroids": "c", "adapters": {"c0": {"served_name": "", "sha256": "1" * 64}}}),
    ("adapters", {"slm": "s", "choice": "c", "centroids": "c", "adapters": {
        "c0": {"served_name": "x", "sha256": "1" * 64}, "c1": {"served_name": "x", "sha256": "2" * 64}}}),
    ("adapters", {"slm": "s", "choice": "c", "centroids": "c", "adapters": {"c0": {"served_name": "x", "sha256": "weights"}}}),
    ("unknown", {}),
])
def test_fact_shapes_are_checked(fact_root, name, payload):
    with pytest.raises(facts.FactError):
        facts.write_fact("J", name, payload)


def test_sha256_dir_identifies_the_directory(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "adapter.safetensors").write_bytes(b"w")
    before = facts.sha256_dir(tmp_path / "a")
    (tmp_path / "a" / "adapter.safetensors").write_bytes(b"x")
    assert facts.sha256_dir(tmp_path / "a") != before and len(before) == 64


def test_empty_adapter_name_never_falls_back_to_the_base():
    with pytest.raises(ConfigError, match="empty adapter"):
        engine_spec(load_config(BASE), "slm:qwen3-8b+lora:")


def test_assign_uses_the_prompt_only_with_cosine_and_ties_by_id(monkeypatch):
    assert clusters.prompt_text([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]) == "user: a\n\nassistant: b"
    assert clusters.nearest([0.6, 0.8], {"b": [0.0, 1.0], "a": [1.0, 0.0]}) == "b"
    assert clusters.nearest([1.0, 1.0], {"b": [1.0, 0.0], "a": [0.0, 1.0]}) == "a"
    assert clusters.nearest([1.0, 1.0], {"a": [10.0, 0.0], "b": [0.5, 0.5]}) == "b"  # cosine; a raw dot product picks a
    with pytest.raises(ValueError, match="dimension"):
        clusters.nearest([1.0, 0.0, 0.0], {"a": [1.0, 0.0]})
    seen = []
    monkeypatch.setattr(clusters, "embed", lambda texts, embedding: seen.extend(texts) or [[1.0, 0.0]])
    clusters.assign(MSG, {"embedding": EMBEDDING, "clusters": {"c0": [1.0, 0.0]}})
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
