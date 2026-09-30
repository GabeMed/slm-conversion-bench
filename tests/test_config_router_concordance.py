"""C2 (configuration), C4 (router) and C3 (agreement)."""
import copy

import pytest
import yaml

from bench import paths
from bench.contracts.calls import CALL_SITES
from bench.contracts.concordance import agree
from bench.contracts.config import ConfigError, config_sha256, load_config, validate_config
from bench.contracts.router import route

BASE = paths.ROOT / "config.yaml"
SMOKE = paths.ROOT / "configs" / "smoke-local.yaml"


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
    lambda c: c["retries"].update(parse_max_attempts=0),
    lambda c: c["data"]["plat_sql_test"].pop("sha256"),
    lambda c: c.pop("seeds"),
])
def test_invalid_configs(mutate):
    config = copy.deepcopy(load_config(BASE))
    mutate(config)
    assert validate_config(config)


# ---------------------------------------------------------------- C4

def test_b0_routes_every_call_site_to_the_production_llm():
    config = load_config(BASE)
    for site in CALL_SITES:
        assert route("B0", site, [{"role": "user", "content": "q"}], config) == "production_llm"


def test_other_arms_are_not_built_yet_and_b2_never_routes():
    config = load_config(BASE)
    for arm in ("B1", "B3", "B4", "B5"):
        with pytest.raises(NotImplementedError):
            route(arm, "filter_column", [], config)
    with pytest.raises(ValueError):
        route("B2", "filter_column", [], config)
    with pytest.raises(ValueError):
        route("B0", "unit_test", [], config)


# ---------------------------------------------------------------- C3

def test_agreement_per_call_site():
    assert agree("extract_keywords", ["a", "b"], ["b", "a"])
    assert not agree("extract_keywords", ["a"], ["a", "b"])
    assert agree("filter_column", {"is_column_information_relevant": "Yes"}, {"is_column_information_relevant": " yes"})
    assert not agree("filter_column", {"is_column_information_relevant": "Yes"}, {"is_column_information_relevant": "No"})
    assert agree("select_tables", {"table_names": ["`Schools`", "frpm"]}, {"table_names": ["frpm", "schools"]})
    assert agree("select_columns",
                 {"chain_of_thought_reasoning": "x", "schools": ["`County`", "cds"]},
                 {"chain_of_thought_reasoning": "y", "Schools": ["cds", "county"]})
    assert not agree("select_columns", {"schools": ["cds"]}, {"schools": ["cds", "county"]})
    assert agree("agent_ss", {"tool": "select_tables"}, {"tool": "select_tables"})
    assert not agree("agent_ss", {"tool": "select_tables"}, {"done": True})


def test_unparsed_never_agrees_and_gold_sites_refuse():
    assert not agree("extract_keywords", None, None)
    for site in ("generate_candidate", "revise"):
        with pytest.raises(ValueError):
            agree(site, {"SQL": "x"}, {"SQL": "x"})
