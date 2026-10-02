"""C2, Front R's keys (design §6.3): the thresholds, the allocation floor and the claims bar every verdict
reads are in config.yaml with the registered values, and a missing or malformed one is refused."""
import copy

import pytest

from bench import paths
from bench.contracts.config import load_config, validate_config, verdict_settings_errors

BASE = load_config(paths.ROOT / "config.yaml")


def test_the_shipped_verdict_rules():
    assert BASE["thresholds"] == {"delta_pp": 5, "selection_delta_pp": 2.5, "concordance_min": 0.95,
                                  "concordance_slack_pp": 2, "format_tolerance_pp": 1}
    assert "delta_cap_pp" not in BASE["thresholds"]  # T3: no cap on a margin derived from d
    assert BASE["allocation"]["min_calls"] == 100 and BASE["claims"]["v3_min_ratio"] == 3
    assert verdict_settings_errors(BASE) == []


@pytest.mark.parametrize("path, value, message", [
    (("thresholds", "delta_pp"), None, "thresholds.delta_pp"),
    (("thresholds", "delta_pp"), -1, "thresholds.delta_pp"),
    (("thresholds", "selection_delta_pp"), "2.5", "thresholds.selection_delta_pp"),
    (("thresholds", "concordance_min"), 95, "thresholds.concordance_min"),
    (("thresholds", "concordance_slack_pp"), None, "thresholds.concordance_slack_pp"),
    (("thresholds", "format_tolerance_pp"), True, "thresholds.format_tolerance_pp"),
    (("allocation", "min_calls"), None, "allocation.min_calls"),
    (("allocation", "min_calls"), 0, "allocation.min_calls"),
    (("allocation", "min_calls"), 2.5, "allocation.min_calls"),
    (("claims", "v3_min_ratio"), 0.5, "claims.v3_min_ratio"),
])
def test_a_malformed_verdict_rule_is_refused(path, value, message):
    config = copy.deepcopy(BASE)
    config[path[0]][path[1]] = value
    assert any(message in error for error in validate_config(config))


@pytest.mark.parametrize("section, key", [("thresholds", "delta_pp"), ("thresholds", "format_tolerance_pp"),
                                          ("allocation", "min_calls"), ("claims", "v3_min_ratio")])
def test_a_missing_verdict_rule_is_refused(section, key):
    config = copy.deepcopy(BASE)
    del config[section][key]
    assert any(f"{section}.{key}" in error for error in validate_config(config))
