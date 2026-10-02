"""C2 · the keys of the stratified pilot and of the cap on the training examples: what `load_config`
refuses, and what the shipped configuration sets."""
import copy

import pytest

from bench import paths
from bench.contracts.config import load_config, validate_config

BASE = paths.ROOT / "config.yaml"


def test_the_shipped_configuration_sets_the_pilot_mix_the_cap_and_the_training_timeout():
    config = load_config(BASE)
    assert config["stats"]["pilot_mix"] == {"simple": 15, "moderate": 25, "challenging": 10}
    assert sum(config["stats"]["pilot_mix"].values()) == config["stats"]["pilot_size"] == 50
    assert config["curation"]["max_per_question_call_site"] == 4
    assert len(set(config["seeds"].values())) == len(config["seeds"]) and "curation_sample" in config["seeds"]
    assert config["train"]["timeout_s"] == 39600
    # a projection that passes preflight leaves the container an hour to load the base and save the adapter
    assert config["preflight"]["schedule"]["train_hours_max"] * 3600 == config["train"]["timeout_s"] - 3600


@pytest.mark.parametrize("mutate, error", [
    (lambda c: c["stats"]["pilot_mix"].update(simple=-1), "stats.pilot_mix"),
    (lambda c: c["stats"]["pilot_mix"].update(simple=1.5), "stats.pilot_mix"),
    (lambda c: c["stats"]["pilot_mix"].update(simple=True), "stats.pilot_mix"),
    (lambda c: c["stats"].update(pilot_mix=[15, 25, 10]), "stats.pilot_mix"),
    (lambda c: c["stats"].update(pilot_mix={}), "stats.pilot_mix"),
    (lambda c: c["curation"].update(max_per_question_call_site=0), "curation.max_per_question_call_site"),
    (lambda c: c["curation"].update(max_per_question_call_site="4"), "curation.max_per_question_call_site"),
    (lambda c: c["curation"].update(max_per_question_call_site=True), "curation.max_per_question_call_site"),
    (lambda c: c["seeds"].update(curation_sample="20260934"), "seeds.curation_sample must be an integer"),
])
def test_malformed_pilot_and_cap_keys_are_refused(mutate, error):
    config = copy.deepcopy(load_config(BASE))
    assert validate_config(config) == []
    mutate(config)
    assert [e for e in validate_config(config) if error in e]


def test_a_difficulty_with_no_pilot_question_is_a_valid_mix():
    config = copy.deepcopy(load_config(BASE))
    config["stats"]["pilot_mix"] = {"simple": 50, "moderate": 0, "challenging": 0}
    assert validate_config(config) == []


def test_the_keys_are_checked_where_they_are_set():
    """A configuration that neither draws the pilot nor curates (other commands' fixtures) loads without them."""
    config = copy.deepcopy(load_config(BASE))
    config["stats"] = {"n_boot": 100}
    del config["curation"], config["seeds"]["curation_sample"]
    assert validate_config(config) == []
