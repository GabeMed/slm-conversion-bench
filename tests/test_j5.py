"""J5 · S3: clusters without the call site, prompt-only unit centroids as a valid fact the router
can use, ARI, and the assignment rate."""
import json
import random

import pytest

pytest.importorskip("sklearn")

from bench import paths  # noqa: E402
from bench.contracts import clusters, facts  # noqa: E402
from bench.embed import run_embed  # noqa: E402
from bench.judge import j5  # noqa: E402
from bench.judge.base import JudgmentError, read_result  # noqa: E402
from fixtures.fake import call, curate_run, example, fake_embed, fake_tokens, prompt, repo, write_run  # noqa: E402

SITES = ("select_tables", "filter_column", "generate_candidate")
SETTINGS = {"k_min": 2, "k_max": 6, "seed": 7, "n_init": 5, "silhouette_sample": 1000}


def build(tmp_path, monkeypatch, sites_of=None):
    """A curated set of 3 call sites x 12 questions, its embedding, and the teacher on calib."""
    config_path, config = repo(tmp_path, monkeypatch, {"clustering": SETTINGS})
    true = [example(str(q), site, detail=f"detail {q}") for q in range(1, 13) for site in SITES]
    examples = [{**e, "call_site": sites_of(e)} for e in true] if sites_of else true
    curate_run("curate-t", examples, originals_of=true)
    train = run_embed("curate-t", str(config_path), embed_fn=fake_embed, count_tokens=fake_tokens).name
    calib_calls = [call("agent-c", str(q), site, messages=prompt(site, f"question {q} about topic {q}", f"detail {q}"),
                        response=f"{site} answer ok {q}", parsed={}) for q in range(100, 106) for site in SITES]
    write_run("agent-c", {"type": "agent", "arm": "B0", "split": "calib"}, calib_calls)
    calib = run_embed("agent-c", str(config_path), embed_fn=fake_embed, count_tokens=fake_tokens).name
    return config, examples, train, calib, calib_calls


def test_clusters_call_sites_writes_a_valid_prompt_space_fact(tmp_path, monkeypatch):
    config, examples, train, calib, calib_calls = build(tmp_path, monkeypatch)
    path, fact = j5.run("curate-t", train, calib, config)
    payload = read_result(path, "J5")
    result = payload["result"]
    assert result["k"] == 3 and result["ari_call_sites"] == pytest.approx(1.0)
    assert result["assignment"]["train_in_sample"] == 1.0 and result["assignment"]["calib"]["rate"] == 1.0
    assert result["assignment"]["calib"]["n"] == len(calib_calls)
    assert sorted(result["sizes"].values()) == [12, 12, 12] and set(result["members"]) == {e["call_id"] for e in examples}
    assert set(payload["reads"]) == {"curate", "embed", "embed_calib"}

    centroids, sha = facts.read_fact(str(fact), "centroids")  # validated: unit, one dimension, pinned embedding
    assert sha == result["centroids"]["sha256"] and centroids["embedding"] == config["clustering"]["embedding"]
    assert set(centroids["clusters"]) == set(result["clusters"]) == {"c0", "c1", "c2"}
    # the router, with the same embedding, sends a new call of a site to that site's cluster
    monkeypatch.setattr(clusters, "embed", fake_embed)
    by_site = {}
    for e in examples:
        by_site.setdefault(e["call_site"], set()).add(result["members"][e["call_id"]])
    for c in calib_calls:
        assert {clusters.assign(c["prompt_messages"], centroids)} == by_site[c["call_site"]]


def test_centroids_are_the_unit_mean_of_the_members_prompts(tmp_path, monkeypatch):
    config, examples, train, _, _ = build(tmp_path, monkeypatch)
    _, fact = j5.run("curate-t", train, None, config)
    centroids, _ = facts.read_fact(str(fact), "centroids")
    result = j5.judge("curate-t", train, None, config["clustering"])[0]
    c0 = [e for e in examples if result["members"][e["call_id"]] == "c0"]
    vectors = fake_embed([clusters.prompt_text(e["prompt"]) for e in c0], None)
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0]))]
    norm = sum(x * x for x in mean) ** 0.5
    assert centroids["clusters"]["c0"] == pytest.approx([x / norm for x in mean], abs=1e-6)
    assert result["assignment"]["calib"] is None


def test_clustering_never_sees_the_call_site(tmp_path, monkeypatch):
    config, _, train, _, _ = build(tmp_path, monkeypatch)
    honest = j5.judge("curate-t", train, None, config["clustering"])[0]
    shuffled = random.Random(0).sample(SITES * 12, 36)
    labels = iter(shuffled)
    config2, _, train2, _, _ = build(tmp_path / "other", monkeypatch, sites_of=lambda e: next(labels))
    lied = j5.judge("curate-t", train2, None, config2["clustering"])[0]
    assert lied["members"] == honest["members"] and lied["centroids"]["sha256"] == honest["centroids"]["sha256"]
    assert lied["ari_call_sites"] < 0.5  # only the validation reads the (now wrong) call sites


def test_nearest_rows_follows_the_contract_tie_rule():
    centres = {"c10": [1.0, 0.0], "c2": [1.0, 0.0], "c3": [0.0, 1.0]}
    assert j5.nearest_rows([[1.0, 0.0], [0.1, 0.9]], centres) == [clusters.nearest([1.0, 0.0], centres), "c3"] == ["c10", "c3"]


def test_refuses_an_embedding_of_another_source(tmp_path, monkeypatch):
    config, _, train, _, _ = build(tmp_path, monkeypatch)
    curate_run("curate-other", [example("1", "select_tables")])
    with pytest.raises(JudgmentError, match="not curate-other"):
        j5.judge("curate-other", train, None, config["clustering"])
    manifest = json.loads((paths.RUNS / "curate-t" / "manifest.json").read_text())
    assert manifest["type"] == "curate"


def test_the_calib_rate_counts_calls_the_prompt_sends_elsewhere(tmp_path, monkeypatch):
    """Calls whose action looks like another call site: their S3 cluster (prompt + action) is not
    where the prompt alone sends them, and the rate counts them as misses."""
    from fixtures.fake import TEMPLATES
    config, _, train, _, calib_calls = build(tmp_path, monkeypatch)
    odd = [call("agent-c2", str(q), "select_tables", messages=prompt("select_tables", f"question {q}", ""),
                response=TEMPLATES["generate_candidate"] * 3, parsed={}) for q in range(200, 203)]
    write_run("agent-c2", {"type": "agent", "arm": "B0", "split": "calib"}, [{**c, "run_id": "agent-c2"} for c in calib_calls] + odd)
    config_path = tmp_path / "config.yaml"
    calib2 = run_embed("agent-c2", str(config_path), embed_fn=fake_embed, count_tokens=fake_tokens).name
    result = j5.judge("curate-t", train, calib2, config["clustering"])[0]
    assert result["assignment"]["calib"]["n"] == len(calib_calls) + 3
    assert result["assignment"]["calib"]["rate"] == pytest.approx(len(calib_calls) / (len(calib_calls) + 3))


def test_the_calib_embedding_must_be_the_teacher_on_calib(tmp_path, monkeypatch):
    config, _, train, _, calib_calls = build(tmp_path, monkeypatch)
    write_run("agent-b1", {"type": "agent", "arm": "B1", "split": "calib"}, [{**c, "run_id": "agent-b1"} for c in calib_calls])
    b1 = run_embed("agent-b1", str(tmp_path / "config.yaml"), embed_fn=fake_embed, count_tokens=fake_tokens).name
    with pytest.raises(JudgmentError, match="arm"):
        j5.judge("curate-t", train, b1, config["clustering"])
    assert not (paths.ROOT / "judgments" / "J5").exists()  # nothing written when a check fails


def test_j5_records_its_parameters(tmp_path, monkeypatch):
    config, _, train, _, _ = build(tmp_path, monkeypatch)
    result = j5.judge("curate-t", train, None, config["clustering"])[0]
    assert result["parameters"] == {"seed": 7, "n_init": 5, "k_range": [2, 6], "silhouette_sample": 1000,
                                    "row_order": "sha256 of the embedded text"}


# relabelling call sites among the ones the success filter does not execute: the same texts,
# a different order in curate's output (it writes call site by call site)
RELABEL = {"extract_keywords": "select_tables", "select_tables": "extract_keywords", "filter_column": "select_columns"}


def curated_and_clustered(tmp_path, monkeypatch, relabel):
    """What curate really writes, in its real order, embedded and clustered."""
    import hashlib
    pytest.importorskip("datasketch")
    import yaml
    from bench.curate import run_curate
    from bench.contracts.config import config_sha256
    from fixtures.world import teacher
    from synthetic import make_repo
    from test_curate import teacher_config
    _, config_path, config = make_repo(tmp_path, monkeypatch)
    config = teacher_config(config)
    config["clustering"].update(SETTINGS)
    config_path.write_text(yaml.safe_dump(config))
    calls = [{**c, "call_site": relabel.get(c["call_site"], c["call_site"])} for c in teacher("agent-tr", "train", ["1", "2", "3"])]
    write_run("agent-tr", {"type": "agent", "arm": "B0", "split": "train", "question_ids": ["1", "2", "3"],
                           "config_sha256": config_sha256(config)}, calls, config)
    curated = run_curate(["agent-tr"], str(config_path)).name
    embedded = run_embed(curated, str(config_path), embed_fn=fake_embed, count_tokens=fake_tokens).name
    result = j5.judge(curated, embedded, None, config["clustering"])[0]
    index = {r["call_id"]: r["action_sha256"] for r in __import__("bench.judge.base", fromlist=["read_jsonl"]).read_jsonl(
        paths.RUNS / embedded / "index.jsonl")}
    by_text = {index[call_id]: cluster for call_id, cluster in result["members"].items()}
    order = [json.loads(line)["call_site"] for line in (paths.RUNS / curated / "examples.jsonl").read_text().splitlines()]
    return result, by_text, order, hashlib


def test_clusters_do_not_depend_on_the_order_curate_writes(tmp_path, monkeypatch):
    honest, honest_labels, honest_order, _ = curated_and_clustered(tmp_path / "a", monkeypatch, {})
    relabelled, labels, order, _ = curated_and_clustered(tmp_path / "b", monkeypatch, RELABEL)
    assert order != honest_order  # curate did write them in another order
    assert relabelled["k"] == honest["k"] and relabelled["centroids"]["sha256"] == honest["centroids"]["sha256"]
    assert labels == honest_labels  # every text in the same cluster, under the same name
    assert sorted(honest["sizes"].values()).count(3) >= 2  # equal sizes: naming would follow row order
