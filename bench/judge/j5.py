"""J5 · S3, task clustering (SPEC §4 S3): clusters, their centroids for the router, and how well
the clusters are discovered and assigned.

Reads: a `curate` execution (the training examples), the `embed` execution of it, and optionally
the `embed` execution of the teacher on `calib`. Writes the `centroids` fact and the J5 result.

- **Clusters** are formed without supervision on the **prompt + action** vectors: k-means on unit
  vectors (so Euclidean distance orders like cosine), with the number of clusters chosen
  automatically as the one of highest cosine silhouette in `[k_min, k_max]` (seeded; the
  silhouette on a seeded sample). `cluster()` receives the vectors and nothing else: the call site
  never reaches it (SPEC S3). K-means rather than a density method because every training example
  needs a cluster (S5 trains one adapter per cluster), and a density method leaves noise out.
- **Centroids for assignment** live in the **prompt-only** space the router embeds in: the unit
  mean of the members' prompt vectors. The fact's `embedding` block is the embed execution's, so
  the router reproduces the same space. Cluster ids are `c0`, `c1`, … by decreasing size.
- **ARI** of the clusters against the call sites (the only place call sites are read), with each
  cluster's composition. ARI ≈ 1 means S3 is trivial for this agent (SPEC S3); read it with the
  fraction of prompts the embedding cut (`truncation`), since a cut that removed everything but
  the template would make ARI ≈ 1 by construction (design §5.1).
- **Assignment rate**: how often the prompt-only nearest centroid (what the router does) is the
  S3 cluster, in sample on `train` and on `calib`, where the S3 cluster of a call is its nearest
  k-means centre in the prompt + action space. The nearest centroid follows `clusters.nearest`:
  highest cosine, ties to the smallest id.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bench import paths
from bench.contracts.facts import write_fact
from bench.judge.base import (JudgmentError, read_jsonl, reference, relative, require_done, write_result)

JUDGMENT = "J5"


def _unit(matrix):
    import numpy as np
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def cluster(vectors, k_min: int, k_max: int, seed: int, n_init: int, silhouette_sample: int
            ) -> Tuple[List[int], Any, Dict[int, float]]:
    """(labels, centres, silhouette per k): k-means on unit vectors, k by the best silhouette."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    x = _unit(np.asarray(vectors, dtype=np.float64))
    n = len(x)
    ks = [k for k in range(k_min, k_max + 1) if 2 <= k < n]
    if not ks:
        raise JudgmentError(f"cannot choose a number of clusters for {n} examples in [{k_min}, {k_max}]")
    best, scores = None, {}
    for k in ks:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x)
        if len(set(model.labels_)) < 2:
            continue
        scores[k] = float(silhouette_score(x, model.labels_, metric="cosine",
                                           sample_size=min(silhouette_sample, n), random_state=seed))
        if best is None or scores[k] > scores[best[0]]:
            best = (k, model)
    if best is None:
        raise JudgmentError("every k gave a single cluster: the vectors do not separate")
    return [int(label) for label in best[1].labels_], _unit(best[1].cluster_centers_), scores


def names(labels: Sequence[int]) -> Dict[int, str]:
    """`c0`, `c1`, … by decreasing size; equal sizes by first member."""
    order = sorted(set(labels), key=lambda label: (-list(labels).count(label), list(labels).index(label)))
    return {label: f"c{i}" for i, label in enumerate(order)}


def nearest_rows(vectors, centres: Dict[str, Sequence[float]]) -> List[str]:
    """`clusters.nearest` for many vectors at once: highest cosine, ties to the smallest id."""
    import numpy as np
    ids = sorted(centres)
    matrix = _unit(np.asarray([centres[c] for c in ids], dtype=np.float64))
    similarity = _unit(np.asarray(vectors, dtype=np.float64)) @ matrix.T
    return [ids[i] for i in similarity.argmax(axis=1)]  # argmax keeps the first maximum: the smallest id


def adjusted_rand(truth: Sequence[str], found: Sequence[str]) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(list(truth), list(found)))


# ---------------------------------------------------------------- reading the executions

def load_embed(run_id: str):
    """(manifest, index rows, prompt matrix, prompt+action matrix) of an embed execution."""
    import numpy as np
    found = require_done(run_id, type="embed")
    directory = paths.RUNS / run_id
    return (found, read_jsonl(directory / "index.jsonl"),
            np.load(directory / "prompt.npy"), np.load(directory / "prompt_action.npy"))


def judge(curate_run_id: str, embed_run_id: str, calib_embed_run_id: Optional[str], settings: Dict[str, Any],
          root=None) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    """The J5 result, what it read, and the path of the centroids fact it wrote."""
    curated = require_done(curate_run_id, type="curate")
    examples = read_jsonl(paths.RUNS / curate_run_id / "examples.jsonl")
    embedded, index, prompts, actions = load_embed(embed_run_id)
    if embedded["source"]["run_id"] != curate_run_id:
        raise JudgmentError(f"{embed_run_id} embedded {embedded['source']['run_id']}, not {curate_run_id}")
    if [row["call_id"] for row in index] != [e["call_id"] for e in examples] or any(r["action_row"] is None for r in index):
        raise JudgmentError(f"{embed_run_id} does not cover every curated example with its action")
    embedding = embedded["embedding"]

    labels, _, silhouettes = cluster(actions[[r["action_row"] for r in index]], settings["k_min"], settings["k_max"],
                                     settings["seed"], settings["n_init"], settings["silhouette_sample"])
    rename = names(labels)
    found = [rename[label] for label in labels]
    ids = sorted(set(found), key=lambda c: int(c[1:]))
    prompt_matrix = prompts[[r["prompt_row"] for r in index]]
    centroids = {c: [float(v) for v in _unit(prompt_matrix[[i for i, f in enumerate(found) if f == c]].mean(axis=0, keepdims=True))[0]]
                 for c in ids}
    action_centres = {c: _unit(actions[[index[i]["action_row"] for i, f in enumerate(found) if f == c]]
                               .mean(axis=0, keepdims=True))[0] for c in ids}  # the k-means centre, in PA space
    fact = write_fact(JUDGMENT, "centroids", {"embedding": embedding, "clusters": centroids}, root)

    sites = [e["call_site"] for e in examples]  # read here, after clustering, for validation only
    composition = {c: {} for c in ids}
    for c, site in zip(found, sites):
        composition[c][site] = composition[c].get(site, 0) + 1
    in_sample = nearest_rows(prompt_matrix, centroids)

    calib = None
    reads = {"curate": reference(curate_run_id), "embed": reference(embed_run_id)}
    if calib_embed_run_id:
        calib_manifest, calib_index, calib_prompts, calib_actions = load_embed(calib_embed_run_id)
        if calib_manifest["embedding"] != embedding:
            raise JudgmentError(f"{calib_embed_run_id} used another embedding than the clusters")
        with_action = [r for r in calib_index if r["action_row"] is not None]
        s3 = nearest_rows(calib_actions[[r["action_row"] for r in with_action]], action_centres)
        routed = nearest_rows(calib_prompts[[r["prompt_row"] for r in with_action]], centroids)
        calib = {"split": calib_manifest.get("split"), "n": len(with_action),
                 "rate": sum(a == b for a, b in zip(s3, routed)) / len(with_action) if with_action else None,
                 "routed_sizes": {c: routed.count(c) for c in ids},
                 "truncation": calib_manifest["tokens"]}
        reads["embed_calib"] = reference(calib_embed_run_id)

    result = {
        "method": "k-means on unit prompt+action vectors, k by cosine silhouette",
        "k": len(ids), "silhouette": {str(k): s for k, s in sorted(silhouettes.items())},
        "clusters": ids, "sizes": {c: found.count(c) for c in ids}, "composition": composition,
        "ari_call_sites": adjusted_rand(sites, found),
        "members": {e["call_id"]: c for e, c in zip(examples, found)},
        "assignment": {"train_in_sample": sum(a == b for a, b in zip(in_sample, found)) / len(found), "calib": calib},
        "embedding": embedding, "truncation": embedded["tokens"],
        "centroids": {"path": relative(fact), "sha256": fact.parent.name},
        "curation": curated["counts"],
    }
    return result, reads, fact


def run(curate_run_id: str, embed_run_id: str, calib_embed_run_id: Optional[str], config: Dict[str, Any]):
    result, reads, fact = judge(curate_run_id, embed_run_id, calib_embed_run_id, config["clustering"])
    return write_result(JUDGMENT, reads, result), fact
