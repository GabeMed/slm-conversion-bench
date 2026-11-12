"""EXPLORATORY, post-registration (per-cluster learning curves). Never read by a verdict.

The c1 curve's subsample (.ref/c1-curve/subsample.py), for any cluster: nested, seeded, call-site-stratified
subsamples of the registered train/datasets/<cluster>.jsonl (read-only, refused on another sha256).

Call site of a row: the curate execution's examples.jsonl joined on (prompt, completion), as c1 does; a row not
found there is a routed-sites row (train/datasets/manifest.json `routed_sites`: the agent's steps and the column
filter are written by that rule, not curated) and is classified by its prompt's opening. Where both exist they
must agree, and the counts per call site must be the ones train/datasets/manifest.json records.

Rank of a row within its call site: sha256(f"{salt}:{row sha256}:{k}"), k the row's occurrence among identical
(prompt, completion) rows of the file, in file order; for k = 0 the ":0" is left out, so a file with no repeated
row (c1) gets exactly the c1 rule, sha256(f"{salt}:{row sha256}"). The first floor(f * n + 0.5) are kept, so 25%
is inside 50%, which is inside the full set. Rows keep the order of the source file. The occurrence index is the
one thing added to c1's rule: c3's 9,164 rows are a few dozen distinct (prompt, completion) pairs, and without it
a subset would hold every copy of some pairs and no copy of the others.

    python3 -I .ref/curves/subsample.py                    # c0, c2, c3 -> .ref/curves/datasets/<cluster>/p{25,50}/<cluster>.jsonl
    python3 -I .ref/curves/subsample.py --dry-run          # counts and hashes only
    python3 -I .ref/curves/subsample.py --check-c1         # recomputes c1's two subsets with c1's salt: must be c1's sha256
Stdlib only, streaming (c0.jsonl is 221 MB): no bench import.
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "train" / "datasets"
CURATE = ROOT / "runs" / "curate-20261110T011030.933406Z" / "examples.jsonl"
FRACTIONS = {"p25": 0.25, "p50": 0.50}
# cluster -> (registered dataset sha256, salt, rows per call site as train/datasets/manifest.json and the c1 plan record them)
CLUSTERS = {
    "c0": ("ae189770741d6d5c955dfa991392fb59808e338be5bf82b00474e87056b71e9f", "c0-curve-20261111",
           {"filter_column": 11982}),
    "c1": ("e73a7dc3178b698a45fc97be968ebd8e011d076a8482c50d9debbbcf01647263", "c1-curve-20261110",
           {"select_tables": 833, "select_columns": 833, "generate_candidate": 826, "filter_column": 18, "revise": 6,
            "extract_keywords": 1}),
    "c2": ("e1e53dcc7ca25c14a994162232990cddb853f18333797dc7a9d1362cbde1736d", "c2-curve-20261111",
           {"extract_keywords": 833}),
    "c3": ("42e2455b94b67d3c756b667b7b67f558285789a5032b792391b2115abe0bd03a", "c3-curve-20261111",
           {"agent_ir": 3336, "agent_ss": 3332, "agent_cg": 2496}),
}
C1_SUBSETS = {"p25": "cc0c3aa5627fb65056550511bd2bed72d2d0559a19691724fef385201e73097a",
              "p50": "8e3cc087c99c7c0dfffff11b8a2bbb087a5fbe2fbfd0286529d107c56c0b0d27"}
OPENINGS = {  # a routed-sites row's call site, by how its prompt opens
    "You are a detail-oriented data scientist tasked with evaluating the relevance": "filter_column",
    "<system>\nHuman: You are the Information Retriever agent": "agent_ir",
    "<system>\nHuman: You are the schema_selector agent": "agent_ss",
    "<system>\nHuman: You are the Candidate Generator agent": "agent_cg",
}
LABEL = ("EXPLORATORY, post-registration learning curve of {cluster}; outside the preregistered plan; "
         "never read by a verdict")


def canonical(payload) -> bytes:  # = bench.contracts.facts.canonical
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def curated_sites() -> dict:
    sites = defaultdict(set)
    with CURATE.open() as f:
        for line in f:
            e = json.loads(line)
            sites[hashlib.sha256(canonical([e["prompt"], e["completion"]])).hexdigest()].add(e["call_site"])
    return sites


def subsample(cluster: str, sites: dict, out: Path, dry_run: bool) -> dict:
    source_sha, salt, expected = CLUSTERS[cluster]
    source = DATASETS / f"{cluster}.jsonl"
    if file_sha256(source) != source_sha:
        sys.exit(f"REFUSED: {source} is not the registered {cluster} dataset {source_sha[:12]}")
    strata = defaultdict(list)  # call site -> [(rank, index)]
    seen = Counter()
    sizes = []
    by_rule = Counter()
    with source.open("rb") as f:
        for i, raw_line in enumerate(f):
            sizes.append(len(raw_line))
            row = json.loads(raw_line)
            key = hashlib.sha256(canonical([row["prompt"], row["completion"]])).hexdigest()
            opening = next((site for start, site in OPENINGS.items() if row["prompt"][0]["content"].startswith(start)), None)
            found = sites.get(key)
            if found:
                if len(found) != 1:
                    sys.exit(f"{cluster} row {i}: ambiguous call sites {sorted(found)}")
                site = next(iter(found))
                if opening is not None and opening != site:
                    sys.exit(f"{cluster} row {i}: curated as {site}, opens as {opening}")
                by_rule["curated"] += 1
            elif opening is not None:
                site = opening  # a routed-sites row (written by the routed sites' rule, not curated)
                by_rule["routed (by opening)"] += 1
            else:
                sys.exit(f"{cluster} row {i}: call site not found")
            k = seen[key]
            seen[key] += 1
            strata[site].append((hashlib.sha256(f"{salt}:{key}{f':{k}' if k else ''}".encode()).hexdigest(), i))
    full = {s: len(v) for s, v in sorted(strata.items())}
    if full != dict(sorted(expected.items())):
        sys.exit(f"REFUSED: {cluster} has {full} rows per call site, the registered manifest says {expected}")
    total = sum(sizes)
    repeated = sum(n - 1 for n in seen.values())
    print(f"{cluster} full rows={len(sizes)} {full} distinct (prompt, completion)={len(seen)} repeated rows={repeated} "
          f"sites from {dict(by_rule)}")
    manifest = {"label": LABEL.format(cluster=cluster),
                "source": {"path": f"train/datasets/{cluster}.jsonl", "sha256": source_sha, "rows": len(sizes), "bytes": total},
                "call_site_from": f"runs/{CURATE.parent.name}/examples.jsonl joined on (prompt, completion); "
                                  "unmatched rows = routed-sites rows, by their prompt's opening",
                "salt": salt,
                "rule": "per call site, the floor(f*n+0.5) rows of smallest sha256(salt:row sha256[:k]), k the row's "
                        "occurrence among identical rows (left out when 0); nested",
                "distinct_rows": len(seen), "repeated_rows": repeated, "full": full, "subsets": {}}
    kept = {}
    for name, fraction in FRACTIONS.items():
        keep, counts = set(), {}
        for site, ranked in sorted(strata.items()):
            k = math.floor(fraction * len(ranked) + 0.5)
            keep |= {i for _, i in sorted(ranked)[:k]}
            counts[site] = k
        kept[name] = keep
        digest, size, distinct = hashlib.sha256(), 0, set()
        target = out / cluster / name / f"{cluster}.jsonl"
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
        sink = None if dry_run else target.open("wb")
        with source.open("rb") as f:
            for i, raw_line in enumerate(f):
                if i in keep:
                    digest.update(raw_line)
                    size += len(raw_line)
                    distinct.add(hashlib.sha256(raw_line).digest())
                    if sink:
                        sink.write(raw_line)
        if sink:
            sink.close()
        sha, share = digest.hexdigest(), size / total
        print(f"{cluster} {name} rows={len(keep)} {counts} distinct={len(distinct)} bytes share {share:.4f} {sha}")
        manifest["subsets"][name] = {"fraction": fraction, "rows": len(keep), "by_call_site": counts, "distinct_rows": len(distinct),
                                     "path": f".ref/curves/datasets/{cluster}/{name}/{cluster}.jsonl", "sha256": sha,
                                     "bytes": size, "bytes_share_of_full": round(share, 4)}
    if not kept["p25"] <= kept["p50"]:
        sys.exit(f"{cluster}: p25 is not inside p50")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", nargs="+", default=["c0", "c2", "c3"], choices=sorted(CLUSTERS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-c1", action="store_true", help="recompute c1's subsets, write nothing, compare with c1's sha256")
    ap.add_argument("--out", default=str(ROOT / ".ref" / "curves" / "datasets"))
    args = ap.parse_args()
    sites = curated_sites()
    if args.check_c1:
        got = {name: s["sha256"] for name, s in subsample("c1", sites, Path(args.out), dry_run=True)["subsets"].items()}
        if got != C1_SUBSETS:
            sys.exit(f"c1 NOT reproduced: {got}")
        print("c1 reproduced: the rule here is c1's rule on c1's file")
        return 0
    if "c1" in args.clusters:
        sys.exit("c1's subsets are .ref/c1-curve/datasets (use --check-c1 to recompute them, written nowhere)")
    out = Path(args.out)
    manifests = {cluster: subsample(cluster, sites, out, args.dry_run) for cluster in args.clusters}
    if not args.dry_run:
        path = out / "manifest.json"
        merged = json.loads(path.read_text()) if path.exists() else {}
        merged.update(manifests)
        path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
