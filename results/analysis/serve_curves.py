"""EXPLORATORY, post-registration (per-cluster learning curves). Never read by a verdict.

.ref/c1-curve/serve_curve.py for the four clusters: `bench serve --on runpod up --candidate qwen3-8b`, unchanged
(runpod_apps.serve.up), with a plan that serves the base and, per cluster, three adapters: the official one (the
100% point) and the two curve adapters (25%, 50%). Twelve adapters on one server when every cluster is asked for.
`up` takes a ready plan (`plan=`); only where the adapters are uploaded from differs (Provider.ensure_adapters reads
train/adapters/<cluster>/adapter, so it is wrapped to read each adapter's own directory). Every adapter is checked
as adapters_to_serve checks one: its bytes are its manifest's sha256, its served name is <cluster>-<sha256[:12]>, it
was trained on the candidate's pinned base and template kwargs; an official one must be the registered adapter.

c1's curve adapters are .ref/c1-curve/adapters/p{25,50}/c1 (trained); the others are
.ref/curves/adapters/<cluster>/p{25,50}/<cluster>. `--plan-only` prints the plan with what exists and names what is
not trained yet; without it, a missing adapter refuses (serve fewer with --clusters / --points).

Refuses while any qwen3-8b serving pod is up (up() itself reuses a recorded one, which would serve the registered
set and not these adapters; serving_pods() refuses an unrecorded one). The base (the 0% point) is always served:
`--engine slm:qwen3-8b`.

    PYTHONPATH=<worktree> .venv-agent/bin/python .ref/curves/serve_curves.py --official-dir <main>/train/adapters --plan-only
    PYTHONPATH=<worktree> .venv-agent/bin/python .ref/curves/serve_curves.py --official-dir <main>/train/adapters --hours 4
    # afterwards: PYTHONPATH=<worktree> .venv-agent/bin/bench serve --on runpod down --candidate qwen3-8b
"""
import argparse
import copy
import json
import sys
from pathlib import Path

CANDIDATE = "qwen3-8b"
SOURCE = "agent-B0-calib-20261109T151141.039667Z"
OFFICIAL = {"c0": "842db723c47e533efa6358df51745f7f045ad8271a2b3e2feba848fab24de9d8",
            "c1": "fa3a060fdb1cfac7e301dc45fcf9a864c3746110382a2d8ba8af1d594bc86865",
            "c2": "64efb1484f8ce42c3c38fbd1c09dfa0f146810db0053b2934dc4593440670720",
            "c3": "22febb722a91e7714add223ac75c59f64fb479da61adf4bcca8a1b0711c72052"}
SITES = {"c0": ["filter_column"], "c1": ["select_tables", "select_columns", "generate_candidate", "revise"],
         "c2": ["extract_keywords"], "c3": ["agent_ir", "agent_ss", "agent_cg"]}
POINTS = ("p25", "p50", "p100")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--official-dir", help="the directory holding the official <cluster>/{manifest.json,adapter/} "
                                           "(default: <root>/train/adapters)")
    ap.add_argument("--clusters", nargs="+", default=sorted(SITES), choices=sorted(SITES))
    ap.add_argument("--points", nargs="+", default=list(POINTS), choices=POINTS)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=16, help="for the replay commands printed")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--expect-root")
    args = ap.parse_args()

    from bench import paths
    from bench.contracts.config import load_config
    from bench.contracts.facts import sha256_dir
    from bench.train import candidate, served_name
    from modal_apps import common

    if args.expect_root and paths.ROOT != Path(args.expect_root).resolve():
        sys.exit(f"REFUSED: bench.paths.ROOT is {paths.ROOT}, not {args.expect_root} (set PYTHONPATH)")
    config = load_config(args.config)
    base_only = copy.deepcopy(config)
    base_only["arms"]["B4"]["adapters"] = None  # adapters_to_serve then globs <root>/train/adapters, absent in the worktree
    plan = common.serve_plan(base_only, CANDIDATE)
    if plan["adapters"]:
        sys.exit(f"REFUSED: the base plan already serves {[a['served_name'] for a in plan['adapters']]}: "
                 "run from a worktree without train/adapters/")
    entry = candidate(config, CANDIDATE)
    trained_for = {"base": dict(entry["hf"]), "chat_template_kwargs": dict(entry.get("chat_template_kwargs") or {})}
    official = Path(args.official_dir) if args.official_dir else paths.ROOT / "train" / "adapters"

    def directory(cluster: str, point: str) -> Path:
        if point == "p100":
            return official / cluster
        if cluster == "c1":  # the c1 curve's own directory, read only
            return paths.ROOT / ".ref" / "c1-curve" / "adapters" / point / "c1"
        return paths.ROOT / ".ref" / "curves" / "adapters" / cluster / point / cluster

    local, served, missing = {}, [], []
    for cluster in args.clusters:
        for point in args.points:
            d = directory(cluster, point)
            if not (d / "manifest.json").is_file():
                missing.append(f"{cluster} {point}")
                print(f"{cluster} {point}: NOT TRAINED YET ({d})")
                continue
            manifest = json.loads((d / "manifest.json").read_text())
            sha = sha256_dir(d / "adapter")
            if manifest.get("cluster") != cluster or sha != manifest["adapter_sha256"] \
                    or manifest["served_name"] != served_name(cluster, sha):
                sys.exit(f"REFUSED: {d} is not the {cluster} adapter its manifest names")
            if {k: manifest.get(k) for k in trained_for} != trained_for:
                sys.exit(f"REFUSED: {d} was trained on other weights or template kwargs")
            if point == "p100" and sha != OFFICIAL[cluster]:
                sys.exit(f"REFUSED: {d} is not the official {cluster} adapter {OFFICIAL[cluster][:12]}")
            if point != "p100" and sha == OFFICIAL[cluster]:
                sys.exit(f"REFUSED: {d} holds the official {cluster} adapter, not a curve point")
            rank = json.loads((d / "adapter" / "adapter_config.json").read_text())["r"]
            plan["adapters"].append({"cluster": cluster, "served_name": manifest["served_name"], "sha256": sha, "r": rank})
            local[sha] = d / "adapter"
            served.append((cluster, point, manifest["served_name"]))
            print(f"{cluster} {point}: {manifest['served_name']} rows={manifest['dataset']['rows']} r={rank} sha256={sha}")
    if len({a["served_name"] for a in plan["adapters"]}) != len(plan["adapters"]):
        sys.exit("REFUSED: two points share an adapter")
    command = common.vllm_command(plan)
    print(f"\n{len(plan['adapters'])} adapters beside the base; "
          + " ".join(f"{flag} {command[command.index(flag) + 1]}" for flag in ("--max-loras", "--max-cpu-loras", "--max-lora-rank")
                     if flag in command))
    print(" ".join(command))
    print(f"\n# replays, one at a time, each adapter at its cluster's call sites only (source {SOURCE}):")
    for cluster, point, name in served:
        print(f"bench replay {SOURCE} --engine slm:{CANDIDATE}+lora:{name} --call-sites {' '.join(SITES[cluster])} "
              f"--workers {args.workers}   # {cluster} {point}")
    for item in missing:
        cluster, point = item.split()
        print(f"# {cluster} {point}: bench replay {SOURCE} --engine slm:{CANDIDATE}+lora:<served_name of {directory(cluster, point)}/manifest.json> "
              f"--call-sites {' '.join(SITES[cluster])} --workers {args.workers}")
    if args.plan_only:
        return 0
    if missing:
        sys.exit(f"REFUSED: not trained yet: {', '.join(missing)} (train them, or serve fewer with --clusters / --points)")

    from runpod_apps import pods as rp
    from runpod_apps import serve
    from runpod_apps.settings import VOLUME

    def ensure_adapters(self, remote, adapters):  # = Provider.ensure_adapters, from each adapter's own directory
        uploaded = []
        for adapter in adapters:
            target = f"{VOLUME}/adapters/{adapter['sha256']}"
            if remote.exists(f"{target}/adapter_config.json"):
                continue
            partial = f"{VOLUME}/adapters/.upload-{adapter['sha256']}"
            remote.check(f"rm -rf {partial} && mkdir -p {VOLUME}/adapters", "preparing the adapter upload")
            remote.shell.put(local[adapter["sha256"]], partial)
            remote.check(f"rm -rf {target} && mv {partial} {target}", "installing the adapter")
            uploaded.append(adapter["sha256"])
        return uploaded

    rp.Provider.ensure_adapters = ensure_adapters
    provider = rp.make_provider(config)
    if provider.state.serving(CANDIDATE) or serve.serving_pods(provider, CANDIDATE):
        sys.exit(f"REFUSED: a {CANDIDATE} pod is up (the registered chain's?): wait for it to go down")
    result = serve.up(provider, CANDIDATE, args.hours, plan=plan)
    print(json.dumps({k: result.get(k) for k in ("pod_id", "base_url", "uploaded_adapters", "cost_per_hour")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
