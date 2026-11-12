"""EXPLORATORY, post-registration (per-cluster learning curves). Never read by a verdict.

.ref/c1-curve/train_curve.py for c0, c2 and c3: `bench train --cluster <c> --on tinker` on a subsample, unchanged
except for where it reads and writes. bench.train.datasets_dir and adapters_dir (and bench.train_tinker's imported
copy of adapters_dir) point at .ref/curves/{datasets,adapters}/<cluster>/<point>/, so the registered
train/datasets/, train/adapters/, the .tinker pending/resume records and the S5 `adapters` fact are never touched.
The cluster keeps its name (training_plan only accepts a centroid cluster), so the adapter is served as
<cluster>-<sha256[:12]> of its own bytes, distinct from the official one. register_adapters() then finds only that
cluster under the curve directory and writes no fact. The manifest's dataset.path is rewritten to the subsample's
real path (the plan, and so the Tinker key, carry it).

    PYTHONPATH=<worktree> .venv-tinker/bin/python <worktree>/.ref/curves/train_curve.py c0 p25 --plan-only   # free
    PYTHONPATH=<worktree> .venv-tinker/bin/python <worktree>/.ref/curves/train_curve.py c0 p25 [--save-every N] [--resume]
c1's two points are .ref/c1-curve's (already trained); this script refuses c1.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

POINTS = {  # (cluster, point) -> the subsample's sha256 (.ref/curves/datasets/manifest.json)
    ("c0", "p25"): "0ea6548b4aabc18eaa65d24db3261e1f129b24ba67c67422d8c2fd640211d01b",
    ("c0", "p50"): "1da493614c4e023cd28fff6c260baff030eecb8f0be477d6b9fe5a937cd99d2c",
    ("c2", "p25"): "89ae452fb0498e1d6f0060e60ab3e6dfd2df21d0712f7a532e86aef3532e12de",
    ("c2", "p50"): "06f29135c3c688c38bbd94a80423c48e2fd58a225a2e461027c990f3ea465e6c",
    ("c3", "p25"): "cde78c7988cb8ddb920fce1324276c626aa7059df7165866626e0b42cb7db51f",
    ("c3", "p50"): "d8eb424f40b0fe406fb48e802e4e5086ba95cb741238035aacad9697dffb8207",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster", choices=sorted({c for c, _ in POINTS}))
    ap.add_argument("point", choices=sorted({p for _, p in POINTS}))
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tinker-config", default="configs/train-tinker.yaml")
    ap.add_argument("--plan-only", action="store_true", help="print the plan, schedule and Tinker key; call nothing")
    ap.add_argument("--save-every", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--expect-root", help="refuse unless bench.paths.ROOT is this directory")
    args = ap.parse_args()
    cluster, point = args.cluster, args.point

    from bench import paths
    import bench.train as T
    import bench.train_tinker as TT

    if args.expect_root and paths.ROOT != Path(args.expect_root).resolve():
        sys.exit(f"REFUSED: bench.paths.ROOT is {paths.ROOT}, not {args.expect_root} (set PYTHONPATH)")
    base = paths.ROOT / ".ref" / "curves"
    data_dir, adapter_dir = base / "datasets" / cluster / point, base / "adapters" / cluster / point
    rel_dataset = f".ref/curves/datasets/{cluster}/{point}/{cluster}.jsonl"
    got = hashlib.sha256((data_dir / f"{cluster}.jsonl").read_bytes()).hexdigest()
    if got != POINTS[cluster, point]:
        sys.exit(f"REFUSED: {rel_dataset} sha256 {got} is not the subsample {POINTS[cluster, point]}")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    T.datasets_dir = lambda: data_dir
    T.adapters_dir = lambda: adapter_dir
    TT.adapters_dir = lambda: adapter_dir
    official_plan = TT.training_plan

    def plan_of(config, which):
        if which != cluster:
            sys.exit(f"REFUSED: a plan for {which}, this run is {cluster}")
        plan, raw = official_plan(config, which)
        if plan["dataset"]["sha256"] != POINTS[cluster, point]:
            sys.exit(f"REFUSED: the plan read {plan['dataset']}, not the subsample: the redirection failed")
        plan["dataset"]["path"] = rel_dataset
        return plan, raw
    TT.training_plan = plan_of
    # fail closed: nothing may resolve to the registered directories
    assert T.datasets_dir() != paths.ROOT / "train" / "datasets" and TT.adapters_dir() != paths.ROOT / "train" / "adapters"

    if args.plan_only:
        from bench.contracts.config import load_config
        config = load_config(args.config)
        plan, raw = plan_of(config, cluster)
        run = TT.tinker_plan(plan, TT.load_settings(args.tinker_config))
        rows = T.parse_dataset(raw, plan["dataset"]["path"])
        run["schedule"] = TT.schedule(len(rows), plan["hyperparameters"]["sft"])
        print(json.dumps({"label": "EXPLORATORY", "root": str(paths.ROOT), "cluster": cluster, "point": point,
                          "dataset": plan["dataset"], "schedule": run["schedule"],
                          "tinker_key": T.plan_id({**plan, "run": run}), "plan_sha256": T.plan_id(plan),
                          "target": str(adapter_dir / cluster)}, indent=2))
        return 0

    result = TT.train_cluster(args.config, cluster, args.tinker_config, resume=args.resume,
                              save_every=TT.SAVE_EVERY if args.save_every is None else args.save_every)
    if result.get("fact") is not None:
        sys.exit(f"UNEXPECTED: an adapters fact was written: {result['fact']}")
    manifest = json.loads(Path(result["manifest"]).read_text())
    (adapter_dir / "EXPLORATORY.json").write_text(json.dumps({
        "label": f"EXPLORATORY, post-registration learning curve of {cluster}; outside the preregistered plan; "
                 "never read by a verdict", "cluster": cluster, "point": point, "served_name": manifest["served_name"],
        "adapter_sha256": manifest["adapter_sha256"], "dataset": manifest["dataset"],
        "tokens_seen": manifest["stats"].get("tokens_seen")}, indent=2) + "\n")
    print(result["manifest"], manifest["served_name"], "tokens_seen", manifest["stats"].get("tokens_seen"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
