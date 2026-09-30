"""The `bench` command (SPEC-aligned seam: each subcommand has one consumer, see the design §3.2)."""
import argparse
import sys

from bench.barrier import TestSplitLocked
from bench.contracts.config import ConfigError
from bench.data import DataError

AGENT_ARMS = ("B0", "B1", "B3", "B4", "B5")
SPLITS = ("train", "calib", "test")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bench")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="download, check and pin the inputs; write data/MANIFEST.json and data/splits.json")
    p.add_argument("--config", default="config.yaml")

    p = sub.add_parser("preprocess", help="CHESS preprocessing of databases with the configured embeddings")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--db", action="append", required=True, help="database id (repeatable)")

    p = sub.add_parser("run", help="an agent execution: one arm on one split")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--arm", required=True, choices=AGENT_ARMS)
    p.add_argument("--split", required=True, choices=SPLITS)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--ids", nargs="+", help="question ids, all inside the split")
    group.add_argument("--limit", type=int, help="the first N questions of the split, by id")

    p = sub.add_parser("eval", help="the eval execution of a run's predictions (paired by question_id)")
    p.add_argument("run_id")

    args = parser.parse_args(argv)
    try:
        if args.command == "data":
            from bench import data
            from bench.contracts.config import load_config
            print(data.run(load_config(args.config)))
        elif args.command == "preprocess":
            from bench.agent.runner import preprocess
            preprocess(args.config, args.db)
        elif args.command == "run":
            from bench.agent.runner import run_agent
            print(run_agent(args.config, args.arm, args.split, ids=args.ids, limit=args.limit))
        elif args.command == "eval":
            from bench.evaluate import evaluate
            print(evaluate(args.run_id))
    except (TestSplitLocked, ConfigError, DataError) as e:
        print(f"bench {args.command}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
