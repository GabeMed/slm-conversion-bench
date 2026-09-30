"""The `bench` command (SPEC-aligned seam: each subcommand has one consumer, see the design §3.2)."""
import argparse
import sys

from bench.agent.hooks import HarnessError
from bench.barrier import TestSplitLocked
from bench.contracts.config import ConfigError
from bench.contracts.facts import FactError
from bench.data import DataError

AGENT_ARMS = ("B0", "B1", "B3", "B4", "B5")
RUN_ARMS = ("B0", "B1", "B2", "B3", "B4", "B5")  # B2: one call per question, no agent (F1)
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
    p.add_argument("--arm", required=True, choices=RUN_ARMS)
    p.add_argument("--split", required=True, choices=SPLITS)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--ids", nargs="+", help="question ids, all inside the split")
    group.add_argument("--limit", type=int, help="the first N questions of the split, by id")
    p.add_argument("--engine", choices=("production_llm", "cheap_alt"), help="B2 only: the engine of the single call")

    # ---- F1: replay and the call-site registry
    p = sub.add_parser("replay", help="resend the first attempt of every invocation of a B0 run to another engine")
    p.add_argument("source_run_id")
    p.add_argument("--config", default="config.yaml")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--engine", help="a fixed engine: production_llm, cheap_alt, slm:<candidate>[+lora:<served name>]")
    group.add_argument("--arm", choices=AGENT_ARMS, help="route each call as this arm does")
    p.add_argument("--call-sites", nargs="+", help="only these call sites")

    p = sub.add_parser("call-sites", help="add the call sites of done train/calib runs to registry/call_sites.json")
    p.add_argument("run_ids", nargs="+")
    # ---- end F1

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
            print(run_agent(args.config, args.arm, args.split, ids=args.ids, limit=args.limit, engine=args.engine))
        elif args.command == "replay":
            from bench.agent.replay import replay
            print(replay(args.config, args.source_run_id, engine=args.engine, arm=args.arm, call_sites=args.call_sites))
        elif args.command == "call-sites":
            from bench.agent.registry import update_call_sites
            print(update_call_sites(args.run_ids))
        elif args.command == "eval":
            from bench.evaluate import evaluate
            print(evaluate(args.run_id))
    except (TestSplitLocked, ConfigError, DataError, FactError, HarnessError) as e:
        print(f"bench {args.command}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
