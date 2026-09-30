"""The `bench` command (SPEC-aligned seam: each subcommand has one consumer, see the design §3.2)."""
import argparse
import sys

from bench.agent.hooks import HarnessError
from bench.barrier import TestSplitLocked
from bench.contracts.config import ConfigError
from bench.contracts.facts import FactError
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

    # F3 · training, load test and preflight (bench/train.py, bench/loadtest.py, bench/preflight.py)
    p = sub.add_parser("train", help="S5: one LoRA adapter for a cluster; registers the adapters fact once every cluster has one")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--cluster", required=True)
    p.add_argument("--on", required=True, choices=("local", "modal"), help="local: this machine; modal: a Modal GPU")
    p = sub.add_parser("loadtest", help="the loadtest execution: AIPerf replays a run's calls on an engine, one run per concurrency")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--engine", required=True, help="an engine of the router, e.g. slm:qwen3-8b+lora:c3-<sha256[:12]>")
    p.add_argument("--source", required=True, help="the run whose calls.jsonl is replayed")
    p.add_argument("--on", required=True, choices=("local", "modal"), help="where the AIPerf client runs")
    p.add_argument("--concurrency", type=int, nargs="+", help="levels (default: loadtest.concurrency)")
    p.add_argument("--tokenizer", help="HF repo of the engine's tokenizer, when the engine is not an SLM candidate")
    p.add_argument("--tokenizer-revision", help="its 40-hex commit")
    p = sub.add_parser("preflight", help="the Day-1 preconditions of SPEC 7.1 and P-4, each with its action")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--parity", metavar="CLUSTER", help="run P-4 on this cluster's adapter against the served candidate")
    p.add_argument("--on", default="modal", choices=("local", "modal"), help="where P-4's HF-PEFT reference runs")

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
        elif args.command in ("train", "loadtest", "preflight"):  # F3: each reports its own errors
            from bench import loadtest, preflight, train
            return {"train": train.cli, "loadtest": loadtest.cli, "preflight": preflight.cli}[args.command](args)
    except (TestSplitLocked, ConfigError, DataError, FactError, HarnessError) as e:
        print(f"bench {args.command}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
