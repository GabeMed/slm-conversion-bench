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

    f4_commands(sub)
    args = parser.parse_args(argv)
    if getattr(args, "f4", None):
        return args.f4(args)
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
    except (TestSplitLocked, ConfigError, DataError, FactError, HarnessError) as e:
        print(f"bench {args.command}: {e}", file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------- F4: S2, S3, the judgments, the report

def _pairs(values):
    """`RUN=EVAL` arguments as {run: eval}."""
    from bench.judge.base import JudgmentError
    pairs = {}
    for value in values or []:
        run_id, sep, eval_run_id = value.partition("=")
        if not sep or not run_id or not eval_run_id:
            raise JudgmentError(f"expected RUN=EVAL, got {value!r}")
        pairs[run_id] = eval_run_id
    return pairs


def _f4(args) -> int:
    from bench.contracts.config import load_config
    from bench.judge.base import JudgmentError
    try:
        config = load_config(args.config)
        if args.command == "curate":
            from bench.curate import run_curate
            print(run_curate(args.source, args.config))
        elif args.command == "datasets":
            from bench.curate import write_datasets
            print(write_datasets(args.curated, args.j5, args.config))
        elif args.command == "embed":
            from bench.embed import run_embed
            print(run_embed(args.source, args.config))
        elif args.command == "report":
            from bench import report
            print(report.run(args.plan, config))
        elif args.judgment == "j2":
            from bench.judge import j2
            from bench.judge.base import write_result
            reads, result = (j2.judge_run(args.run) if args.run else
                             j2.judge_replay(args.replay, args.replay_eval, args.teacher_eval))
            print(write_result(j2.JUDGMENT, reads, result))
        elif args.judgment == "j3":
            from bench.judge import j3
            print(j3.run(args.run, args.eval, args.j8, config))
        elif args.judgment == "j5":
            from bench.judge import j5
            print(*j5.run(args.curated, args.embed, args.calib_embed, config), sep="\n")
        elif args.judgment == "j6":
            from bench.judge import j6
            print(*j6.run(_pairs(args.zeroshot), args.teacher_eval, config), sep="\n")
        elif args.judgment == "j7":
            from bench.judge import j7
            replays = {engine: next(iter(_pairs([value]).items())) for engine, value in
                       (("cheap_alt", args.cheap_alt), ("slm", args.slm))}
            print(*j7.run(args.centroids, args.adapters, replays, args.teacher_eval, args.j8, args.j6, config), sep="\n")
        elif args.judgment == "j8":
            from bench.judge import j8
            print(j8.run(args.loadtest, config))
    except (JudgmentError, FactError, ConfigError, DataError, TestSplitLocked) as e:
        print(f"bench {args.command}: {e}", file=sys.stderr)
        return 2
    return 0


def f4_commands(sub) -> None:
    p = sub.add_parser("curate", help="S2: curate the teacher's train logs (masking, success filter, dedup)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", action="append", required=True, help="a B0 train execution (repeatable)")
    p.set_defaults(f4=_f4)

    p = sub.add_parser("datasets", help="S2/S3: train/datasets/<cluster>.jsonl from a curate execution and J5")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--curated", required=True, help="the curate execution")
    p.add_argument("--j5", required=True, help="the J5 result that clustered it")
    p.set_defaults(f4=_f4)

    p = sub.add_parser("embed", help="the embed execution: prompt and prompt+action vectors (S3)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", required=True, help="a curate, agent or replay execution")
    p.set_defaults(f4=_f4)

    p = sub.add_parser("report", help="R: the SPEC section 8 report from the judgments")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--plan", required=True, help="a YAML plan naming the executions and judgments (bench/report.py)")
    p.set_defaults(f4=_f4)

    judge = sub.add_parser("judge", help="the judgments J2, J3, J5-J8 (pure functions over executions)")
    judgments = judge.add_subparsers(dest="judgment", required=True)
    p = judgments.add_parser("j2", help="per call site: a replay against its teacher, or one run's format validity")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--replay", help="a replay execution")
    group.add_argument("--run", help="an agent or replay execution: format validity only")
    p.add_argument("--replay-eval", help="the per-call eval of the replay")
    p.add_argument("--teacher-eval", help="the per-call eval of the replay's source")
    p = judgments.add_parser("j3", help="cost of an execution")
    p.add_argument("--run", required=True)
    p.add_argument("--eval", help="its eval execution, for the cost per correct query")
    p.add_argument("--j8", help="the J8 result, when the execution has SLM calls")
    p = judgments.add_parser("j5", help="S3: clusters and the centroids fact")
    p.add_argument("--curated", required=True)
    p.add_argument("--embed", required=True, help="the embed execution of the curated examples")
    p.add_argument("--calib-embed", help="the embed execution of the teacher on calib")
    p = judgments.add_parser("j6", help="S4: the choice fact")
    p.add_argument("--zeroshot", action="append", required=True, metavar="REPLAY=EVAL")
    p.add_argument("--teacher-eval", required=True)
    p = judgments.add_parser("j7", help="S6: the allocation fact")
    p.add_argument("--centroids", required=True)
    p.add_argument("--adapters", required=True)
    p.add_argument("--cheap-alt", required=True, metavar="REPLAY=EVAL")
    p.add_argument("--slm", required=True, metavar="REPLAY=EVAL", help="the calib replay routed as B4")
    p.add_argument("--teacher-eval", required=True)
    p.add_argument("--j8", required=True)
    p.add_argument("--j6", required=True, help="the J6 result: its chosen candidate's zero-shot replay is the pilot")
    p = judgments.add_parser("j8", help="load: SLM cost per request at each utilization")
    p.add_argument("--loadtest", action="append", required=True)
    for p in judgments.choices.values():
        p.add_argument("--config", default="config.yaml")
    judge.set_defaults(f4=_f4)


if __name__ == "__main__":
    sys.exit(main())
