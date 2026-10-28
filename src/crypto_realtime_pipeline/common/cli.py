import argparse
from pathlib import Path

from crypto_realtime_pipeline.common.log import RunContext


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--bundle", required=True, help="Bundle name, for log context")
    parser.add_argument("--target", required=True, help="Bundle target, for log context")
    parser.add_argument("--run-id", required=True, help="Job run id, for log context")
    return parser


def landing_parser(description: str) -> argparse.ArgumentParser:
    parser = base_parser(description)
    parser.add_argument(
        "--landing-root", required=True, type=Path, help="Root path of the landing volume"
    )
    return parser


def run_context(args: argparse.Namespace, workload: str) -> RunContext:
    return RunContext(bundle=args.bundle, target=args.target, workload=workload, run_id=args.run_id)
