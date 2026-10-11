"""Inspect a declared pipeline without execution."""
import argparse
import json
from .config import ConfigError, load_pipeline
from .planning import explain


def main():
    parser = argparse.ArgumentParser(prog="cpg")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Validate YAML and print a plan without execution")
    inspect.add_argument("config")
    inspect.add_argument("--height", required=True, type=int, help="Input height")
    inspect.add_argument("--width", required=True, type=int, help="Input width")
    inspect.add_argument("--backend", choices=("reference", "cuda"), default="reference")
    inspect.add_argument("--explain", action="store_true", help="Show static execution groups and pixel-buffer sizes; no measurements")
    args = parser.parse_args()
    try:
        pipeline = load_pipeline(args.config)
        shape = (args.height, args.width, 3)
        plan = explain(pipeline, shape, backend=args.backend) if args.explain else pipeline.plan(shape, backend=args.backend)
    except (ConfigError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2))
