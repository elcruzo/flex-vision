"""Inspect a declared pipeline without execution."""
import argparse
import json
from .config import ConfigError, load_pipeline


def main():
    parser = argparse.ArgumentParser(prog="cpg")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Validate YAML and print the reference-only plan")
    inspect.add_argument("config")
    inspect.add_argument("--height", required=True, type=int, help="Input height")
    inspect.add_argument("--width", required=True, type=int, help="Input width")
    args = parser.parse_args()
    try:
        plan = load_pipeline(args.config).plan((args.height, args.width, 3))
    except (ConfigError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2))
