#!/usr/bin/env python3
"""Read-only model-B production config audit; never loads a checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mini_llm.production_profile_0105 import audit, parse_cli_options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("training_flags", nargs=argparse.REMAINDER,
                        help="training CLI flags following --")
    args = parser.parse_args()
    flags = args.training_flags
    if flags and flags[0] == "--":
        flags = flags[1:]
    result = audit(os.environ, parse_cli_options(flags))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
