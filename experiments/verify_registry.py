"""Verify the research registry's hash chain. Exit code 1 if it is corrupt."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research_governance import verify_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiments/registry.csv")
    args = parser.parse_args()
    result = verify_registry(ROOT / args.registry)
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "corrupt" else 0


if __name__ == "__main__":
    raise SystemExit(main())
