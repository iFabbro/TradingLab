"""Finalize the immutable research verdict after robustness confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research_governance import evaluate_acceptance, update_registry_decision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--registry", default="experiments/registry.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    confirmation_path = Path(args.confirmation)
    metadata_path = Path(args.metadata)
    report_path = run_dir / "report.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))

    # The confirmation result is authoritative for promotion. No numerical
    # experiment output is changed here; only the governance verdict is
    # recomputed after the pre-registered confirmation step has completed.
    report["robustness_confirmation"] = confirmation
    acceptance = evaluate_acceptance(report, metadata)
    report["acceptance"] = acceptance
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    run_id = metadata["run_id"]
    update_registry_decision(
        ROOT / args.registry,
        run_id,
        acceptance["decision"],
        acceptance["reasons"],
        acceptance["warnings"],
    )

    print(json.dumps({
        "run_id": run_id,
        "confirmation_status": confirmation.get("status"),
        "decision": acceptance["decision"],
        "reasons": acceptance["reasons"],
        "warnings": acceptance["warnings"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
