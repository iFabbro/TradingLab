from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_PROTOCOL.md"
PROTOCOL_VERSION_RE = re.compile(r"\*\*Version:\*\*\s*([^\s]+)")

# Registry v1.0 (legacy) columns. Rows written under protocol v1.0 keep these values unchanged.
LEGACY_REGISTRY_FIELDS = [
    "run_id", "timestamp_utc", "git_commit", "protocol_version", "protocol_sha256",
    "dataset_sha256", "strategy", "symbol", "requested_start", "requested_end",
    "provider", "interval", "windows", "nominal_candidate_trials", "oos_observations",
    "oos_total_return", "oos_sharpe", "oos_psr", "decision", "reasons_json", "warnings_json",
]

# Registry v1.1 adds: a unique execution id, the record type, the buy-and-hold benchmark over the
# same OOS days, and a SHA-256 hash chain that makes edits and deletions in the middle detectable.
REGISTRY_FIELDS = LEGACY_REGISTRY_FIELDS + [
    "execution_id", "record_type",
    "benchmark", "benchmark_total_return", "benchmark_sharpe", "excess_total_return",
    "prev_record_sha256", "record_sha256",
]

RECORD_RUN = "run"                # one execution of an experiment; decision is PENDING until finalised
RECORD_DECISION = "decision"      # verdict appended after confirmation; never edits the run record
RECORD_LEGACY_RUN = "legacy_run"  # row written under protocol v1.0 (decision already final)
RECORD_TYPES = {RECORD_RUN, RECORD_DECISION, RECORD_LEGACY_RUN}
DECISION_PENDING = "PENDING"
FINAL_DECISIONS = {"CANDIDATE", "FAILED"}
_JSON_FIELDS = {"reasons_json", "warnings_json"}


def protocol_fingerprint(path: Path = PROTOCOL_PATH) -> dict[str, str]:
    data = path.read_bytes()
    text = data.decode("utf-8")
    match = PROTOCOL_VERSION_RE.search(text)
    if not match:
        raise ValueError("RESEARCH_PROTOCOL.md does not declare a version")
    return {"protocol_version": match.group(1), "protocol_sha256": hashlib.sha256(data).hexdigest()}


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def evaluate_acceptance(report: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    hard_fail = False

    provenance = report.get("provenance", metadata)
    protocol = provenance.get("research_protocol", {})
    if not protocol.get("protocol_version") or not protocol.get("protocol_sha256"):
        reasons.append("research protocol fingerprint is missing")
        hard_fail = True

    frictions = provenance.get("frictions", {})
    if _finite(frictions.get("transaction_cost_bps")) != 5.0:
        reasons.append("transaction_cost_bps must equal the protocol baseline of 5 bps")
        hard_fail = True
    if _finite(frictions.get("slippage_bps")) != 2.0:
        reasons.append("slippage_bps must equal the protocol baseline of 2 bps")
        hard_fail = True
    if not provenance.get("dataset_sha256"):
        reasons.append("dataset provenance hash is missing")
        hard_fail = True
    if not provenance.get("git_commit"):
        warnings.append("git commit provenance is unavailable")

    # A preliminary walk-forward result is not sufficient for CANDIDATE status.
    # The pre-registered untouched confirmation must explicitly complete.
    confirmation = report.get("robustness_confirmation")
    if not isinstance(confirmation, dict):
        reasons.append("robustness confirmation result is missing")
        hard_fail = True
    elif confirmation.get("status") != "confirmed_run":
        status = confirmation.get("status") or "missing"
        reason = confirmation.get("reason") or "no confirmed frozen parameter was produced"
        reasons.append(f"robustness confirmation is not confirmed ({status}): {reason}")
        hard_fail = True

    test_metrics = report.get("test_metrics", [])
    if isinstance(test_metrics, dict):
        test_metrics = [test_metrics]
    returns = [_finite(row.get("total_return")) for row in test_metrics if isinstance(row, dict)]
    returns = [x for x in returns if x is not None]
    if not returns:
        reasons.append("no finite OOS total-return metrics are available")
        hard_fail = True
    else:
        oos_return = _finite(
            report.get("robustness", {})
            .get("oos_daily_block_bootstrap", {})
            .get("total_return", {})
            .get("estimate")
        )
        if oos_return is None:
            reasons.append("aggregate OOS total-return estimate is missing")
            hard_fail = True
            oos_return = sum(returns)
        if oos_return <= 0:
            reasons.append("aggregate net OOS performance is not positive")
            hard_fail = True

        positive_windows = sum(x > 0 for x in returns)
        if len(returns) < 2:
            reasons.append("candidate status requires at least two OOS windows")
            hard_fail = True
        elif positive_windows < 2:
            reasons.append("OOS result is positive in fewer than two windows")
            hard_fail = True

    robustness = report.get("robustness", {})
    if not robustness.get("oos_daily_block_bootstrap"):
        reasons.append("OOS bootstrap diagnostics are missing")
        hard_fail = True
    if not robustness.get("oos_confirmation_psr"):
        reasons.append("OOS PSR diagnostic is missing")
        hard_fail = True
    if not robustness.get("validation_dsr"):
        reasons.append("validation DSR diagnostics are missing")
        hard_fail = True
    if not robustness.get("multiple_testing"):
        reasons.append("multiple-testing diagnostics are missing")
        hard_fail = True

    sharpe_ci = robustness.get("oos_daily_block_bootstrap", {}).get("sharpe", {})
    if isinstance(sharpe_ci, dict) and _finite(sharpe_ci.get("lower")) is not None and sharpe_ci["lower"] <= 0:
        warnings.append("OOS Sharpe bootstrap confidence interval includes zero")

    psr = robustness.get("oos_confirmation_psr", {})
    if isinstance(psr, dict) and _finite(psr.get("probability")) is not None and psr["probability"] < 0.95:
        warnings.append("OOS PSR probability is below 0.95")

    decision = "FAILED" if hard_fail else "CANDIDATE"
    return {
        "decision": decision,
        "reasons": reasons,
        "warnings": warnings,
        "rules_version": f"RESEARCH_PROTOCOL.md v{protocol.get('protocol_version') or 'unknown'}",
    }


def _cell(field: str, value: Any) -> str:
    if value is None:
        return ""
    if field in _JSON_FIELDS and not isinstance(value, str):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _record_hash(cells: dict[str, str]) -> str:
    """SHA-256 over every field except the hash itself, in the fixed registry column order."""
    payload = json.dumps(
        [[field, cells.get(field, "")] for field in REGISTRY_FIELDS if field != "record_sha256"],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(path: Path) -> tuple[list[str] | None, list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return None, []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: (value or "") for key, value in row.items()} for row in reader]
        return list(reader.fieldnames or []), rows


def migrate_legacy_registry(path: Path) -> bool:
    """One-time, deterministic upgrade of a v1.0 registry to the v1.1 column set.

    Every original value is preserved byte for byte. New columns are added; each legacy row gets
    ``record_type=legacy_run``, a unique ``execution_id`` (``run_id@timestamp_utc``) and a place in
    the hash chain. The chain therefore protects legacy rows only from the migration onwards.
    Returns True if a migration was performed.
    """
    header, rows = _read(path)
    if header != LEGACY_REGISTRY_FIELDS:
        return False
    previous = ""
    migrated = []
    for row in rows:
        cells = {field: row.get(field, "") for field in LEGACY_REGISTRY_FIELDS}
        cells["execution_id"] = f"{cells['run_id']}@{cells['timestamp_utc']}"
        cells["record_type"] = RECORD_LEGACY_RUN
        cells["prev_record_sha256"] = previous
        cells["record_sha256"] = _record_hash(cells)
        previous = cells["record_sha256"]
        migrated.append(cells)
    temp = path.with_suffix(path.suffix + ".migrating")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in REGISTRY_FIELDS} for row in migrated)
    temp.replace(path)
    return True


def ensure_current_format(path: Path) -> None:
    header, _ = _read(path)
    if header is None or header == REGISTRY_FIELDS:
        return
    if header == LEGACY_REGISTRY_FIELDS:
        migrate_legacy_registry(path)
        return
    raise ValueError(f"unrecognised registry header in {path}; refusing to modify it")


def _append(path: Path, record: dict[str, Any]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_current_format(path)
    _, rows = _read(path)
    record_type = record.get("record_type")
    if record_type not in RECORD_TYPES:
        raise ValueError(f"unknown record_type: {record_type!r}")
    execution_id = str(record.get("execution_id") or "")
    if not execution_id:
        raise ValueError("execution_id is required")
    known = {r["execution_id"] for r in rows if r["record_type"] in {RECORD_RUN, RECORD_LEGACY_RUN}}
    if record_type == RECORD_DECISION and execution_id not in known:
        raise ValueError(f"decision refers to an unknown execution_id: {execution_id}")
    if record_type != RECORD_DECISION and execution_id in known:
        raise ValueError(f"execution_id already registered: {execution_id}")

    cells = {field: _cell(field, record.get(field)) for field in REGISTRY_FIELDS}
    cells["prev_record_sha256"] = rows[-1]["record_sha256"] if rows else ""
    cells["record_sha256"] = _record_hash(cells)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(cells)
    return cells


def append_registry(path: Path, row: dict[str, Any]) -> dict[str, str]:
    """Append one execution record. Existing records are never modified."""
    record = dict(row)
    record.setdefault("record_type", RECORD_RUN)
    record.setdefault("decision", DECISION_PENDING)
    return _append(path, record)


def append_decision(
    path: Path,
    execution_id: str,
    decision: str,
    reasons: list[str],
    warnings: list[str],
    *,
    run_id: str,
    timestamp_utc: str,
    git_commit: str | None = None,
    protocol_version: str | None = None,
    protocol_sha256: str | None = None,
) -> dict[str, str]:
    """Append the final verdict for an execution as a new record (the run record is not edited)."""
    if decision not in FINAL_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(FINAL_DECISIONS)}")
    return _append(path, {
        "record_type": RECORD_DECISION,
        "execution_id": execution_id,
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "decision": decision,
        "reasons_json": reasons,
        "warnings_json": warnings,
    })


def resolve_verdicts(path: Path) -> dict[str, dict[str, Any]]:
    """Current verdict per execution: the last decision record, else the run record's own value."""
    _, rows = _read(path)
    verdicts: dict[str, dict[str, Any]] = {}
    for row in rows:
        execution_id = row.get("execution_id", "")
        if row["record_type"] in {RECORD_RUN, RECORD_LEGACY_RUN} or row["record_type"] == RECORD_DECISION:
            verdicts[execution_id] = {
                "decision": row["decision"],
                "reasons": json.loads(row["reasons_json"]) if row["reasons_json"] else [],
                "warnings": json.loads(row["warnings_json"]) if row["warnings_json"] else [],
                "source": row["record_type"],
            }
    return verdicts


def verify_registry(path: Path) -> dict[str, Any]:
    """Check the registry's structure and hash chain.

    status: ``empty`` | ``legacy`` (v1.0 file, not chained) | ``ok`` | ``corrupt``.
    The chain is tamper-*evident*: it reveals edits and deletions before the last record. It cannot
    detect truncation of the tail on its own; git history is the external anchor for that.
    """
    header, rows = _read(path)
    if header is None:
        return {"status": "empty", "records": 0, "problems": []}
    if header == LEGACY_REGISTRY_FIELDS:
        return {"status": "legacy", "records": len(rows), "problems": []}
    if header != REGISTRY_FIELDS:
        return {"status": "corrupt", "records": len(rows), "problems": ["unrecognised registry header"]}

    problems: list[str] = []
    previous = ""
    seen_runs: set[str] = set()
    for number, row in enumerate(rows, 1):
        if row["record_type"] not in RECORD_TYPES:
            problems.append(f"record {number}: unknown record_type {row['record_type']!r}")
        if row["prev_record_sha256"] != previous:
            problems.append(f"record {number}: broken chain (prev hash does not match the previous record)")
        if row["record_sha256"] != _record_hash(row):
            problems.append(f"record {number}: content does not match its hash (edited?)")
        execution_id = row["execution_id"]
        if row["record_type"] in {RECORD_RUN, RECORD_LEGACY_RUN}:
            if execution_id in seen_runs:
                problems.append(f"record {number}: duplicate execution_id {execution_id}")
            seen_runs.add(execution_id)
        elif row["record_type"] == RECORD_DECISION and execution_id not in seen_runs:
            problems.append(f"record {number}: decision for unknown execution_id {execution_id}")
        previous = row["record_sha256"]
    return {"status": "corrupt" if problems else "ok", "records": len(rows), "problems": problems}
