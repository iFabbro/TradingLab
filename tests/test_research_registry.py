import csv

import pytest

from src.research_governance import (
    LEGACY_REGISTRY_FIELDS,
    REGISTRY_FIELDS,
    append_decision,
    append_registry,
    migrate_legacy_registry,
    resolve_verdicts,
    verify_registry,
)


def _run(execution_id="spy_x@t1", **extra):
    return {"execution_id": execution_id, "run_id": "spy_x", "timestamp_utc": "t1", "strategy": "x", **extra}


def _decide(path, execution_id="spy_x@t1", decision="FAILED"):
    return append_decision(
        path, execution_id, decision, ["why"], [], run_id="spy_x", timestamp_utc="t2"
    )


def _rewrite(path, mutate):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_legacy(path, count=2):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEGACY_REGISTRY_FIELDS)
        writer.writeheader()
        for n in range(count):
            row = {field: "" for field in LEGACY_REGISTRY_FIELDS}
            row.update(run_id="spy_old", timestamp_utc=f"2026-09-2{n}", protocol_version="1.0",
                       decision="FAILED", reasons_json='["r"]', warnings_json="[]", oos_total_return="0.4565")
            writer.writerow(row)


def test_append_builds_a_valid_chain_and_pending_run_then_decision(tmp_path):
    path = tmp_path / "registry.csv"
    first = append_registry(path, _run())
    assert first["decision"] == "PENDING" and first["record_type"] == "run" and first["prev_record_sha256"] == ""
    second = _decide(path)
    assert second["prev_record_sha256"] == first["record_sha256"]
    assert verify_registry(path) == {"status": "ok", "records": 2, "problems": []}
    assert resolve_verdicts(path)["spy_x@t1"]["decision"] == "FAILED"


def test_decision_never_edits_the_run_record(tmp_path):
    path = tmp_path / "registry.csv"
    append_registry(path, _run())
    before = path.read_text(encoding="utf-8").splitlines()
    _decide(path)
    after = path.read_text(encoding="utf-8").splitlines()
    assert after[: len(before)] == before and len(after) == len(before) + 1


def test_same_run_id_can_run_twice_but_execution_id_cannot_repeat(tmp_path):
    path = tmp_path / "registry.csv"
    append_registry(path, _run("spy_x@t1"))
    append_registry(path, _run("spy_x@t3"))
    with pytest.raises(ValueError, match="already registered"):
        append_registry(path, _run("spy_x@t3"))
    assert verify_registry(path)["status"] == "ok"


def test_decision_for_unknown_execution_is_rejected(tmp_path):
    path = tmp_path / "registry.csv"
    append_registry(path, _run())
    with pytest.raises(ValueError, match="unknown execution_id"):
        _decide(path, "other@t9")
    with pytest.raises(ValueError, match="decision must be one of"):
        _decide(path, decision="PENDING")


def test_editing_a_record_is_detected(tmp_path):
    path = tmp_path / "registry.csv"
    append_registry(path, _run())
    _decide(path, decision="FAILED")
    append_registry(path, _run("spy_x@t5"))
    _rewrite(path, lambda rows: rows[1].update(decision="CANDIDATE"))
    result = verify_registry(path)
    assert result["status"] == "corrupt"
    assert any("record 2" in problem and "hash" in problem for problem in result["problems"])


def test_deleting_a_middle_record_is_detected(tmp_path):
    path = tmp_path / "registry.csv"
    append_registry(path, _run())
    _decide(path)
    append_registry(path, _run("spy_x@t5"))
    _rewrite(path, lambda rows: rows.pop(1))
    result = verify_registry(path)
    assert result["status"] == "corrupt"
    assert any("broken chain" in problem for problem in result["problems"])


def test_legacy_registry_is_migrated_without_changing_original_values(tmp_path):
    path = tmp_path / "registry.csv"
    _write_legacy(path)
    assert verify_registry(path)["status"] == "legacy"
    assert migrate_legacy_registry(path) is True
    assert migrate_legacy_registry(path) is False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["record_type"] for r in rows] == ["legacy_run", "legacy_run"]
    assert rows[0]["decision"] == "FAILED" and rows[0]["oos_total_return"] == "0.4565"
    assert rows[0]["execution_id"] != rows[1]["execution_id"]
    assert verify_registry(path)["status"] == "ok"


def test_append_to_legacy_registry_migrates_first(tmp_path):
    path = tmp_path / "registry.csv"
    _write_legacy(path)
    append_registry(path, _run())
    result = verify_registry(path)
    assert result["status"] == "ok" and result["records"] == 3


def test_unknown_header_is_refused(tmp_path):
    path = tmp_path / "registry.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    assert verify_registry(path)["status"] == "corrupt"
    with pytest.raises(ValueError, match="unrecognised"):
        append_registry(path, _run())
