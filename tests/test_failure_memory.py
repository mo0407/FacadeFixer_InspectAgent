"""Local failure ledger semantics; no input readers, models or network calls."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from facade_mcp.failure_memory import FailureMemory, REASONS


def record(memory, *, job="job-1", occurrence="failure-1", code="input_invalid", **kwargs):
    return memory.record(job_id=job, stage="fixture-stage", reason_code=code,
                         occurrence_key=occurrence, **kwargs)


def test_catalog_is_fixed_and_kind_cannot_be_supplied(tmp_path):
    assert {code for code, value in REASONS.items() if value["kind"] == "runtime"} == {
        "input_missing", "input_invalid", "source_changed", "playbook_changed",
        "model_weights_missing", "model_runtime_error", "mcp_connection_error",
        "api_timeout", "api_unauthorized", "api_rate_limited", "api_http_error",
        "api_transport_error", "model_output_invalid", "invalid_inspection_action",
        "model_version_changed", "expert_unavailable",
        "candidate_budget_exceeded", "final_review_invalid", "unclassified_runtime_error"}
    assert {code for code, value in REASONS.items() if value["kind"] == "quality"} == {
        "missed_defect", "false_positive", "wrong_class", "bbox_inaccurate", "mask_inaccurate",
        "scope_error", "report_error", "duplicate_defect", "metric_below_target"}
    assert {code for code, value in REASONS.items() if value["kind"] == "observation"} == {
        "unknown_class", "scope_uncertain", "evidence_downsampled", "harness_override", "candidate_rejected"}
    assert all(set(value) == {"kind", "title", "summary", "recommended_action"} for value in REASONS.values())
    with pytest.raises(TypeError):
        REASONS["input_invalid"]["kind"] = "observation"
    with pytest.raises(TypeError):
        record(FailureMemory(tmp_path), kind="observation")


def test_occurrence_retry_is_idempotent_and_never_overwrites(tmp_path):
    memory = FailureMemory(tmp_path)
    first = record(memory, summary="First observation", context={"model_profile": "yolo11m"})
    retry = record(memory, summary="Changed text must not replace first", context={"model_profile": "other"})
    assert first["inserted"] is True and retry["inserted"] is False
    assert retry["event_id"] == first["event_id"]
    assert retry["recorded_at"] == first["recorded_at"]
    assert retry["summary"] == "First observation"
    assert retry["context"] == {"model_profile": "yolo11m"}
    assert memory.summary()["totals"]["events"] == 1
    with pytest.raises(ValueError, match="different reason_code"):
        record(memory, code="input_missing")
    assert memory.summary()["totals"]["events"] == 1


def test_same_reason_events_and_affected_jobs_are_separate_counts(tmp_path):
    memory = FailureMemory(tmp_path)
    record(memory, job="job-a", occurrence="one")
    record(memory, job="job-a", occurrence="two")
    record(memory, job="job-b", occurrence="one")
    summary = memory.summary()
    reason = summary["reasons"][0]
    assert reason["reason_code"] == "input_invalid"
    assert reason["count"] == 3 and reason["affected_jobs"] == 2 and reason["repeat_count"] == 2
    assert reason["first_seen"] <= reason["last_seen"]
    assert summary["totals"] == dict(runtime_failures=3, quality_failures=0,
                                     observations=0, events=3, affected_jobs=2)
    assert summary["latest_event"] == summary["events"][0]
    assert [e["recorded_at"] for e in summary["events"]] == sorted(
        [e["recorded_at"] for e in summary["events"]], reverse=True)
    scoped = memory.summary("job-a")
    assert len(scoped["events"]) == 2 and scoped["totals"]["affected_jobs"] == 1
    assert scoped["reasons"][0]["repeat_count"] == 1


def test_observations_never_count_as_failures_and_quality_is_explicit(tmp_path):
    memory = FailureMemory(tmp_path)
    for index, code in enumerate(("unknown_class", "scope_uncertain", "evidence_downsampled",
                                  "harness_override", "candidate_rejected")):
        record(memory, occurrence=str(index), code=code)
    record(memory, occurrence="quality", code="missed_defect", source="reviewer",
           evidence_refs=["review_job/request.json#candidate-1"], cause_status="confirmed")
    summary = memory.summary()
    assert summary["totals"] == dict(runtime_failures=0, quality_failures=1,
                                     observations=5, events=6, affected_jobs=1)
    assert "不计入失败" in summary["interpretation"]["observation"]
    assert "reviewer/evaluator" in summary["interpretation"]["quality"]
    assert "不代表" in summary["interpretation"]["quality"]


def test_quality_requires_evidence_but_never_reads_reference_targets(tmp_path):
    memory = FailureMemory(tmp_path)
    with pytest.raises(ValueError, match="evidence reference"):
        record(memory, code="false_positive")
    with pytest.raises(ValueError, match="evidence reference"):
        record(memory, code="false_positive", evidence_refs=[])
    event = record(memory, code="false_positive", source="evaluator",
                   evidence_refs=["does-not-exist/request.json#candidate-1", "evidence:id"])
    assert event["evidence_refs"] == ["does-not-exist/request.json#candidate-1", "evidence:id"]
    # An occurrence retry is still idempotent when no new evidence is supplied.
    retry = record(memory, code="false_positive")
    assert retry["inserted"] is False and retry["evidence_refs"] == event["evidence_refs"]
    assert memory.summary()["totals"]["quality_failures"] == 1


def test_concurrent_eight_writes_and_retries_are_transactional(tmp_path):
    memory = FailureMemory(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        events = list(pool.map(lambda _: record(memory), range(8)))
    assert sum(event["inserted"] for event in events) == 1
    assert len({event["event_id"] for event in events}) == 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        events = list(pool.map(lambda index: record(
            FailureMemory(tmp_path), job=f"job-{index}", occurrence="distinct"), range(8)))
    assert all(event["inserted"] for event in events)
    summary = memory.summary()
    assert summary["totals"]["events"] == 9
    assert summary["totals"]["affected_jobs"] == 8
    assert summary["reasons"][0]["count"] == 9


def test_restart_preserves_record_and_deduplication(tmp_path):
    first = record(FailureMemory(tmp_path), code="api_timeout", cause_status="hypothesis",
                   hypothesis="Provider completion is unknown", context={"timeout_seconds": 90})
    restarted = FailureMemory(tmp_path)
    assert restarted.db_path == tmp_path / "failure_memory.sqlite3"
    retry = record(restarted, code="api_timeout")
    assert retry["event_id"] == first["event_id"] and retry["inserted"] is False
    assert retry["cause_status"] == "hypothesis"
    assert retry["hypothesis"] == "Provider completion is unknown"
    assert restarted.summary()["totals"]["events"] == 1


def test_unknown_code_and_invalid_cause_status_are_rejected(tmp_path):
    memory = FailureMemory(tmp_path)
    with pytest.raises(ValueError, match="Unknown reason_code"):
        record(memory, code="make-it-up")
    with pytest.raises(ValueError, match="cause_status"):
        record(memory, cause_status="automatic-ground-truth")
    assert memory.summary()["totals"]["events"] == 0


@pytest.mark.parametrize("kwargs", [
    {"summary": "sk-fixtureSecret123456789"},
    {"summary": "Bearer fixture-token"},
    {"context": {"api_key": "fixture"}},
    {"context": {"nested": {"authorization": "fixture"}}},
    {"context": {"exception": "Raw exception text"}},
    {"context": {"value": RuntimeError("raw")}},
    {"context": {"value": float("nan")}},
    {"context": {"value": float("inf")}},
    {"context": {"value": "x" * 2049}},
    {"context": {"value": [None] * 513}},
    {"summary": "x" * 2001},
    {"evidence_refs": [""]},
    {"evidence_refs": [123]},
])
def test_unsafe_or_unbounded_fields_are_rejected_without_persisting(tmp_path, kwargs):
    memory = FailureMemory(tmp_path)
    with pytest.raises(ValueError):
        record(memory, **kwargs)
    assert memory.summary()["events"] == []


def test_records_use_catalog_summary_and_safe_json_roundtrip(tmp_path):
    memory = FailureMemory(tmp_path)
    event = record(memory, context={"http_status": 429, "model": "fixture-model",
                                    "values": [True, False, None, .5]})
    assert event["summary"] == REASONS["input_invalid"]["summary"]
    assert event["kind"] == "runtime" and event["reason_code"] == "input_invalid"
    assert json.loads(json.dumps(memory.summary(), ensure_ascii=False, allow_nan=False))["events"][0]["context"] == event["context"]


def test_html_escapes_every_caller_controlled_field(tmp_path):
    memory = FailureMemory(tmp_path / "workspace")
    event = memory.record(job_id="<script>alert(1)</script>", stage='<img src=x onerror="bad">',
        reason_code="report_error", occurrence_key="html-case", summary="<b>Untrusted report</b>",
        evidence_refs=["<iframe>evidence</iframe>"], context={"note": "<svg onload=bad>"},
        source="<reviewer>", cause_status="hypothesis", hypothesis="<em>Maybe</em>")
    output = tmp_path / "exports" / "统计.html"
    exported = memory.export_html(output, event["job_id"])
    document = output.read_text(encoding="utf-8")
    assert exported["totals"]["quality_failures"] == 1 and exported["path"] == str(output)
    for raw in ("<script>", "<img ", "<b>Untrusted", "<iframe>", "<svg ", "<reviewer>", "<em>Maybe"):
        assert raw not in document
    for escaped in ("&lt;script&gt;", "&lt;img ", "&lt;b&gt;Untrusted", "&lt;iframe&gt;", "&lt;svg "):
        assert escaped in document
    assert "原因频次" in document and "事件记录" in document and "观察项不计入失败" in document


def test_empty_and_unmatched_job_summary(tmp_path):
    memory = FailureMemory(tmp_path)
    summary = memory.summary("missing-job")
    assert summary["events"] == [] and summary["reasons"] == [] and summary["latest_event"] is None
    assert summary["totals"] == dict(runtime_failures=0, quality_failures=0,
                                     observations=0, events=0, affected_jobs=0)
    output = tmp_path / "empty.html"
    assert memory.export_html(output)["totals"]["events"] == 0
    assert output.is_file()


def test_publish_summary_returns_canonical_snapshot_with_metadata(tmp_path):
    memory = FailureMemory(tmp_path / "workspace")
    record(memory)
    output = tmp_path / "published" / "summary.json"
    published = memory.publish_summary(output, job_id="job-1", extra={
        "mock_observations_omitted": True,
        "insights": [{"hypothesis": "Review proposal only", "confirmed": False}]})
    assert published == json.loads(output.read_text(encoding="utf-8"))
    assert published["job_id"] == "job-1" and published["totals"]["events"] == 1
    assert published["mock_observations_omitted"] is True
    assert published["insights"][0]["confirmed"] is False
    assert not list(output.parent.glob("*.tmp"))


@pytest.mark.parametrize("field", ["events", "reasons", "totals", "job_id", "latest_event", "interpretation"])
def test_publish_summary_rejects_canonical_field_overrides(tmp_path, field):
    memory = FailureMemory(tmp_path)
    output = tmp_path / "summary.json"
    with pytest.raises(ValueError, match="canonical"):
        memory.publish_summary(output, extra={field: "overwrite"})
    assert not output.exists()


@pytest.mark.parametrize("extra", [{"score": float("nan")}, {"exception": RuntimeError("raw")}, []])
def test_publish_summary_requires_finite_json_metadata(tmp_path, extra):
    with pytest.raises(ValueError):
        FailureMemory(tmp_path).publish_summary(tmp_path / "summary.json", extra=extra)


def test_eight_concurrent_record_publish_writers_never_regress_or_break_json(tmp_path):
    memory = FailureMemory(tmp_path / "workspace")
    output = tmp_path / "summary.json"
    memory.publish_summary(output)
    barrier = threading.Barrier(8)
    stop = threading.Event()
    observed_counts = []
    read_errors = []

    def reader():
        while not stop.is_set():
            try:
                snapshot = json.loads(output.read_text(encoding="utf-8"))
                observed_counts.append(snapshot["totals"]["events"])
                assert len(snapshot["events"]) == snapshot["totals"]["events"]
            except PermissionError:
                # Windows can temporarily deny opening a destination while an
                # atomic rename is in progress; this is not a partial JSON read.
                stop.wait(.001)
            except Exception as error:
                read_errors.append(error)
                break

    def writer(index):
        local = FailureMemory(tmp_path / "workspace")
        barrier.wait(timeout=10)
        for occurrence in range(3):
            record(local, job="same-job", occurrence=f"writer-{index}-{occurrence}")
            snapshot = local.publish_summary(output)
            assert snapshot["totals"]["events"] >= occurrence + 1

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(writer, range(8)))
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not thread.is_alive() and not read_errors
    assert observed_counts and observed_counts == sorted(observed_counts)
    final = json.loads(output.read_text(encoding="utf-8"))
    assert final == memory.summary()
    assert final["totals"]["events"] == 24


def test_source_hash_catalog_reports_inconsistency_without_claiming_mutation():
    assert REASONS["source_changed"]["title"] == "图像证据不一致"
    assert REASONS["source_changed"]["summary"] == "原图、快照或专家结果的图像哈希校验不一致。"


def test_publish_review_metadata_is_not_subject_to_ledger_context_limits(tmp_path):
    memory = FailureMemory(tmp_path)
    insights = [dict(metric_id=f"metric-{index}", hypothesis="x" * 3000,
                     evidence_ids=[f"evidence-{number}" for number in range(30)]) for index in range(20)]
    summary = memory.publish_summary(tmp_path / "review-summary.json", extra={"insights": insights})
    assert summary["insights"] == insights
    assert len(json.dumps(summary)) > 16384
