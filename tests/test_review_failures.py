import asyncio
import json
from pathlib import Path

import httpx
import pytest

from facade_mcp.failure_memory import FailureMemory
from facade_mcp.model_gateway import GatewayError, ModelGateway
from facade_mcp.playbook import Playbook, Rule
from facade_mcp.review_cycle import ReviewRequest, run_review
from test_model_gateway import config


def review_request(metrics):
    return ReviewRequest.model_validate({
        "task": "Review supplied facade metrics", "evidence": {"reviewed-frame": "fixture evidence"},
        "metrics": metrics, "acceptance_version": "acceptance-v1", "playbook_version": "P0-test",
    })


def metric(**changes):
    return {"id": "wall-recall", "value": .6, "threshold": .9,
            "direction": "min", "required": True, "reference_verified": True,
            "evidence_ids": ["reviewed-frame"], **changes}


def playbook():
    return Playbook(version="P0-test", rules=[Rule(
        id="scope-1", condition="Scope is unverified", recommendation="Check wall scope first")])


def read_summary(result):
    path = Path(result["failure_summary_path"])
    assert path.parent == Path(result["output"])
    return json.loads(path.read_text(encoding="utf-8"))


def metric_events(summary):
    return [event for event in summary["events"] if event["reason_code"] == "metric_below_target"]


def test_verified_failed_metric_recorded_once_and_rules_unchanged(tmp_path):
    book = playbook()
    before = book.model_dump_json()
    result = asyncio.run(run_review(config(), review_request([metric()]), tmp_path, playbook=book))
    assert result["acceptance_status"] == "optimizing"
    events = metric_events(read_summary(result))
    assert len(events) == 1
    event = events[0]
    assert event["stage"] == "evaluation" and event["source"] == "evaluator"
    assert event["occurrence_key"] == "metric:wall-recall"
    assert event["cause_status"] == "observed"
    assert event["hypothesis"] is None
    assert event["context"]["value"] == .6
    assert event["context"]["threshold"] == .9
    assert event["context"]["reference_verified"] is True
    assert "evidence:reviewed-frame" in event["evidence_refs"]
    assert any(ref.endswith("/evaluation.json#metrics/0") for ref in event["evidence_refs"])
    assert before == book.model_dump_json()


@pytest.mark.parametrize("changes", [
    {"reference_verified": False}, {"value": None}, {"threshold": None},
])
def test_pending_metric_is_not_a_failure(tmp_path, changes):
    result = asyncio.run(run_review(config(), review_request([metric(**changes)]), tmp_path,
                                    playbook=playbook()))
    assert result["acceptance_status"] == "pending_review"
    assert metric_events(read_summary(result)) == []


def test_missing_evidence_never_becomes_failure_even_when_reflection_is_rejected(tmp_path):
    # The existing reflector validation rejects an insight with empty evidence.
    # That runtime rejection must not turn the evaluator's pending metric into fail.
    with pytest.raises(ValueError, match="Reflection lacks valid metric/evidence"):
        asyncio.run(run_review(config(), review_request([metric(evidence_ids=[])]), tmp_path,
                               playbook=playbook()))
    folder = next(tmp_path.glob("review_*"))
    evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "failure_summary.json").read_text(encoding="utf-8"))
    assert evaluation["metrics"][0]["status"] == "pending"
    assert metric_events(summary) == []


def test_optional_failure_is_recorded_without_changing_acceptance(tmp_path):
    request = review_request([metric(id="required-pass", value=.95),
                              metric(id="optional-fail", required=False)])
    result = asyncio.run(run_review(config(), request, tmp_path, playbook=playbook()))
    assert result["acceptance_status"] == "pass"
    events = metric_events(read_summary(result))
    assert len(events) == 1 and events[0]["context"]["metric_id"] == "optional-fail"


def test_same_metric_in_separate_jobs_counts_separate_occurrences(tmp_path):
    results = [asyncio.run(run_review(config(), review_request([metric()]), tmp_path,
                                     playbook=playbook())) for _ in range(2)]
    assert all(len(metric_events(read_summary(result))) == 1 for result in results)
    global_summary = FailureMemory(tmp_path).summary()
    events = metric_events(global_summary)
    assert len(events) == 2
    assert len({event["job_id"] for event in events}) == 2
    reason = next(row for row in global_summary["reasons"] if row["reason_code"] == "metric_below_target")
    assert reason["count"] == 2 and reason["affected_jobs"] == 2 and reason["repeat_count"] == 1


def test_reflector_hypothesis_is_preserved_without_becoming_failure_cause(tmp_path):
    result = asyncio.run(run_review(config(), review_request([metric()]), tmp_path,
                                    playbook=playbook()))
    summary = read_summary(result)
    assert len(metric_events(summary)) == 1
    insight = summary["insights"][0]
    assert insight["hypothesis"]
    assert insight["status"] == "proposal"
    assert insight["cause_status"] == "hypothesis" and insight["confirmed"] is False
    assert metric_events(summary)[0]["cause_status"] == "observed"
    assert metric_events(summary)[0]["hypothesis"] is None


def test_many_evidence_references_keep_complete_request_snapshot(tmp_path):
    data = review_request([metric()]).model_dump()
    evidence_ids = [f"frame-{index}" for index in range(70)]
    data["evidence"] = {key: "fixture evidence" for key in evidence_ids}
    data["metrics"][0]["evidence_ids"] = evidence_ids
    result = asyncio.run(run_review(config(), ReviewRequest.model_validate(data), tmp_path,
                                    playbook=playbook()))
    events = metric_events(read_summary(result))
    assert len(events) == 1
    assert events[0]["context"]["evidence_reference_count"] == 70
    request_snapshot = json.loads((Path(result["output"]) / "request.json").read_text(encoding="utf-8"))
    assert request_snapshot["metrics"][0]["evidence_ids"] == evidence_ids


def test_evaluation_failure_survives_vlm_exception(tmp_path, monkeypatch):
    async def unavailable(*args, **kwargs):
        raise GatewayError("Provider unavailable")
    monkeypatch.setattr(ModelGateway, "invoke", unavailable)
    with pytest.raises(GatewayError, match="unavailable"):
        asyncio.run(run_review(config(), review_request([metric()]), tmp_path,
                               playbook=playbook()))
    folder = next(tmp_path.glob("review_*"))
    summary = json.loads((folder / "failure_summary.json").read_text(encoding="utf-8"))
    assert len(metric_events(summary)) == 1
    assert summary["insights"] == []
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["error"] == "Review rejected; inspect sanitized trace. No automatic retry."


@pytest.mark.parametrize("failure_kind,reason_code", [
    ("timeout", "api_timeout"), ("invalid_json", "model_output_invalid"),
])
@pytest.mark.parametrize("verified,quality_count", [(True, 1), (False, 0)])
def test_runtime_and_evaluator_failures_are_distinct(tmp_path, monkeypatch, failure_kind,
                                                     reason_code, verified, quality_count):
    monkeypatch.setenv("TEST_FACADE_ENDPOINT", "https://fixture.invalid/chat/completions")
    monkeypatch.setenv("TEST_FACADE_KEY", "fixture-key")

    def handler(request):
        if failure_kind == "timeout":
            raise httpx.ReadTimeout("Fixture provider timeout", request=request)
        return httpx.Response(200, json={
            "model":"fixture-version",
            "choices": [{"finish_reason": "stop", "message": {"content": "not valid JSON"}}],
        })

    with pytest.raises(GatewayError):
        asyncio.run(run_review(config(True), review_request([metric(reference_verified=verified)]),
                               tmp_path, playbook=playbook(), allow_api=True,
                               transport=httpx.MockTransport(handler)))
    folder = next(tmp_path.glob("review_*"))
    summary = json.loads((folder / "failure_summary.json").read_text(encoding="utf-8"))
    assert len(metric_events(summary)) == quality_count
    runtime_events = [event for event in summary["events"] if event["kind"] == "runtime"]
    assert len(runtime_events) == 1
    assert runtime_events[0]["reason_code"] == reason_code
    assert runtime_events[0]["stage"] == "review_generator"
    assert summary["totals"]["events"] == quality_count + 1
    evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["metrics"][0]["status"] == ("fail" if verified else "pending")
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["failure_recorded"] is True and state["failure_reason_code"] == reason_code
    trace = json.loads((folder / "trace.json").read_text(encoding="utf-8"))
    assert trace[-1]["error_code"] == reason_code


def test_runtime_logging_failure_cannot_replace_original_exception(tmp_path, monkeypatch):
    async def unavailable(*args, **kwargs):
        raise GatewayError("Original provider failure")

    def broken_log(*args, **kwargs):
        raise OSError("Failure log unavailable")

    monkeypatch.setattr(ModelGateway, "invoke", unavailable)
    monkeypatch.setattr("facade_mcp.review_cycle.record_runtime_failure", broken_log)
    with pytest.raises(GatewayError, match="Original provider failure"):
        asyncio.run(run_review(config(), review_request([metric()]), tmp_path,
                               playbook=playbook()))
    folder = next(tmp_path.glob("review_*"))
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["failure_recorded"] is False
    assert state["error"] == "Review rejected; inspect sanitized trace. No automatic retry."
