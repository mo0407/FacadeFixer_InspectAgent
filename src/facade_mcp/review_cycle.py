"""One bounded G/R/C review cycle over supplied evaluation data, not video inference."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from .failure_memory import FailureMemory
from .image_failures import progress, record_runtime_failure
from .model_gateway import ModelConfig, ModelGateway, Strict
from .playbook import Playbook, RuleFeedback, build_delta
from .storage import ArtifactStore


Stage = Literal["scope", "sampling", "detection", "association", "segmentation", "report"]


class Metric(Strict):
    id: str
    value: float | None = None
    threshold: float | None = None
    direction: Literal["min", "max"] = "min"
    required: bool = True
    evidence_ids: list[str] = Field(default_factory=list)
    reference_verified: bool = False

    @model_validator(mode="after")
    def finite(self):
        if any(x is not None and not math.isfinite(x) for x in [self.value, self.threshold]):
            raise ValueError("Metric values must be finite")
        return self


class ReviewRequest(Strict):
    task: str
    evidence: dict[str, str]
    metrics: list[Metric]
    acceptance_version: str
    playbook_version: str

    @model_validator(mode="after")
    def references(self):
        if len({m.id for m in self.metrics}) != len(self.metrics):
            raise ValueError("Metric IDs must be unique")
        if not self.metrics or not any(m.required for m in self.metrics):
            raise ValueError("At least one required metric is needed")
        for m in self.metrics:
            if any(k not in self.evidence for k in m.evidence_ids):
                raise ValueError("Unknown evidence ID")
        return self


class Plan(Strict):
    stages: list[Stage]
    rationale: str
    used_rule_ids: list[str] = Field(default_factory=list)


class Insight(Strict):
    metric_id: str
    evidence_ids: list[str]
    hypothesis: str
    proposed_stage: Stage
    controlled_change: str


class Reflection(Strict):
    insights: list[Insight]
    rule_feedback: list[RuleFeedback] = Field(default_factory=list)


class Experience(Strict):
    action: Literal["add", "revise", "deprecate"]
    condition: str
    recommendation: str
    evidence_ids: list[str]
    limitations: str
    target_rule_id: str | None = None


class Curation(Strict):
    candidates: list[Experience]


def evaluate(request: ReviewRequest):
    rows = []
    for m in request.metrics:
        if m.threshold is None or m.value is None or not m.reference_verified or not m.evidence_ids:
            status = "pending"
        else:
            ok = m.value >= m.threshold if m.direction == "min" else m.value <= m.threshold
            status = "pass" if ok else "fail"
        rows.append(dict(metric_id=m.id, value=m.value, threshold=m.threshold,
                         status=status, required=m.required))
    required = [r["status"] for r in rows if r["required"]]
    status = "optimizing" if "fail" in required else "pending_review" if "pending" in required else "pass"
    return {"status": status, "metrics": rows,
            "scope": "supplied metric values only; not independently measured by this command"}


async def run_review(config: ModelConfig, request: ReviewRequest, output: Path, *, allow_api=False, transport=None,
                     playbook: Playbook | None = None):
    snapshot = (playbook or Playbook(version=request.playbook_version)).model_copy(deep=True)
    if snapshot.version != request.playbook_version:
        raise ValueError('Playbook version differs from request')
    folder = output / ("review_" + uuid4().hex)
    folder.mkdir(parents=True, exist_ok=False)
    gateway = ModelGateway(config, allow_api=allow_api, transport=transport)
    failures = FailureMemory(output)
    failure_insights = []
    review_failed = False
    evaluation = evaluate(request)
    state = dict(status="running", mode="review_only_no_video_execution",
                 evaluation=evaluation, acceptance_version=request.acceptance_version,
                 playbook_version=request.playbook_version, api_enabled=config.api_enabled,
                 failure_summary_path=str((folder / "failure_summary.json").resolve()))
    def save(name, value):
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        save("request.json", request.model_dump())
        save("config.json", config.model_dump())
        save("playbook_snapshot.json", snapshot.model_dump())
        save("evaluation.json", evaluation)
        # Only evaluator-confirmed failures become occurrences. Incomplete or
        # unverified metrics stay pending, regardless of later model opinions.
        for index, (metric, measured) in enumerate(zip(request.metrics, evaluation["metrics"])):
            if measured["status"] != "fail":
                continue
            comparison = ">=" if metric.direction == "min" else "<="
            # The ledger accepts 64 references; the request snapshot preserves
            # the complete list when a metric cites many reviewed images.
            failures.record(
                job_id=folder.name, stage="evaluation", reason_code="metric_below_target",
                occurrence_key="metric:" + metric.id,
                summary=f"Metric {metric.id} missed target: value={metric.value}; required {comparison} {metric.threshold}.",
                evidence_refs=[f"{folder.name}/request.json#metrics/{index}",
                               f"{folder.name}/evaluation.json#metrics/{index}",
                               *["evidence:" + evidence_id for evidence_id in metric.evidence_ids[:62]]],
                context={"metric_id": metric.id, "value": metric.value,
                         "threshold": metric.threshold, "direction": metric.direction,
                         "acceptance_version": request.acceptance_version,
                         "playbook_version": snapshot.version,
                         "reference_verified": metric.reference_verified,
                         "evidence_reference_count": len(metric.evidence_ids),
                         "complete_evidence_list_ref": f"{folder.name}/request.json#metrics/{index}/evidence_ids"},
                source="evaluator", cause_status="observed", hypothesis=None)
        task = {"request": request.model_dump(), "evaluation": evaluation,
                "playbook": snapshot.model_dump(),
                "instruction": "Propose only; metrics are supplied, images have not been provided. "
                "Generator must cite only actually used rule IDs. Reflector may tag cited rules "
                "helpful/harmful/neutral with evidence. Curator emits small add/revise/deprecate deltas, "
                "using target_rule_id for revise/deprecate. Preserve unrelated rules and acceptance thresholds."}
        progress(folder, "review_generator", playbook_version=snapshot.version)
        plan = await gateway.invoke("generator", task, Plan, mock_output={
            "stages": ["scope", "detection", "segmentation", "report"],
            "rationale": "Mock plan: demonstrate structured review without executing video tools."})
        save("plan.json", plan.model_dump())
        build_delta(snapshot, [], [], plan.used_rule_ids, set(request.evidence))
        pending = [m for m in request.metrics if next(r for r in evaluation["metrics"] if r["metric_id"] == m.id)["status"] != "pass"]
        progress(folder, "review_reflector", playbook_version=snapshot.version)
        reflection = await gateway.invoke("reflector", {**task, "plan": plan.model_dump()}, Reflection,
            mock_output={"insights": [dict(metric_id=m.id, evidence_ids=m.evidence_ids,
                hypothesis="Missing acceptance evidence or failed supplied metric; verify before changing models.",
                proposed_stage="scope", controlled_change="Review target-building scope annotations.") for m in pending]})
        metric_ids = {m.id for m in request.metrics}
        for insight in reflection.insights:
            if insight.metric_id not in metric_ids or not insight.evidence_ids or any(k not in request.evidence for k in insight.evidence_ids):
                raise ValueError("Reflection lacks valid metric/evidence references")
        # Reflections propose explanations; they neither create failures nor
        # overwrite the evaluator's observed reason or its occurrence count.
        failure_insights = [{**insight.model_dump(), "source": "reflector", "status": "proposal",
                             "cause_status": "hypothesis", "confirmed": False}
                            for insight in reflection.insights]
        save("reflection.json", reflection.model_dump())
        build_delta(snapshot, [], reflection.rule_feedback, plan.used_rule_ids, set(request.evidence))
        progress(folder, "review_curator", playbook_version=snapshot.version)
        curation = await gateway.invoke("curator", {**task, "reflection": reflection.model_dump()}, Curation,
            mock_output={"candidates": [dict(action="add", condition="Building identity is uncertain",
                recommendation="Resolve scope before accepting facade findings.",
                evidence_ids=list(request.evidence)[:1], limitations="Mock candidate; no measured improvement or independent validation.")] if request.evidence else []})
        for item in curation.candidates:
            if not item.evidence_ids or any(k not in request.evidence for k in item.evidence_ids):
                raise ValueError("Experience lacks valid evidence references")
        delta = build_delta(snapshot, curation.candidates, reflection.rule_feedback,
                            plan.used_rule_ids, set(request.evidence))
        save("context_delta.json", delta)
        save("trajectory.json", {
            "schema": "facade-review-trajectory/v1",
            "acceptance_version": request.acceptance_version,
            "playbook_version": snapshot.version,
            "used_rule_ids": plan.used_rule_ids,
            "plan": plan.model_dump(), "evaluation": evaluation,
            "reflection": reflection.model_dump(),
            "context_delta": "context_delta.json",
        })
        save("candidate_experience.json", {"status": "candidate_only", "promoted": False,
             "mock_generated": any(c.adapter == "mock" for c in config.connections.values()),
             "parent_playbook_version": request.playbook_version, **curation.model_dump()})
        state.update(status="review_completed", acceptance_status=evaluation["status"],
                     report_generated=False, experiments_executed=0, experience_promoted=False)
    except Exception as error:
        review_failed = True
        state.update(status="failed", error="Review rejected; inspect sanitized trace. No automatic retry.")
        try:
            failure = record_runtime_failure(ArtifactStore(output), folder, error, gateway.trace)
            state.update(failure_reason_code=failure["reason_code"], failure_recorded=True,
                         failure_snapshots_written=failure["snapshots_written"])
        except Exception:
            # Failure telemetry is secondary to the original review exception.
            state["failure_recorded"] = False
        raise
    finally:
        try:
            save("trace.json", gateway.trace)
            failures.publish_summary(folder/"failure_summary.json",job_id=folder.name,
                                     extra={"insights":failure_insights})
            save("state.json", state)
        except Exception:
            if not review_failed:
                raise
    return {**state, "output": str(folder.resolve())}
