"""Bounded controller; execution and measurement are explicit injected adapters."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable
from uuid import uuid4

from .model_gateway import ModelConfig
from .optimization_dashboard import RoundDetails, render_dashboard
from .review_cycle import ReviewRequest, evaluate, run_review
from .playbook import Playbook


def contract(request: ReviewRequest):
    return (request.task, request.acceptance_version, request.playbook_version,
            sorted((m.id, m.threshold, m.direction, m.required) for m in request.metrics))


@dataclass(frozen=True)
class RoundResult:
    request: ReviewRequest
    details: RoundDetails


async def run_optimization(
    config: ModelConfig, initial: ReviewRequest, output: Path,
    execute_and_measure: Callable[[dict, ReviewRequest], Awaitable[ReviewRequest | RoundResult]],
    *, max_experiments: int, allow_api=False, transport=None,
    initial_details: RoundDetails | None = None,
    playbook: Playbook | None = None,
):
    """Run adapter, remeasure, gate. Never execute model-proposed shell commands.

    The caller's adapter must implement allowed changes and independent evaluation.
    A missing reference blocks execution; the numeric target contract is immutable.
    """
    if isinstance(max_experiments, bool) or not isinstance(max_experiments, int) or not 1 <= max_experiments <= 20:
        raise ValueError("max_experiments must be an integer between 1 and 20")
    snapshot = (playbook or Playbook(version=initial.playbook_version)).model_copy(deep=True)
    if snapshot.version != initial.playbook_version:
        raise ValueError('Playbook version differs from frozen run')
    folder = output / ("optimization_" + uuid4().hex)
    folder.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "experiments_executed": 0,
             "experience_promoted": False, "history": [], "output": str(folder.resolve())}
    fixed_contract = contract(initial)
    current = initial.model_copy(deep=True)
    details = initial_details

    def save():
        path = folder / "state.json"
        temporary = folder / "state.tmp"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        dashboard = folder / 'dashboard.tmp'
        dashboard.write_text(render_dashboard(state, folder, snapshot.model_dump()), encoding='utf-8')
        dashboard.replace(folder / 'dashboard.html')

    try:
        while True:
            evaluation = evaluate(current)
            entry = {"experiment": state["experiments_executed"],
                     "evaluation": evaluation, "request": current.model_dump()}
            if details is not None:
                entry["details"] = details.model_dump(mode="json")
            state["history"].append(entry)
            required_pending = any(r["required"] and r["status"] == "pending" for r in evaluation["metrics"])
            if required_pending:
                state.update(status="pending_review", stop_reason="missing_reference_value_or_threshold")
                break
            if evaluation["status"] == "pass":
                state.update(status="targets_met", stop_reason="all_required_supplied_metrics_pass")
                break
            if state["experiments_executed"] >= max_experiments:
                state.update(status="budget_exhausted", stop_reason="targets_not_met_within_experiment_limit")
                break
            save()
            review = await run_review(config, current, folder, allow_api=allow_api, transport=transport,
                                      playbook=snapshot)
            review_path = Path(review["output"])
            proposal = {"plan": json.loads((review_path / "plan.json").read_text(encoding="utf-8")),
                        "reflection": json.loads((review_path / "reflection.json").read_text(encoding="utf-8")),
                        "review_output": str(review_path), "run_folder": str(folder.resolve())}
            state["history"][-1]["review_output"] = str(review_path)
            state["status"] = "executing"
            save()
            updated = await execute_and_measure(proposal, current.model_copy(deep=True))
            state["experiments_executed"] += 1
            details = updated.details if isinstance(updated, RoundResult) else None
            if isinstance(updated, RoundResult):
                updated = updated.request
            # Revalidate even if an adapter constructs an unchecked Pydantic copy.
            updated = ReviewRequest.model_validate(updated.model_dump())
            if contract(updated) != fixed_contract:
                raise ValueError("Execution adapter changed the frozen acceptance contract")
            current = updated
            state["status"] = "running"
    except Exception:
        state.update(status="failed", stop_reason="adapter_or_review_failed_no_automatic_retry")
        raise
    finally:
        save()
    return state
