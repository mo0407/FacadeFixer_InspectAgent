"""Image workflow contract tests; no weights, network, or paid inference needed."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from facade_mcp.image_demo import run_image_demo
from facade_mcp.image_inspection import (
    ImageInspectionRequest, crop_box, execute_image_inspection, merge_candidates,
)
from facade_mcp.model_gateway import ModelConfig, ModelGateway
from facade_mcp.playbook import Playbook, Rule
from facade_mcp.storage import ArtifactStore, sha256


def row(code="C", box=(10, 10, 30, 30), confidence=.9, **extra):
    return dict(class_code=code, class_name="fixture " + code,
                bbox_xyxy=list(box), confidence=confidence, **extra)


class FakeExperts:
    """Preserve source hashes/dimensions just as the MCP detector must."""

    def __init__(self, full=None, local=None, hook=None, method="test-fixture"):
        self.full = full or []
        self.local = local or []
        self.hook = hook
        self.method = method
        self.calls = []

    async def call(self, name, arguments):
        assert name == "facade_models_detect"
        path = Path(arguments["image_paths"][0])
        with Image.open(path) as image:
            pixels = np.asarray(image).copy()
            width, height = image.size
        self.calls.append(dict(arguments=copy.deepcopy(arguments), path=path, pixels=pixels))
        if self.hook:
            self.hook(self.calls)
        rows = self.local if path.name.startswith("local-") else self.full
        if callable(rows):
            rows = rows(self.calls[-1])
        return dict(status="completed", images=[dict(
            source_sha256=sha256(path), width=width, height=height,
            defects=copy.deepcopy(rows), unmapped_defects=[])],
            source=dict(method=self.method, model_profile=arguments["model_profile"]))


class FakeVLM:
    """Stage-driven fake matching the gateway's structured async interface."""

    def __init__(self, actions=None, final=None, scope="wall"):
        self.actions = list(actions or [])
        self.final = final
        self.scope = scope
        self.calls = []

    async def invoke(self, role, task, schema, **kwargs):
        assert role == "generator"
        assert kwargs["image_paths"]
        assert all(Path(path).is_file() for path in kwargs["image_paths"])
        self.calls.append(copy.deepcopy(task))
        if task["stage"] == "visual_anchoring":
            return schema.model_validate({'regions':[{'bbox_normalized':[0,0,1,1], 'role':'context',
                'description':'Synthetic full-image context; no semantic wall assertion.'}], 'summary':'Fixture context'})
        if task["stage"] == "choose_inspection":
            output = self.actions.pop(0) if self.actions else dict(action="finish", reason="Finished fixture")
        else:
            assert task["stage"] == "final_verification"
            output = dict(summary="Fixture visual review", decisions=[dict(
                candidate_id=c["candidate_id"], decision="keep", scope_decision=self.scope,
                reason="Fixture evidence supports this decision") for c in task["candidates"]])
            if self.final:
                output = self.final(task, output)
        return schema.model_validate(output)


def action(anchor="grid-1-1", profile="yolo11m", scale=1):
    return dict(action="inspect", anchor_id=anchor, model_profile=profile,
                scale=scale, reason="Inspect fixture pixels")


def setup(tmp_path, *, mask=None, **request_options):
    # Unique x/y channels make accidental resizing or crop displacement detectable.
    yy, xx = np.indices((96, 120))
    pixels = np.stack((xx, yy, (xx + yy) % 256), axis=-1).astype(np.uint8)
    source = tmp_path / "facade.png"
    Image.fromarray(pixels).save(source)
    mask_path = None
    if mask is not None:
        mask_path = tmp_path / "scope.png"
        Image.fromarray(mask.astype(np.uint8)).save(mask_path)
    store = ArtifactStore(tmp_path / "artifacts", [tmp_path])
    options = dict(image_path=str(source), detector_profiles=["yolo11m"], max_inspections=0)
    if mask_path:
        options["scope_mask_path"] = str(mask_path)
    options.update(request_options)
    request = ImageInspectionRequest(**options)
    _, folder = store.create_job("fixture", request.model_dump())
    return request, store, folder, pixels, source


def execute(request, store, folder, experts, vlm=None, playbook=None):
    return asyncio.run(execute_image_inspection(
        request, store, experts, vlm or FakeVLM(), playbook or Playbook(version="P0-test"), folder))


def model_config(adapter="mock", *, api_enabled=False):
    connection = dict(adapter=adapter, model="fixture", supports_images=True)
    if adapter != "mock":
        connection.update(endpoint_env="NO_NETWORK_ENDPOINT", api_key_env="NO_NETWORK_KEY")
    return ModelConfig.model_validate(dict(
        schema_version="model-gateway-v1", api_enabled=api_enabled,
        connections={"fixture": connection},
        roles={role: "fixture" for role in ("generator", "reflector", "curator")}))


def test_original_pixel_crop_and_offset_are_exact(tmp_path):
    request, store, folder, pixels, original = setup(
        tmp_path, max_inspections=1, grid_rows=2, grid_cols=2)
    original_hash = sha256(original)
    experts = FakeExperts(local=[row(box=(2, 3, 20, 22))])
    result = execute(request, store, folder, experts,
                     FakeVLM(actions=[action("grid-2-2")]))
    assert crop_box([60, 48, 120, 96], 1, 120, 96) == [60, 48, 120, 96]
    assert crop_box([1, 1, 11, 11], 3, 120, 96) == [0, 0, 21, 21]
    assert np.array_equal(experts.calls[1]["pixels"], pixels[48:96, 60:120])
    candidate = result["candidates"][0]
    assert candidate["bbox_xyxy"] == [62, 51, 80, 70]
    assert candidate["local_bbox_xyxy"] == [2, 3, 20, 22]
    assert candidate["crop_origin_xy"] == [60, 48]
    assert result["inspections"][0]["crop_bbox_original"] == [60, 48, 120, 96]
    assert result["original_input_sha256"] == original_hash == sha256(original)
    assert sha256(folder / "input.png") == original_hash
    assert result["source_sha256"] == sha256(folder / "source.png")
    with Image.open(folder / "source.png") as snapshot:
        assert np.array_equal(np.asarray(snapshot), pixels)


def test_same_class_nms_preserves_members_but_never_merges_cross_class_or_unknown():
    observations = [dict(observation_id=name, class_code=code, bbox_xyxy=box, confidence=score)
        for name, code, box, score in [
            ("low", "C", [11, 11, 31, 31], .6),
            ("high", "C", [10, 10, 30, 30], .9),
            ("other-class", "S", [10, 10, 30, 30], .8),
            ("unknown-a", "unknown", [10, 10, 30, 30], .7),
            ("unknown-b", "unknown", [10, 10, 30, 30], .5)]]
    merged = merge_candidates(observations, .5)
    assert len(merged) == 4
    crack = next(c for c in merged if c["class_code"] == "C")
    assert crack["bbox_xyxy"] == [10, 10, 30, 30]
    assert [m["observation_id"] for m in crack["members"]] == ["high", "low"]
    assert len([c for c in merged if c["class_code"] == "unknown"]) == 2
    assert len(observations) == 5 and all("members" not in c for c in observations)


def test_p_s_canonical_mapping_keeps_every_original_class(tmp_path):
    request, store, folder, _, _ = setup(tmp_path)
    experts = FakeExperts(full=[
        row("P", confidence=.9, original_class_name="Original peeling"),
        row("S", confidence=.8, original_class_code="S", original_class_name="Original spalling"),
        row("S", confidence=.7, original_class_code="P", original_class_name="Upstream normalized peeling"),
    ])
    result = execute(request, store, folder, experts)
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["class_code"] == "S"
    assert [m["source_class_code"] for m in candidate["members"]] == ["P", "S", "P"]
    assert [m["source_class_name"] for m in candidate["members"]] == [
        "Original peeling", "Original spalling", "Upstream normalized peeling"]
    persisted = json.loads((folder / "observations.json").read_text(encoding="utf-8"))
    assert [c["source_class_code"] for c in persisted] == ["P", "S", "P"]


def test_inspection_budget_stops_at_two_even_if_vlm_wants_more(tmp_path):
    request, store, folder, _, _ = setup(tmp_path, max_inspections=2)
    experts = FakeExperts()
    vlm = FakeVLM(actions=[action("grid-1-1"), action("grid-1-2"), action("grid-1-3")])
    result = execute(request, store, folder, experts, vlm)
    assert result["inspections_used"] == 2
    assert len(experts.calls) == 3  # One full-image call, then two local calls.
    assert [c["remaining_inspections"] for c in vlm.calls if c["stage"] == "choose_inspection"] == [2, 1]
    assert len(vlm.actions) == 1
    with pytest.raises(ValidationError):
        ImageInspectionRequest(image_path="fixture.png", max_inspections=3)


@pytest.mark.parametrize("invalid", [action("made-up-anchor"), action(profile="yolo12m")])
def test_invalid_anchor_or_noneligible_expert_is_rejected_before_local_call(tmp_path, invalid):
    request, store, folder, _, _ = setup(tmp_path, max_inspections=2)
    experts = FakeExperts()
    with pytest.raises(ValueError, match="Invalid anchor or expert profile"):
        execute(request, store, folder, experts, FakeVLM(actions=[invalid]))
    assert len(experts.calls) == 1


def test_duplicate_inspection_action_is_rejected(tmp_path):
    request, store, folder, _, _ = setup(tmp_path, max_inspections=2)
    experts = FakeExperts()
    with pytest.raises(ValueError, match="Duplicate inspection action"):
        execute(request, store, folder, experts, FakeVLM(actions=[action(), action()]))
    assert len(experts.calls) == 2


def test_final_review_decides_every_candidate_once_across_batches(tmp_path):
    request, store, folder, _, _ = setup(tmp_path)
    rows = [row(box=(i * 20 + 1, 10, i * 20 + 10, 20), confidence=.99 - i * .01) for i in range(5)]
    vlm = FakeVLM()
    result = execute(request, store, folder, FakeExperts(full=rows), vlm)
    final_calls = [c for c in vlm.calls if c["stage"] == "final_verification"]
    assert [len(c["candidates"]) for c in final_calls] == [4, 1]
    assert [c["batch"] for c in final_calls] == [1, 2]
    assert all(c["batch_count"] == 2 for c in final_calls)
    ids = [c["candidate_id"] for call in final_calls for c in call["candidates"]]
    assert len(ids) == len(set(ids)) == 5
    assert {c["candidate_id"] for c in result["candidates"]} == set(ids)
    assert result["counts"]["keep"] == 5
    for call in final_calls:
        assert set(call["candidate_evidence"]) == {c["candidate_id"] for c in call["candidates"]}
        assert all(e["native_resolution"] for e in call["candidate_evidence"].values())


@pytest.mark.parametrize("fault", ["unknown", "missing", "duplicate"])
def test_final_review_rejects_invented_omitted_or_duplicate_ids(tmp_path, fault):
    request, store, folder, _, _ = setup(tmp_path)

    def faulty_final(task, output):
        if fault == "unknown":
            output["decisions"][0]["candidate_id"] = "candidate-invented"
        elif fault == "missing":
            output["decisions"].pop()
        else:
            output["decisions"].append(copy.deepcopy(output["decisions"][0]))
        return output

    with pytest.raises(ValueError, match="every batch candidate exactly once"):
        execute(request, store, folder, FakeExperts(full=[row()]), FakeVLM(final=faulty_final))
    assert not (folder / "result.json").exists()


def test_partial_and_outside_scope_mask_cannot_be_kept(tmp_path):
    mask = np.zeros((96, 120), dtype=np.uint8)
    mask[:, :60] = 255
    request, store, folder, _, _ = setup(tmp_path, mask=mask)
    result = execute(request, store, folder, FakeExperts(full=[
        row(box=(10, 10, 30, 30), confidence=.9),
        row(box=(50, 10, 70, 30), confidence=.8),
        row(box=(80, 10, 100, 30), confidence=.7)]))
    by_scope = {c["scope"]["status"]: c for c in result["candidates"]}
    assert by_scope["inside"]["decision"] == "keep"
    assert by_scope["partial"]["scope"]["wall_fraction"] == .5
    for status in ("partial", "outside"):
        assert by_scope[status]["decision"] == "defer"
        assert by_scope[status]["proposed_decision"] == "keep"
    assert result["scope_verified_by_mask"] is True


@pytest.mark.parametrize("visual_scope", ["uncertain", "excluded"])
def test_unclear_or_excluded_visual_scope_cannot_be_kept(tmp_path, visual_scope):
    request, store, folder, _, _ = setup(tmp_path)
    result = execute(request, store, folder, FakeExperts(full=[row()]), FakeVLM(scope=visual_scope))
    candidate = result["candidates"][0]
    assert candidate["scope"]["status"] == "not_reviewed"
    assert candidate["visual_scope_decision"] == visual_scope
    assert candidate["decision"] == "defer" and candidate["proposed_decision"] == "keep"


def test_unknown_class_cannot_be_kept_even_when_vlm_says_wall(tmp_path):
    request, store, folder, _, _ = setup(tmp_path)
    result = execute(request, store, folder, FakeExperts(full=[row("unmapped-checkpoint-id")]))
    assert result["candidates"][0]["class_code"] == "unknown"
    assert result["candidates"][0]["decision"] == "defer"


def test_p0_is_a_deep_frozen_snapshot_not_the_callers_live_object(tmp_path):
    request, store, folder, _, _ = setup(tmp_path, max_inspections=1)
    playbook = Playbook(version="P0-original", rules=[Rule(
        id="wall-only", condition="wall", recommendation="Original recommendation",
        evidence_ids=["original-evidence"])])
    expected = playbook.model_dump(mode="json")
    expected_hash = hashlib.sha256(playbook.model_dump_json().encode()).hexdigest()

    def mutate_original(calls):
        if len(calls) == 1:
            playbook.version = "P0-mutated"
            playbook.rules[0].recommendation = "Changed externally"
            playbook.rules[0].evidence_ids.append("new-evidence")

    vlm = FakeVLM()
    result = execute(request, store, folder, FakeExperts(full=[row()], hook=mutate_original), vlm, playbook)
    assert playbook.version == "P0-mutated"
    assert all(call["p0"] == expected for call in vlm.calls)
    assert json.loads((folder / "playbook.json").read_text(encoding="utf-8")) == expected
    assert result["playbook_sha256"] == expected_hash
    assert result["playbook_version"] == "P0-original" and result["playbook_updated"] is False


def test_mock_vlm_marks_even_expert_backed_results_as_demo(tmp_path):
    request, store, folder, _, _ = setup(tmp_path)
    # The expert reports an inference-style source; only VLM review is mocked.
    experts = FakeExperts(full=[row()], method="ultralytics-test-double")
    result = execute(request, store, folder, experts, ModelGateway(model_config()))
    assert result["demo"] is True
    assert result["acceptance_status"] == "pending_review" and result["expert_validated"] is False
    assert result["counts"] == {"keep": 0, "reject": 0, "defer": 1}
    assert "演示 / mock" in (folder / "report.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("api_enabled,allow_api", [(False, False), (True, False), (False, True)])
def test_missing_api_opt_in_fails_before_expert_execution(tmp_path, api_enabled, allow_api):
    request, store, folder, _, _ = setup(tmp_path)
    experts = FakeExperts(full=[row()])
    vlm = ModelGateway(model_config("responses", api_enabled=api_enabled), allow_api=allow_api)
    with pytest.raises(ValueError, match="requires api_enabled and --allow-api"):
        execute(request, store, folder, experts, vlm)
    assert experts.calls == [] and vlm.trace == []
    assert not (folder / "source.png").exists()


def test_one_grid_mock_finishes_without_repeating_inspection(tmp_path):
    request, store, folder, _, _ = setup(tmp_path, grid_rows=1, grid_cols=1, max_inspections=2)
    experts = FakeExperts(full=[row()])
    gateway = ModelGateway(model_config())
    result = execute(request, store, folder, experts, gateway)
    assert result["inspections_used"] == 1
    assert [step["action"] for step in result["trajectory"]] == ["inspect", "finish"]
    assert len(experts.calls) == 2
    assert len(gateway.trace) == 4
    assert all(t["status"] == "completed" and t["images"] for t in gateway.trace)


def test_synthetic_demo_produces_auditable_report_and_artifacts(tmp_path):
    store = ArtifactStore(tmp_path / "workspace", [tmp_path])
    result = asyncio.run(run_image_demo(tmp_path / "demo-inputs", store))
    assert result["demo"] is True and result["inspections_used"] == 2
    assert result["counts"] == {"keep": 0, "reject": 0, "defer": 2}
    assert {c["class_code"] for c in result["candidates"]} == {"C", "S"}
    assert all(len(c["members"]) == 2 for c in result["candidates"])
    folder = Path(result["report_path"]).parent
    assert (folder / "report.html").is_file()
    assert (folder / "synthetic_tool_calls.json").is_file()
    anchors = json.loads((folder / "anchors.json").read_text(encoding="utf-8"))["anchors"]
    assert {a["kind"] for a in anchors} == {"Box", "Ghost", "Grid", "Env"}
    trace = json.loads((folder / "model_trace.json").read_text(encoding="utf-8"))
    assert len(trace) == 3 and all(c["adapter"] == "mock" for c in trace)
    assert store.job_status(result["job_id"])["status"] == "completed"
    assert {"report", "result", "anchors", "trace", "playbook"} <= set(result["artifacts"])
