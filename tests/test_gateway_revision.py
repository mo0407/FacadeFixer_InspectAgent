"""Identity/reasoning/vision checks using synthetic pixels and MockTransport only."""
import asyncio
import base64
import copy
import io
import json

import httpx
from PIL import Image
import pytest
from pydantic import ValidationError

from facade_mcp import model_gateway as module
from facade_mcp.model_gateway import (
    Connection, GatewayError, ModelConfig, ModelGateway, Strict, validate_vision_connection,
)


class Answer(Strict):
    answer: str


def config(adapter="chat_completions", **options):
    connection = dict(adapter=adapter, model="model-alias", supports_images=True, **options)
    if adapter != "mock":
        connection.update(endpoint_env="REVISION_ENDPOINT", api_key_env="REVISION_KEY")
    return ModelConfig.model_validate(dict(schema_version="model-gateway-v1", api_enabled=True,
        connections={"one": connection}, roles={role: "one" for role in ("generator", "reflector", "curator")}))


@pytest.fixture(autouse=True)
def test_credentials(monkeypatch):
    monkeypatch.setenv("REVISION_ENDPOINT", "https://fixture.invalid/v1/fixture")
    monkeypatch.setenv("REVISION_KEY", "never-persist-revision-key")


def response(adapter, content=None, *, model="model-revision-1"):
    content = {"answer": "ok"} if content is None else content
    if adapter == "responses":
        body = dict(status="completed", output=[dict(type="message", content=[
            dict(type="output_text", text=json.dumps(content))])])
    else:
        body = dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps(content)))])
    if model is not None:
        body["model"] = model
    return httpx.Response(200, json=body)


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max", "none"])
def test_all_requested_effort_values_are_supported(effort):
    assert config("responses", reasoning_effort=effort).connections["one"].reasoning_effort == effort


def test_vendor_mode_and_official_effort_cannot_be_combined():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        config("responses", reasoning_mode="standard", reasoning_effort="medium")
    with pytest.raises(ValidationError, match="responses"):
        config("chat_completions", reasoning_mode="standard")


@pytest.mark.parametrize("adapter", ["responses", "chat_completions"])
@pytest.mark.parametrize("effort", [None, "high", "none"])
def test_effort_and_token_limit_use_adapter_specific_fields(adapter, effort):
    def handle(request):
        payload = json.loads(request.content)
        if adapter == "responses":
            assert payload["max_output_tokens"] == 2000
            assert "max_tokens" not in payload and "max_completion_tokens" not in payload
            if effort is not None:
                assert payload["reasoning"] == {"effort": effort}
            else:
                assert "reasoning" not in payload
        elif effort is not None:
            assert payload["reasoning_effort"] == effort
            assert payload["max_completion_tokens"] == 2000 and "max_tokens" not in payload
        else:
            assert payload["max_tokens"] == 2000 and "max_completion_tokens" not in payload
        return response(adapter)

    gateway = ModelGateway(config(adapter, reasoning_effort=effort), allow_api=True,
                           transport=httpx.MockTransport(handle))
    assert asyncio.run(gateway.invoke("generator", {}, Answer)).answer == "ok"


@pytest.mark.parametrize("adapter", ["responses", "chat_completions"])
def test_identity_is_pinned_per_connection_across_roles(adapter):
    models = iter(["model-revision-1", "model-revision-1", "model-revision-2"])
    gateway = ModelGateway(config(adapter), allow_api=True,
        transport=httpx.MockTransport(lambda request: response(adapter, model=next(models))))

    async def run():
        await gateway.invoke("generator", {}, Answer)
        await gateway.invoke("reflector", {}, Answer)
        with pytest.raises(GatewayError, match="identity"):
            await gateway.invoke("curator", {}, Answer)

    asyncio.run(run())
    assert gateway.model_identities == {"one": "model-revision-1"}
    assert all(row["model_identity_verified"] for row in gateway.trace[:2])
    assert gateway.trace[-1]["error_code"] == "model_version_changed"
    assert gateway.trace[-1]["status"] == "failed"
    assert gateway.trace[-1]["actual_model"] == "model-revision-2"
    identities = gateway.model_identities
    identities["one"] = "caller-cannot-replace-pin"
    assert gateway.model_identities["one"] == "model-revision-1"


@pytest.mark.parametrize("actual", [None, "", " ", "unexpected-revision"])
def test_missing_or_unexpected_actual_model_is_rejected(actual):
    gateway = ModelGateway(config(expected_model="model-revision-1"), allow_api=True,
        transport=httpx.MockTransport(lambda request: response("chat_completions", model=actual)))
    with pytest.raises(GatewayError, match="identity"):
        asyncio.run(gateway.invoke("generator", {}, Answer))
    assert gateway.trace[-1]["error_code"] == "model_version_changed"
    assert gateway.model_identities == {}


def test_distinct_connections_have_independent_identity_pins():
    data = config().model_dump()
    data["connections"]["two"] = {**data["connections"]["one"], "model": "other-alias"}
    data["roles"]["reflector"] = "two"

    def handle(request):
        payload = json.loads(request.content)
        actual = "other-revision" if payload["model"] == "other-alias" else "model-revision-1"
        return response("chat_completions", model=actual)

    gateway = ModelGateway(ModelConfig.model_validate(data), allow_api=True, transport=httpx.MockTransport(handle))

    async def run():
        await gateway.invoke("generator", {}, Answer)
        await gateway.invoke("reflector", {}, Answer)

    asyncio.run(run())
    assert gateway.model_identities == {"one": "model-revision-1", "two": "other-revision"}


def test_caller_cannot_change_gateway_configuration_after_creation():
    settings = config(expected_model="model-revision-1")

    def handle(request):
        assert json.loads(request.content)["model"] == "model-alias"
        return response("chat_completions")

    gateway = ModelGateway(settings, allow_api=True, transport=httpx.MockTransport(handle))
    settings.connections["one"].model = "changed-alias"
    settings.connections["one"].expected_model = "changed-revision"
    assert asyncio.run(gateway.invoke("generator", {}, Answer)).answer == "ok"


def test_mock_does_not_claim_identity_or_vision_verified(tmp_path):
    image = tmp_path / "fixture.png"
    Image.new("RGB", (32, 32), "blue").save(image)
    gateway = ModelGateway(config("mock", expected_model="not-a-real-provider"),
        transport=httpx.MockTransport(lambda request: pytest.fail("mock must not send HTTP")))
    assert asyncio.run(gateway.invoke("generator", {}, Answer, mock_output={"answer": "fixture"},
                                      image_paths=[image])).answer == "fixture"
    assert gateway.model_identities == {}
    assert gateway.trace[0]["model_identity_verified"] is False
    assert gateway.trace[0]["vision_capability_verified"] is False


def capture_probe(monkeypatch):
    captured = {}
    original = module._create_vision_probe

    def wrapped(path):
        expected, anchors = original(path)
        captured.update(expected=expected, anchors=anchors)
        return expected, anchors

    monkeypatch.setattr(module, "_create_vision_probe", wrapped)
    return captured


def correct_probe_answer(captured, inspect=False):
    action = dict(action="finish", reason="The four probe blocks have been read")
    if inspect:
        action.update(action="inspect", anchor_id="probe-tl", model_profile="synthetic-probe", scale=2)
    return dict(blocks=copy.deepcopy(captured["expected"]), action=action)


@pytest.mark.parametrize("adapter", ["responses", "chat_completions"])
@pytest.mark.parametrize("inspect", [False, True])
def test_simulated_vision_probe_checks_pixels_numbers_and_legal_action(tmp_path, monkeypatch, adapter, inspect):
    captured = capture_probe(monkeypatch)
    calls = []

    def handle(request):
        calls.append(request)
        payload = json.loads(request.content)
        user = payload["input" if adapter == "responses" else "messages"][1]["content"]
        text = json.loads(user[0]["text"])
        assert text["stage"] == "vision_capability_check"
        assert "expected_observations" not in text and "blocks" not in text
        url = user[1]["image_url"] if adapter == "responses" else user[1]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
            assert image.size == (512, 512) and image.format == "PNG"
            assert len({image.getpixel((anchor["bbox_xyxy"][0]+4, anchor["bbox_xyxy"][1]+4))
                        for anchor in captured["anchors"]}) == 4
        return response(adapter, correct_probe_answer(captured, inspect))

    output = tmp_path / "vision-check.json"
    result = asyncio.run(validate_vision_connection(config(adapter), output, allow_api=True,
        transport=httpx.MockTransport(handle)))
    assert len(calls) == 1 and result["status"] == "simulated_pass"
    assert result["checks_passed"] is True and all(result["checks"].values())
    assert result["capability_verified"] is False and result["execution_mode"] == "simulated_http"
    assert result["actual_model"] == "model-revision-1"
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert (output.parent / result["probe"]["file"]).is_file()
    assert "never-persist-revision-key" not in output.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("fault", ["wrong_number", "wrong_color", "duplicate_position", "invalid_action"])
def test_probe_rejects_incorrect_visual_answers_and_invalid_actions(tmp_path, monkeypatch, fault):
    captured = capture_probe(monkeypatch)

    def handle(request):
        answer = correct_probe_answer(captured)
        if fault == "wrong_number":
            number = answer["blocks"][0]["number"]
            answer["blocks"][0]["number"] = 11 if number != 11 else 12
        elif fault == "wrong_color":
            color = answer["blocks"][0]["color"]
            answer["blocks"][0]["color"] = "blue" if color != "blue" else "red"
        elif fault == "duplicate_position":
            answer["blocks"][0]["position"] = answer["blocks"][1]["position"]
        else:
            answer["action"] = dict(action="inspect", reason="Missing required inspection fields")
        return response("chat_completions", answer)

    result = asyncio.run(validate_vision_connection(config(), tmp_path / "failed.json", allow_api=True,
        transport=httpx.MockTransport(handle)))
    assert result["status"] == "failed" and result["capability_verified"] is False
    assert result["error_code"] == "model_output_invalid"


def test_mock_probe_cannot_be_reported_as_actual_capability_verification(tmp_path):
    result = asyncio.run(validate_vision_connection(config("mock"), tmp_path / "mock.json",
        transport=httpx.MockTransport(lambda request: pytest.fail("mock must not send HTTP"))))
    assert result["status"] == "mock_unverified"
    assert result["capability_verified"] is False and result["checks_passed"] is False
    assert result["actual_model"] is None and result["checks"]["actual_model_reported"] is False


@pytest.mark.parametrize("case", ["disabled", "no_image_declaration"])
def test_probe_requires_opt_in_and_explicit_image_declaration(tmp_path, case):
    settings = config()
    if case == "no_image_declaration":
        settings.connections["one"].supports_images = False
    result = asyncio.run(validate_vision_connection(settings, tmp_path / "check.json", allow_api=case != "disabled",
        transport=httpx.MockTransport(lambda request: pytest.fail("preflight must not send HTTP"))))
    assert result["status"] == ("not_run" if case == "disabled" else "failed")
    assert result["capability_verified"] is False and result["trace"] == []
    assert "probe" not in result


def test_probe_requires_reported_actual_model_id(tmp_path, monkeypatch):
    captured = capture_probe(monkeypatch)
    result = asyncio.run(validate_vision_connection(config(), tmp_path / "missing-id.json", allow_api=True,
        transport=httpx.MockTransport(lambda request: response("chat_completions", correct_probe_answer(captured), model=None))))
    assert result["status"] == "failed" and result["error_code"] == "model_version_changed"
    assert result["capability_verified"] is False


def test_probe_timeout_is_recorded_without_retry(tmp_path):
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("secret timeout body must not persist", request=request)

    output = tmp_path / "timeout.json"
    result = asyncio.run(validate_vision_connection(config(), output, allow_api=True,
                                                   transport=httpx.MockTransport(handle)))
    assert len(calls) == 1 and result["status"] == "failed" and result["error_code"] == "api_timeout"
    assert result["trace"][0]["status"] == "request_outcome_unknown"
    assert "secret timeout body" not in output.read_text(encoding="utf-8")
