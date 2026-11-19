import asyncio
import base64
import hashlib
import io
import json

import httpx
import pytest
from PIL import Image
from pydantic import BaseModel

from facade_mcp.model_gateway import (
    GatewayError, MAX_IMAGE_BYTES, MAX_TOTAL_IMAGE_BYTES, ModelConfig, ModelGateway,
)


class Answer(BaseModel):
    answer: str


def config(adapter="mock", *, supports_images=True):
    connection = dict(adapter=adapter, model="fixture-model", supports_images=supports_images)
    if adapter != "mock":
        connection.update(endpoint_env="IMAGE_TEST_ENDPOINT", api_key_env="IMAGE_TEST_KEY")
    return ModelConfig.model_validate(dict(
        schema_version="model-gateway-v1", api_enabled=adapter != "mock",
        connections={"fixture": connection},
        roles={role: "fixture" for role in ("generator", "reflector", "curator")},
    ))


def make_image(tmp_path, image_format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", (13, 7), (80, 100, 120)).save(stream, format=image_format)
    path = tmp_path / ("private-snapshot.png" if image_format == "PNG" else "private-snapshot.jpg")
    path.write_bytes(stream.getvalue())
    return path


def invoke(gateway, paths, **kwargs):
    return asyncio.run(gateway.invoke("generator", {"instruction": "检查墙面"}, Answer,
                                      image_paths=paths, **kwargs))


@pytest.mark.parametrize("adapter", ["chat_completions", "responses"])
@pytest.mark.parametrize("image_format,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_multimodal_payload_and_private_trace(tmp_path, monkeypatch, adapter, image_format, mime):
    path = make_image(tmp_path, image_format)
    image_bytes = path.read_bytes()
    monkeypatch.setenv("IMAGE_TEST_ENDPOINT", "https://fixture.invalid/v1/" + adapter)
    monkeypatch.setenv("IMAGE_TEST_KEY", "secret-fixture-key")
    calls = []

    def handle(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer secret-fixture-key"
        payload = json.loads(request.content)
        messages = payload["input" if adapter == "responses" else "messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        text, image = messages[1]["content"]
        if adapter == "responses":
            assert text["type"] == "input_text"
            assert image["type"] == "input_image"
            image_url = image["image_url"]
            assert payload["store"] is False
        else:
            assert text["type"] == "text"
            assert image["type"] == "image_url"
            image_url = image["image_url"]["url"]
        assert json.loads(text["text"]) == {"instruction": "检查墙面"}
        assert image_url.startswith("data:" + mime + ";base64,")
        assert base64.b64decode(image_url.split(",", 1)[1], validate=True) == image_bytes
        if adapter == "responses":
            body = dict(id="fixture-id", model="fixture-version", status="completed", output=[
                {"type": "message", "content": [{"type": "output_text", "text": '{"answer":"ok"}'}]}])
        else:
            body = dict(id="fixture-id", model="fixture-version", choices=[
                {"finish_reason": "stop", "message": {"content": '{"answer":"ok"}'}}])
        return httpx.Response(200, json=body)

    gateway = ModelGateway(config(adapter), allow_api=True, transport=httpx.MockTransport(handle))
    assert invoke(gateway, [path]).answer == "ok"
    assert len(calls) == 1
    assert gateway.trace[0]["images"] == [dict(
        sha256=hashlib.sha256(image_bytes).hexdigest(), width=13, height=7,
        size_bytes=len(image_bytes), mime_type=mime)]
    trace = json.dumps(gateway.trace)
    for private_value in (str(path), path.name, "secret-fixture-key", base64.b64encode(image_bytes).decode()):
        assert private_value not in trace


def test_mock_validates_images_without_creating_client(tmp_path, monkeypatch):
    def forbidden_client(*args, **kwargs):
        pytest.fail("Mock must not create an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_client)
    gateway = ModelGateway(config())
    path = make_image(tmp_path)
    assert invoke(gateway, [path], mock_output={"answer": "fixture"}).answer == "fixture"
    assert gateway.trace[0]["images"][0]["width"] == 13
    assert gateway.trace[0]["status"] == "completed"


@pytest.mark.parametrize("adapter", ["mock", "chat_completions", "responses"])
def test_images_require_connection_capability(tmp_path, adapter):
    gateway = ModelGateway(config(adapter, supports_images=False), allow_api=True)
    with pytest.raises(GatewayError, match="does not support image"):
        invoke(gateway, [make_image(tmp_path)], mock_output={"answer": "fixture"})
    assert gateway.trace[0]["status"] == "failed"


@pytest.mark.parametrize("content", [b"not-an-image", b"\x89PNG\r\n\x1a\ninvalid-data"])
def test_invalid_images_are_rejected(tmp_path, content):
    path = tmp_path / "invalid.png"
    path.write_bytes(content)
    gateway = ModelGateway(config())
    with pytest.raises(GatewayError, match="valid PNG or JPEG"):
        invoke(gateway, [path], mock_output={"answer": "fixture"})
    assert gateway.trace[0]["status"] == "failed"


def test_unsupported_image_and_truncated_jpeg_are_rejected(tmp_path):
    gif = tmp_path / "unsupported.png"
    Image.new("RGB", (13, 7)).save(gif, format="GIF")
    jpeg = make_image(tmp_path, "JPEG")
    jpeg.write_bytes(jpeg.read_bytes()[:-20])
    for path in (gif, jpeg):
        with pytest.raises(GatewayError, match="valid PNG or JPEG"):
            invoke(ModelGateway(config()), [path], mock_output={"answer": "fixture"})


def test_missing_file_error_does_not_leak_path(tmp_path):
    path = tmp_path / "missing-private-image.png"
    gateway = ModelGateway(config())
    with pytest.raises(GatewayError, match="could not be read") as failure:
        invoke(gateway, [path], mock_output={"answer": "fixture"})
    assert path.name not in str(failure.value) + json.dumps(gateway.trace)


def test_image_count_limit_and_boundary(tmp_path):
    path = make_image(tmp_path)
    gateway = ModelGateway(config())
    assert invoke(gateway, [path] * 8, mock_output={"answer": "fixture"}).answer == "fixture"
    with pytest.raises(GatewayError, match="At most 8"):
        invoke(gateway, [path] * 9, mock_output={"answer": "fixture"})


def test_per_image_byte_limit(tmp_path):
    path = make_image(tmp_path)
    path.write_bytes(path.read_bytes().ljust(MAX_IMAGE_BYTES + 1, b"\0"))
    with pytest.raises(GatewayError, match="at most 10 MB"):
        invoke(ModelGateway(config()), [path], mock_output={"answer": "fixture"})


def test_total_byte_limit_and_boundary(tmp_path):
    path = make_image(tmp_path)
    # PNG remains decodable with trailing bytes; every byte is counted and hashed.
    path.write_bytes(path.read_bytes().ljust(MAX_IMAGE_BYTES, b"\0"))
    assert MAX_TOTAL_IMAGE_BYTES == MAX_IMAGE_BYTES * 3
    gateway = ModelGateway(config())
    assert invoke(gateway, [path] * 3, mock_output={"answer": "fixture"}).answer == "fixture"
    with pytest.raises(GatewayError, match="at most 30 MB"):
        invoke(gateway, [path] * 4, mock_output={"answer": "fixture"})


@pytest.mark.parametrize("paths", [None, []])
def test_no_image_input_keeps_text_payload_compatible(monkeypatch, paths):
    monkeypatch.setenv("IMAGE_TEST_ENDPOINT", "https://fixture.invalid/v1/chat/completions")
    monkeypatch.setenv("IMAGE_TEST_KEY", "secret-fixture-key")

    def handle(request):
        payload = json.loads(request.content)
        assert isinstance(payload["messages"][1]["content"], str)
        assert json.loads(payload["messages"][1]["content"]) == {"instruction": "检查墙面"}
        return httpx.Response(200, json={"model": "fixture-version", "choices": [
            {"finish_reason": "stop", "message": {"content": '{"answer":"ok"}'}}]})

    gateway = ModelGateway(config("chat_completions", supports_images=False), allow_api=True,
                           transport=httpx.MockTransport(handle))
    assert invoke(gateway, paths).answer == "ok"
    assert "images" not in gateway.trace[0]
