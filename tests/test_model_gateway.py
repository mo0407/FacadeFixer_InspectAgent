import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from facade_mcp.model_gateway import ModelConfig, ModelGateway, GatewayError
from facade_mcp.review_cycle import Plan, ReviewRequest, evaluate, run_review


def config(remote=False):
    connection = {"adapter": "chat_completions" if remote else "mock", "model": "fixture"}
    if remote:
        connection.update(endpoint_env="TEST_FACADE_ENDPOINT", api_key_env="TEST_FACADE_KEY")
    return ModelConfig.model_validate(dict(schema_version="model-gateway-v1", api_enabled=remote,
        connections={"one": connection}, roles={r:"one" for r in ["generator","reflector","curator"]}))


def request(value=None, threshold=None, verified=False):
    return ReviewRequest.model_validate(dict(task="fixture", evidence={"e1":"fixture evidence"},
        metrics=[dict(id="recall",value=value,threshold=threshold,reference_verified=verified,evidence_ids=["e1"])],
        acceptance_version="fixture",playbook_version="empty"))


@pytest.mark.parametrize("value,threshold,verified,status",[
    (None,None,False,"pending_review"),(1,None,True,"pending_review"),
    (1,.9,False,"pending_review"),(.8,.9,True,"optimizing"),(1,.9,True,"pass")])
def test_gate(value,threshold,verified,status):
    assert evaluate(request(value,threshold,verified))["status"]==status


def test_invalid_reference_and_nonfinite():
    with pytest.raises(ValidationError):request(float("nan"),.9,True)
    data=request().model_dump();data["metrics"][0]["evidence_ids"]=["unknown"]
    with pytest.raises(ValidationError):ReviewRequest.model_validate(data)


def test_mock_cycle_cannot_promote(tmp_path):
    out=asyncio.run(run_review(config(),request(),tmp_path))
    assert out["acceptance_status"]=="pending_review"
    assert out["report_generated"] is False and out["experiments_executed"]==0
    from pathlib import Path
    folder=Path(out["output"])
    assert json.loads((folder/'candidate_experience.json').read_text())["promoted"] is False
    assert len(json.loads((folder/'trace.json').read_text()))==3


def test_remote_requires_explicit_opt_in():
    gateway=ModelGateway(config(True))
    with pytest.raises(GatewayError,match="disabled"):
        asyncio.run(gateway.invoke("generator",{},Plan))


def test_remote_response_and_trace_no_secrets(monkeypatch):
    monkeypatch.setenv("TEST_FACADE_ENDPOINT","https://fixture.invalid/chat/completions")
    monkeypatch.setenv("TEST_FACADE_KEY","never-log-this-key")
    def handler(req):
        assert req.headers['authorization']=='Bearer never-log-this-key'
        return httpx.Response(200,json={"id":"r1","model":"fixture-version","choices":[{
            "finish_reason":"stop","message":{"content":json.dumps({"stages":["scope"],"rationale":"check"})}}]})
    g=ModelGateway(config(True),allow_api=True,transport=httpx.MockTransport(handler))
    result=asyncio.run(g.invoke("generator",{},Plan))
    assert result.stages==["scope"]
    assert g.trace[0]['actual_model']=='fixture-version'
    assert 'never-log-this-key' not in json.dumps(g.trace)


def test_timeout_does_not_retry(monkeypatch):
    monkeypatch.setenv("TEST_FACADE_ENDPOINT","https://fixture.invalid/chat/completions")
    monkeypatch.setenv("TEST_FACADE_KEY","secret")
    calls=[]
    def handler(req):
        calls.append(req);raise httpx.ReadTimeout("secret body",request=req)
    g=ModelGateway(config(True),allow_api=True,transport=httpx.MockTransport(handler))
    with pytest.raises(GatewayError,match="unknown"):
        asyncio.run(g.invoke("generator",{},Plan))
    assert len(calls)==1 and g.trace[0]['status']=='request_outcome_unknown'


def test_responses_standard_mode_and_structured_output(monkeypatch):
    monkeypatch.setenv('TEST_FACADE_ENDPOINT', 'https://fixture.invalid/v1/responses')
    monkeypatch.setenv('TEST_FACADE_KEY', 'test-key')
    data = config(True).model_dump()
    data['connections']['one'].update(adapter='responses', model='gpt-6.1-sol', reasoning_mode='standard')

    def handler(req):
        payload = json.loads(req.content)
        assert payload['reasoning'] == {'mode': 'standard'}
        assert payload['store'] is False and payload['max_output_tokens'] == 2000
        assert payload['input'][0]['role'] == 'system' and 'messages' not in payload
        return httpx.Response(200, json={
            'status': 'completed', 'model': 'gpt-6.1-sol',
            'output': [{'type': 'reasoning'}, {'type': 'message', 'content': [
                {'type': 'output_text', 'text': '{"stages":["scope"],"rationale":"check"}'}]}],
        })

    gateway = ModelGateway(ModelConfig.model_validate(data), allow_api=True, transport=httpx.MockTransport(handler))
    assert asyncio.run(gateway.invoke('generator', {}, Plan)).stages == ['scope']
    assert gateway.trace[0]['reasoning_mode'] == 'standard'


@pytest.mark.parametrize("content",['{"stages":["scope"],"rationale":"x","acceptance_status":"pass"}', '{}'])
def test_invalid_output_rejected(monkeypatch,content):
    monkeypatch.setenv("TEST_FACADE_ENDPOINT","https://fixture.invalid/chat/completions")
    monkeypatch.setenv("TEST_FACADE_KEY","secret")
    transport=httpx.MockTransport(lambda req:httpx.Response(200,json={"choices":[{
        "finish_reason":"stop","message":{"content":content}}]}))
    with pytest.raises(GatewayError):
        asyncio.run(ModelGateway(config(True),allow_api=True,transport=transport).invoke("generator",{},Plan))
