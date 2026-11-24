"""Local Ollama contracts using synthetic in-process HTTP only; no real inference."""
import asyncio
import base64
import json

import httpx
from PIL import Image
import pytest
from pydantic import ValidationError

from facade_mcp.model_gateway import Connection, GatewayError, ModelConfig, ModelGateway, Strict


DIGEST = 'a' * 64


class Answer(Strict):
    answer: str


def config(**changes):
    connection = {'adapter': 'ollama', 'model': 'llava:34b', 'supports_images': True,
                  'endpoint_env': 'TEST_OLLAMA_URL', 'expected_model': 'llava:34b',
                  'expected_digest': DIGEST, 'max_output_tokens': 128}
    connection.update(changes)
    return ModelConfig.model_validate({'schema_version':'model-gateway-v1','api_enabled':True,
        'connections':{'local':connection},
        'roles':{'generator':'local','reflector':'local','curator':'local'}})


def tags(digest=DIGEST):
    return httpx.Response(200,json={'models':[{'name':'llava:34b','model':'llava:34b','digest':digest}]})


def result(*, model='llava:34b', done=True, reason='stop'):
    return httpx.Response(200,json={'model':model,'done':done,'done_reason':reason,
        'message':{'role':'assistant','content':'{"answer":"ok"}'},
        'prompt_eval_count':10,'eval_count':3})


def invoke(gateway, paths=None):
    return asyncio.run(gateway.invoke('generator',{'stage':'fixture'},Answer,image_paths=paths))


def test_loopback_ollama_sends_images_schema_and_pins_digest(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_OLLAMA_URL','http://127.0.0.1:11434/api/chat')
    path=tmp_path/'source.png'
    Image.new('RGB',(24,16),'red').save(path)
    calls=[]
    def handle(request):
        calls.append((request.method,request.url.path))
        assert 'authorization' not in request.headers
        if request.method == 'GET': return tags()
        payload=json.loads(request.content)
        assert payload['model']=='llava:34b' and payload['stream'] is False
        assert payload['format']==Answer.model_json_schema()
        assert payload['options']=={'temperature':0,'num_predict':128}
        assert json.loads(payload['messages'][1]['content'])=={'stage':'fixture'}
        assert base64.b64decode(payload['messages'][1]['images'][0])==path.read_bytes()
        return result()
    gateway=ModelGateway(config(),allow_api=True,transport=httpx.MockTransport(handle))
    assert invoke(gateway,[path]).answer=='ok'
    assert calls==[('GET','/api/tags'),('POST','/api/chat'),('GET','/api/tags')]
    row=gateway.trace[0]
    assert row['status']=='completed' and row['actual_model']=='llava:34b'
    assert row['model_digest']==DIGEST and row['model_identity_verified'] is True
    assert row['usage']=={'input_tokens':10,'output_tokens':3}
    assert row['vision_capability_verified'] is False
    assert str(path) not in json.dumps(row) and 'images' in row
    assert gateway.model_identities=={'local':'llava:34b'}


@pytest.mark.parametrize('url',[
    'http://example.com:11434/api/chat','http://localhost:11434/api/chat',
    'https://127.0.0.1:11434/api/chat','http://127.0.0.1:11434/api/chat?x=1',
    'http://user:pass@127.0.0.1:11434/api/chat','http://127.0.0.1:11434/v1/chat',
    'http://127.0.0.1/api/chat','http://127.0.0.1:99999/api/chat',
])
def test_endpoint_rejects_non_loopback_or_unexpected_urls(monkeypatch,url):
    monkeypatch.setenv('TEST_OLLAMA_URL',url)
    calls=[]
    gateway=ModelGateway(config(),allow_api=True,transport=httpx.MockTransport(lambda request: calls.append(request)))
    with pytest.raises(GatewayError): invoke(gateway)
    assert not calls


def test_ollama_requires_explicit_invocation_permission(monkeypatch):
    monkeypatch.setenv('TEST_OLLAMA_URL','http://127.0.0.1:11434/api/chat')
    calls=[]
    gateway=ModelGateway(config(),allow_api=False,transport=httpx.MockTransport(lambda request: calls.append(request)))
    with pytest.raises(GatewayError,match='disabled'): invoke(gateway)
    assert not calls


@pytest.mark.parametrize('changes',[
    {'expected_digest':None},{'expected_digest':'bad'},{'expected_model':None},
    {'expected_model':'other-model'},{'endpoint_env':None},{'api_key_env':'SECRET_KEY'},
    {'reasoning_effort':'low'},
])
def test_ollama_configuration_requires_frozen_model_without_remote_credentials(changes):
    with pytest.raises(ValidationError): config(**changes)


def test_digest_mismatch_before_request_rejects_without_post(monkeypatch):
    monkeypatch.setenv('TEST_OLLAMA_URL','http://127.0.0.1:11434/api/chat')
    calls=[]
    def handle(request):
        calls.append(request.method)
        return tags('b'*64)
    gateway=ModelGateway(config(),allow_api=True,transport=httpx.MockTransport(handle))
    with pytest.raises(GatewayError,match='pinned digest'): invoke(gateway)
    assert calls==['GET'] and gateway.trace[0]['error_code']=='model_version_changed'
    assert gateway.trace[0]['model_identity_verified'] is False


def test_digest_change_after_inference_rejects_result(monkeypatch):
    monkeypatch.setenv('TEST_OLLAMA_URL','http://127.0.0.1:11434/api/chat')
    counter={'gets':0}
    def handle(request):
        if request.method=='POST': return result()
        counter['gets']+=1
        return tags(DIGEST if counter['gets']==1 else 'b'*64)
    gateway=ModelGateway(config(),allow_api=True,transport=httpx.MockTransport(handle))
    with pytest.raises(GatewayError,match='pinned digest'): invoke(gateway)
    assert gateway.trace[0]['status']=='failed'
    assert gateway.trace[0]['model_identity_verified'] is False


@pytest.mark.parametrize('payload',[
    {'model':'other-model'}, {'done':False}, {'reason':'length'},
])
def test_model_mismatch_or_incomplete_output_is_rejected(monkeypatch,payload):
    monkeypatch.setenv('TEST_OLLAMA_URL','http://127.0.0.1:11434/api/chat')
    gateway=ModelGateway(config(),allow_api=True,transport=httpx.MockTransport(
        lambda request: tags() if request.method=='GET' else result(**payload)))
    with pytest.raises(GatewayError): invoke(gateway)
    assert gateway.trace[0]['status']=='failed'


def test_ollama_does_not_follow_redirects(monkeypatch):
    monkeypatch.setenv('TEST_OLLAMA_URL','http://127.0.0.1:11434/api/chat')
    calls=[]
    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(302,headers={'Location':'http://example.com/'})
    gateway=ModelGateway(config(),allow_api=True,transport=httpx.MockTransport(handle))
    with pytest.raises(GatewayError): invoke(gateway)
    assert len(calls)==1 and calls[0]=='http://127.0.0.1:11434/api/tags'


def test_timeout_is_unknown_and_never_retried(monkeypatch):
    monkeypatch.setenv('TEST_OLLAMA_URL','http://127.0.0.1:11434/api/chat')
    calls=[]
    def handle(request):
        calls.append(request.method)
        if request.method=='GET': return tags()
        raise httpx.ReadTimeout('synthetic timeout')
    gateway=ModelGateway(config(),allow_api=True,transport=httpx.MockTransport(handle))
    with pytest.raises(GatewayError,match='timed out'): invoke(gateway)
    assert calls==['GET','POST']
    assert gateway.trace[0]['status']=='request_outcome_unknown'
