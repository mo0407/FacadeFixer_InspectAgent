import asyncio
import gzip
import hashlib
import json
import os
import stat

import httpx
import pytest
from pydantic import BaseModel

from facade_mcp.model_gateway import GatewayError, ModelConfig
from facade_mcp.p0_v2_gateway import RawCaptureGateway


class Output(BaseModel):
    answer: int


def _config():
    return ModelConfig.model_validate({'schema_version':'model-gateway-v1','api_enabled':True,
        'connections':{'remote':{'adapter':'responses','model':'model-a','expected_model':'model-a',
          'endpoint_env':'TEST_P0V2_URL','api_key_env':'TEST_P0V2_KEY'}},
        'roles':{'generator':'remote','reflector':'remote','curator':'remote'}})


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv('TEST_P0V2_URL','https://example.test/v1/responses')
    monkeypatch.setenv('TEST_P0V2_KEY','do-not-store-this-key')


def _body(model='model-a',output='{"answer": 3}'):
    return json.dumps({'id':'request-1','model':model,'status':'completed',
        'usage':{'input_tokens':8,'output_tokens':5},
        'output':[{'type':'message','content':[{'type':'output_text','text':output}]}]}).encode()


def _invoke(gateway,path):
    return asyncio.run(gateway.invoke('curator',{'evidence':'test'},Output,capture_path=path))


def test_exact_response_saved_before_success(tmp_path):
    body=_body(); path=tmp_path/'response.body'
    gateway=RawCaptureGateway(_config(),allow_api=True,transport=httpx.MockTransport(lambda req: httpx.Response(200,content=body)))
    result=_invoke(gateway,path)
    assert result.answer==3 and path.read_bytes()==body
    assert stat.S_IMODE(path.stat().st_mode)==0o600
    assert gateway.trace[-1]['usage']['input_tokens']==8 and gateway.model_identities=={'remote':'model-a'}
    assert b'do-not-store-this-key' not in path.read_bytes()
    with pytest.raises(GatewayError,match='exclusive'):_invoke(gateway,path)
    assert path.read_bytes()==body


def test_invalid_output_and_model_still_capture_raw_bytes(tmp_path):
    for name,body in [('invalid',_body(output='not json')),('model',_body(model='wrong'))]:
        path=tmp_path/name
        gateway=RawCaptureGateway(_config(),allow_api=True,transport=httpx.MockTransport(lambda req: httpx.Response(200,content=body)))
        with pytest.raises(GatewayError):_invoke(gateway,path)
        assert path.read_bytes()==body
        assert gateway.trace[-1]['status']=='failed'


def test_http_failure_is_captured_and_classified(tmp_path):
    body=b'provider refused';path=tmp_path/'refused'
    gateway=RawCaptureGateway(_config(),allow_api=True,transport=httpx.MockTransport(lambda req: httpx.Response(429,content=body)))
    with pytest.raises(GatewayError):_invoke(gateway,path)
    assert path.read_bytes()==body and gateway.trace[-1]['error_code']=='api_rate_limited'


def test_timeout_is_unknown_without_retry(tmp_path):
    calls=[]
    def handler(request):
        calls.append(1);raise httpx.ReadTimeout('timeout',request=request)
    gateway=RawCaptureGateway(_config(),allow_api=True,transport=httpx.MockTransport(handler));path=tmp_path/'unknown'
    with pytest.raises(GatewayError,match='unknown'):_invoke(gateway,path)
    assert calls==[1] and path.read_bytes()==b''
    assert gateway.trace[-1]['status']=='request_outcome_unknown'


def test_capture_failure_is_unknown_and_cannot_be_accepted(tmp_path,monkeypatch):
    gateway=RawCaptureGateway(_config(),allow_api=True,transport=httpx.MockTransport(lambda req: httpx.Response(200,content=_body())))
    real_write=os.write
    def fail(fd,data): raise OSError('full')
    monkeypatch.setattr(os,'write',fail)
    with pytest.raises(GatewayError,match='capture failed'):_invoke(gateway,tmp_path/'failed')
    monkeypatch.setattr(os,'write',real_write)
    assert gateway.trace[-1]['status']=='request_outcome_unknown'
    assert gateway.trace[-1]['error_code']=='response_capture_failed'


def test_encoded_wire_body_is_captured_before_httpx_decoding(tmp_path):
    raw=gzip.compress(_body())
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield raw[:7]
            yield raw[7:]
    transport=httpx.MockTransport(lambda req:httpx.Response(200,headers={'content-encoding':'gzip'},stream=Stream()))
    gateway=RawCaptureGateway(_config(),allow_api=True,transport=transport);path=tmp_path/'encoded'
    assert _invoke(gateway,path).answer==3
    assert path.read_bytes()==raw
    assert gateway.trace[-1]['response_content_encoding']=='gzip'
    assert gateway.trace[-1]['response_body_sha256']==hashlib.sha256(raw).hexdigest()


def test_explicit_proxy_environment_fails_closed(monkeypatch):
    from facade_mcp.p0_v2_gateway import RawCaptureGateway
    from facade_mcp.model_gateway import GatewayError
    monkeypatch.setenv('HTTPS_PROXY','http://127.0.0.1:9999')
    with pytest.raises(GatewayError, match='proxy'):
        RawCaptureGateway(None,allow_api=True)
