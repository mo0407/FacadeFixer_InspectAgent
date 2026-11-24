"""Figure-1 runner entrypoint contracts; no real models or network calls."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys

from PIL import Image
import pytest

from facade_mcp import figure1_runner as runner


def image(path, color='white'):
    Image.new('RGB', (20, 20), color).save(path)
    return path


def args_for(tmp_path, *extra):
    (tmp_path/'source').mkdir(exist_ok=True)
    (tmp_path/'experts.json').write_text('{}')
    return runner.parser().parse_args(['--images-dir', str(tmp_path/'images'),
        '--expected-count', '2', '--workspace', str(tmp_path/'workspace'),
        '--output', str(tmp_path/'workspace'/'experiment'), '--source-repo', str(tmp_path/'source'),
        '--source-python', sys.executable, '--experts-config', str(tmp_path/'experts.json'), *extra])


@pytest.fixture
def dataset(tmp_path):
    directory=tmp_path/'images'
    directory.mkdir()
    image(directory/'b.jpg', 'blue')
    image(directory/'a.png', 'red')
    (directory/'a.txt').write_text('not an image; must never be read')
    return directory


def test_experts_only_uses_sorted_images_and_never_constructs_vlm(dataset,tmp_path,monkeypatch):
    calls=[]
    class FakeGateway:
        async def call(self,name,arguments):
            calls.append(('call',name,arguments))
            return {'profiles':{'fixture':{}}}
    @asynccontextmanager
    async def gateway(workspace,roots,trace,model_env):
        calls.append(('roots',workspace,roots,trace,model_env))
        yield FakeGateway()
    async def batch(paths,folder,expert,caps,config,**kwargs):
        calls.append(('batch',paths,folder,caps,config,kwargs))
        return {'status':'experts_only','completed':[{'path':str(p)} for p in paths],
                'failed':[],'simulation':False,'vlm_called':False}
    monkeypatch.setattr(runner,'_gateway',gateway)
    monkeypatch.setattr(runner,'run_baseline_experts_batch',batch)
    monkeypatch.setattr(runner,'ModelGateway',lambda *a,**kw:pytest.fail('VLM must not be constructed'))
    args=args_for(tmp_path,'--experts-only')
    result=asyncio.run(runner.run(args))
    assert result['status']=='experts_only' and result['vlm_called'] is False
    entry=next(x for x in calls if x[0]=='batch')
    assert [p.name for p in entry[1]]==['a.png','b.jpg']
    assert entry[4].expected_count==2 and entry[4].expert_input_size==640
    roots=next(x for x in calls if x[0]=='roots')[2]
    assert dataset.resolve() in roots and args.workspace.resolve() in roots
    model_env=next(x for x in calls if x[0]=='roots')[4]
    assert model_env['FACADE_SOURCE_REPO']==str((tmp_path/'source').resolve())
    assert model_env['FACADE_SOURCE_PYTHON']==str(Path(sys.executable).resolve())
    assert model_env['FACADE_EXPERTS_CONFIG']==str((tmp_path/'experts.json').resolve())
    assert model_env['FACADE_SOURCE_DEVICE']=='cuda:0'


def test_exact_count_and_duplicate_image_paths_are_checked(dataset,tmp_path):
    (tmp_path/'source').mkdir()
    (tmp_path/'experts.json').write_text('{}')
    options=['--source-repo',str(tmp_path/'source'),'--source-python',sys.executable,
             '--experts-config',str(tmp_path/'experts.json')]
    args=runner.parser().parse_args(['--images-dir',str(dataset),'--expected-count','203',
        '--workspace',str(tmp_path/'work'),'--output',str(tmp_path/'work'/'run'),'--experts-only',*options])
    with pytest.raises(ValueError,match='Expected 203 images; found 2'):
        runner._prepare(args)
    path=dataset/'a.png'
    args=runner.parser().parse_args(['--image',str(path),'--image',str(path),'--expected-count','2',
        '--workspace',str(tmp_path/'work'),'--output',str(tmp_path/'work'/'run'),'--experts-only',*options])
    with pytest.raises(ValueError,match='unique'):
        runner._prepare(args)


def test_complete_requires_explicit_real_model_and_api_authorization(dataset,tmp_path):
    args=args_for(tmp_path,'--complete')
    with pytest.raises(ValueError,match='--model-config and --allow-api'):
        runner._prepare(args)
    config=tmp_path/'mock.json'
    config.write_text(json.dumps({'schema_version':'model-gateway-v1','api_enabled':True,
        'connections':{'mock':{'adapter':'mock','model':'fixture','supports_images':True,'expected_model':'fixture'}},
        'roles':{'generator':'mock','reflector':'mock','curator':'mock'}}))
    args=args_for(tmp_path,'--complete','--allow-api','--model-config',str(config))
    with pytest.raises(ValueError,match='real enabled model'):
        runner._prepare(args)


def test_explicit_remote_model_paths_override_stale_copied_environment(dataset,tmp_path,monkeypatch):
    monkeypatch.setenv('FACADE_SOURCE_REPO','/home/ubuntu/old-project')
    monkeypatch.setenv('FACADE_SOURCE_PYTHON','/home/ubuntu/old-python')
    monkeypatch.setenv('FACADE_EXPERTS_CONFIG','/home/ubuntu/old-experts.json')
    monkeypatch.setenv('FACADE_WORKSPACE','/home/ubuntu/old-workspace')
    monkeypatch.setenv('FACADE_IMAGE_API_KEY','synthetic-fixture')
    args=args_for(tmp_path,'--experts-only')
    images,workspace,_,_,_,model_env=runner._prepare(args)
    config=runner._model_server_config(workspace,[p.parent for p in images],model_env)
    assert set(config['mcpServers'])=={'model'}
    env=config['mcpServers']['model']['env']
    assert env['FACADE_WORKSPACE']==str(workspace)
    assert env['FACADE_SOURCE_REPO']==str((tmp_path/'source').resolve())
    assert env['FACADE_SOURCE_PYTHON']==str(Path(sys.executable).resolve())
    assert env['FACADE_EXPERTS_CONFIG']==str((tmp_path/'experts.json').resolve())
    assert 'FACADE_IMAGE_API_KEY' not in env


def test_complete_passes_real_mode_and_recovery_flags(dataset,tmp_path,monkeypatch):
    config=tmp_path/'real.json'
    config.write_text(json.dumps({'schema_version':'model-gateway-v1','api_enabled':True,
        'connections':{'remote':{'adapter':'responses','model':'fixture','expected_model':'fixture',
            'endpoint_env':'TEST_ENDPOINT','api_key_env':'TEST_KEY','supports_images':True}},
        'roles':{'generator':'remote','reflector':'remote','curator':'remote'}}))
    seen={}
    class FakeGateway:
        async def call(self,name,arguments): return {'profiles':{}}
    @asynccontextmanager
    async def gateway(*args): yield FakeGateway()
    async def batch(images,folder,experts,vlm,capabilities,config,**kwargs):
        seen.update(kwargs)
        seen['allow_api']=vlm.allow_api
        return {'completed':[{'path':str(p)} for p in images], 'failed':[], 'simulation':False}
    monkeypatch.setattr(runner,'_gateway',gateway)
    monkeypatch.setattr(runner,'run_baseline_batch',batch)
    args=args_for(tmp_path,'--complete','--allow-api','--model-config',str(config),
                  '--retry-failed','--acknowledge-unknown')
    asyncio.run(runner.run(args))
    assert seen=={'simulation':False,'retry_failed':True,'acknowledge_unknown':True,'allow_api':True}


@pytest.mark.skipif(os.name != 'posix', reason='POSIX flock is required by the batch runner')
def test_rejects_unrelated_output_and_concurrent_runner(tmp_path):
    output=tmp_path/'output'
    output.mkdir()
    (output/'keep.txt').write_text('user data')
    with pytest.raises(ValueError,match='nonempty directory'):
        with runner._output_lock(output): pass
    assert (output/'keep.txt').read_text()=='user data'
    assert not (output/'.figure1.lock').exists()
    (output/'expert-manifest.json').write_text('{}')
    with runner._output_lock(output):
        with pytest.raises(RuntimeError,match='Another Figure-1 runner'):
            with runner._output_lock(output): pass


def test_main_returns_failure_for_incomplete_batch(dataset,tmp_path,monkeypatch,capsys):
    async def incomplete(args):
        return {'status':'experts_only','completed':[{}],'failed':[{}],
                'simulation':False,'halted_reason':'consecutive_expert_failures'}
    monkeypatch.setattr(runner,'run',incomplete)
    exit_code=runner.main(['--images-dir',str(dataset),'--expected-count','2',
        '--workspace',str(tmp_path/'work'),'--output',str(tmp_path/'work'/'run'),'--experts-only',
        '--source-repo',str(tmp_path),'--source-python',sys.executable,'--experts-config',str(tmp_path/'missing.json')])
    assert exit_code==1
    result=json.loads(capsys.readouterr().out)
    assert result['completed']==1 and result['failed']==1 and result['halted_reason']=='consecutive_expert_failures'
