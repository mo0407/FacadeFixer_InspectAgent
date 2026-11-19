"""Synthetic contracts only: no real model checkpoints, downloads or paid APIs."""
import json
from pathlib import Path
import subprocess
import sys
import venv

import numpy as np
from PIL import Image
import pytest

from facade_mcp.expert_registry import load_registry
from facade_mcp.model_service import ModelService
from facade_mcp.storage import ArtifactStore, sha256
from facade_mcp.taxonomy import CLASS_NAMES, opencv_code


def registration(tmp_path, **changes):
    path = tmp_path/'experts.json'
    weights = tmp_path/'checkpoint.pt'
    weights.write_bytes(b'fixture only: not trained weights')
    profile = dict(backend='ultralytics',weights_path='checkpoint.pt',task='segment',
                   class_map={'crack':'C','10':'B'},class_codes=['C','B'],
                   local_eligible=True,full_image=True,reference_name='Synthetic fixture')
    profile.update(changes)
    path.write_text(json.dumps({'profiles':{'fixture_seg':profile}}),encoding='utf-8')
    return path, weights


@pytest.fixture
def configured(tmp_path, monkeypatch):
    config, weights = registration(tmp_path)
    image = tmp_path/'image.png'
    Image.new('RGB',(80,60),'gray').save(image)
    service = ModelService(ArtifactStore(tmp_path/'workspace',[tmp_path]),
                           repo_path=tmp_path/'absent_legacy_repo',python_path=Path(sys.executable),experts_config=config)
    yield service,image,weights
    service.masks.close()


def test_registry_confines_paths_and_profile_names(tmp_path, monkeypatch):
    config,_ = registration(tmp_path, weights_path='../outside.pt')
    with pytest.raises(ValueError,match='paths must remain'):
        load_registry(config,default_python=Path(sys.executable))
    monkeypatch.setenv('FACADE_EXPERTS_ALLOWED_ROOTS',str(tmp_path.parent))
    assert Path(load_registry(config,default_python=Path(sys.executable))['fixture_seg']['weights_path']).parent == tmp_path.parent
    data=json.loads(config.read_text())
    data['profiles']['../evil'] = data['profiles'].pop('fixture_seg')
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='profile names'):
        load_registry(config,default_python=Path(sys.executable))


@pytest.mark.parametrize('changes',[
    {'class_codes':['C','C']}, {'class_map':{'crack':'C'}}, {'class_map':{'crack':'Z'}}, {'backend':'shell'},
    {'backend':'yolov9','task':'segment'}, {'command':'arbitrary.exe'},
])
def test_registry_rejects_unsupported_or_inconsistent_config(tmp_path,changes):
    config,_ = registration(tmp_path,**changes)
    with pytest.raises(ValueError):
        load_registry(config,default_python=Path(sys.executable))


def test_registry_cannot_replace_builtin(tmp_path):
    config,_=registration(tmp_path)
    with pytest.raises(ValueError,match='not replace builtins'):
        load_registry(config,default_python=Path(sys.executable),reserved=['fixture_seg'])


def test_runtime_source_paths_are_confined_and_included_in_expert_definition(tmp_path):
    config, weights = registration(tmp_path, implementation_files={'custom_block': 'runtime/block.py'})
    runtime = tmp_path/'runtime/block.py'
    runtime.parent.mkdir()
    runtime.write_text('# original runtime implementation\n')
    service = ModelService(ArtifactStore(tmp_path/'workspace', [tmp_path]),
                           repo_path=tmp_path/'absent', python_path=Path(sys.executable), experts_config=config)
    try:
        before = service.capabilities()['profiles']['fixture_seg']
        assert before['ready']
        assert before['implementation_sha256']['expert_runtime/custom_block'] == sha256(runtime)
        runtime.write_text('# modified runtime implementation\n')
        after = service.capabilities()['profiles']['fixture_seg']
        assert before['weights_sha256'] == after['weights_sha256'] == sha256(weights)
        assert before['definition_sha256'] != after['definition_sha256']
        runtime.unlink()
        assert service.capabilities()['profiles']['fixture_seg']['reason'] == 'EXPERT_IMPLEMENTATION_NOT_FOUND'
    finally:
        service.masks.close()
    data = json.loads(config.read_text())
    data['profiles']['fixture_seg']['implementation_files']['custom_block'] = '../outside.py'
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='paths must remain'):
        load_registry(config, default_python=Path(sys.executable))


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX venv symlink layout')
def test_registry_preserves_venv_interpreter_symlink(tmp_path, monkeypatch):
    env = tmp_path/'model-env'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(env)
    interpreter = env/'bin/python'
    monkeypatch.setenv('FACADE_EXPERTS_ALLOWED_ROOTS', str(interpreter.resolve().parent))
    config, _ = registration(tmp_path, python_path=str(interpreter))
    profile = load_registry(config, default_python=Path(sys.executable))['fixture_seg']
    assert Path(profile['python_path']) == interpreter
    output = subprocess.check_output([profile['python_path'], '-c', 'import sys; print(sys.prefix)'], text=True)
    assert Path(output.strip()) == env


def test_capabilities_never_probe_and_report_actual_weights(configured,monkeypatch):
    service,_,weights=configured
    monkeypatch.setattr(service,'_run',lambda *a,**k:pytest.fail('Capability inventory must not probe inference'))
    row=service.capabilities()['profiles']['fixture_seg']
    assert row['ready'] and row['weights_sha256']==sha256(weights)
    assert row['class_codes']==['C','B'] and row['task']=='segment'
    assert row['model_load_verified'] is False
    weights.unlink()
    missing=service.capabilities()['profiles']['fixture_seg']
    assert not missing['ready'] and missing['reason']=='WEIGHTS_NOT_FOUND'
    assert service.detect(['not-an-image'],'fixture_seg')['error']['code']=='WEIGHTS_NOT_FOUND'


def fake_inference(service, monkeypatch, *, escape=False, bad_hash=False):
    def run(directory,action,**arguments):
        assert action=='detect'
        directory.mkdir(parents=True,exist_ok=True)
        mask=np.zeros((60,80),dtype=np.uint8)
        mask[5:25,10:30]=255
        path=(directory.parent if escape else directory)/'mask.png'
        Image.fromarray(mask).save(path)
        result=dict(status='completed',defects=[
            dict(instance_id='one',class_name='crack',class_id=0,bbox_xyxy=[10,5,30,25],confidence=.9,
                 mask_path=str(path),mask_sha256=sha256(path)),
            dict(instance_id='unknown',class_name='corrosion',class_id=200,bbox_xyxy=[35,5,50,25],confidence=.7,
                 mask_path=str(path),mask_sha256=sha256(path))],
            model_provenance=dict(weights_sha256='0'*64 if bad_hash else sha256(Path(service.registry['fixture_seg']['weights_path'])),
                                  source_image_sha256=sha256(Path(arguments['image_path']))))
        service.store.write_json(directory/'worker_output.json',result)
        return result
    monkeypatch.setattr(service,'_run',run)


def test_registered_segment_expert_exports_original_masks_and_explicit_mapping(configured,monkeypatch):
    service,image,weights=configured
    fake_inference(service,monkeypatch)
    result=service.detect([str(image)],'fixture_seg',inference_size=960)
    assert result['status']=='completed',result
    row=result['images'][0]
    assert row['model_provenance']['weights_sha256']==sha256(weights)
    expected_definition=service.capabilities()['profiles']['fixture_seg']['definition_sha256']
    assert row['model_provenance']['definition_sha256']==expected_definition
    assert result['source']['definition_sha256']==expected_definition
    assert row['defects'][0]['class_code']=='C'
    mask=row['defects'][0]['mask_artifact']
    assert Path(mask['path']).is_file() and Image.open(mask['path']).size==(80,60)
    assert row['defects'][0]['area_pixels']==400
    # The old normalizer knows 'corrosion', but custom experts cannot claim a
    # class outside their server-owned map merely because a label looks familiar.
    assert row['unmapped_defects'][0]['original_class_name']=='corrosion'
    assert len(row['defects'])==1


def test_explicit_registered_map_cannot_escape_via_legacy_label_heuristic(configured,monkeypatch):
    service,image,_=configured
    service.registry['fixture_seg']['class_map']={'crack':'L'}
    service.registry['fixture_seg']['class_codes']=['L']
    fake_inference(service,monkeypatch)
    # Legacy 'efflorescence' labels normally force L->E. Explicit server maps
    # must remain authoritative instead of inventing an undeclared capability.
    original=service._run
    def run(*args,**kwargs):
        result=original(*args,**kwargs)
        result['defects'][0]['class_name']='efflorescence'
        return result
    service.registry['fixture_seg']['class_map']={'efflorescence':'L'}
    monkeypatch.setattr(service,'_run',run)
    result=service.detect([str(image)],'fixture_seg')
    assert result['status']=='completed'
    assert result['images'][0]['defects'][0]['class_code']=='L'


@pytest.mark.parametrize('option,expected',[('escape','MODEL_OUTPUT_ESCAPE'),('bad_hash','CHECKPOINT_HASH_MISMATCH')])
def test_registered_evidence_and_hash_guards(configured,monkeypatch,option,expected):
    service,image,_=configured
    fake_inference(service,monkeypatch,**{option:True})
    result=service.detect([str(image)],'fixture_seg')
    assert result['status']=='failed' and result['error']['code']==expected


def test_one_detection_job_never_switches_checkpoint_between_images(configured,monkeypatch,tmp_path):
    service,image,weights=configured
    second=tmp_path/'second.png'
    Image.new('RGB',(80,60),'white').save(second)
    fake_inference(service,monkeypatch)
    original=service._run
    calls=[]
    def run(*args,**kwargs):
        result=original(*args,**kwargs)
        calls.append(result)
        return result
    # Simulate a replacement after first-image artifact export, before second
    # worker starts; replacing the prior immutable snapshot is not involved.
    original_snapshot=service._snapshot
    def snapshot(path,directory):
        result=original_snapshot(path,directory)
        if len(calls)==1:
            weights.write_bytes(b'replacement fixture weights')
        return result
    monkeypatch.setattr(service,'_run',run)
    monkeypatch.setattr(service,'_snapshot',snapshot)
    result=service.detect([str(image),str(second)],'fixture_seg')
    assert result['error']['code']=='CHECKPOINT_HASH_MISMATCH' and len(calls)==1


@pytest.mark.parametrize('size',[False,0,True,321])
def test_registered_inference_size_is_not_silently_defaulted(configured,size):
    service,image,_=configured
    assert service.detect([str(image)],'fixture_seg',inference_size=size)['error']['code']=='INVALID_INFERENCE_SIZE'


def test_checkpoint_pin_optional_in_config_is_enforced(tmp_path):
    config,_=registration(tmp_path,checkpoint_sha256='0'*64)
    service=ModelService(ArtifactStore(tmp_path/'workspace',[tmp_path]),repo_path=tmp_path/'absent',python_path=Path(sys.executable),experts_config=config)
    try:
        assert service.capabilities()['profiles']['fixture_seg']['reason']=='CHECKPOINT_HASH_MISMATCH'
    finally:
        service.masks.close()


def test_worker_offline_mock_preserves_mask_and_weight_provenance(configured,monkeypatch,tmp_path):
    from facade_mcp import registered_expert_worker as worker
    service,image,weights=configured
    monkeypatch.setattr(worker,'_ultralytics',lambda *args:[('crack',0,.9,[10,5,30,25],np.ones((60,80),bool))])
    output=tmp_path/'worker'
    result=worker.run_registered(dict(expert=service.registry['fixture_seg'],image_path=str(image),
        output_dir=str(output),expected_weights_sha256=sha256(weights),model_profile='fixture_seg',threshold=.1,device='cpu'))
    assert result['model_provenance']['weights_sha256']==sha256(weights)
    assert result['model_provenance']['network_access'] is False
    assert result['defects'][0]['class_code']=='C'
    assert Image.open(result['defects'][0]['mask_path']).size==(80,60)


def test_bulging_is_business_class_and_not_an_opencv_mask():
    assert CLASS_NAMES['B']=='Bulging'
    with pytest.raises(ValueError,match='does not support B'):
        opencv_code({'class_code':'B'})


def test_builtin_composite_hashes_preserve_member_names(tmp_path):
    from facade_mcp.model_service import PROFILES
    repo=tmp_path/'source'
    (repo/'model_runner').mkdir(parents=True)
    (repo/'backend').mkdir()
    for name in ('yolo11m','yolo12m','rtdetr'):
        path=repo/PROFILES[name]['weights'][0]
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(name.encode())
    service=ModelService(ArtifactStore(tmp_path/'workspace',[tmp_path]),repo_path=repo,python_path=Path(sys.executable))
    try:
        profiles=service.capabilities()['profiles']
        hashes=profiles['ensemble']['weights_sha256']
        assert hashes=={name:profiles[name]['weights_sha256'] for name in ('yolo11m','yolo12m','rtdetr')}
        assert len(profiles['ensemble']['checkpoint_identity'])==64
    finally:
        service.masks.close()


def test_yolov9_official_api_unletterboxes_mask_to_original_pixels(configured,monkeypatch,tmp_path):
    """Mock official module API, exercise real crop/pad/mask/export geometry."""
    from contextlib import nullcontext
    from types import ModuleType
    from facade_mcp import registered_expert_worker as worker
    service,image,weights=configured
    repo=tmp_path/'official-yolov9-fixture'
    def module(name,**members):
        value=ModuleType(name)
        value.__file__=str(repo/Path(*name.split('.')).with_suffix('.py'))
        value.__path__=[]
        for key,entry in members.items():
            setattr(value,key,entry)
        monkeypatch.setitem(sys.modules,name,value)
        return value
    class Tensor:
        def __init__(self,array): self.array=array
        @property
        def shape(self): return self.array.shape
        def to(self,*a): return self
        def float(self): return self
        def __truediv__(self,number): return Tensor(self.array/number)
        def __getitem__(self,key): return Tensor(self.array[key])
    class Backend:
        stride=32
        names={0:'crack'}
        device='cpu'
        def __init__(self,*a,**k): pass
        def __call__(self,*a,**k): return np.zeros((1,1,38)),np.zeros((1,32,25,25))
    det=np.zeros((1,38),np.float32)
    det[0,:6]=[20,25,40,45,.9,0]
    def masks(*a,**k):
        mask=np.zeros((1,100,100),bool)
        mask[0,25:45,20:40]=True
        return mask
    def nms(*a,**k):
        assert k['nm']==32
        return [det]
    module('torch',from_numpy=lambda x:Tensor(x),inference_mode=nullcontext)
    module('models')
    module('models.common',DetectMultiBackend=Backend)
    module('utils')
    module('utils.augmentations',letterbox=lambda *a,**k:(np.zeros((100,100,3),np.uint8),(1.,1.),(10.,20.)))
    module('utils.general',check_img_size=lambda *a,**k:(100,100),non_max_suppression=nms,
           scale_boxes=lambda shape,box,target:np.asarray(box)-[10,20,10,20])
    module('utils.segment')
    module('utils.segment.general',process_mask=masks)
    module('utils.torch_utils',select_device=lambda *a:'cpu')
    expert={**service.registry['fixture_seg'],'backend':'yolov9','repo_path':str(repo)}
    result=worker.run_registered(dict(expert=expert,image_path=str(image),output_dir=str(tmp_path/'y9-output'),
        expected_weights_sha256=sha256(weights),model_profile='fixture_seg',threshold=.1,device='cpu'))
    row=result['defects'][0]
    assert row['bbox_xyxy']==[10.,5.,30.,25.]
    mask=np.asarray(Image.open(row['mask_path']))
    assert mask.shape==(60,80) and np.count_nonzero(mask)==400
    assert np.all(mask[5:25,10:30]==255)


def test_registered_subprocess_has_fixed_script_and_no_process_keys(configured,monkeypatch):
    import subprocess
    service,image,weights=configured
    monkeypatch.setenv('FACADE_IMAGE_API_KEY','not-for-child-process')
    def run(command,**options):
        assert command[0]==str(service.python)
        assert Path(command[1]).name=='registered_expert_worker.py'
        assert options['env'].get('FACADE_IMAGE_API_KEY') is None
        payload=json.loads(Path(command[3]).read_text())
        assert payload['expert']['weights_path']==str(weights)
        assert payload['expected_weights_sha256']==sha256(weights)
        result={'status':'completed','defects':[],
                'model_provenance':{'weights_sha256':sha256(weights),'source_image_sha256':sha256(Path(payload['image_path']))}}
        Path(command[5]).write_text(json.dumps(result))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(subprocess,'run',run)
    assert service.detect([str(image)],'fixture_seg')['status']=='completed'


def test_same_checkpoint_with_changed_class_map_has_distinct_definition(configured):
    service,_,_=configured
    before=service.capabilities()['profiles']['fixture_seg']
    service.registry['fixture_seg']['class_map']={'crack':'S','10':'B'}
    service.registry['fixture_seg']['class_codes']=['S','B']
    after=service.capabilities()['profiles']['fixture_seg']
    assert before['weights_sha256']==after['weights_sha256']
    assert before['definition_sha256']!=after['definition_sha256']
    service.registry['fixture_seg']['class_map']={'10':'B','crack':'S'}
    service.registry['fixture_seg']['class_codes']=['B','S']
    assert service.capabilities()['profiles']['fixture_seg']['definition_sha256']==after['definition_sha256']


def test_definition_tracks_yolov9_implementation_not_paths(tmp_path):
    config,weights=registration(tmp_path,backend='yolov9',repo_path='official')
    repo=tmp_path/'official'
    for relative in ('models/common.py','utils/augmentations.py','utils/general.py','utils/segment/general.py','utils/torch_utils.py'):
        path=repo/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# Synthetic official-source fixture\n')
    service=ModelService(ArtifactStore(tmp_path/'workspace',[tmp_path]),repo_path=tmp_path/'absent',
                         python_path=Path(sys.executable),experts_config=config)
    try:
        before=service.capabilities()['profiles']['fixture_seg']
        assert before['ready']
        (repo/'utils/segment/general.py').write_text('# Changed mask processing implementation\n')
        after=service.capabilities()['profiles']['fixture_seg']
        assert before['weights_sha256']==after['weights_sha256']==sha256(weights)
        assert before['definition_sha256']!=after['definition_sha256']
        assert before['implementation_sha256']['yolov9/utils/segment/general.py']!=after['implementation_sha256']['yolov9/utils/segment/general.py']
    finally:
        service.masks.close()


def test_builtin_definition_tracks_source_adapter_bytes(tmp_path):
    from facade_mcp.model_service import PROFILES
    repo=tmp_path/'source'
    path=repo/PROFILES['yolo11m']['weights'][0]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b'unchanged synthetic weights')
    adapter=repo/'model_runner/adapters/detection/yolo.py'
    adapter.parent.mkdir(parents=True)
    adapter.write_text('# original source\n')
    (repo/'backend').mkdir()
    service=ModelService(ArtifactStore(tmp_path/'workspace',[tmp_path]),repo_path=repo,python_path=Path(sys.executable))
    try:
        before=service.capabilities()['profiles']['yolo11m']
        adapter.write_text('# modified source\n')
        after=service.capabilities()['profiles']['yolo11m']
        assert before['weights_sha256']==after['weights_sha256']
        assert before['definition_sha256']!=after['definition_sha256']
    finally:
        service.masks.close()


def test_mutating_expert_definition_during_inference_rejects_job(configured,monkeypatch):
    service,image,_=configured
    fake_inference(service,monkeypatch)
    original=service._run
    def run(*args,**kwargs):
        result=original(*args,**kwargs)
        service.registry['fixture_seg']['full_image']=False
        return result
    monkeypatch.setattr(service,'_run',run)
    result=service.detect([str(image)],'fixture_seg')
    assert result['error']['code']=='EXPERT_DEFINITION_CHANGED'


def test_mutating_runtime_source_during_inference_rejects_job(configured, monkeypatch, tmp_path):
    service, image, _ = configured
    runtime = tmp_path/'runtime.py'
    runtime.write_text('# original implementation\n')
    service.registry['fixture_seg']['implementation_files'] = {'block': str(runtime)}
    fake_inference(service, monkeypatch)
    original = service._run
    def run(*args, **kwargs):
        result = original(*args, **kwargs)
        runtime.write_text('# changed implementation\n')
        return result
    monkeypatch.setattr(service, '_run', run)
    result = service.detect([str(image)], 'fixture_seg')
    assert result['error']['code'] == 'EXPERT_DEFINITION_CHANGED'
