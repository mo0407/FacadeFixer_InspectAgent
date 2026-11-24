"""Figure-1 baseline contracts with synthetic images and no external calls."""
import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from facade_mcp.baseline import (BaselineConfig, UncertainGatewayOutcome, execute_baseline_image, _overview, _review_evidence,
                                 run_baseline_batch, run_baseline_experts_batch)
from facade_mcp.model_gateway import GatewayError
from facade_mcp.storage import sha256


PROFILES=['yolo11m','cubit_yolov9e','cubit_efficientvit_m5']
CAPS={p:{'ready':True,'full_image':True,'local_eligible':True,'class_codes':['C','S','T'],
         'weights_sha256':str(i)*64,'definition_sha256':str(i+3)*64}
      for i,p in enumerate(PROFILES,1)}


class Experts:
    def __init__(self): self.calls=[]
    async def call(self,name,args):
        assert name=='facade_models_detect'
        self.calls.append(args)
        path=Path(args['image_paths'][0]); profile=args['model_profile']
        with Image.open(path) as image: assert image.size==(640,640)
        if path.name.startswith('review-'):
            entries=[dict(class_code='C',class_name='crack',bbox_xyxy=[64,64,128,128],confidence=.8)]
        elif profile==PROFILES[0]:
            entries=[dict(class_code='C',class_name='crack',bbox_xyxy=[64,192,320,448],confidence=.25)]
        elif profile==PROFILES[1]:
            entries=[dict(class_code='S',class_name='spalling',bbox_xyxy=[64,192,320,448],confidence=.8)]
        else:
            entries=[dict(class_code='T',class_name='stain',bbox_xyxy=[64,192,320,448],confidence=.9)]
        p=CAPS[profile]
        return {'status':'completed','coordinate_space':'original_pixels','bbox_convention':'zero_based_xyxy_max_exclusive',
                'images':[{'width':640,'height':640,'source_sha256':sha256(path),'defects':entries,'unmapped_defects':[],
                           'model_provenance':{'weights_sha256':p['weights_sha256'], 'definition_sha256':p['definition_sha256'],
                                               'backend':'synthetic'}}]}


class VLM:
    def __init__(self,review=False,multiple=False):
        self.review=review;self.multiple=multiple;self.calls=[];self.trace=[];self.model_identities={}
    async def invoke(self,role,task,schema,**kwargs):
        assert role=='generator'
        assert all(Path(p).is_file() for p in kwargs['image_paths'])
        self.calls.append(task)
        self.trace.append({'role':'generator','adapter':'mock','status':'completed'})
        decisions=[]
        for i,c in enumerate(task['candidates']):
            do_review=task['stage']=='baseline_judge' and self.review and (i==0 or self.multiple)
            decisions.append(dict(candidate_id=c['candidate_id'],decision='review' if do_review else 'accept',
                                  reason='synthetic review',review_profile=PROFILES[0] if do_review else None,
                                  review_scale=1 if do_review else None))
        return schema.model_validate({'decisions':decisions})


def fixture_image(path, color='white'):
    Image.new('RGB',(100,50),color).save(path)
    return path


def test_each_vlm_call_receives_only_its_numbered_grid_and_roi_evidence(tmp_path):
    class ManyExperts(Experts):
        async def call(self,name,args):
            result=await super().call(name,args)
            defects=result['images'][0]['defects']
            if not Path(args['image_paths'][0]).name.startswith('review-'):
                defects[:]=[dict(class_code='C',class_name='crack',
                    bbox_xyxy=[64+i*8,192+i*4,320+i*8,448+i*4],confidence=.3) for i in range(5)]
            return result
    class EvidenceVLM(VLM):
        def __init__(self): super().__init__(review=True); self.images=[]
        async def invoke(self,role,task,schema,**kwargs):
            self.images.append([Path(p) for p in kwargs['image_paths']])
            answer=await super().invoke(role,task,schema,**kwargs)
            self.review=False
            return answer
    source=fixture_image(tmp_path/'source.png'); out=tmp_path/'out'; vlm=EvidenceVLM()
    result=asyncio.run(execute_baseline_image(source,out,ManyExperts(),vlm,CAPS,BaselineConfig(),simulation=True))
    assert [len(t['candidates']) for t in vlm.calls]==[8,7,2]
    assert len(result['vlm_evidence'])==len(vlm.calls)
    for entry,task,paths in zip(result['vlm_evidence'],vlm.calls,vlm.images):
        chunk=task['candidates']
        assert entry['candidates']==[{'candidate_id':c['candidate_id'],'overlay_label':f'#{i}'}
                                     for i,c in enumerate(chunk,1)]
        assert [row['file'] for row in entry['images']]==[p.name for p in paths]
        assert all(row['sha256']==sha256(p)==result['artifacts_sha256'][p.name]
                   for row,p in zip(entry['images'],paths))
        expected=tmp_path/'expected.jpg'
        with Image.open(source) as image: _overview(image.convert('RGB'),chunk,expected,compact=True)
        assert expected.read_bytes()==paths[0].read_bytes()
        if task['stage']=='baseline_review':
            with Image.open(out/'review-original.png') as roi:
                _review_evidence(roi.convert('RGB'),task['crop_bbox_original'],None,chunk,expected,compact=True)
            assert expected.read_bytes()==paths[1].read_bytes()
    assert len({c['candidate_id'] for t in vlm.calls[:2] for c in t['candidates']})==15


def test_tampered_per_call_visual_evidence_cannot_resume(tmp_path):
    image=fixture_image(tmp_path/'a.png');folder=tmp_path/'batch';experts=Experts();vlm=VLM()
    config=BaselineConfig(expected_count=1)
    summary=asyncio.run(run_baseline_batch([image],folder,experts,vlm,CAPS,config,simulation=True))
    result_path=Path(summary['completed'][0]['result_file'])
    result=json.loads(result_path.read_text())
    (result_path.parent/result['vlm_evidence'][0]['images'][0]['file']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='artifacts changed'):
        asyncio.run(run_baseline_batch([image],folder,experts,vlm,CAPS,config,simulation=True))


def test_original_coordinates_and_exact_boundary_with_one_roi(tmp_path):
    image=fixture_image(tmp_path/'input.png');experts=Experts();vlm=VLM(review=True)
    result=asyncio.run(execute_baseline_image(image,tmp_path/'output',experts,vlm,CAPS,BaselineConfig(),simulation=True))
    assert result['simulation'] is True and result['review_count']==1
    assert len(experts.calls)==4 and all(call['inference_size']==640 and call['threshold']==.001 for call in experts.calls)
    first=result['experts'][PROFILES[0]]['candidates'][0]
    assert first['bbox_xyxy']==[10,5,50,45] and first['kind']=='Ghost' and first['confidence']==.25
    assert result['experts'][PROFILES[1]]['candidates'][0]['kind']=='Normal'
    assert result['context_candidates'][0]['kind']=='Env'
    assert all(c['decision']=='accept' for c in result['candidates'])
    local=next(c for c in result['candidates'] if c['candidate_id'].startswith('review-'))
    assert local['bbox_xyxy']==[14,9,18,13]
    assert next(r for r in result['trajectory'] if r['stage']=='review')['crop_bbox_original']==[10,5,50,45]
    with Image.open(tmp_path/'output/baseline-overview.jpg') as overview: assert overview.size==(1024,680)
    with Image.open(tmp_path/'output/full-input-640.png') as square: assert square.size==(640,640)


def test_batch_resume_requires_same_inputs_and_intact_evidence(tmp_path):
    a=fixture_image(tmp_path/'a.png');b=fixture_image(tmp_path/'b.png','red')
    experts=Experts();vlm=VLM();folder=tmp_path/'batch';cfg=BaselineConfig(expected_count=2)
    first=asyncio.run(run_baseline_batch([a,b],folder,experts,vlm,CAPS,cfg,simulation=True))
    assert len(first['completed'])==2 and not first['failed'] and len(experts.calls)==6
    again=asyncio.run(run_baseline_batch([a,b],folder,experts,vlm,CAPS,cfg,simulation=True))
    assert all(x['resumed'] for x in again['completed']) and len(experts.calls)==6
    result=Path(again['completed'][0]['result_file'])
    (result.parent/'baseline-overview.jpg').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='artifacts changed'):
        asyncio.run(run_baseline_batch([a,b],folder,experts,vlm,CAPS,cfg,simulation=True))


def test_expert_only_stage_resumes_into_vlm_without_repeating_experts(tmp_path):
    a=fixture_image(tmp_path/'a.png');b=fixture_image(tmp_path/'b.png','red')
    experts=Experts();vlm=VLM();folder=tmp_path/'batch';cfg=BaselineConfig(expected_count=2)
    first=asyncio.run(run_baseline_experts_batch([a,b],folder,experts,CAPS,cfg))
    assert first['status']=='experts_only' and first['vlm_called'] is False
    assert len(first['completed'])==2 and len(experts.calls)==6 and not vlm.calls
    stage=json.loads(Path(first['completed'][0]['result_file']).read_text())
    assert stage['status']=='experts_completed' and stage['vlm_called'] is False
    assert stage['schema']=='facade-baseline-experts/v1' and 'candidates' not in stage
    second=asyncio.run(run_baseline_experts_batch([a,b],folder,experts,CAPS,cfg))
    assert all(row['resumed'] for row in second['completed']) and len(experts.calls)==6
    final=asyncio.run(run_baseline_batch([a,b],folder,experts,vlm,CAPS,cfg,simulation=True))
    assert len(final['completed'])==2 and len(experts.calls)==6 and len(vlm.calls)==2


@pytest.mark.parametrize('change', ['edit_decision', 'remove_checksum'])
def test_batch_resume_rejects_changed_final_decisions(tmp_path,change):
    image=fixture_image(tmp_path/'a.png'); experts=Experts(); vlm=VLM();folder=tmp_path/'batch'
    config=BaselineConfig(expected_count=1)
    initial=asyncio.run(run_baseline_batch([image],folder,experts,vlm,CAPS,config,simulation=True))
    result=Path(initial['completed'][0]['result_file'])
    assert (result.parent/'baseline-result.sha256.json').is_file()
    if change=='edit_decision':
        edited=json.loads(result.read_text())
        edited['candidates'][0]['decision']='reject'
        result.write_text(json.dumps(edited))
    else:
        (result.parent/'baseline-result.sha256.json').unlink()
    with pytest.raises(ValueError,match='integrity record|metadata changed'):
        asyncio.run(run_baseline_batch([image],folder,experts,vlm,CAPS,config,simulation=True))
    assert len(vlm.calls)==1 and len(experts.calls)==3


def test_expert_only_tampered_evidence_cannot_be_promoted(tmp_path):
    a=fixture_image(tmp_path/'a.png');experts=Experts();folder=tmp_path/'batch';cfg=BaselineConfig(expected_count=1)
    initial=asyncio.run(run_baseline_experts_batch([a],folder,experts,CAPS,cfg))
    result=Path(initial['completed'][0]['result_file'])
    (result.parent/'full-01-expert.json').write_text('{}')
    final=asyncio.run(run_baseline_batch([a],folder,experts,VLM(),CAPS,cfg,simulation=True))
    assert final['completed']==[] and final['failed'][0]['error_type']=='ValueError'
    assert len(experts.calls)==3


def test_expert_only_stage_metadata_tamper_blocks_resume(tmp_path):
    image=fixture_image(tmp_path/'a.png');experts=Experts();folder=tmp_path/'batch'
    cfg=BaselineConfig(expected_count=1)
    initial=asyncio.run(run_baseline_experts_batch([image],folder,experts,CAPS,cfg))
    stage=Path(initial['completed'][0]['result_file'])
    changed=json.loads(stage.read_text())
    changed['observations'][0]['confidence']=.99
    stage.write_text(json.dumps(changed))
    resumed=asyncio.run(run_baseline_experts_batch([image],folder,experts,CAPS,cfg))
    assert resumed['completed']==[] and resumed['failed'][0]['error_type']=='ValueError'
    assert len(experts.calls)==3
    final=asyncio.run(run_baseline_batch([image],folder,experts,VLM(),CAPS,cfg,simulation=True))
    assert final['completed']==[] and final['failed'][0]['error_type']=='ValueError'


def test_no_candidates_still_calls_vlm_and_records_auditable_empty_verdict(tmp_path):
    class EmptyExperts(Experts):
        async def call(self,name,args):
            result=await super().call(name,args)
            result['images'][0]['defects']=[]
            return result
    image=fixture_image(tmp_path/'a.png'); experts=EmptyExperts(); vlm=VLM()
    result=asyncio.run(execute_baseline_image(image,tmp_path/'output',experts,vlm,CAPS,
                                               BaselineConfig(),simulation=True))
    assert result['status']=='completed' and result['candidates']==[]
    assert len(vlm.calls)==1 and vlm.calls[0]['stage']=='baseline_no_candidates'
    assert vlm.calls[0]['candidates']==[] and vlm.calls[0]['review_allowed'] is False
    assert len(result['model_trace'])==1 and result['trajectory'][0]['decision_count']==0
    assert 'baseline-expert-result.sha256.json' in result['artifacts_sha256']


def test_expert_batch_stops_after_repeated_failures(tmp_path):
    paths=[fixture_image(tmp_path/f'{i}.png',(i,0,0)) for i in range(5)]
    class FailedExperts(Experts):
        async def call(self,name,args):
            self.calls.append(args)
            raise RuntimeError('synthetic model failure')
    experts=FailedExperts()
    result=asyncio.run(run_baseline_experts_batch(paths,tmp_path/'batch',experts,CAPS,
                                                   BaselineConfig(expected_count=5)))
    assert result['halted_reason']=='consecutive_expert_failures'
    assert len(result['failed'])==3 and len(experts.calls)==3


def test_expert_batch_stops_immediately_when_runtime_changes_between_images(tmp_path):
    paths=[fixture_image(tmp_path/f'{i}.png',(i,0,0)) for i in range(3)]
    class DriftingExperts(Experts):
        async def call(self,name,args):
            result=await super().call(name,args)
            if len(self.calls)>3:
                result['images'][0]['model_provenance']['backend']='changed-runtime'
            return result
    experts=DriftingExperts()
    result=asyncio.run(run_baseline_experts_batch(paths,tmp_path/'batch',experts,CAPS,
                                                   BaselineConfig(expected_count=3)))
    assert result['halted_reason']=='expert_implementation_changed'
    assert len(result['completed'])==1 and len(result['failed'])==1
    assert len(experts.calls)==6
    assert set(result['implementation_identities'])==set(PROFILES)


def test_expert_batch_resume_checks_implementation_identity_across_cached_images(tmp_path):
    paths=[fixture_image(tmp_path/f'{i}.png',(i,0,0)) for i in range(2)]
    experts=Experts(); folder=tmp_path/'batch'; config=BaselineConfig(expected_count=2)
    result=asyncio.run(run_baseline_experts_batch(paths,folder,experts,CAPS,config))
    second=Path(result['completed'][1]['result_file'])
    changed=json.loads(second.read_text())
    changed['implementation_identities'][PROFILES[0]]='f'*64
    second.write_text(json.dumps(changed))
    (second.parent/'baseline-expert-result.sha256.json').write_text(
        json.dumps({'sha256':hashlib.sha256(second.read_bytes()).hexdigest()}))
    resumed=asyncio.run(run_baseline_experts_batch(paths,folder,experts,CAPS,config))
    assert resumed['halted_reason']=='expert_implementation_changed'
    assert len(resumed['completed'])==1 and resumed['completed'][0]['resumed'] is True
    assert len(experts.calls)==6


def test_vlm_batch_checks_expert_drift_before_any_new_vlm_request(tmp_path):
    paths=[fixture_image(tmp_path/f'{i}.png',(i,0,0)) for i in range(2)]
    class DriftingExperts(Experts):
        async def call(self,name,args):
            result=await super().call(name,args)
            if len(self.calls)>3:
                result['images'][0]['model_provenance']['backend']='changed-runtime'
            return result
    experts=DriftingExperts(); vlm=VLM()
    result=asyncio.run(run_baseline_batch(paths,tmp_path/'batch',experts,vlm,CAPS,
                                           BaselineConfig(expected_count=2),simulation=True))
    assert result['halted_reason']=='expert_implementation_changed'
    assert len(result['completed'])==1 and len(vlm.calls)==1 and len(experts.calls)==6


def test_review_budget_violation_records_failure_without_fake_success(tmp_path):
    image=fixture_image(tmp_path/'a.png')
    result=asyncio.run(run_baseline_batch([image],tmp_path/'batch',Experts(),VLM(review=True,multiple=True),CAPS,
                                           BaselineConfig(expected_count=1),simulation=True))
    assert result['completed']==[] and result['failed'][0]['error_type']=='GatewayError'
    assert result['halted_reason']=='vlm_gateway_failure'
    assert not list((tmp_path/'batch').glob('*/baseline-result.json'))
    assert list((tmp_path/'batch').glob('*/baseline-failure.json'))


def test_unknown_vlm_outcome_halts_batch_and_is_not_automatically_retried(tmp_path):
    image=fixture_image(tmp_path/'a.png'); second=fixture_image(tmp_path/'b.png','red')
    class TimeoutVLM(VLM):
        async def invoke(self, role, task, schema, **kwargs):
            self.calls.append(task)
            self.trace.append({'status':'request_outcome_unknown','error_code':'api_timeout'})
            raise GatewayError('Request timed out; billing/outcome unknown')
    vlm=TimeoutVLM(); experts=Experts();folder=tmp_path/'batch';cfg=BaselineConfig(expected_count=2)
    first=asyncio.run(run_baseline_batch([image,second],folder,experts,vlm,CAPS,cfg,simulation=True))
    assert first['halted_reason']=='vlm_gateway_failure' and len(vlm.calls)==1 and len(experts.calls)==3
    second_run=asyncio.run(run_baseline_batch([image,second],folder,experts,vlm,CAPS,cfg,simulation=True))
    assert second_run['halted_reason']=='previous_api_outcome_unknown' and len(vlm.calls)==1
    with pytest.raises(ValueError,match='explicit acknowledgement'):
        asyncio.run(run_baseline_batch([image,second],folder,experts,vlm,CAPS,cfg,simulation=True,retry_failed=True))


def test_completed_vlm_decisions_survive_interruption_without_resending(tmp_path, monkeypatch):
    import facade_mcp.baseline as module
    image=fixture_image(tmp_path/'a.png');out=tmp_path/'output';experts=Experts();vlm=VLM()
    original=module._json
    def fail_before_result(path,value):
        if Path(path).name=='baseline-result.json': raise OSError('simulated interrupted writer')
        return original(path,value)
    monkeypatch.setattr(module,'_json',fail_before_result)
    with pytest.raises(OSError,match='interrupted writer'):
        asyncio.run(execute_baseline_image(image,out,experts,vlm,CAPS,BaselineConfig(),simulation=True))
    assert len(vlm.calls)==1 and (out/'vlm-call-0001.json').is_file()
    monkeypatch.setattr(module,'_json',original)
    result=asyncio.run(execute_baseline_image(image,out,experts,vlm,CAPS,BaselineConfig(),simulation=True))
    assert len(vlm.calls)==1 and len(result['model_trace'])==1
    assert result['artifacts_sha256']['vlm-call-0001.json']==sha256(out/'vlm-call-0001.json')


def test_unknown_interrupted_vlm_request_needs_acknowledgement(tmp_path):
    image=fixture_image(tmp_path/'a.png');out=tmp_path/'output';experts=Experts();vlm=VLM()
    asyncio.run(execute_baseline_image(image,out,experts,vlm,CAPS,BaselineConfig(),simulation=True))
    (out/'baseline-result.json').unlink()
    (out/'vlm-call-0001.json').rename(out/'vlm-call-0001.pending.json')
    with pytest.raises(UncertainGatewayOutcome):
        asyncio.run(execute_baseline_image(image,out,experts,vlm,CAPS,BaselineConfig(),simulation=True))
    assert len(vlm.calls)==1
    asyncio.run(execute_baseline_image(image,out,experts,vlm,CAPS,BaselineConfig(),simulation=True,
                                       acknowledge_unknown=True))
    assert len(vlm.calls)==2 and (out/'vlm-call-0001.pending-ack-1.json').is_file()
