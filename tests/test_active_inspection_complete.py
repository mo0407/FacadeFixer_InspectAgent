import asyncio
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from facade_mcp.image_inspection import run_image_inspection
from facade_mcp.playbook import Playbook
from test_image_inspection import FakeExperts, FakeVLM, action, execute, model_config, row, setup


class LockedExperts(FakeExperts):
    def __init__(self,*args,drift=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.drift=drift

    async def call(self,name,arguments):
        response=await super().call(name,arguments)
        digest='b'*64 if self.drift and len(self.calls)>1 else 'a'*64
        response['images'][0]['model_provenance']={'weights_sha256':digest,'library':'fixture','library_version':'1',
                                                 'source_files_sha256':{'fixture.py':'c'*64}}
        return response


def caps(**entries):
    return {'profiles':{name:{'ready':True,'full_image':True,'local_eligible':True,
        'class_codes':['C','S','B'],'weights_sha256':'a'*64,**changes} for name,changes in entries.items()}}


def test_new_local_candidate_becomes_next_selectable_anchor_and_pixel_crop(tmp_path):
    request,store,folder,pixels,_=setup(tmp_path,max_inspections=2,grid_rows=2,grid_cols=2)
    experts=FakeExperts(local=[row(box=(2,3,12,13))])
    vlm=FakeVLM(actions=[action('grid-2-2'),action('local-01-0001',scale=2)])
    result=execute(request,store,folder,experts,vlm)
    assert result['inspections_used']==2
    calls=[call for call in vlm.calls if call['stage']=='choose_inspection']
    assert 'local-01-0001' not in {a['id'] for a in calls[0]['anchors']}
    second_anchor=next(a for a in calls[1]['anchors'] if a['id']=='local-01-0001')
    assert second_anchor['bbox_xyxy']==[62,51,72,61]
    assert np.array_equal(experts.calls[2]['pixels'],pixels[46:66,57:77])
    assert (folder/'overview-00.png').is_file() and (folder/'overview-01.png').is_file()
    assert calls[0]['images'][0]=='overview-00.png' and calls[1]['images'][0]=='overview-01.png'
    final_anchors=json.loads((folder/'anchors.json').read_text(encoding='utf-8'))['anchors']
    assert 'local-02-0001' in {a['id'] for a in final_anchors}


def test_scope_mask_generates_wall_env_without_extra_visual_model_call(tmp_path):
    mask=np.zeros((96,120),dtype=np.uint8);mask[20:80,30:90]=255
    request,store,folder,_,_=setup(tmp_path,mask=mask)
    vlm=FakeVLM()
    execute(request,store,folder,FakeExperts(),vlm)
    assert not any(call['stage']=='visual_anchoring' for call in vlm.calls)
    regions=json.loads((folder/'scene_context.json').read_text(encoding='utf-8'))['regions']
    assert regions[0]['bbox_xyxy']==[30,20,90,80]
    assert regions[0]['scope_role']=='target_wall' and regions[0]['source']=='scope_mask'


def test_vlm_automatic_context_uses_normalized_coordinates_and_excludes_window_roi(tmp_path):
    request,store,folder,_,_=setup(tmp_path,max_inspections=1)
    class SceneVLM(FakeVLM):
        async def invoke(self,role,task,schema,**kwargs):
            if task['stage']=='visual_anchoring':
                return schema.model_validate({'summary':'Window context', 'regions':[
                    {'bbox_normalized':[.25,.25,.5,.5],'role':'window','description':'Visible window'}]})
            return await super().invoke(role,task,schema,**kwargs)
    experts=FakeExperts()
    with pytest.raises(ValueError,match='not eligible'):
        execute(request,store,folder,experts,SceneVLM(actions=[action('env-auto-01')]))
    anchors=json.loads((folder/'anchors.json').read_text(encoding='utf-8'))['anchors']
    region=next(a for a in anchors if a['id']=='env-auto-01')
    assert region['bbox_xyxy']==[30,24,60,48] and region['eligible_profiles']==[]
    assert len(experts.calls)==1


def test_expert_class_filter_controls_local_routing_without_falsifying_full_predictions(tmp_path):
    request,store,folder,_,_=setup(tmp_path,max_inspections=1,
        detector_profiles=['yolo11m','crack_specialist'],expert_class_filters={'yolo11m':['S']})
    experts=LockedExperts(full=[row('C')])
    vlm=FakeVLM(actions=[action('full-01-0001',profile='crack_specialist')])
    from facade_mcp.image_inspection import execute_image_inspection
    asyncio.run(execute_image_inspection(request,store,experts,vlm,Playbook(version='fixture'),folder,
        expert_capabilities=caps(yolo11m={},crack_specialist={'class_codes':['C'],'full_image':False,'backend':'ultralytics'})))
    choose=next(c for c in vlm.calls if c['stage']=='choose_inspection')
    target=next(a for a in choose['anchors'] if a['id']=='full-01-0001')
    assert target['eligible_profiles']==['crack_specialist']
    assert [c['arguments']['model_profile'] for c in experts.calls]==['yolo11m','crack_specialist']
    assert experts.calls[1]['arguments']['inference_size']==request.inference_size


def test_profile_outside_anchor_specialty_is_rejected_before_extra_inference(tmp_path):
    request,store,folder,_,_=setup(tmp_path,max_inspections=1,expert_class_filters={'yolo11m':['S']})
    with pytest.raises(ValueError,match='not eligible'):
        execute(request,store,folder,FakeExperts(full=[row('C')]),FakeVLM(actions=[action('full-01-0001')]))


def test_entire_run_checks_checkpoint_identity_not_just_each_worker_call(tmp_path):
    request,store,folder,_,_=setup(tmp_path,max_inspections=1)
    from facade_mcp.image_inspection import execute_image_inspection
    with pytest.raises(ValueError,match='checkpoint changed'):
        asyncio.run(execute_image_inspection(request,store,LockedExperts(drift=True),FakeVLM(actions=[action()]),
            Playbook(version='fixture'),folder,expert_capabilities=caps(yolo11m={})))


def test_missing_server_identity_cannot_be_treated_as_frozen_model(tmp_path):
    request,store,folder,_,_=setup(tmp_path)
    from facade_mcp.image_inspection import execute_image_inspection
    with pytest.raises(ValueError,match='identity missing'):
        asyncio.run(execute_image_inspection(request,store,FakeExperts(),FakeVLM(),Playbook(version='fixture'),folder,
            expert_capabilities=caps(yolo11m={})))


def test_latest_verified_p0_is_loaded_unless_explicit_snapshot_is_given(tmp_path,monkeypatch):
    request,store,_,_,_=setup(tmp_path)
    from facade_mcp.rule_lifecycle import RuleLifecycle
    monkeypatch.setattr(RuleLifecycle,'latest',lambda self:Playbook(version='adopted-p0'))
    result=asyncio.run(run_image_inspection(request,store,model_config(),experts=FakeExperts()))
    assert result['playbook_version']=='adopted-p0'
    explicit=asyncio.run(run_image_inspection(request,store,model_config(),Playbook(version='explicit'),experts=FakeExperts()))
    assert explicit['playbook_version']=='explicit'


def test_ensemble_member_weight_swap_cannot_pass_identity_lock():
    from facade_mcp.image_inspection import _lock_expert
    declared={'yolo11m':'a'*64,'yolo12m':'b'*64}
    changed={'models':[{'profile':'yolo11m','weights_sha256':'b'*64},
                       {'profile':'yolo12m','weights_sha256':'a'*64}]}
    with pytest.raises(ValueError,match='checkpoint changed'):
        _lock_expert('ensemble',changed,{}, {'weights_sha256':declared},True,{})


def test_same_weights_but_changed_definition_or_implementation_are_rejected():
    from facade_mcp.image_inspection import _lock_expert
    provenance={'weights_sha256':'a'*64,'library':'fixture','library_version':'1','definition_sha256':'c'*64}
    capability={'weights_sha256':'a'*64,'definition_sha256':'c'*64}
    locks,implementations={},{}
    _lock_expert('fixture',provenance,locks,capability,True,implementations)
    with pytest.raises(ValueError,match='definition differs'):
        _lock_expert('fixture',{**provenance,'definition_sha256':'d'*64},locks,capability,True,implementations)
    with pytest.raises(ValueError,match='implementation drifted'):
        _lock_expert('fixture',{**provenance,'library_version':'2'},locks,capability,True,implementations)
