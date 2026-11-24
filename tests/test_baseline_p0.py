import asyncio
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from facade_mcp.baseline_p0 import match_baseline_batch, match_baseline_case, propose_initial_p0
from facade_mcp.baseline_p0_coverage import propose_full_coverage_p0
from facade_mcp.failure_memory import FailureMemory
from facade_mcp.model_gateway import ModelConfig
from facade_mcp.rule_lifecycle import RuleLifecycle
from facade_mcp.storage import ArtifactStore, sha256


def fixture_case(tmp_path, *, simulation=False, invalid=False, color='gray'):
    image=tmp_path/'input.jpg'
    Image.new('RGB',(100,80),color).save(image)
    labels=tmp_path/'input.txt'
    lines=['0 .1 .1 .4 .1 .4 .4 .1 .4',
           '1 .5 .1 .7 .1 .7 .3 .5 .3',
           '1 .1 .6 .3 .6 .3 .8 .1 .8']
    if invalid:
        lines.append('0 .5 .2 .5 .3 .5 .4')
    labels.write_text('\n'.join(lines)+'\n')
    def candidate(identity, cls, box, decision):
        return dict(candidate_id=identity,class_code=cls,bbox_xyxy=box,
                    class_name=cls,confidence=.9,model_profile='expert',decision=decision)
    artifact_names=['full-01-expert.json','full-02-expert.json','full-03-expert.json']
    for name in artifact_names:
        (tmp_path/name).write_text('{}')
    Image.new('RGB',(100,80),color).save(tmp_path/'baseline-overview.jpg')
    artifact_names.append('baseline-overview.jpg')
    data={'schema':'facade-baseline-image/v1','status':'completed','simulation':simulation,
        'image':{'path':str(image),'sha256':sha256(image),'width':100,'height':80},
        'experts':{name:{'weight_identity':{'checkpoint':digest*64},'response_file':path} for name,digest,path in
                   [('yolo11m','a',artifact_names[0]),('cubit_yolov9e','b',artifact_names[1]),
                    ('cubit_efficientvit_m5','c',artifact_names[2])]},
        'artifacts_sha256':{name:sha256(tmp_path/name) for name in artifact_names},
        'overview_file':'baseline-overview.jpg',
        'experiment_sha256':'d'*64,
        'model_trace':[{'adapter':'chat_completions','status':'completed','actual_model':'fixture-vlm',
                        'model_identity_verified':True}],
        'candidates':[candidate('tp','C',[10,8,40,32],'accept'),
                      candidate('fp','C',[70,40,90,60],'accept'),
                      candidate('deferred','S',[50,8,70,24],'defer'),
                      candidate('context','V',[1,1,5,5],'accept'),
                      candidate('rejected','C',[1,1,5,5],'reject')],
        'trajectory':[{'action':'review','candidate_id':'deferred'}]}
    baseline=tmp_path/'baseline-result.json'
    baseline.write_text(json.dumps(data))
    (tmp_path/'baseline-result.sha256.json').write_text(json.dumps({'sha256':sha256(baseline)}))
    return ArtifactStore(tmp_path/'workspace',[tmp_path]),image,labels,baseline


def evaluate(store,labels,baseline,**kwargs):
    return match_baseline_case(store,str(baseline),str(labels),{'0':'C','1':'S'},**kwargs)


def test_per_instance_matching_respects_defer_and_unannotated_classes(tmp_path):
    store,_,labels,baseline=fixture_case(tmp_path)
    summary=evaluate(store,labels,baseline,reference_complete=True)
    assert summary['counts']=={'tp':1,'fp':1,'fn':1,'unverified_predictions':0,
                               'unresolved_references':1,'reference_issues':0}
    report=json.loads(open(summary['match_path']).read())
    assert report['matches'][0]['candidate_id']=='tp'
    assert report['false_positives'][0]['candidate_id']=='fp'
    assert report['false_negatives'][0]['reference_id']=='gt-0003'
    assert report['unresolved_references'][0]['reference_id']=='gt-0002'
    assert {row['candidate_id'] for row in report['excluded']}=={'context','rejected'}
    events=FailureMemory(store.root).summary()['events']
    assert {event['reason_code'] for event in events}=={'false_positive','missed_defect'}
    assert all(event['context']['source_sha256']==report['image']['sha256'] for event in events)
    assert evaluate(store,labels,baseline,reference_complete=True)['event_ids']==summary['event_ids']
    assert len(FailureMemory(store.root).summary()['events'])==2


def test_unmatched_predictions_are_not_called_fp_without_completeness(tmp_path):
    store,_,labels,baseline=fixture_case(tmp_path)
    summary=evaluate(store,labels,baseline)
    assert summary['counts']['fp']==0
    assert summary['counts']['unverified_predictions']==1
    assert {event['reason_code'] for event in FailureMemory(store.root).summary()['events']}=={'missed_defect'}


def test_invalid_reference_suspends_whole_image_instead_of_inventing_errors(tmp_path):
    store,_,labels,baseline=fixture_case(tmp_path,invalid=True)
    summary=evaluate(store,labels,baseline,reference_complete=True)
    assert summary['status']=='needs_reference_review'
    assert summary['counts']['reference_issues']==1
    assert summary['counts']['fp']==summary['counts']['fn']==0
    assert summary['event_ids']==[]
    report=json.loads(open(summary['match_path']).read())
    assert report['labels']['issues']==[{'line':4,'reason':'degenerate_polygon'}]


def test_changed_image_or_missing_explicit_class_mapping_is_rejected(tmp_path):
    store,image,labels,baseline=fixture_case(tmp_path)
    with pytest.raises(ValueError,match='class-ID'):
        match_baseline_case(store,str(baseline),str(labels),{})
    Image.new('RGB',(100,80),'white').save(image)
    with pytest.raises(ValueError,match='image bytes'):
        evaluate(store,labels,baseline)


def test_real_baseline_requires_intact_result_sidecar(tmp_path):
    store,_,labels,baseline=fixture_case(tmp_path)
    sidecar=baseline.with_name('baseline-result.sha256.json')
    sidecar.write_text(json.dumps({'sha256':'0'*64}))
    with pytest.raises(ValueError,match='sidecar hash'):
        evaluate(store,labels,baseline)
    sidecar.unlink()
    with pytest.raises(ValueError,match='requires its separate'):
        evaluate(store,labels,baseline)


def mock_settings():
    return ModelConfig.model_validate({'schema_version':'model-gateway-v1','api_enabled':False,
        'connections':{'mock':{'adapter':'mock','model':'mock-test','supports_images':True}},
        'roles':{'generator':'mock','reflector':'mock','curator':'mock'}})


def test_candidate_p0_reuses_evidence_and_is_not_published(tmp_path):
    store,_,labels,baseline=fixture_case(tmp_path)
    match=evaluate(store,labels,baseline,reference_complete=True)
    proposal=asyncio.run(propose_initial_p0(store,mock_settings(),[match['job_id']]))
    assert proposal['status']=='candidate_only' and not proposal['promoted']
    assert proposal['mock_generated']
    assert RuleLifecycle(store).latest() is None
    metadata=json.loads(open(proposal['candidate_metadata_path']).read())
    assert set(metadata['evidence_event_ids'])==set(match['event_ids'])


def test_mock_or_unverified_baseline_cannot_create_real_history(tmp_path):
    store,_,labels,baseline=fixture_case(tmp_path,simulation=True)
    match=evaluate(store,labels,baseline,reference_complete=True)
    assert match['simulation']
    with pytest.raises(ValueError,match='non-mock observed failure'):
        asyncio.run(propose_initial_p0(store,mock_settings(),[match['job_id']]))


def test_mutating_match_evidence_rejects_proposal(tmp_path):
    store,_,labels,baseline=fixture_case(tmp_path)
    match=evaluate(store,labels,baseline,reference_complete=True)
    report=json.loads(Path(match['match_path']).read_text())
    frozen_labels=Path(report['labels']['path'])
    frozen_labels.write_text(frozen_labels.read_text()+'0 .1 .1 .2 .1 .2 .2\n')
    with pytest.raises(ValueError,match='evidence hash changed'):
        asyncio.run(propose_initial_p0(store,mock_settings(),[match['job_id']]))


def test_original_reference_edit_preserves_historical_snapshot(tmp_path):
    store,_,labels,baseline=fixture_case(tmp_path)
    match=evaluate(store,labels,baseline,reference_complete=True)
    original=labels.read_text()
    labels.write_text(original+'0 .1 .1 .2 .1 .2 .2\n')
    result=asyncio.run(propose_initial_p0(store,mock_settings(),[match['job_id']]))
    assert result['status']=='candidate_only'
    report=json.loads(Path(match['match_path']).read_text())
    assert Path(report['labels']['path']).read_text()==original


def test_expert_artifacts_are_hash_checked_at_match_and_proposal(tmp_path):
    store,_,labels,baseline=fixture_case(tmp_path)
    match=evaluate(store,labels,baseline,reference_complete=True)
    (tmp_path/'full-01-expert.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='evidence hash changed'):
        asyncio.run(propose_initial_p0(store,mock_settings(),[match['job_id']]))
    with pytest.raises(ValueError,match='artifact changed'):
        evaluate(store,labels,baseline,reference_complete=True)


def test_synthetic_figure1_output_feeds_matcher_without_becoming_real_p0(tmp_path):
    from facade_mcp.baseline import BaselineConfig, execute_baseline_image
    names=['yolo11m','cubit_yolov9e','cubit_efficientvit_m5']
    caps={name:{'ready':True,'full_image':True,'local_eligible':True,'class_codes':['C','S'],
                'weights_sha256':str(i)*64,'definition_sha256':str(i+3)*64}
          for i,name in enumerate(names,1)}
    class Experts:
        async def call(self, tool, args):
            path=Path(args['image_paths'][0]); profile=args['model_profile']; p=caps[profile]
            return {'status':'completed','coordinate_space':'original_pixels',
                'bbox_convention':'zero_based_xyxy_max_exclusive',
                'images':[{'width':640,'height':640,'source_sha256':sha256(path),
                    'defects':[{'class_code':'C','class_name':'crack','bbox_xyxy':[64,192,320,448],
                                'confidence':.8}], 'unmapped_defects':[],
                    'model_provenance':{'weights_sha256':p['weights_sha256'],
                                        'definition_sha256':p['definition_sha256'],'backend':'fixture'}}]}
    class VLM:
        def __init__(self): self.trace=[]; self.model_identities={}
        async def invoke(self, role, task, schema, **kwargs):
            self.trace.append({'role':role,'adapter':'mock','status':'completed'})
            return schema.model_validate({'decisions':[{'candidate_id':x['candidate_id'],
                'decision':'accept','reason':'synthetic','review_profile':None,'review_scale':None}
                for x in task['candidates']]})
    image=tmp_path/'input.png'
    Image.new('RGB',(100,50),'gray').save(image)
    labels=tmp_path/'input.txt'
    labels.write_text('0 .1 .1 .5 .1 .5 .9 .1 .9\n')
    out=tmp_path/'baseline'
    asyncio.run(execute_baseline_image(image,out,Experts(),VLM(),caps,BaselineConfig(),simulation=True,
                                       experiment_sha256='d'*64))
    store=ArtifactStore(tmp_path/'workspace',[tmp_path])
    matched=match_baseline_case(store,str(out/'baseline-result.json'),str(labels),{'0':'C'},reference_complete=True)
    assert matched['counts']['tp']==1 and matched['counts']['fp']==2
    report=json.loads(Path(matched['match_path']).read_text())
    assert all(x['match_diagnostic']=='duplicate_of_matched_reference' for x in report['false_positives'])
    assert all(x['competing_reference_id']=='gt-0001' for x in report['false_positives'])
    assert matched['simulation']
    with pytest.raises(ValueError,match='non-mock observed failure'):
        asyncio.run(propose_initial_p0(store,mock_settings(),[matched['job_id']]))


def test_batch_requires_all_frozen_cases_and_records_reference_review(tmp_path, monkeypatch):
    import facade_mcp.baseline_p0 as module
    store=ArtifactStore(tmp_path/'workspace',[tmp_path]);batch=tmp_path/'batch';batch.mkdir()
    labels=tmp_path/'labels';labels.mkdir()
    images=[]
    for name,color in [('a','white'),('b','black')]:
        image=tmp_path/(name+'.png');Image.new('RGB',(10,10),color).save(image)
        (labels/(name+'.txt')).write_text('0 .1 .1 .5 .1 .5 .5\n')
        images.append(image)
    frozen={'schema':'facade-baseline-batch/v1','image_inputs':[
        {'path':str(image),'sha256':sha256(image)} for image in images]}
    exp=hashlib.sha256(json.dumps(frozen,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    (batch/'baseline-manifest.json').write_text(json.dumps({**frozen,'experiment_sha256':exp}))
    completed=[]
    for index,image in enumerate(images):
        result=tmp_path/f'result-{index}.json'
        result.write_text(json.dumps({'experiment_sha256':exp,'image':{'path':str(image),'sha256':sha256(image)}}))
        completed.append({'path':str(image),'result_file':str(result)})
    summary={'schema':'facade-baseline-summary/v1','experiment_sha256':exp,'completed':list(completed),'failed':[]}
    summary_path=batch/'baseline-summary.json';summary_path.write_text(json.dumps(summary))
    state={'needs_review':True}
    def fake_match(store,baseline_path,labels_path,class_map,**kwargs):
        issue=int(Path(labels_path).stem=='b' and state['needs_review'])
        return {'status':'needs_reference_review' if issue else 'completed','job_id':Path(labels_path).stem,
                'match_path':baseline_path,'event_ids':[],'simulation':True,
                'counts':{'tp':1-issue,'fp':0,'fn':0,'unverified_predictions':0,
                          'unresolved_references':0,'reference_issues':issue}}
    monkeypatch.setattr(module,'match_baseline_case',fake_match)
    result=match_baseline_batch(store,str(summary_path),str(labels),{'0':'C'},expected_count=2)
    assert result['matched_count']==2 and result['status']=='needs_reference_review'
    assert result['counts']['tp']==1 and result['counts']['reference_issues']==1
    assert Path(result['match_batch_path']).is_file()
    old_path=Path(result['match_batch_path']);old_hash=sha256(old_path)
    state['needs_review']=False
    updated=match_baseline_batch(store,str(summary_path),str(labels),{'0':'C'},expected_count=2)
    assert updated['status']=='completed' and updated['match_batch_path']!=str(old_path)
    assert old_path.is_file() and sha256(old_path)==old_hash
    summary['completed'].pop();summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='incomplete'):
        match_baseline_batch(store,str(summary_path),str(labels),{'0':'C'},expected_count=2)
    summary['completed']=completed;summary_path.write_text(json.dumps(summary))
    (batch/'baseline-manifest.json').write_text(json.dumps({**frozen,'tampered':True,'experiment_sha256':exp}))
    with pytest.raises(ValueError,match='manifest has changed'):
        match_baseline_batch(store,str(summary_path),str(labels),{'0':'C'},expected_count=2)


def full_coverage_fixture(tmp_path, *, event_count=None, extra_artifacts=0):
    store=ArtifactStore(tmp_path/'workspace',[tmp_path]); cases=[]
    values=[('one','white'),('two','black')] if event_count is None else [('one','white')]
    for name,color in values:
        case=tmp_path/name;case.mkdir()
        _,image,labels,baseline=fixture_case(case,color=color)
        if extra_artifacts:
            data=json.loads(baseline.read_text())
            for i in range(extra_artifacts):
                artifact=case/f'call-evidence-{i:04d}.json'
                artifact.write_text(json.dumps({'call':i}))
                data['artifacts_sha256'][artifact.name]=sha256(artifact)
            baseline.write_text(json.dumps(data))
            baseline.with_name('baseline-result.sha256.json').write_text(json.dumps({'sha256':sha256(baseline)}))
        if event_count is not None:
            labels.write_text('')
            data=json.loads(baseline.read_text())
            data['candidates']=[{'candidate_id':f'candidate-{i:04d}','class_code':'C',
                'class_name':'crack','bbox_xyxy':[1,1,10,10],'confidence':.9,
                'model_profile':'expert','decision':'accept'} for i in range(event_count)]
            baseline.write_text(json.dumps(data))
            baseline.with_name('baseline-result.sha256.json').write_text(json.dumps({'sha256':sha256(baseline)}))
        row=match_baseline_case(store,str(baseline),str(labels),{'0':'C','1':'S'},reference_complete=True)
        cases.append({'image_path':str(image),**row})
    summary=tmp_path/'baseline-summary.json';summary.write_text('{}')
    manifest=tmp_path/'baseline-manifest.json';manifest.write_text('{}')
    batch={'schema':'facade-baseline-match-batch/v1','status':'completed','simulation':False,
        'reference_complete':True,'expected_count':len(cases),'matched_count':len(cases),
        'experiment_sha256':'d'*64,'cases':cases,'baseline_summary':str(summary),
        'baseline_summary_sha256':sha256(summary),'baseline_manifest':str(manifest),
        'baseline_manifest_sha256':sha256(manifest)}
    path=store.root/'match-batch.json';store.write_json(path,batch)
    return store,path,cases


def test_dense_baseline_evidence_is_transitively_bound_and_tampering_is_rejected(tmp_path):
    store,batch,cases=full_coverage_fixture(tmp_path,event_count=1,extra_artifacts=115)
    report=json.loads(Path(cases[0]['match_path']).read_text())
    assert len(report['artifacts'])==119
    event=FailureMemory(store.root).summary(cases[0]['job_id'])['events'][0]
    assert event['context']['evidence_reference_mode']=='manifest_chain'
    assert len(event['evidence_refs'])<=64
    with pytest.raises(ValueError,match='full-coverage'):
        asyncio.run(propose_initial_p0(store,mock_settings(),[cases[0]['job_id']]))
    with pytest.raises(ValueError,match='full-coverage'):
        asyncio.run(RuleLifecycle(store).propose(mock_settings(),job_ids=[cases[0]['job_id']]))
    proposal=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=1))
    assert proposal['status']=='candidate_only'
    (tmp_path/'one'/'call-evidence-0100.json').write_text('{"tampered":true}')
    with pytest.raises(ValueError,match='artifact changed'):
        asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=1))


def test_dense_coverage_rejects_artifact_symlink_escaping_baseline_directory(tmp_path):
    store,batch,_=full_coverage_fixture(tmp_path,event_count=1,extra_artifacts=65)
    artifact=tmp_path/'one'/'call-evidence-0060.json'
    external=tmp_path/'same-bytes.json'
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(external)
    with pytest.raises(ValueError,match='escaped its directory'):
        asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=1))


def test_promoted_diagnostic_full_source_chain_is_frozen_and_checked(tmp_path):
    from copy import deepcopy
    from facade_mcp.baseline_p0 import _diagnostic_evidence
    parent=tmp_path/'batch'; parent.mkdir()
    case=parent/'image'; case.mkdir()
    _,image,labels,baseline=fixture_case(case)
    store=ArtifactStore(tmp_path/'workspace',[tmp_path])
    manifest=parent/'baseline-manifest.json'; manifest.write_text('{}')
    cfg=store.root/'figure1-apinebula.local.json'; cfg.write_text('{}')
    script=store.root/'figure1_diagnose_fixture.py'; script.write_text('# diagnostic fixture\n')
    for name in ('baseline-expert-result.json','vlm-call-0003-grid.jpg',
                 'vlm-call-0001.json','vlm-call-0002.json',
                 'vlm-call-0003.failed-1.json','vlm-call-0003.failed-2.json'):
        (case/name).write_text('{}')
    inputs={'manifest':manifest,'stage':case/'baseline-expert-result.json',
            'image':case/'vlm-call-0003-grid.jpg','model_config':cfg,
            'prior_call_1':case/'vlm-call-0001.json','prior_call_2':case/'vlm-call-0002.json',
            'failure_1':case/'vlm-call-0003.failed-1.json','failure_2':case/'vlm-call-0003.failed-2.json'}
    diag=store.root/'figure1-diagnostic-fixture';diag.mkdir()
    request='f'*64; started='start'; completed='end'
    attempt={'status':'request_may_be_sent','request_sha256':request,'experiment_sha256':'d'*64,
             'diagnostic_sha256':sha256(script),'started_utc':started,
             'source_hashes':{name:sha256(path) for name,path in inputs.items()}}
    response={'decisions':[]}; trace=[{'adapter':'responses','status':'completed'}]
    outcome={'status':'protocol_valid','request_sha256':request,'response':response,
             'model_trace':trace,'completed_utc':completed}
    (diag/'attempt.json').write_text(json.dumps(attempt))
    (diag/'outcome.json').write_text(json.dumps(outcome))
    call={'request_sha256':request,'status':'completed','response':response,'model_trace':trace,
          'diagnostic_origin':{'attempt_sha256':sha256(diag/'attempt.json'),
              'outcome_sha256':sha256(diag/'outcome.json'),'diagnostic_sha256':sha256(script),
              'started_utc':started,'completed_utc':completed}}
    (case/'vlm-call-0003.json').write_text(json.dumps(call))
    data=json.loads(baseline.read_text())
    data['artifacts_sha256']['vlm-call-0003.json']=sha256(case/'vlm-call-0003.json')
    baseline.write_text(json.dumps(data))
    baseline.with_name('baseline-result.sha256.json').write_text(json.dumps({'sha256':sha256(baseline)}))
    row=match_baseline_case(store,str(baseline),str(labels),{'0':'C','1':'S'},reference_complete=True)
    detail=json.loads(Path(row['match_path']).read_text())
    assert len(detail['diagnostic_provenance'])==1
    assert len(detail['diagnostic_provenance'][0]['files'])==11
    summary=tmp_path/'baseline-summary.json';summary.write_text('{}')
    batch={'schema':'facade-baseline-match-batch/v1','status':'completed','simulation':False,
           'reference_complete':True,'expected_count':1,'matched_count':1,'experiment_sha256':'d'*64,
           'cases':[{'image_path':str(image),**row}],'baseline_summary':str(summary),
           'baseline_summary_sha256':sha256(summary),'baseline_manifest':str(manifest),
           'baseline_manifest_sha256':sha256(manifest)}
    path=store.root/'match-batch.json';store.write_json(path,batch)
    proposal=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(path),expected_count=1))
    assert proposal['status']=='candidate_only'

    call_path=case/'vlm-call-0003.json'
    original_call=call_path.read_bytes()
    original_outcome=(diag/'outcome.json').read_bytes()
    altered=deepcopy(outcome); altered['response']={'decisions':[{'unexpected':True}]}
    (diag/'outcome.json').write_text(json.dumps(altered))
    altered_call=deepcopy(call); altered_call['diagnostic_origin']['outcome_sha256']=sha256(diag/'outcome.json')
    call_path.write_text(json.dumps(altered_call))
    with pytest.raises(ValueError,match='provenance does not match'):
        _diagnostic_evidence(store,baseline,data)
    (diag/'outcome.json').write_bytes(original_outcome)
    altered_call=deepcopy(call); altered_call['diagnostic_origin']['started_utc']=''
    call_path.write_text(json.dumps(altered_call))
    with pytest.raises(ValueError,match='Incomplete promoted diagnostic'):
        _diagnostic_evidence(store,baseline,data)
    call_path.write_bytes(original_call)
    duplicate=store.root/'figure1-diagnostic-duplicate'; duplicate.mkdir()
    (duplicate/'attempt.json').write_bytes((diag/'attempt.json').read_bytes())
    (duplicate/'outcome.json').write_bytes(original_outcome)
    with pytest.raises(ValueError,match='missing or ambiguous'):
        _diagnostic_evidence(store,baseline,data)
    (duplicate/'attempt.json').unlink(); (duplicate/'outcome.json').unlink(); duplicate.rmdir()
    original_failure=inputs['failure_2'].read_bytes(); inputs['failure_2'].unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _diagnostic_evidence(store,baseline,data)
    inputs['failure_2'].write_bytes(original_failure)

    inputs['failure_1'].write_text('{"changed":true}')
    with pytest.raises(ValueError,match='diagnostic source changed'):
        asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(path),expected_count=1))


def test_full_coverage_resumes_completed_batches_and_does_not_publish(tmp_path,monkeypatch):
    from facade_mcp.model_gateway import ModelGateway
    store,batch,cases=full_coverage_fixture(tmp_path)
    calls=[];original=ModelGateway.invoke
    async def counted(self, *args, **kwargs):
        calls.append(args[0]);return await original(self,*args,**kwargs)
    monkeypatch.setattr(ModelGateway,'invoke',counted)
    result=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=2))
    assert result['status']=='candidate_only' and not result['promoted']
    assert not result['registered_with_rule_lifecycle'] and result['simulation']
    assert result['case_count']==2 and result['eligible_event_count']==4
    assert result['accounted_event_count']==4 and len(calls)==4
    assert RuleLifecycle(store).latest() is None
    again=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=2))
    assert again['candidate_playbook_file_sha256']==result['candidate_playbook_file_sha256']
    assert len(calls)==4


def test_full_coverage_handles_more_than_lifecycle_event_limit(tmp_path):
    store,batch,_=full_coverage_fixture(tmp_path,event_count=205)
    result=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=1))
    assert result['eligible_event_count']==result['accounted_event_count']==205
    manifest=json.loads(Path(result['manifest_path']).read_text())
    assert sum(len(group['event_ids']) for group in manifest['batches'])==205
    assert all(len(group['event_ids'])<=6 for group in manifest['batches'])


def test_no_disagreements_yields_empty_candidate_without_inventing_rules(tmp_path,monkeypatch):
    from facade_mcp.model_gateway import ModelGateway
    store,batch,_=full_coverage_fixture(tmp_path,event_count=0)
    calls=[]
    async def unexpected(*args,**kwargs): calls.append(1);raise AssertionError('must not call model')
    monkeypatch.setattr(ModelGateway,'invoke',unexpected)
    result=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=1))
    assert result['status']=='candidate_only' and result['coverage_status']=='complete'
    assert result['eligible_event_count']==result['rule_count']==0 and not calls


def test_full_coverage_rejects_invalid_case_and_changed_evidence_before_calls(tmp_path, monkeypatch):
    from facade_mcp.model_gateway import ModelGateway
    store,batch,cases=full_coverage_fixture(tmp_path)
    calls=[]
    async def should_not_call(*args,**kwargs): calls.append(1);raise AssertionError('must not call model')
    monkeypatch.setattr(ModelGateway,'invoke',should_not_call)
    data=json.loads(batch.read_text());data['status']='needs_reference_review';store.write_json(batch,data)
    with pytest.raises(ValueError,match='requires every real completed case'):
        asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=2))
    data['status']='completed';store.write_json(batch,data)
    Path(cases[0]['match_path']).write_text('{}')
    with pytest.raises(ValueError,match='Match report changed'):
        asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=2))
    assert not calls


def test_full_coverage_does_not_silently_drop_recorded_failure(tmp_path,monkeypatch):
    from facade_mcp.model_gateway import ModelGateway
    store,batch,_=full_coverage_fixture(tmp_path)
    data=json.loads(batch.read_text());data['cases'][0]['event_ids'].pop();store.write_json(batch,data)
    calls=[]
    async def must_not_call(*args,**kwargs): calls.append(1);raise AssertionError('must not call model')
    monkeypatch.setattr(ModelGateway,'invoke',must_not_call)
    with pytest.raises(ValueError,match='coverage disagree'):
        asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=2))
    assert not calls


def test_curator_cannot_cite_reflection_with_insufficient_evidence(tmp_path,monkeypatch):
    from facade_mcp.model_gateway import ModelGateway
    store,batch,_=full_coverage_fixture(tmp_path,event_count=2)
    async def fake(self,role,task,schema,**kwargs):
        self.trace.append({'status':'completed','adapter':'mock'})
        ids=[row['event_id'] for row in task['events']]
        if role=='reflector':
            return schema.model_validate({'assessments':[{'event_id':x,'status':'insufficient_evidence',
                'hypothesis':'uncertain','controlled_change':'none'} for x in ids]})
        return schema.model_validate({'candidates':[{'condition':'unsure','recommendation':'inspect',
            'limitations':'unknown','evidence_ids':[ids[0]]}],'no_rule_event_ids':ids[1:]})
    monkeypatch.setattr(ModelGateway,'invoke',fake)
    with pytest.raises(ValueError,match='insufficient evidence'):
        asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=1))


def test_call_budget_pause_resumes_without_resending_completed_requests(tmp_path,monkeypatch):
    from facade_mcp.model_gateway import ModelGateway
    store,batch,_=full_coverage_fixture(tmp_path)
    calls=[];original=ModelGateway.invoke
    async def counted(self,*args,**kwargs):
        calls.append(args[0]);return await original(self,*args,**kwargs)
    monkeypatch.setattr(ModelGateway,'invoke',counted)
    for expected_calls in (1,2,3):
        result=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=2,max_calls=1))
        assert result['status']=='paused_call_budget'
        assert result['model_calls_this_run']==1 and len(calls)==expected_calls
    result=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=2,max_calls=1))
    assert result['status']=='candidate_only' and len(calls)==4
    assert result['eligible_event_count']==result['accounted_event_count']==4


def test_concurrent_same_request_sends_at_most_once(tmp_path):
    from facade_mcp.baseline_p0_coverage import CoverageReflection, _phase
    store=ArtifactStore(tmp_path/'workspace',[tmp_path]);folder=store.root/'journal';folder.mkdir()
    class SlowGateway:
        def __init__(self): self.trace=[];self.calls=0;self.started=asyncio.Event();self.release=asyncio.Event()
        async def invoke(self, role, task, schema, **kwargs):
            self.calls+=1;self.started.set();await self.release.wait()
            self.trace.append({'status':'completed','adapter':'mock'})
            return schema.model_validate({'assessments':[]})
    async def run():
        gateway=SlowGateway()
        first=asyncio.create_task(_phase(store,folder,'reflector','same',{},CoverageReflection,gateway))
        await gateway.started.wait()
        with pytest.raises(ValueError,match='already in progress'):
            await _phase(store,folder,'reflector','same',{},CoverageReflection,gateway)
        assert gateway.calls==1
        gateway.release.set();await first
        await _phase(store,folder,'reflector','same',{},CoverageReflection,gateway)
        assert gateway.calls==1
    asyncio.run(run())


def test_missed_reference_reflection_includes_overlapping_rejected_trajectory():
    from facade_mcp.baseline_p0_coverage import _related_baseline_evidence
    event={'context':{'class_code':'C','bbox_xyxy':[10,10,30,30],'candidate_id':None}}
    baseline={'candidates':[{'candidate_id':'rejected','class_code':'C','bbox_xyxy':[12,12,29,29],
                             'decision':'reject','confidence':.8,'reason':'not a defect'},
                            {'candidate_id':'far','class_code':'C','bbox_xyxy':[40,40,50,50],
                             'decision':'accept','confidence':.8}],
              'trajectory':[{'stage':'baseline_judge','candidate_id':'rejected','decision':'reject'},
                            {'stage':'baseline_judge','candidate_id':'far','decision':'accept'}]}
    related=_related_baseline_evidence(event,baseline)
    assert [x['candidate_id'] for x in related['overlapping_candidates']]==['rejected']
    assert related['overlapping_candidates'][0]['decision']=='reject'
    assert related['related_trajectory'][0]['candidate_id']=='rejected'
    assert 'not be inferred absent' in related['scope']


def test_provider_error_halts_and_preserves_unknown_request(tmp_path,monkeypatch):
    from facade_mcp.model_gateway import GatewayError, ModelGateway
    store,batch,_=full_coverage_fixture(tmp_path,event_count=1)
    async def failed(self,role,*args,**kwargs):
        self.trace.append({'status':'failed','adapter':'mock','error_code':'api_rate_limited','usage':None})
        raise GatewayError('sanitized provider error')
    monkeypatch.setattr(ModelGateway,'invoke',failed)
    result=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=1))
    assert result['status']=='halted_provider_error' and result['model_calls_this_run']==1
    assert result['usage']['calls_without_complete_usage']==1
    with pytest.raises(ValueError,match='explicit acknowledgement'):
        asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=1))


def test_partial_reference_review_then_repair_reuses_valid_case_calls(tmp_path,monkeypatch):
    from facade_mcp.model_gateway import ModelGateway
    store,batch,cases=full_coverage_fixture(tmp_path)
    label=tmp_path/'two'/'input.txt';original_label=label.read_text()
    label.write_text(original_label+'0 .5 .2 .5 .3 .5 .4\n')
    baseline=tmp_path/'two'/'baseline-result.json'
    partial=match_baseline_case(store,str(baseline),str(label),{'0':'C','1':'S'},reference_complete=True)
    assert partial['status']=='needs_reference_review'
    data=json.loads(batch.read_text());data['status']='needs_reference_review'
    data['cases'][1]={'image_path':str(tmp_path/'two'/'input.jpg'),**partial}
    partial_batch=store.root/'match-partial.json';store.write_json(partial_batch,data)
    with pytest.raises(ValueError,match='requires every real completed case'):
        asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(partial_batch),expected_count=2))
    calls=[];original=ModelGateway.invoke
    async def counted(self,*args,**kwargs):
        calls.append(args[0]);return await original(self,*args,**kwargs)
    monkeypatch.setattr(ModelGateway,'invoke',counted)
    draft=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(partial_batch),expected_count=2,allow_incomplete=True))
    assert draft['status']==draft['coverage_status']=='partial_reference_review'
    assert draft['case_count']==2 and draft['reviewed_case_count']==1 and draft['excluded_case_count']==1
    assert draft['excluded_cases'][0]['reference_issues'][0]['line']==4
    assert draft['excluded_cases'][0]['pending_candidate_ids']
    assert not draft['promoted'] and not draft['registered_with_rule_lifecycle'] and len(calls)==2
    # Restoring a separately reviewed correct reference produces a new match
    # and manifest; the unaffected case's two completed model calls are reused.
    label.write_text(original_label)
    repaired=match_baseline_case(store,str(baseline),str(label),{'0':'C','1':'S'},reference_complete=True)
    data['status']='completed';data['cases'][1]={'image_path':str(tmp_path/'two'/'input.jpg'),**repaired}
    repaired_batch=store.root/'match-repaired.json';store.write_json(repaired_batch,data)
    final=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(repaired_batch),expected_count=2))
    assert final['status']=='candidate_only' and final['coverage_status']=='complete'
    assert final['eligible_event_count']==final['accounted_event_count']==4
    assert len(calls)==4
    assert Path(draft['manifest_path']).is_file() and draft['manifest_path'] != final['manifest_path']
    old_manifest=json.loads(Path(draft['manifest_path']).read_text())
    assert all(sha256(Path(path))==digest for path,digest in old_manifest['source_files_sha256'].items())


def test_unreviewed_reference_assumption_is_recorded_in_candidate(tmp_path):
    store,batch,_=full_coverage_fixture(tmp_path,event_count=0)
    data=json.loads(batch.read_text())
    data['reference_assumption']='Supplied annotations are provisionally exhaustive; completeness is unreviewed.'
    store.write_json(batch,data)
    result=asyncio.run(propose_full_coverage_p0(store,mock_settings(),str(batch),expected_count=1))
    assert result['status']=='candidate_only'
    assert result['coverage_status']=='conditional_reference_assumption'
    assert result['conditional_on_unreviewed_reference'] is True
    assert result['reference_assumption']==data['reference_assumption']
    assert not result['promoted'] and not result['registered_with_rule_lifecycle']
    manifest=json.loads(Path(result['manifest_path']).read_text())
    assert manifest['reference_assumption']==data['reference_assumption']


def test_cli_requires_explicit_assumption_and_propagates_it(tmp_path,monkeypatch,capsys):
    import facade_mcp.baseline_p0_cli as cli
    calls=[]
    def match(store,summary,labels,class_map,**kwargs):
        calls.append(kwargs)
        return {'status':'completed','match_batch_path':'/test/batch.json','matched_count':203,'expected_count':203,
                'counts':{},'simulation':False,'cases':[], 'reference_complete':kwargs['reference_complete'],
                'reference_assumption':kwargs['reference_assumption']}
    monkeypatch.setattr(cli,'match_baseline_batch',match)
    base=['--workspace',str(tmp_path/'ws'),'match','--summary','/test/summary','--labels','/test/labels',
          '--class-map','{"0":"C","1":"S"}']
    assert cli.main(base)==2
    assert json.loads(capsys.readouterr().out)['status']=='needs_completeness_declaration'
    assert calls[-1]['reference_complete'] is False and calls[-1]['reference_assumption'] is None
    assert cli.main(base+['--assume-reference-complete'])==0
    report=json.loads(capsys.readouterr().out)
    assert calls[-1]['reference_complete'] is True
    assert 'unreviewed assumption' in report['reference_assumption']


def test_cli_curation_defaults_to_bounded_no_api_and_no_partial(tmp_path,monkeypatch,capsys):
    import facade_mcp.baseline_p0_cli as cli
    config=tmp_path/'config.json';config.write_text(mock_settings().model_dump_json())
    calls=[]
    async def curate(store,settings,match_batch,**kwargs):
        calls.append(kwargs)
        return {'status':'candidate_only'}
    monkeypatch.setattr(cli,'propose_full_coverage_p0',curate)
    assert cli.main(['--workspace',str(tmp_path/'ws'),'--allow-root',str(tmp_path),'curate',
                     '--match-batch',str(tmp_path/'batch.json'),'--model-config',str(config)])==0
    capsys.readouterr()
    assert calls[0]['expected_count']==203 and calls[0]['max_calls']==1024
    assert not calls[0]['allow_api'] and not calls[0]['allow_incomplete'] and not calls[0]['acknowledge_unknown']
