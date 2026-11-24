import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from facade_mcp import baseline_p0_coverage as v1
from facade_mcp import p0_v2_reuse as reuse
from facade_mcp.model_gateway import ModelConfig, _load_images
from facade_mcp.storage import ArtifactStore, sha256


def _settings():
    return ModelConfig.model_validate({'schema_version':'model-gateway-v1','api_enabled':True,
        'connections':{'remote':{'adapter':'responses','model':'gpt-6.2-sol','expected_model':'gpt-6.2-sol',
            'supports_images':True,'endpoint_env':'TEST_V2_ENDPOINT','api_key_env':'TEST_V2_KEY'}},
        'roles':{'generator':'remote','reflector':'remote','curator':'remote'}})


def _cache(folder, phase, key, request_sha, output, image_meta=None):
    trace={'role':phase,'status':'completed','http_status':200,'adapter':'responses',
           'requested_model':'gpt-6.2-sol','actual_model':'gpt-6.2-sol','model_identity_verified':True,
           'usage':{'input_tokens':10,'output_tokens':2,'total_tokens':12}}
    if image_meta is not None: trace['images']=image_meta
    row={'status':'completed','identity':key,'request_sha256':request_sha,'output':output,
         'output_sha256':v1._digest(output),'trace':trace}
    path=folder/(phase+'.json');path.write_text(json.dumps(row),encoding='utf-8')
    return row


@pytest.fixture
def sample(tmp_path,monkeypatch):
    root=tmp_path/'workspace';root.mkdir();store=ArtifactStore(root,[tmp_path])
    monkeypatch.setenv('TEST_V2_ENDPOINT','https://example.test/v1/responses')
    settings=_settings();match=root/'match.json'
    doc={'schema':'facade-baseline-match-batch/v1','experiment_sha256':'d'*64,'reference_assumption':'unreviewed'}
    match.write_text(json.dumps(doc));db=root/'failure_memory.sqlite3';db.write_bytes(b'fixture-db')
    groups=[];events={};reports=[]
    for batch_index in range(2):
        case=tmp_path/f'case{batch_index}';case.mkdir()
        image=case/'source.jpg';Image.new('RGB',(40,30),'white').save(image)
        overview=case/'overview.jpg';Image.new('RGB',(40,30),'white').save(overview)
        baseline=case/'baseline.json';baseline.write_text(json.dumps({'overview_file':'overview.jpg',
            'artifacts_sha256':{'overview.jpg':sha256(overview)},'candidates':[],'trajectory':[]}))
        report=case/'report.json';report.write_text(json.dumps({'baseline':{'path':str(baseline)},
            'image':{'path':str(image),'sha256':sha256(image),'width':40,'height':30}}))
        ids=[f'{batch_index*2+1:032x}',f'{batch_index*2+2:032x}']
        for event_id in ids:
            events[event_id]={'event_id':event_id,'reason_code':'false_positive',
                'context':{'bbox_xyxy':[1,2,10,12],'class_code':'C','candidate_id':event_id}}
        groups.append({'batch_id':f'0001-{batch_index+1:04d}','match_path':str(report),
            'match_sha256':sha256(report),'event_ids':ids,
            'event_sha256':{i:v1._digest(events[i]) for i in ids}})
        reports.append(report)
    sources={str(match):sha256(match)};cases=[]
    endpoints={role:hashlib.sha256(b'https://example.test/v1/responses').hexdigest() for role in ('reflector','curator')}
    parent={'schema':'facade-baseline-p0-full/v1','match_batch_sha256':sha256(match),'experiment_sha256':'d'*64,
            'case_count':1,'cases':cases,'batches':groups,'reference_assumption':'unreviewed',
            'source_files_sha256':sources,'model_config_sha256':v1._digest(settings.model_dump(mode='json')),
            'endpoint_hashes':endpoints,'implementation_sha256':sha256(Path(v1.__file__)),'chunk_size':6,
            'allow_incomplete':True,'simulation':False}
    identity=v1._digest(parent);parent['manifest_sha256']=identity
    directory=root/'baseline_p0_full'/identity;directory.mkdir(parents=True)
    manifest=directory/'manifest.json';manifest.write_text(json.dumps(parent))
    (directory/'progress.json').write_text(json.dumps({'schema':'facade-baseline-p0-progress/v1',
        'status':'running','manifest_sha256':identity,'completed_batches':1,'accounted_event_count':2,
        'total_batches':2,'total_event_count':4}))
    shared=root/'baseline_p0_full'/'shared-cache';shared.mkdir()
    for i,group in enumerate(groups):
        crops=directory/'crops'/group['batch_id'];crops.mkdir(parents=True)
        for index,event_id in enumerate(group['event_ids'],1):
            Image.new('RGB',(15,15),'white').save(crops/f'{index:02d}-{event_id}.jpg')
        folder=shared/reuse._group_key(parent,group);folder.mkdir()
        observations,task,images,metadata=reuse._tasks_and_images(store,directory,group,[events[e] for e in group['event_ids']])
        request=v1._digest({'task':task,'images':[x['sha256'] for x in metadata]})
        ids=group['event_ids'];reflection={'assessments':[
            {'event_id':ids[0],'status':'hypothesis','hypothesis':'h','controlled_change':'c'},
            {'event_id':ids[1],'status':'insufficient_evidence','hypothesis':'h','controlled_change':'c'}]}
        _cache(folder,'reflector',folder.name,request,reflection,metadata)
        model=reuse._valid_reflection(group,reflection);ctask=reuse._curator_task(group,observations,model)
        curation={'candidates':[{'condition':'condition','recommendation':'review','limitations':'limited',
                   'evidence_ids':[ids[0] if i==0 else ids[1]]}],
                  'no_rule_event_ids':[ids[1] if i==0 else ids[0]]}
        _cache(folder,'curator',folder.name,v1._digest({'task':ctask,'images':[]}),curation)
    monkeypatch.setattr(reuse,'_readonly_prepare',lambda *args:(doc,cases,groups,sources,events))
    return store,manifest,settings,groups,parent,shared


def _verify(sample):
    store,manifest,settings,groups,parent,shared=sample
    return reuse.verify_parent_prefix(store,str(manifest),settings,expected_parent_file_sha256=sha256(manifest),
                                      expected_prefix=1,expected_count=1)


def test_reuses_only_verified_prefix_and_next_reflector(sample):
    result=_verify(sample)
    assert len(result['prefix'])==1 and result['prefix'][0]['batch_id']=='0001-0001'
    assert len(result['prefix'][0]['observations'])==2
    assert result['prefix'][0]['provenance']['curator']['source']=='p0-v1-original-cache'
    assert result['next_reflection']['batch_id']=='0001-0002' and 'curation' not in result['next_reflection']
    assert len(result['next_reflection']['provenance']['reflector']['images'])==3
    assert result['rejected_history'][-1]['kind']=='stopped_invalid_curator'
    assert result['usage']['reported_input_tokens']==40
    assert result['limitations']


def test_crop_tampering_or_invalid_prefix_curation_blocks_reuse(sample):
    store,manifest,settings,groups,parent,shared=sample
    folder=manifest.parent/'crops'/groups[0]['batch_id'];crop=next(folder.glob('*.jpg'))
    Image.new('RGB',(15,15),'black').save(crop)
    with pytest.raises(ValueError,match='reflected image metadata changed|identity, request'):
        _verify(sample)


def test_stopped_curator_must_really_be_invalid(sample):
    store,manifest,settings,groups,parent,shared=sample
    folder=shared/reuse._group_key(parent,groups[1]);path=folder/'curator.json';row=json.loads(path.read_text())
    ids=groups[1]['event_ids'];row['output']['candidates'][0]['evidence_ids']=[ids[0]]
    row['output']['no_rule_event_ids']=[ids[1]];row['output_sha256']=v1._digest(row['output']);path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='unexpectedly valid'):
        _verify(sample)


def test_wrong_model_pending_or_parent_hash_block_reuse(sample):
    store,manifest,settings,groups,parent,shared=sample
    with pytest.raises(ValueError,match='manifest file hash'):
        reuse.verify_parent_prefix(store,str(manifest),settings,expected_parent_file_sha256='0'*64,
                                   expected_prefix=1,expected_count=1)
    folder=shared/reuse._group_key(parent,groups[0]);pending=folder/'curator.pending.json';pending.write_text('{}')
    with pytest.raises(ValueError,match='pending'):_verify(sample)
    pending.unlink();path=folder/'curator.json';row=json.loads(path.read_text());row['trace']['actual_model']='other'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='model trace'):_verify(sample)


def test_rejects_endpoint_or_progress_drift(sample,monkeypatch):
    store,manifest,settings,groups,parent,shared=sample
    monkeypatch.setenv('TEST_V2_ENDPOINT','https://other.test/v1/responses')
    with pytest.raises(ValueError,match='endpoint identity'):_verify(sample)
    monkeypatch.setenv('TEST_V2_ENDPOINT','https://example.test/v1/responses')
    progress=manifest.parent/'progress.json';row=json.loads(progress.read_text());row['completed_batches']=0
    progress.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='progress'):_verify(sample)


def test_rejected_archive_is_bound_and_usage_preserved(sample):
    store,manifest,settings,groups,parent,shared=sample
    folder=shared/reuse._group_key(parent,groups[1]); source=folder/'curator.json'
    row=json.loads(source.read_text()); archive=folder/'curator.semantic-rejected-0001.json'
    archive.write_bytes(source.read_bytes())
    plan=folder/'curator.semantic-plan-0001.json'; recovery_hash='f'*64
    plan.write_text(json.dumps({'manifest_sha256':parent['manifest_sha256'],
        'source_file_sha256':sha256(archive),'request_sha256':row['request_sha256'],
        'phase':'curator','batch_id':groups[1]['batch_id'],
        'action':'preserve_rejected_phase_only','automatic_api_calls':0,'recovery_script_sha256':recovery_hash}))
    commit=folder/'curator.semantic-commit-0001.json'
    commit.write_text(json.dumps({'status':'rejected_phase_preserved','plan_sha256':sha256(plan),
        'archive_sha256':sha256(archive),'recovery_script_sha256':recovery_hash,'committed_utc':'2026-01-01T00:00:00Z'}))
    result=_verify(sample)
    rejected=[r for r in result['rejected_history'] if r['kind']=='semantic_rejected']
    assert len(rejected)==1 and rejected[0]['files'][0]['sha256']==sha256(archive)
    assert result['usage']['reported_input_tokens']==50
    archive.unlink();archive.hardlink_to(source)
    with pytest.raises(ValueError,match='share an inode'):_verify(sample)
    archive.unlink();archive.write_bytes(source.read_bytes())
    plan.write_text(plan.read_text()+' ')
    with pytest.raises(ValueError,match='rejection provenance'):_verify(sample)


def test_legacy_curator_commit_is_accepted_only_with_matching_action(sample):
    _,_,_,groups,parent,shared=sample
    folder=shared/reuse._group_key(parent,groups[1]); source=folder/'curator.json'
    row=json.loads(source.read_text()); archive=folder/'curator.semantic-rejected-0001.json'
    archive.write_bytes(source.read_bytes()); plan=folder/'curator.semantic-plan-0001.json'; recovery_hash='f'*64
    plan_row={'manifest_sha256':parent['manifest_sha256'],'source_file_sha256':sha256(archive),
        'curator_request_sha256':row['request_sha256'],'batch_id':groups[1]['batch_id'],
        'action':'preserve_rejected_curator_cache_only','automatic_api_calls':0,'recovery_script_sha256':recovery_hash}
    plan.write_text(json.dumps(plan_row)); commit=folder/'curator.semantic-commit-0001.json'
    commit_row={'status':'rejected_curator_preserved','plan_sha256':sha256(plan),
        'archive_sha256':sha256(archive),'recovery_script_sha256':recovery_hash,'committed_utc':'2026-01-01T00:00:00Z'}
    commit.write_text(json.dumps(commit_row))
    assert any(r['kind']=='semantic_rejected' for r in _verify(sample)['rejected_history'])
    commit_row['status']='rejected_phase_preserved';commit.write_text(json.dumps(commit_row))
    with pytest.raises(ValueError,match='rejection provenance'):_verify(sample)
    plan_row['action']='preserve_rejected_phase_only';plan_row['request_sha256']=plan_row.pop('curator_request_sha256')
    plan.write_text(json.dumps(plan_row));commit_row['plan_sha256']=sha256(plan)
    commit_row['status']='rejected_curator_preserved';commit.write_text(json.dumps(commit_row))
    with pytest.raises(ValueError,match='rejection provenance'):_verify(sample)


def test_memory_is_read_only_and_uncheckpointed_wal_is_rejected(tmp_path):
    memory=reuse.FailureMemory(tmp_path)
    original=memory.record(job_id='job',stage='match',reason_code='false_positive',
        occurrence_key='one',evidence_refs=['report.json'])
    before=sha256(memory.db_path)
    rows=reuse._ReadOnlyMemory(tmp_path).summary()['events']
    assert len(rows)==1 and rows[0]['event_id']==original['event_id']
    assert sha256(memory.db_path)==before
    Path(str(memory.db_path)+'-wal').write_bytes(b'not-a-complete-wal')
    with pytest.raises(ValueError,match='WAL'):
        reuse._ReadOnlyMemory(tmp_path).summary()


def test_retry_history_identity_sequence_and_unknown_fail_closed(sample):
    store,manifest,settings,groups,parent,shared=sample
    folder=shared/reuse._group_key(parent,groups[0]);current=json.loads((folder/'curator.json').read_text())
    row={'status':'request_failed_or_unknown','identity':folder.name,'request_sha256':current['request_sha256'],
         'trace':{'status':'failed','role':'curator','adapter':'responses','requested_model':'gpt-6.2-sol',
                  'http_status':502,'error_code':'api_http_error'}}
    retry=folder/'curator.failed-retry-001.json';retry.write_text(json.dumps(row))
    assert any(r['kind']=='failed_or_unknown' for r in _verify(sample)['rejected_history'])
    row['trace']['role']='reflector';retry.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='history differs'):_verify(sample)
    row['trace']['role']='curator';retry.write_text(json.dumps(row));retry.rename(folder/'curator.failed-retry-002.json')
    with pytest.raises(ValueError,match='retry history'):_verify(sample)
    (folder/'curator.failed-retry-002.json').unlink()
    (folder/'curator.unknown-1.json').write_text('{}')
    with pytest.raises(ValueError,match='unknown'):_verify(sample)


def test_repeated_provider_id_is_not_double_counted(sample):
    _,_,_,groups,parent,shared=sample
    for group in groups:
        path=shared/reuse._group_key(parent,group)/'reflector.json';row=json.loads(path.read_text())
        row['trace']['request_id']='same-provider-id';path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='request identity'):_verify(sample)
