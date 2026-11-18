import asyncio
import json
from pathlib import Path
import sys

import pytest

from facade_mcp.failure_memory import FailureMemory
from facade_mcp.image_failures import FailureFeedback, classify, progress, record_feedback, record_observations, record_runtime_failure
from facade_mcp.image_inspection import ImageInspectionRequest, run_image_inspection
from facade_mcp.storage import ArtifactStore
from test_image_inspection import FakeExperts, FakeVLM, execute, model_config, row, setup


def test_repeated_runtime_failures_are_automatic_and_original_error_survives(tmp_path):
    store=ArtifactStore(tmp_path/'workspace',[tmp_path])
    request=ImageInspectionRequest(image_path=str(tmp_path/'missing.png'))
    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            asyncio.run(run_image_inspection(request,store,model_config(),experts=FakeExperts()))
    summary=FailureMemory(store.root).summary()
    assert summary['totals']['runtime_failures']==2
    cause=summary['reasons'][0]
    assert (cause['reason_code'],cause['count'],cause['affected_jobs'],cause['repeat_count'])==('input_missing',2,2,1)
    for folder in (store.root/'jobs').iterdir():
        job=json.loads((folder/'job.json').read_text(encoding='utf-8'))
        assert job['status']=='failed' and job['error']['failure_recorded'] is True
        assert (folder/'failure.json').is_file() and (folder/'failure_summary.json').is_file()
        event=FailureMemory(store.root).summary(job_id=folder.name)['events'][0]
        assert event['stage']=='input_snapshot'
        # Retrying persistence for the SAME terminal event does not add a failure.
        record_runtime_failure(store,folder,FileNotFoundError('irrelevant provider prose'),[])
    assert FailureMemory(store.root).summary()['totals']['runtime_failures']==2


@pytest.mark.parametrize('trace,code',[
    ({'status':'failed','http_status':401},'api_unauthorized'),
    ({'status':'failed','http_status':429},'api_rate_limited'),
    ({'status':'failed','http_status':503},'api_http_error'),
    ({'status':'request_outcome_unknown'},'api_timeout'),
    ({'status':'failed','error_code':'api_transport_error'},'api_transport_error'),
    ({'status':'failed','error_code':'model_output_invalid'},'model_output_invalid'),
])
def test_api_classification_uses_structured_trace_without_claiming_root_cause(tmp_path,trace,code):
    progress(tmp_path,'final_verification',batch=1)
    observed,_=classify(RuntimeError('Provider secret text must not be copied'),tmp_path,[trace])
    assert observed==code


def test_completed_old_trace_does_not_misclassify_new_harness_error(tmp_path):
    progress(tmp_path,'final_review_validation')
    cause,_=classify(ValueError('Final review must decide every batch candidate exactly once'),tmp_path,
                     [{'status':'completed','http_status':200}])
    assert cause=='final_review_invalid'


def test_final_rule_reference_and_unstructured_mcp_failures_are_not_overdiagnosed(tmp_path):
    progress(tmp_path,'final_review_validation')
    assert classify(ValueError('VLM cites unknown or duplicate P0 rule IDs'),tmp_path,[])[0]=='final_review_invalid'
    progress(tmp_path,'full_detection')
    trace=tmp_path/'tool_calls.jsonl'
    trace.write_text(json.dumps({'status':'failed','error':'opaque transport error'})+'\n',encoding='utf-8')
    assert classify(RuntimeError('irrelevant'),tmp_path,[])[0]=='unclassified_runtime_error'
    trace.write_text(json.dumps({'status':'failed','output':{'error':{'code':'INVALID_INPUT'}}})+'\n',encoding='utf-8')
    assert classify(RuntimeError('irrelevant'),tmp_path,[])[0]=='input_invalid'


def test_cli_configuration_rejection_is_recorded_without_running_models(tmp_path,monkeypatch):
    from facade_mcp.cli import main
    workspace=tmp_path/'workspace'
    monkeypatch.setattr(sys,'argv',['facade-image','--workspace',str(workspace),'image-inspect',
        '--config',str(tmp_path/'missing-request.json'),'--model-config',str(tmp_path/'missing-models.json')])
    with pytest.raises(FileNotFoundError):
        main()
    summary=FailureMemory(workspace).summary()
    assert summary['totals']['runtime_failures']==1
    assert summary['events'][0]['stage']=='preflight'
    assert summary['events'][0]['reason_code']=='input_missing'


def test_telemetry_write_failures_cannot_replace_original_exception(tmp_path,monkeypatch):
    store=ArtifactStore(tmp_path/'workspace',[tmp_path])
    request=ImageInspectionRequest(image_path=str(tmp_path/'missing.png'))
    original_write=store.write_json
    def write(path,data):
        if Path(path).name=='model_trace.json':
            raise OSError('trace unavailable')
        original_write(path,data)
    monkeypatch.setattr(store,'write_json',write)
    monkeypatch.setattr(store,'fail_job',lambda *args,**kwargs:(_ for _ in ()).throw(OSError('status unavailable')))
    with pytest.raises(FileNotFoundError):
        asyncio.run(run_image_inspection(request,store,model_config(),experts=FakeExperts()))
    assert FailureMemory(store.root).summary()['totals']['runtime_failures']==1


def test_committed_ledger_event_stays_recorded_when_snapshot_publication_fails(tmp_path,monkeypatch):
    store=ArtifactStore(tmp_path/'workspace',[tmp_path])
    request=ImageInspectionRequest(image_path=str(tmp_path/'missing.png'))
    def unavailable(*args,**kwargs):
        raise OSError('snapshot unavailable')
    monkeypatch.setattr(FailureMemory,'publish_summary',unavailable)
    with pytest.raises(FileNotFoundError):
        asyncio.run(run_image_inspection(request,store,model_config(),experts=FakeExperts()))
    event=FailureMemory(store.root).summary()['events'][0]
    status=store.job_status(event['job_id'])
    assert status['error']['failure_recorded'] is True
    assert status['error']['failure_snapshots_written'] is False


def test_opaque_error_text_is_not_persisted(tmp_path):
    request,store,_,_,_=setup(tmp_path)
    marker='opaque-private-provider-message'
    class Broken(FakeExperts):
        async def call(self,*args,**kwargs):
            raise ValueError(marker)
    with pytest.raises(ValueError,match=marker):
        asyncio.run(run_image_inspection(request,store,model_config(),experts=Broken()))
    memory=FailureMemory(store.root)
    event=memory.summary()['events'][0]
    assert event['reason_code']=='model_output_invalid'
    assert marker not in json.dumps(event)
    folder=store.job_dir(event['job_id'])
    assert marker not in (folder/'failure.json').read_text(encoding='utf-8')


def completed_fixture(tmp_path):
    request,store,_,_,_=setup(tmp_path)
    result=asyncio.run(run_image_inspection(request,store,model_config(),experts=FakeExperts(full=[row()])))
    return store,result,Path(result['report_path']).parent


def feedback(result,**changes):
    return FailureFeedback.model_validate({'job_id':result['job_id'],'reason_code':'bbox_inaccurate',
        'occurrence_key':'review-1:candidate-0001','summary':'人工复核发现边界覆盖了邻近墙砖。',
        'evidence_refs':['candidate-0001-evidence.png'],'reviewer':'fixture-reviewer','candidate_id':'candidate-0001',**changes})


def test_human_feedback_requires_local_evidence_and_is_idempotent(tmp_path):
    store,result,folder=completed_fixture(tmp_path)
    spec=feedback(result)
    first=record_feedback(store,spec)
    second=record_feedback(store,spec)
    assert first['event']['inserted'] is True and second['event']['inserted'] is False
    assert first['event']['event_id']==second['event']['event_id']
    summary=FailureMemory(store.root).summary()
    assert summary['totals']['quality_failures']==1 and summary['totals']['runtime_failures']==0
    assert first['event']['context']['evidence_sha256']
    assert first['event']['cause_status']=='confirmed'
    assert first['event']['context']['confirmation'].startswith('reviewer_declared')
    assert json.loads((folder/'failure_summary.json').read_text(encoding='utf-8'))['totals']['quality_failures']==1
    assert result['playbook_updated'] is False
    assert store.resolve_input(result['artifacts']['failure_summary']['uri']).is_file()
    assert Path(first['event_path']).is_file()


@pytest.mark.parametrize('evidence',['../outside.png','missing.png','result.json#invented','failure_summary.json'])
def test_feedback_rejects_missing_escaped_or_mutable_evidence(tmp_path,evidence):
    store,result,_=completed_fixture(tmp_path)
    (tmp_path/'outside.png').write_bytes(b'not evidence')
    with pytest.raises(ValueError):
        record_feedback(store,feedback(result,evidence_refs=[evidence]))
    assert FailureMemory(store.root).summary()['totals']['quality_failures']==0


def test_feedback_rejects_unknown_candidate(tmp_path):
    store,result,_=completed_fixture(tmp_path)
    with pytest.raises(ValueError,match='Unknown feedback candidate'):
        record_feedback(store,feedback(result,candidate_id='candidate-9999'))


def test_defer_and_harness_override_are_observations_not_quality_failures(tmp_path):
    request,store,folder,_,_=setup(tmp_path)
    result=execute(request,store,folder,FakeExperts(full=[row()]),FakeVLM(scope='uncertain'))
    summary=record_observations(store,folder,result)
    assert result['counts']['defer']==1
    assert {e['reason_code'] for e in summary['events']}=={'scope_uncertain','harness_override'}
    assert summary['totals']['runtime_failures']==summary['totals']['quality_failures']==0
    assert summary['totals']['observations']==2
    record_observations(store,folder,result)
    assert FailureMemory(store.root).summary()['totals']['observations']==2
