import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from facade_mcp.model_gateway import GatewayError
from facade_mcp.p0_v2 import (StopV2, constrained_curation_schema, curator_request,
                               run_phase, validate_curation)


def reflection():
    return {'assessments':[{'event_id':'a','status':'hypothesis','hypothesis':'h','controlled_change':'c'},
                           {'event_id':'b','status':'insufficient_evidence','hypothesis':'h','controlled_change':'c'}]}


def curation(omitted=('b',)):
    return {'candidates':[{'condition':'c','recommendation':'r','limitations':'l','evidence_ids':['a']}],
            'no_rule_event_ids':list(omitted)}


def test_v2_curator_eligibility_is_explicit_and_schema_rejects_bad_id():
    task, eligible, insufficient=curator_request({'batch_id':'0001-0001','event_ids':['a','b']},[],reflection())
    assert eligible==['a'] and insufficient==['b']
    assert 'Never cite an event marked insufficient_evidence' in task['instruction']
    schema=constrained_curation_schema(['a','b'],eligible)
    assert schema.model_validate(curation()).model_dump(mode='json')==curation()
    bad=curation();bad['candidates'][0]['evidence_ids']=['b']
    with pytest.raises(ValidationError): schema.model_validate(bad)
    with pytest.raises(ValueError,match='insufficient'): validate_curation(['a','b'],reflection(),bad)


class FakeGateway:
    def __init__(self, outputs, *, timeout=False):
        self.outputs=list(outputs); self.trace=[]; self.timeout=timeout; self.calls=0

    async def invoke(self, role, task, schema, **kwargs):
        self.calls+=1
        if self.timeout:
            self.trace.append({'role':role,'status':'request_outcome_unknown','error_code':'api_timeout'})
            raise GatewayError('timeout')
        payload=json.dumps(self.outputs[0]).encode()
        kwargs['capture_path'].write_bytes(payload)
        self.trace.append({'response_body_sha256':hashlib.sha256(payload).hexdigest(),
                           'role':role,'status':'completed','adapter':'responses','http_status':200,
                           'model_identity_verified':True,'requested_model':'gpt-6.2-sol',
                           'actual_model':'gpt-6.2-sol','usage':{'input_tokens':10,'output_tokens':2}})
        return schema.model_validate(self.outputs.pop(0))


def run(coro): return asyncio.run(coro)


def setup_phase(tmp_path, gateway, budget, *, sleep=None):
    task,eligible,_=curator_request({'batch_id':'0001-0001','event_ids':['a','b']},[],reflection())
    schema=constrained_curation_schema(['a','b'],eligible)
    kwargs={'simulation':False}
    if sleep is not None: kwargs['sleep']=sleep
    return run_phase(tmp_path,'curator',task,schema,[],'f'*64,gateway,
                     lambda value:validate_curation(['a','b'],reflection(),value),
                     'gpt-6.2-sol',budget,**kwargs)


def test_successful_cache_is_reused_before_budget_and_identity_is_bound(tmp_path):
    gateway=FakeGateway([curation()]); first=run(setup_phase(tmp_path,gateway,{'calls':0,'limit':1}))
    assert first['execution']=='model' and gateway.calls==1
    result=run(setup_phase(tmp_path,gateway,{'calls':0,'limit':0}))
    assert result==first and gateway.calls==1
    done=tmp_path/'curator.json'; data=json.loads(done.read_text()); data['output']['no_rule_event_ids']=[]
    done.write_text(json.dumps(data))
    with pytest.raises(StopV2,match='sidecar'): run(setup_phase(tmp_path,gateway,{'calls':0,'limit':1}))


def test_semantic_rejections_are_preserved_and_retried_with_backoff(tmp_path):
    bad=curation(omitted=()); gateway=FakeGateway([bad,curation()]); waits=[]
    async def sleep(seconds): waits.append(seconds)
    record=run(setup_phase(tmp_path,gateway,{'calls':0,'limit':4},sleep=sleep))
    rejected=json.loads((tmp_path/'curator.rejected-0001.json').read_text())
    assert waits==[30] and gateway.calls==2 and rejected['kind']=='semantic'
    assert rejected['response']==bad and record['output']==curation()
    assert (tmp_path/'curator.rejected-0001.json.sha256').is_file()
    assert not (tmp_path/'curator.pending.json').exists()


def test_three_retries_are_bounded_and_each_rejection_is_retained(tmp_path):
    gateway=FakeGateway([curation(omitted=()) for _ in range(4)]); waits=[]
    async def sleep(seconds): waits.append(seconds)
    with pytest.raises(StopV2) as exc: run(setup_phase(tmp_path,gateway,{'calls':0,'limit':10},sleep=sleep))
    assert exc.value.status=='halted_retry_limit' and waits==[30,90,180] and gateway.calls==4
    assert len(list(tmp_path.glob('curator.rejected-*.json')))==4
    assert not (tmp_path/'curator.pending.json').exists()


def test_timeout_keeps_durable_pending_and_never_retries_on_resume(tmp_path):
    gateway=FakeGateway([],timeout=True)
    with pytest.raises(StopV2) as exc: run(setup_phase(tmp_path,gateway,{'calls':0,'limit':10}))
    assert exc.value.status=='halted_unknown_outcome' and gateway.calls==1
    assert (tmp_path/'curator.pending.json').is_file()
    with pytest.raises(StopV2) as exc: run(setup_phase(tmp_path,gateway,{'calls':0,'limit':10}))
    assert exc.value.status=='halted_unknown_outcome' and gateway.calls==1


def test_budget_pause_keeps_rejection_and_resumes_same_request(tmp_path):
    gateway=FakeGateway([curation(omitted=()),curation()]); waits=[]
    async def sleep(seconds): waits.append(seconds)
    with pytest.raises(StopV2) as exc: run(setup_phase(tmp_path,gateway,{'calls':0,'limit':1},sleep=sleep))
    assert exc.value.status=='paused_call_budget' and gateway.calls==1
    result=run(setup_phase(tmp_path,gateway,{'calls':0,'limit':1},sleep=sleep))
    assert result['status']=='completed' and waits==[30] and gateway.calls==2


def test_revision_completes_with_reused_reflection_and_deterministic_curation(tmp_path):
    from types import SimpleNamespace
    from facade_mcp.p0_v2 import run_revision
    from facade_mcp.baseline_p0_coverage import _digest
    event={'event_id':'a','context':{}}
    group={'batch_id':'0001-0001','event_ids':['a'],'event_sha256':{'a':_digest(event)}}
    reflected={'assessments':[{'event_id':'a','status':'insufficient_evidence','hypothesis':'h','controlled_change':'c'}]}
    verified={'batches':[group],'events':{'a':event},'prefix':[],
              'next_reflection':{'batch_id':group['batch_id'],'observations':[], 'reflection':reflected},
              'cases':[{'event_ids':['a']}]}
    manifest={'revision_sha256':'f'*64,'batches':[group],'reuse_v1_phase_count':1,
              'excluded_cases':[{'excluded_reason':'review'}],'source_files_sha256':{},
              'package_sources_sha256':{},'reference_assumption':'unreviewed','limitations':[]}
    (tmp_path/'manifest.json').write_text('{}')
    gateway=FakeGateway([])
    state=run(run_revision(SimpleNamespace(),SimpleNamespace(),manifest,tmp_path,verified,gateway=gateway))
    assert state['status']=='partial_reference_review' and state['completed_batches']==1
    assert state['accounted_event_count']==1 and state['new_calls_this_run']==0 and gateway.calls==0
    metadata=json.loads((tmp_path/'candidate-metadata.json').read_text())
    assert metadata['promoted'] is False and metadata['revision_sha256']=='f'*64
    assert metadata['candidate_playbook_file_sha256']==__import__('hashlib').sha256((tmp_path/'candidate-playbook.json').read_bytes()).hexdigest()
    assert json.loads((tmp_path/'candidate-playbook.json').read_text())['rules']==[]


def test_completed_phase_rechecks_prior_rejected_and_raw_identity(tmp_path):
    gateway=FakeGateway([curation(omitted=()),curation()])
    async def no_wait(_): pass
    run(setup_phase(tmp_path,gateway,{'calls':0,'limit':4},sleep=no_wait))
    old=tmp_path/'curator.rejected-0001.json';old.write_text('{}')
    with pytest.raises(StopV2) as exc:
        run(setup_phase(tmp_path,gateway,{'calls':0,'limit':4},sleep=no_wait))
    assert exc.value.status=='halted_integrity_error' and gateway.calls==2


def test_any_unresolved_future_phase_blocks_new_requests(tmp_path):
    from types import SimpleNamespace
    from facade_mcp.p0_v2 import run_revision
    future=tmp_path/'phases'/'9999-0001-0001';future.mkdir(parents=True)
    (future/'reflector.pending.json').write_text('{}')
    manifest={'revision_sha256':'f'*64,'batches':[],'reuse_v1_phase_count':225,'excluded_cases':[]}
    verified={'batches':[],'events':{},'prefix':[]}
    gateway=FakeGateway([])
    state=run(run_revision(SimpleNamespace(),SimpleNamespace(),manifest,tmp_path,verified,gateway=gateway))
    assert state['status']=='halted_unknown_outcome' and gateway.calls==0


def _layout_with_half_batch(tmp_path):
    from facade_mcp.baseline_p0_coverage import CoverageReflection
    from facade_mcp.p0_v2 import validate_reflection
    groups=[{'batch_id':'0001-0001','event_ids':['a','b']},{'batch_id':'0002-0001','event_ids':['a','b']}]
    first=tmp_path/'phases'/'0001-0001-0001';first.mkdir(parents=True)
    run(setup_phase(first,FakeGateway([curation()]),{'calls':0,'limit':1}))
    second=tmp_path/'phases'/'0002-0002-0001';second.mkdir(parents=True)
    gateway=FakeGateway([reflection()])
    run(run_phase(second,'reflector',{'task':'reflection'},CoverageReflection,[],'f'*64,gateway,
                  lambda x:validate_reflection(['a','b'],x),'gpt-6.2-sol',{'calls':0,'limit':1}))
    return groups,first,second


def test_layout_allows_budget_pause_after_half_batch(tmp_path):
    from facade_mcp.p0_v2 import _audit_phase_layout
    groups,first,second=_layout_with_half_batch(tmp_path)
    _audit_phase_layout(tmp_path,groups,0,{'completed_batches':1,'accounted_event_count':2,'new_completed_calls':2,'new_rejected_calls':0})


@pytest.mark.parametrize('victim',['curator.json','curator.sha256','curator.raw-0001.response'])
def test_layout_rejects_missing_earlier_phase_evidence(tmp_path,victim):
    from facade_mcp.p0_v2 import _audit_phase_layout
    groups,first,second=_layout_with_half_batch(tmp_path)
    (first/victim).unlink()
    with pytest.raises(StopV2) as exc:
        _audit_phase_layout(tmp_path,groups,0,{'completed_batches':1,'accounted_event_count':2,'new_completed_calls':2,'new_rejected_calls':0})
    assert exc.value.status=='halted_integrity_error'


def test_layout_rejects_deleted_phase_even_if_all_its_files_are_missing(tmp_path):
    from facade_mcp.p0_v2 import _audit_phase_layout
    groups,first,second=_layout_with_half_batch(tmp_path)
    for path in first.iterdir(): path.unlink()
    with pytest.raises(StopV2) as exc:
        _audit_phase_layout(tmp_path,groups,0,{'completed_batches':1,'accounted_event_count':2,'new_completed_calls':2,'new_rejected_calls':0})
    assert exc.value.status=='halted_integrity_error'


def test_layout_rejects_raw_response_tampering_before_new_call(tmp_path):
    from facade_mcp.p0_v2 import _audit_phase_layout
    groups,first,second=_layout_with_half_batch(tmp_path)
    (second/'reflector.raw-0001.response').write_bytes(b'changed')
    with pytest.raises(StopV2) as exc:
        _audit_phase_layout(tmp_path,groups,0,{'completed_batches':1,'accounted_event_count':2,'new_completed_calls':2,'new_rejected_calls':0})
    assert exc.value.status=='halted_integrity_error'


def test_replay_preserves_prior_progress_high_water_through_crash(tmp_path):
    from facade_mcp.p0_v2 import _audit_phase_layout, _progress
    groups,first,second=_layout_with_half_batch(tmp_path)
    manifest={'revision_sha256':'f'*64,'batches':groups,'reuse_v1_phase_count':0,'excluded_cases':[]}
    initial=_progress(tmp_path,manifest,'paused_call_budget',1,2,2)
    replay=_progress(tmp_path,manifest,'running',0,0,0)
    assert replay['completed_batches']==1 and replay['accounted_event_count']==2
    assert replay['new_completed_calls']==2 and replay['new_rejected_calls']==0
    # Simulate a crash immediately after the replay write, then losing the
    # half-completed final phase before the next invocation.
    for path in second.iterdir(): path.unlink()
    preserved=json.loads((tmp_path/'progress.json').read_text())
    with pytest.raises(StopV2) as exc: _audit_phase_layout(tmp_path,groups,0,preserved)
    assert exc.value.status=='halted_integrity_error'
    with pytest.raises(StopV2): _progress(tmp_path,manifest,'running',0,0,0)
    assert json.loads((tmp_path/'progress.json').read_text())==preserved


def test_usage_reports_missing_input_output_even_when_total_is_reported():
    from facade_mcp.p0_v2 import _usage_with_missing
    result=_usage_with_missing([{'usage':{'total_tokens':50}},
                                {'usage':{'input_tokens':20,'output_tokens':4}},
                                {'usage':None}])
    assert result['reported_total_tokens']==74
    assert result['calls_missing_input_or_output']==2
