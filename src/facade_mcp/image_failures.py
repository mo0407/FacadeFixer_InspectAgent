"""Evidence-based failure classification and image-run feedback integration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from .failure_memory import FailureMemory, REASONS
from .model_gateway import Strict
from .storage import ArtifactStore, sha256, utc_now


def progress(folder: Path, stage: str, **context):
    """Only callers' explicit, safe metadata; never stringify arbitrary errors."""
    row = {'stage':stage,'updated_at':utc_now(),**context}
    ArtifactStore.write_json(folder/'progress.json',row)
    with (folder/'stage_events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def _read(path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _last_tool(folder):
    path = folder/'tool_calls.jsonl'
    if not path.is_file():
        return {}
    last = ''
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                last = line
    return json.loads(last) if last else {}


def classify(error, folder, model_trace):
    """Return an observed reason, never guess root causes from provider prose."""
    state = _read(folder/'progress.json',{'stage':'initialization'})
    stage = state['stage']
    trace = model_trace[-1] if model_trace else {}
    # Only the most recent operation's trace may explain the current failure.
    if stage in ('visual_anchoring','choose_inspection','final_verification','review_generator','review_reflector','review_curator') and trace.get('status') != 'completed':
        code = trace.get('error_code')
        if code in REASONS and REASONS[code]['kind'] == 'runtime':
            return code,state
        status = trace.get('http_status')
        if trace.get('status') == 'request_outcome_unknown':
            return 'api_timeout',state
        if status in (401,403):
            return 'api_unauthorized',state
        if status == 429:
            return 'api_rate_limited',state
        if status is not None and status >= 400:
            return 'api_http_error',state
    if stage in ('full_detection','local_detection'):
        tool = _last_tool(folder)
        if tool.get('status') == 'failed':
            code = tool.get('output',{}).get('error',{}).get('code')
            if code == 'EXPERT_DEFINITION_CHANGED':
                return 'model_version_changed',state
            if code == 'WEIGHTS_NOT_FOUND':
                return 'model_weights_missing',state
            if code in ('SOURCE_REPO_NOT_FOUND','SOURCE_PYTHON_NOT_FOUND','INVALID_DEVICE','INVALID_TIMEOUT'):
                return 'input_invalid',state
            if code in ('SOURCE_CHANGED','MODEL_SOURCE_MISMATCH'):
                return 'source_changed',state
            if code == 'INVALID_INPUT':
                return 'input_invalid',state
            if code == 'INVALID_MODEL_OUTPUT':
                return 'model_output_invalid',state
            if code in ('MODEL_BUSY','MODEL_TIMEOUT','MODEL_PROCESS_FAILED','MODEL_EXECUTION_FAILED'):
                return 'model_runtime_error',state
            return 'unclassified_runtime_error',state
    # These are exact errors owned by our harness, not raw exception text saved
    # to the ledger. Known observed failures get stable catalog summaries.
    message = str(error)
    if message.startswith('Expert checkpoint changed'):
        return 'model_version_changed',state
    if message.startswith('Expert unavailable'):
        return 'expert_unavailable',state
    if message.startswith('Expert is not eligible'):
        return 'invalid_inspection_action',state
    if message.startswith('Candidate budget exceeded'):
        return 'candidate_budget_exceeded',state
    if message.startswith(('Frozen P0 changed','Published P0')):
        return 'playbook_changed',state
    if message.startswith(('Source changed','Source pixels changed','Scope input changed','Expert result source hash')):
        return 'source_changed',state
    if message.startswith('Final review must'):
        return 'final_review_invalid',state
    if message.startswith(('Invalid anchor','Duplicate inspection','Inspect requires','Finish has no')):
        return 'invalid_inspection_action',state
    if message.startswith('VLM cites unknown'):
        return ('final_review_invalid' if stage in ('final_verification','final_review_validation') else 'invalid_inspection_action'),state
    if stage == 'expert_startup':
        return 'mcp_connection_error',state
    if stage in ('preflight','input_snapshot','scope_snapshot'):
        return ('input_missing' if isinstance(error,FileNotFoundError) else 'input_invalid'),state
    if stage in ('full_detection','local_detection') and isinstance(error,ValueError):
        return 'model_output_invalid',state
    if stage == 'choose_inspection':
        return 'invalid_inspection_action',state
    if stage == 'final_verification':
        return 'final_review_invalid',state
    if stage.startswith('review_'):
        return 'model_output_invalid',state
    return 'unclassified_runtime_error',state


def record_runtime_failure(store, folder, error, model_trace):
    code,state = classify(error,folder,model_trace)
    request = _read(folder/'request.json',{})
    config = _read(folder/'model-config.json',_read(folder/'config.json',{}))
    connection = config.get('connections',{}).get(config.get('roles',{}).get('generator'),{})
    playbook = _read(folder/'playbook.json',_read(folder/'playbook_snapshot.json',{}))
    safe_context = {key:state[key] for key in ('stage','model_profile','step','batch','source_sha256','playbook_version','simulation','demo') if key in state}
    safe_context.update(exception_type=type(error).__name__,detector_profiles=request.get('detector_profiles',[]),
                        vlm_model=connection.get('model'),vlm_adapter=connection.get('adapter'),
                        playbook_version=playbook.get('version',safe_context.get('playbook_version')))
    if model_trace:
        safe_context.update(vlm_role=model_trace[-1].get('role'),vlm_model=model_trace[-1].get('requested_model',safe_context['vlm_model']))
    refs = [folder.name+'/'+name for name in ('progress.json','stage_events.jsonl','model_trace.json','tool_calls.jsonl',
            'request.json','playbook.json','model-config.json') if (folder/name).is_file()]
    memory = FailureMemory(store.root)
    event = memory.record(job_id=folder.name,stage=state['stage'],reason_code=code,occurrence_key='terminal-runtime',
                         evidence_refs=refs,context=safe_context,source='harness',cause_status='observed')
    failure = {'schema':'image-failure/v1','event':event,'reason_code':code,'summary':REASONS[code]['summary'],
               'recommended_action':REASONS[code]['recommended_action'],'stage':state['stage'],
               'ledger_recorded':True,'snapshots_written':False,
               'global_occurrences_at_recording':None,'global_affected_jobs_at_recording':None,
               'failure_summary_path':str(folder/'failure_summary.json'),
               'root_cause_verified':False,'automatic_retry':False,'playbook_updated':False}
    # The ledger transaction has committed. Snapshot IO must never cause a
    # caller to report that the event was not counted, or hide the run's error.
    try:
        total = next(r for r in memory.summary()['reasons'] if r['reason_code']==code)
        failure.update(global_occurrences_at_recording=total['count'],global_affected_jobs_at_recording=total['affected_jobs'])
        memory.publish_summary(folder/'failure_summary.json',job_id=folder.name)
        failure['snapshots_written']=True
        store.write_json(folder/'failure.json',failure)
    except Exception as snapshot_error:
        failure.update(snapshots_written=False,snapshot_error_type=type(snapshot_error).__name__)
    return failure


def record_observations(store, folder, result):
    memory = FailureMemory(store.root)
    # A mock's defer/reject output is not visual evidence about a real defect.
    if not result['demo']:
        for candidate in result['candidates']:
            cid = candidate['candidate_id']
            codes = []
            if candidate['class_code'] == 'unknown':
                codes.append('unknown_class')
            if candidate['scope']['status'] in ('partial','outside') or candidate['visual_scope_decision'] != 'wall':
                codes.append('scope_uncertain')
            if not candidate['review_evidence']['native_resolution']:
                codes.append('evidence_downsampled')
            if candidate.get('proposed_decision') == 'keep':
                codes.append('harness_override')
            if candidate['decision'] == 'reject':
                codes.append('candidate_rejected')
            for code in codes:
                memory.record(job_id=folder.name,stage='final_verification',reason_code=code,
                    occurrence_key=f'observation:{cid}:{code}',source='harness',cause_status='observed',
                    evidence_refs=[folder.name+'/result.json#'+cid,folder.name+'/'+candidate['review_evidence']['file']],
                    context={'candidate_id':cid,'class_code':candidate['class_code'],'decision':candidate['decision'],
                             'playbook_version':result['playbook_version'],'source_sha256':result['source_sha256']})
    summary = memory.publish_summary(folder/'failure_summary.json',job_id=folder.name,
                                     extra={'mock_observations_omitted':result['demo']})
    return summary


QUALITY_CODES = Literal['missed_defect','false_positive','wrong_class','bbox_inaccurate','mask_inaccurate',
                        'scope_error','report_error','duplicate_defect']


class FailureFeedback(Strict):
    job_id: str = Field(pattern=r'^image_inspection_[a-f0-9]{32}$')
    reason_code: QUALITY_CODES
    occurrence_key: str = Field(min_length=1,max_length=160)
    summary: str = Field(min_length=1,max_length=1000)
    evidence_refs: list[str] = Field(min_length=1,max_length=30)
    reviewer: str = Field(min_length=1,max_length=100)
    candidate_id: str | None = Field(default=None,pattern=r'^candidate-[0-9]{4}$')


def record_feedback(store: ArtifactStore, feedback: FailureFeedback):
    folder = store.job_dir(feedback.job_id)
    status = store.job_status(feedback.job_id)
    if status['kind'] != 'image_inspection' or status['status'] not in ('completed','failed'):
        raise ValueError('Feedback requires a completed or failed image-inspection job')
    result = _read(folder/'result.json',{})
    if feedback.candidate_id and feedback.candidate_id not in {c['candidate_id'] for c in result.get('candidates',[])}:
        raise ValueError('Unknown feedback candidate ID')
    refs, hashes = [], {}
    for reference in feedback.evidence_refs:
        if '#' in reference:
            raise ValueError('Use evidence filenames, not unchecked fragments')
        relative = Path(reference)
        path = (folder/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError('Feedback evidence must be an existing file inside this job')
        normalized = path.relative_to(folder).as_posix()
        if normalized in ('failure_summary.json','job.json','failure.json'):
            raise ValueError('Use stable image/result evidence instead of mutable failure summaries')
        refs.append(folder.name+'/'+normalized)
        hashes[normalized] = sha256(path)
    memory = FailureMemory(store.root)
    event = memory.record(job_id=feedback.job_id,stage='human_review',reason_code=feedback.reason_code,
        occurrence_key='feedback:'+feedback.occurrence_key,summary=feedback.summary,evidence_refs=refs,
        context={'reviewer':feedback.reviewer,'candidate_id':feedback.candidate_id,'evidence_sha256':hashes,
                 'source_sha256':result.get('source_sha256'),'playbook_version':result.get('playbook_version'),
                 'demo':result.get('demo',False),'confirmation':'reviewer_declared; not independently audited'},
        source='reviewer',cause_status='confirmed')
    event_path = folder/f'feedback-{event["event_id"]}.json'
    if not event_path.exists():
        store.write_json(event_path,{key:value for key,value in event.items() if key != 'inserted'})
    summary = memory.publish_summary(folder/'failure_summary.json',job_id=feedback.job_id)
    return {'event':event,'event_path':str(event_path),'job_summary':summary,'playbook_updated':False,'automatic_retry':False}


def record_preflight_failure(store: ArtifactStore, error: Exception):
    """Account for rejected config/request files before a normal run can start."""
    job=None
    ledger_recorded=False
    try:
        job,folder=store.create_job('image_inspection',{'configuration_load':'rejected_before_inspection'})
        progress(folder,'preflight')
        store.write_json(folder/'model_trace.json',[])
        failure=record_runtime_failure(store,folder,error,[])
        ledger_recorded=True
        store.fail_job(job,{'type':type(error).__name__,'reason_code':failure['reason_code'],
                           'message':failure['summary'],'failure_recorded':True,
                           'failure_snapshots_written':failure['snapshots_written']})
    except Exception as recording_error:
        if job:
            try:
                store.fail_job(job,{'type':type(error).__name__,'failure_recorded':ledger_recorded,
                                   'recording_error_type':type(recording_error).__name__})
            except Exception:
                pass
        error.add_note('Preflight failure memory could not be updated; inspect local job status.')
