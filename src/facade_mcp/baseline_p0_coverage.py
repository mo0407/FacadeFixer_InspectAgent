"""Bounded, resumable reflection over every Figure-1 mismatch.

This is a candidate-generation bridge, not rule publication.  The output is
not registered as a RuleLifecycle proposal and cannot be promoted directly.
All source evidence remains local and is hash checked; only the current batch's
overview, crops and sanitized discrepancy descriptions go to the chosen VLM.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw
from pydantic import Field, model_validator

from .baseline_p0 import _diagnostic_evidence
from .failure_memory import FailureMemory
from .model_gateway import GatewayError, ModelGateway, Strict
from .playbook import Playbook, Rule, fingerprint
from .rule_lifecycle import _eligible_event, _iou
from .storage import ArtifactStore, sha256


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,
        separators=(',',':')).encode('utf-8')).hexdigest()


class EventAssessment(Strict):
    event_id: str = Field(min_length=1,max_length=128)
    status: Literal['hypothesis', 'insufficient_evidence']
    hypothesis: str = Field(min_length=1,max_length=1500)
    controlled_change: str = Field(min_length=1,max_length=1500)


class CoverageReflection(Strict):
    assessments: list[EventAssessment] = Field(max_length=6)


class EvidenceRule(Strict):
    condition: str = Field(min_length=1,max_length=2000)
    recommendation: str = Field(min_length=1,max_length=3000)
    limitations: str = Field(min_length=1,max_length=2000)
    evidence_ids: list[str] = Field(min_length=1,max_length=6)

    @model_validator(mode='after')
    def distinct_ids(self):
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError('Rule evidence IDs must be unique')
        return self


class CoverageCuration(Strict):
    candidates: list[EvidenceRule] = Field(default_factory=list,max_length=30)
    no_rule_event_ids: list[str] = Field(default_factory=list,max_length=6)


class CallBudgetReached(Exception):
    """Bounded run can be safely resumed without resending completed calls."""


def _usage_summary(traces):
    total_input=total_output=total_tokens=missing=0
    for row in traces:
        usage=row.get('usage') if isinstance(row,dict) else None
        if not isinstance(usage,dict):
            missing+=1
            continue
        inp=usage.get('input_tokens',usage.get('prompt_tokens'))
        out=usage.get('output_tokens',usage.get('completion_tokens'))
        total=usage.get('total_tokens')
        valid=lambda v:type(v) is int and v>=0
        if valid(inp): total_input+=inp
        if valid(out): total_output+=out
        if valid(total): total_tokens+=total
        elif valid(inp) and valid(out): total_tokens+=inp+out
        else: missing+=1
    return {'reported_input_tokens':total_input,'reported_output_tokens':total_output,
            'reported_total_tokens':total_tokens,'calls_without_complete_usage':missing}


def _file_map(store, event, checked=None):
    expected=event.get('context',{}).get('evidence_sha256')
    if not isinstance(expected,dict) or not expected:
        raise ValueError('Mismatch event lacks evidence hash bindings')
    found={}
    for ref in event.get('evidence_refs',[]):
        path=store.resolve_input(ref)
        if path.name in found:
            raise ValueError('Ambiguous same-name files in mismatch evidence')
        found[path.name]=path
    if set(found)!=set(expected):
        raise ValueError('Mismatch evidence references do not match their manifest')
    for name,path in found.items():
        if not re.fullmatch(r'[a-f0-9]{64}',str(expected[name])):
            raise ValueError('Invalid mismatch evidence hash')
        previous=checked.get(str(path)) if checked is not None else None
        if previous is not None and previous != expected[name]:
            raise ValueError('Inconsistent evidence identity across events')
        if previous is None and sha256(path)!=expected[name]:
            raise ValueError('Mismatch evidence changed since matching')
        if checked is not None:
            checked[str(path)]=expected[name]
    return {str(path):expected[name] for name,path in found.items()}


def _assert_files(files):
    for name,digest in files.items():
        path=Path(name)
        if not path.is_file() or sha256(path)!=digest:
            raise ValueError('Frozen source evidence changed or is missing')


def _prepare(store, batch_path: str, chunk_size: int, expected_count: int, allow_incomplete: bool):
    if type(chunk_size) is not int or not 1 <= chunk_size <= 6:
        raise ValueError('Choose 1–6 mismatch events per visual batch')
    source=store.resolve_input(batch_path)
    document=json.loads(source.read_text(encoding='utf-8-sig'))
    if not isinstance(document,dict) or document.get('schema')!='facade-baseline-match-batch/v1':
        raise ValueError('Expected a Figure-1 match batch')
    cases=document.get('cases')
    allowed_status=('completed','needs_reference_review') if allow_incomplete else ('completed',)
    if (type(expected_count) is not int or expected_count < 1 or document.get('expected_count') != expected_count
            or document.get('status') not in allowed_status or document.get('simulation') is not False
            or document.get('reference_complete') is not True or not isinstance(cases,list)
            or len(cases)!=document.get('expected_count') or len(cases)!=document.get('matched_count')):
        raise ValueError('P0 coverage requires every real completed case and an explicit complete-reference declaration or assumption')
    all_events={e['event_id']:e for e in FailureMemory(store.root).summary()['events']}
    seen=set(); seen_sources=set(); seen_jobs=set(); batches=[]; files={str(source):sha256(source)}; case_rows=[]
    for key in ('baseline_summary','baseline_manifest'):
        path=store.resolve_input(document[key])
        if sha256(path)!=document.get(key+'_sha256'):
            raise ValueError('Frozen baseline summary or manifest changed')
        files[str(path)]=document[key+'_sha256']
    for index,case in enumerate(cases,1):
        if (not isinstance(case,dict) or case.get('status') not in allowed_status or case.get('simulation') is not False
                or not isinstance(case.get('event_ids'),list)):
            raise ValueError('Every baseline case must be real, complete and reference-reviewed')
        report=store.resolve_input(case['match_path'])
        if sha256(report)!=case.get('match_sha256'):
            raise ValueError('Match report changed since batch matching')
        files[str(report)]=case['match_sha256']
        detail=json.loads(report.read_text(encoding='utf-8-sig'))
        source_sha=detail.get('image',{}).get('sha256')
        if (detail.get('status')!=case.get('status') or detail.get('simulation') is not False
                or detail.get('reference_complete') is not True or detail.get('job_id')!=case.get('job_id')
                or source_sha!=case.get('source_sha256') or source_sha in seen_sources or case['job_id'] in seen_jobs):
            raise ValueError('Match case identity is inconsistent')
        seen_sources.add(source_sha);seen_jobs.add(case['job_id'])
        # Freeze every case, including cases with zero discrepancies; no case
        # may disappear merely because it produced no FP/FN event.
        for evidence in ('image','baseline','labels'):
            item=detail.get(evidence,{})
            path=store.resolve_input(item.get('path',''))
            if sha256(path)!=item.get('sha256'):
                raise ValueError('Case input or reference changed')
            files[str(path)]=item['sha256']
        baseline_data=json.loads(Path(detail['baseline']['path']).read_text(encoding='utf-8-sig'))
        if baseline_data.get('experiment_sha256')!=document.get('experiment_sha256'):
            raise ValueError('Case baseline belongs to a different experiment')
        if (not isinstance(detail.get('artifacts'),dict)
                or detail['artifacts']!=baseline_data.get('artifacts_sha256')):
            raise ValueError('Match report artifact manifest differs from its baseline')
        diagnostic=_diagnostic_evidence(store,Path(detail['baseline']['path']),baseline_data)
        if detail.get('diagnostic_provenance',[])!=diagnostic:
            raise ValueError('Match report diagnostic provenance changed')
        for record in diagnostic:
            for item in record['files']:
                if item['path'] in files and files[item['path']]!=item['sha256']:
                    raise ValueError('Conflicting diagnostic evidence identity')
                files[item['path']]=item['sha256']
        sidecar=store.resolve_input(detail['baseline'].get('sidecar_path',''))
        if (sha256(sidecar)!=detail['baseline'].get('sidecar_sha256')
                or json.loads(sidecar.read_text(encoding='utf-8-sig')) != {'sha256':detail['baseline']['sha256']}):
            raise ValueError('Case baseline sidecar changed')
        files[str(sidecar)]=detail['baseline']['sidecar_sha256']
        for name,digest in detail.get('artifacts',{}).items():
            if Path(name).name!=name:
                raise ValueError('Invalid baseline artifact name')
            parent=Path(detail['baseline']['path']).parent.resolve()
            path=store.resolve_input(str(parent/name))
            if not path.is_relative_to(parent) or sha256(path)!=digest:
                raise ValueError('Case baseline artifact changed or escaped its directory')
            files[str(path)]=digest
        ids=case['event_ids']
        if case['status']=='needs_reference_review':
            issues=detail.get('labels',{}).get('issues')
            if not allow_incomplete or not isinstance(issues,list) or not issues or ids:
                raise ValueError('Invalid unresolved reference case')
            case_rows.append({'job_id':case['job_id'],'source_sha256':source_sha,'match_sha256':case['match_sha256'],
                              'image_path':detail['image']['path'],'event_ids':[], 'no_disagreements':False,
                              'excluded_reason':'reference_review','reference_issues':issues,
                              'pending_candidate_ids':[x['candidate_id'] for x in
                                  [*detail.get('accepted',[]),*detail.get('deferred',[])]],
                              'pending_reference_ids':[x['reference_id'] for x in detail.get('references',[])]})
            continue
        if len(ids)!=len(set(ids)) or seen.intersection(ids):
            raise ValueError('Mismatch evidence IDs must be globally unique')
        seen.update(ids)
        expected_items={(kind,item[key]) for kind,items,key in (
            ('false_positive',detail.get('false_positives',[]),'candidate_id'),
            ('missed_defect',detail.get('false_negatives',[]),'reference_id')) for item in items}
        observed_items=set()
        for identity in ids:
            event=all_events.get(identity)
            if (not event or not _eligible_event(event) or event.get('job_id')!=case['job_id']
                    or event.get('context',{}).get('match_sha256')!=case['match_sha256']
                    or event.get('context',{}).get('source_sha256')!=source_sha
                    or event.get('reason_code') not in ('false_positive','missed_defect')):
                raise ValueError('Mismatch event is missing, simulated or belongs to another case')
            reason=event['reason_code'];ctx=event['context']
            observed_items.add((reason,ctx.get('candidate_id' if reason=='false_positive' else 'reference_id')))
            for name,digest in _file_map(store,event,files).items():
                if name in files and files[name]!=digest:
                    raise ValueError('Inconsistent evidence identity across events')
                files[name]=digest
        if observed_items != expected_items or len(ids)!=len(expected_items):
            raise ValueError('Match report and recorded mismatch coverage disagree')
        case_rows.append({'job_id':case['job_id'],'source_sha256':source_sha,'match_sha256':case['match_sha256'],
                          'image_path':detail['image']['path'],'event_ids':ids,'no_disagreements':not bool(ids),
                          'excluded_reason':None})
        for start in range(0,len(ids),chunk_size):
            selected=ids[start:start+chunk_size]
            batches.append({'batch_id':f'{index:04d}-{start//chunk_size+1:04d}',
                'job_id':case['job_id'],'match_path':str(report),'match_sha256':case['match_sha256'],
                'event_ids':selected,'event_sha256':{i:_digest(all_events[i]) for i in selected}})
    if sha256(source)!=files[str(source)]:
        raise ValueError('Match batch changed while constructing P0 coverage')
    return document,case_rows,batches,files,all_events


def _crop_evidence(store, group, events, output):
    report=json.loads(Path(group['match_path']).read_text(encoding='utf-8-sig'))
    source=store.resolve_input(report['image']['path'])
    if sha256(source)!=report['image']['sha256']:
        raise ValueError('Source image changed before P0 reflection')
    baseline=store.resolve_input(report['baseline']['path'])
    baseline_data=json.loads(baseline.read_text(encoding='utf-8-sig'))
    overview_name=baseline_data.get('overview_file')
    if not isinstance(overview_name,str) or Path(overview_name).name!=overview_name:
        raise ValueError('Figure-1 overview is missing')
    overview=store.resolve_input(str(baseline.parent/overview_name))
    if sha256(overview)!=baseline_data.get('artifacts_sha256',{}).get(overview_name):
        raise ValueError('Figure-1 overview differs from its baseline identity')
    images=[overview]; crops=[]
    folder=output/'crops'/group['batch_id'];folder.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as opened:
        opened.load(); original=opened.convert('RGB'); w,h=original.size
    for index,event in enumerate(events,1):
        box=event.get('context',{}).get('bbox_xyxy')
        if (not isinstance(box,list) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float))
                or not math.isfinite(v) for v in box) or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h)):
            raise ValueError('Mismatch event has invalid source pixel coordinates')
        pad_x,pad_y=max(16.,(box[2]-box[0])*.25),max(16.,(box[3]-box[1])*.25)
        bounds=[max(0,math.floor(box[0]-pad_x)),max(0,math.floor(box[1]-pad_y)),
                min(w,math.ceil(box[2]+pad_x)),min(h,math.ceil(box[3]+pad_y))]
        crop=original.crop(tuple(bounds)); draw=ImageDraw.Draw(crop)
        draw.rectangle([box[0]-bounds[0],box[1]-bounds[1],box[2]-bounds[0],box[3]-bounds[1]],
                       outline=(255,0,0),width=max(1,min(crop.size)//200))
        crop.thumbnail((1280,1280))
        target=folder/(f'{index:02d}-{event["event_id"]}.jpg')
        temp=target.with_suffix('.tmp')
        crop.save(temp,format='JPEG',quality=90);os.replace(temp,target)
        images.append(target)
        crops.append({'event_id':event['event_id'],'image_index':index,'crop_bbox_original':bounds,
                      'path':str(target),'sha256':sha256(target)})
    if sha256(source)!=report['image']['sha256']:
        raise ValueError('Source image changed while creating P0 evidence')
    return images,crops,baseline_data


def _related_baseline_evidence(event, baseline):
    """Retain nearby same-class rejected/accepted observations for FN review.

    This bounded context is not a complete list of every candidate: callers
    must not infer that an omitted expert candidate never existed.
    """
    ctx=event['context'];box=ctx.get('bbox_xyxy');candidate_id=ctx.get('candidate_id')
    rows=[]
    for row in baseline.get('candidates',[]):
        if not isinstance(row,dict) or row.get('class_code')!=ctx.get('class_code'):
            continue
        candidate_box=row.get('bbox_xyxy')
        if not isinstance(candidate_box,list) or len(candidate_box)!=4:
            continue
        overlap=_iou(box,candidate_box)
        if overlap > 0 or row.get('candidate_id')==candidate_id:
            rows.append((overlap,{key:row.get(key) for key in ('candidate_id','class_code','bbox_xyxy',
                'confidence','model_profile','kind','decision','reason') if key in row}))
    rows.sort(key=lambda pair:(-pair[0],str(pair[1].get('candidate_id',''))))
    total=len(rows); selected=[{**row,'event_bbox_iou':overlap} for overlap,row in rows[:20]]
    ids={row['candidate_id'] for row in selected if row.get('candidate_id')}
    trajectory=baseline.get('trajectory',[])
    related=[row for row in trajectory if isinstance(row,dict)
             and (row.get('candidate_id') in ids or row.get('stage')=='review')]
    return {'overlapping_candidates':selected,'overlapping_candidate_count':total,
            'candidate_limit':20,'candidates_truncated':total>20,
            'related_trajectory':related[:100],'trajectory_count':len(related),'trajectory_limit':100,
            'trajectory_truncated':len(related)>100,
            'scope':'same-class candidates with overlapping bbox and their related decisions; '
                    'non-overlapping or truncated candidates are not shown and must not be inferred absent'}


async def _phase_unlocked(store, folder, phase, identity, task, schema, gateway, *, image_paths=None,
                          mock_output=None, acknowledge_unknown=False, call_budget=None):
    done=folder/(phase+'.json'); pending=folder/(phase+'.pending.json')
    request_sha=_digest({'task':task,'images':[sha256(path) for path in image_paths or []]})
    if done.is_file():
        cached=json.loads(done.read_text(encoding='utf-8-sig'))
        if (cached.get('request_sha256')!=request_sha or cached.get('identity')!=identity
                or _digest(cached.get('output'))!=cached.get('output_sha256')):
            raise ValueError('Saved P0 response does not match the frozen batch')
        return schema.model_validate(cached['output']),cached['trace']
    if call_budget is not None and call_budget['calls'] >= call_budget['limit']:
        raise CallBudgetReached()
    if pending.exists():
        if not acknowledge_unknown:
            raise ValueError('Previous P0 request outcome unknown; explicit acknowledgement required before retry')
        n=1
        while (folder/(phase+f'.unknown-{n}.json')).exists(): n+=1
        os.replace(pending,folder/(phase+f'.unknown-{n}.json'))
    store.write_json(pending,{'identity':identity,'request_sha256':request_sha,'status':'request_pending'})
    if call_budget is not None:
        call_budget['calls']+=1
    before=len(gateway.trace)
    try:
        answer=await gateway.invoke(phase,task,schema,mock_output=mock_output,image_paths=image_paths)
    except GatewayError:
        trace=gateway.trace[-1] if len(gateway.trace)>before else {}
        store.write_json(pending,{'identity':identity,'request_sha256':request_sha,
                                  'status':'request_failed_or_unknown','trace':trace})
        raise
    trace=gateway.trace[-1]
    output=answer.model_dump(mode='json')
    store.write_json(pending,{'identity':identity,'request_sha256':request_sha,'status':'completed',
                               'output':output,'output_sha256':_digest(output),'trace':trace})
    os.replace(pending,done)
    return answer,trace


async def _phase(store, folder, phase, *args, **kwargs):
    # The lock is held across the provider call. A concurrent process must not
    # overwrite the pending marker or send the same paid request a second time.
    # POSIX flock is released automatically if a process exits or is killed.
    try:
        import fcntl
    except ImportError:
        raise RuntimeError('P0 request locking requires a POSIX runtime') from None
    flags=os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0)
    fd=os.open(str(folder/(phase+'.lock')),flags,0o600)
    try:
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('The same P0 model request is already in progress; no duplicate sent') from None
        try:
            return await _phase_unlocked(store,folder,phase,*args,**kwargs)
        finally:
            fcntl.flock(fd,fcntl.LOCK_UN)
    finally:
        os.close(fd)


async def propose_full_coverage_p0(store: ArtifactStore, settings, match_batch_path: str, *,
                                   allow_api=False, transport=None, chunk_size=6,
                                   expected_count=203, acknowledge_unknown=False, max_calls=1024,
                                   allow_incomplete=False) -> dict:
    """Reflect/curate every real discrepancy, without silently sampling history.

    Results are resumable by immutable manifest. Calls that have no known outcome
    are never retried without explicit acknowledgement. The resulting Playbook
    is an unregistered candidate; independent validation/publication is future
    work. User annotation review is required before a flawed GT case is accepted.
    """
    if type(max_calls) is not int or not 1<=max_calls<=1024:
        raise ValueError('Per-run model call limit must be an integer in [1,1024]')
    if type(allow_incomplete) is not bool:
        raise ValueError('allow_incomplete must be explicitly true or false')
    doc,cases,batches,files,events=_prepare(store,match_batch_path,chunk_size,expected_count,allow_incomplete)
    excluded=[case for case in cases if case.get('excluded_reason')]
    if bool(excluded) != (doc['status']=='needs_reference_review'):
        raise ValueError('Batch reference-review status does not match its cases')
    simulation=transport is not None or any(settings.connections[settings.roles[role]].adapter=='mock'
                                           for role in ('reflector','curator'))
    if not simulation and batches:
        if not allow_api or not settings.api_enabled:
            raise ValueError('Explicitly enable configured model calls before reflecting')
        for role in ('reflector','curator'):
            con=settings.connections[settings.roles[role]]
            if not con.expected_model:
                raise ValueError('Pin the expected reflector and curator model IDs')
    if batches and not settings.connections[settings.roles['reflector']].supports_images:
        raise ValueError('The reflector must accept original-image crop evidence')
    endpoint_hashes={}
    for role in ('reflector','curator'):
        connection=settings.connections[settings.roles[role]]
        endpoint=os.environ.get(connection.endpoint_env,'') if connection.endpoint_env else ''
        endpoint_hashes[role]=hashlib.sha256(endpoint.encode()).hexdigest() if endpoint else None
    manifest={'schema':'facade-baseline-p0-full/v1','match_batch_sha256':files[str(store.resolve_input(match_batch_path))],
        'experiment_sha256':doc['experiment_sha256'],'case_count':len(cases),'cases':cases,'batches':batches,
        'reference_assumption':doc.get('reference_assumption'),
        'source_files_sha256':files,'model_config_sha256':_digest(settings.model_dump(mode='json')),
        'endpoint_hashes':endpoint_hashes,'implementation_sha256':sha256(Path(__file__)),'chunk_size':chunk_size,
        'allow_incomplete':allow_incomplete,'simulation':simulation}
    identity=_digest(manifest)
    folder=store.root/'baseline_p0_full'/identity;folder.mkdir(parents=True,exist_ok=True)
    manifest_path=folder/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8-sig')) != {**manifest,'manifest_sha256':identity}:
            raise ValueError('Existing P0 manifest differs from the frozen inputs')
    else:
        store.write_json(manifest_path,{**manifest,'manifest_sha256':identity})
    gateway=ModelGateway(settings,allow_api=allow_api,transport=transport)
    rules={}; traces=[]; counted=[]; call_budget={'calls':0,'limit':max_calls}
    total_events=sum(len(group['event_ids']) for group in batches)
    def halt(status,reason):
        state={'schema':'facade-baseline-p0-progress/v1','status':status,'reason':reason,
            'manifest_sha256':identity,'manifest_path':str(manifest_path),
            'completed_batches':len(counted_batches),'total_batches':len(batches),
            'accounted_event_count':len(counted),'total_event_count':total_events,
            'coverage_status':'partial_reference_review' if excluded else 'complete',
            'excluded_case_count':len(excluded),'excluded_cases':excluded,
            'model_calls_this_run':call_budget['calls'],'completed_or_failed_calls_seen':len(traces),
            'usage':_usage_summary(traces),'simulation':simulation}
        store.write_json(folder/'progress.json',state)
        return state
    counted_batches=[]
    for group in batches:
        source_events=[events[key] for key in group['event_ids']]
        checked={}
        for event in source_events:
            if _digest(event)!=group['event_sha256'][event['event_id']]:
                raise ValueError('Frozen mismatch event changed')
            _file_map(store,event,checked)
        images,crops,baseline_data=_crop_evidence(store,group,source_events,folder)
        observations=[]
        for event,crop in zip(source_events,crops):
            ctx=event['context']; cid=ctx.get('candidate_id')
            observations.append({'event_id':event['event_id'],'reason_code':event['reason_code'],
                'class_code':ctx.get('class_code'),'bbox_xyxy':ctx.get('bbox_xyxy'),
                'candidate_id':cid,'reference_id':ctx.get('reference_id'),
                'match_diagnostic':ctx.get('match_diagnostic'),'competing_reference_id':ctx.get('competing_reference_id'),
                'crop_image_index':crop['image_index'],'crop_bbox_original':crop['crop_bbox_original'],
                'related_baseline':_related_baseline_evidence(event,baseline_data),
                'note':'A matching disagreement is conditional on the provided reference, not a proven cause.'})
        task={'stage':'baseline_failure_reflection','batch_id':group['batch_id'],'events':observations,
              'image_order':'image 0 is the Figure-1 overview; each numbered crop is an original-pixel region with a red bbox',
              'instruction':'Assess every event once. Treat explanations as hypotheses; use insufficient_evidence if unsure. '
                            'Suggest only a controlled inspection change; do not invent observations or measurements.'}
        mock={'assessments':[{'event_id':e['event_id'],'status':'insufficient_evidence',
            'hypothesis':'Synthetic mock; no cause established.','controlled_change':'No proposed change.'} for e in source_events]}
        # Cache by immutable case evidence + model identity rather than by the
        # overall match batch: after one invalid GT is reviewed, valid cases
        # reuse their completed calls without overwriting the earlier result.
        cache_key=_digest({'group':group,'model_config_sha256':manifest['model_config_sha256'],
                           'endpoint_hashes':endpoint_hashes,'implementation_sha256':manifest['implementation_sha256'],
                           'simulation':simulation})
        phase_dir=store.root/'baseline_p0_full'/'shared-cache'/cache_key;phase_dir.mkdir(parents=True,exist_ok=True)
        try:
            reflected,trace=await _phase(store,phase_dir,'reflector',cache_key,task,CoverageReflection,gateway,
                image_paths=images,mock_output=mock,acknowledge_unknown=acknowledge_unknown,call_budget=call_budget)
        except CallBudgetReached:
            return halt('paused_call_budget','Per-run model call limit reached; safely resume to continue')
        except GatewayError:
            if gateway.trace: traces.append(gateway.trace[-1])
            return halt('halted_provider_error','Model call failed or outcome is unknown; no candidate was completed')
        traces.append(trace)
        actual=[x.event_id for x in reflected.assessments]
        if len(actual)!=len(set(actual)) or set(actual)!=set(group['event_ids']):
            raise ValueError('Reflector must assess every mismatch exactly once')
        ctask={'stage':'baseline_rule_curation','batch_id':group['batch_id'],'events':observations,
               'reflection':reflected.model_dump(mode='json'),
               'instruction':'Propose only evidence-cited reusable inspection rules. List every event that does not '
                             'support a rule in no_rule_event_ids. Rules remain unverified candidates.'}
        mock_curation={'candidates':[],'no_rule_event_ids':group['event_ids']}
        try:
            curated,ctrace=await _phase(store,phase_dir,'curator',cache_key,ctask,CoverageCuration,gateway,
                mock_output=mock_curation,acknowledge_unknown=acknowledge_unknown,call_budget=call_budget)
        except CallBudgetReached:
            return halt('paused_call_budget','Per-run model call limit reached; safely resume to continue')
        except GatewayError:
            if gateway.trace: traces.append(gateway.trace[-1])
            return halt('halted_provider_error','Model call failed or outcome is unknown; no candidate was completed')
        traces.append(ctrace)
        used=set(); allowed=set(group['event_ids'])
        supported={assessment.event_id for assessment in reflected.assessments
                   if assessment.status=='hypothesis'}
        for item in curated.candidates:
            if not set(item.evidence_ids)<=allowed:
                raise ValueError('Curator cited an event from outside the current batch')
            if not set(item.evidence_ids)<=supported:
                raise ValueError('Curator cannot create a rule from insufficient evidence')
            used.update(item.evidence_ids)
            key=fingerprint(item.condition,item.recommendation)
            if key in rules:
                old=rules[key]
                old.evidence_ids=list(dict.fromkeys(old.evidence_ids+item.evidence_ids))
                if item.limitations not in old.limitations:
                    old.limitations+='\n'+item.limitations
            else:
                rules[key]=Rule(id='rule-'+key[:24],condition=item.condition,recommendation=item.recommendation,
                                limitations=item.limitations,evidence_ids=item.evidence_ids)
        omitted=curated.no_rule_event_ids
        if len(omitted)!=len(set(omitted)) or not set(omitted)<=allowed or set(omitted)&used or used|set(omitted)!=allowed:
            raise ValueError('Curator must account for every event, without invented IDs')
        counted.extend(group['event_ids']);counted_batches.append(group['batch_id'])
        store.write_json(folder/'progress.json',{'schema':'facade-baseline-p0-progress/v1','status':'running',
            'manifest_sha256':identity,'completed_batches':len(counted_batches),'total_batches':len(batches),
            'accounted_event_count':len(counted),'total_event_count':total_events,
            'coverage_status':'partial_reference_review' if excluded else 'complete',
            'excluded_case_count':len(excluded),'excluded_cases':excluded,
            'model_calls_this_run':call_budget['calls'],'completed_or_failed_calls_seen':len(traces),
            'usage':_usage_summary(traces)})
    _assert_files(files)
    all_ids=[e for case in cases for e in case['event_ids']]
    if len(counted)!=len(set(counted)) or set(counted)!=set(all_ids):
        raise ValueError('P0 coverage audit did not account for every mismatch')
    book=Playbook(version='P0-candidate-'+identity[:24],rules=list(rules.values()))
    book_path=folder/'candidate-playbook.json';store.write_json(book_path,book.model_dump(mode='json'))
    coverage_status=('partial_reference_review' if excluded else
                     'conditional_reference_assumption' if doc.get('reference_assumption') else 'complete')
    result={'schema':'facade-baseline-p0-candidate/v1','status':coverage_status if excluded else 'candidate_only','promoted':False,
        'registered_with_rule_lifecycle':False,'simulation':simulation,'case_count':len(cases),
        'reference_assumption':doc.get('reference_assumption'),
        'conditional_on_unreviewed_reference':bool(doc.get('reference_assumption')),
        'coverage_status':coverage_status,'reviewed_case_count':len(cases)-len(excluded),
        'excluded_case_count':len(excluded),'excluded_cases':excluded,
        'zero_disagreement_cases':sum(case['no_disagreements'] for case in cases),
        'eligible_event_count':len(all_ids),'accounted_event_count':len(counted),'rule_count':len(book.rules),
        'model_calls_this_run':call_budget['calls'],'completed_calls_seen':len(traces),'usage':_usage_summary(traces),
        'manifest_path':str(manifest_path),'manifest_sha256':identity,'manifest_file_sha256':sha256(manifest_path),
        'candidate_playbook_path':str(book_path),
        'candidate_playbook_file_sha256':sha256(book_path),'model_trace':traces,
        'limitations':['All observed mismatches in included cases were sent for assessment; this does not prove causal correctness.',
                       'Cases pending reference review are excluded and must not be counted as covered.',
                       'No mismatch events means an empty candidate: no rules are invented without evidence.',
                       'P0 is unverified, is not registered for publication and requires independent validation.',
                       'The 203 source images contributed to development and are not independent P0 test data.']}
    store.write_json(folder/'candidate-metadata.json',result)
    store.write_json(folder/'progress.json',{'schema':'facade-baseline-p0-progress/v1','status':coverage_status if excluded else 'candidate_only',
        'manifest_sha256':identity,'completed_batches':len(batches),'total_batches':len(batches),
        'accounted_event_count':len(counted),'total_event_count':len(all_ids),
        'coverage_status':coverage_status,'excluded_case_count':len(excluded),'excluded_cases':excluded,
        'model_calls_this_run':call_budget['calls'],'completed_calls_seen':len(traces),'usage':_usage_summary(traces)})
    return result
