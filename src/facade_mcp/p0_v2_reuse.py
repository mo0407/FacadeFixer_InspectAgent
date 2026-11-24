"""Read-only, evidence-bound selection of a completed P0-v1 prefix.

This module never sends model requests or writes P0-v1 artifacts.  A caller
must create a separate, explicitly versioned v2 manifest and caches; returned
v1 outputs remain attributed to their original task and implementation.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import types
from urllib.parse import quote

from . import baseline_p0_coverage as v1
from .baseline_p0_coverage import CoverageCuration, CoverageReflection
from .failure_memory import FailureMemory
from .model_gateway import ModelConfig, _load_images
from .storage import ArtifactStore, sha256

_HASH = re.compile(r'^[a-f0-9]{64}$')
_BATCH_ID = re.compile(r'^\d{4}-\d{4}$')
_EVENT_ID = re.compile(r'^[a-f0-9]{32}$')


class _ReadOnlyMemory:
    """Use v1's row decoder, but never change the original SQLite database."""
    def __init__(self, workspace):
        self.path=Path(workspace)/'failure_memory.sqlite3'

    def summary(self):
        if not self.path.is_file() or Path(str(self.path)+'-wal').exists():
            raise ValueError('Reference ledger missing or has an uncheckpointed WAL')
        uri='file:'+quote(str(self.path.resolve()))+'?mode=ro&immutable=1'
        with closing(sqlite3.connect(uri,uri=True)) as db:
            db.row_factory=sqlite3.Row
            rows=db.execute('SELECT * FROM events ORDER BY recorded_at DESC, event_id DESC').fetchall()
        return {'events':[FailureMemory._event(row) for row in rows]}


def _readonly_prepare(store, match_path, expected_count):
    """Re-execute the frozen v1 verifier with a read-only ledger dependency.

    A private function object avoids monkeypatching v1 globals in a process
    that might also run a gateway or another inspection concurrently.
    """
    bindings=dict(v1._prepare.__globals__)
    bindings['FailureMemory']=_ReadOnlyMemory
    verifier=types.FunctionType(v1._prepare.__code__,bindings,v1._prepare.__name__,
                                v1._prepare.__defaults__,v1._prepare.__closure__)
    return verifier(store,str(match_path),6,expected_count,True)


def _load_json(path):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(data,dict):
        raise ValueError('Expected a JSON object')
    return data


def _inside(store, path, root):
    resolved=store.resolve_input(str(path)); root=Path(root).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('P0-v1 evidence escaped its frozen directory')
    return resolved


def _source_path(store, root, child):
    path=_inside(store,Path(root)/child,root)
    if not path.is_file():
        raise ValueError('Required P0-v1 evidence is missing')
    return path


def _group_key(parent,group):
    return v1._digest({'group':group,'model_config_sha256':parent['model_config_sha256'],
        'endpoint_hashes':parent['endpoint_hashes'],'implementation_sha256':parent['implementation_sha256'],
        'simulation':parent['simulation']})


def _tasks_and_images(store, parent_dir, group, events):
    """Rebuild exact v1 task inputs without regenerating or editing crop files."""
    if not _BATCH_ID.fullmatch(str(group.get('batch_id',''))):
        raise ValueError('Invalid frozen batch identifier')
    report_path=store.resolve_input(group['match_path'])
    if sha256(report_path)!=group.get('match_sha256'):
        raise ValueError('P0-v1 match report changed')
    report=_load_json(report_path)
    baseline_path=store.resolve_input(report['baseline']['path']); baseline=_load_json(baseline_path)
    name=baseline.get('overview_file')
    if not isinstance(name,str) or Path(name).name!=name:
        raise ValueError('P0-v1 overview identity is invalid')
    overview=_source_path(store,baseline_path.parent,name)
    if sha256(overview)!=baseline.get('artifacts_sha256',{}).get(name):
        raise ValueError('P0-v1 overview changed')
    width,height=report['image']['width'],report['image']['height']
    if type(width) is not int or type(height) is not int or width<=0 or height<=0:
        raise ValueError('P0-v1 image dimensions are invalid')
    source=store.resolve_input(report['image']['path'])
    if sha256(source)!=report['image']['sha256']:
        raise ValueError('P0-v1 original image changed')
    observations=[]; images=[overview]
    crop_base=(parent_dir/'crops').resolve()
    crops_root=(crop_base/group['batch_id']).resolve()
    if not crop_base.is_relative_to(parent_dir) or not crops_root.is_relative_to(crop_base):
        raise ValueError('P0-v1 crop directory escaped its experiment')
    for index,event in enumerate(events,1):
        identity=event.get('event_id')
        if not isinstance(identity,str) or not _EVENT_ID.fullmatch(identity):
            raise ValueError('P0-v1 event identifier is invalid')
        if v1._digest(event)!=group.get('event_sha256',{}).get(identity):
            raise ValueError('P0-v1 event changed')
        ctx=event.get('context',{}); box=ctx.get('bbox_xyxy')
        if (not isinstance(box,list) or len(box)!=4 or any(isinstance(x,bool) or
                not isinstance(x,(int,float)) or not math.isfinite(x) for x in box)
                or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height)):
            raise ValueError('P0-v1 event bounding box is invalid')
        px,py=max(16.,(box[2]-box[0])*.25),max(16.,(box[3]-box[1])*.25)
        bounds=[max(0,math.floor(box[0]-px)),max(0,math.floor(box[1]-py)),
                min(width,math.ceil(box[2]+px)),min(height,math.ceil(box[3]+py))]
        crop=_source_path(store,crops_root,f'{index:02d}-{identity}.jpg')
        images.append(crop)
        observations.append({'event_id':identity,'reason_code':event['reason_code'],
            'class_code':ctx.get('class_code'),'bbox_xyxy':box,'candidate_id':ctx.get('candidate_id'),
            'reference_id':ctx.get('reference_id'),'match_diagnostic':ctx.get('match_diagnostic'),
            'competing_reference_id':ctx.get('competing_reference_id'),'crop_image_index':index,
            'crop_bbox_original':bounds,'related_baseline':v1._related_baseline_evidence(event,baseline),
            'note':'A matching disagreement is conditional on the provided reference, not a proven cause.'})
    task={'stage':'baseline_failure_reflection','batch_id':group['batch_id'],'events':observations,
        'image_order':'image 0 is the Figure-1 overview; each numbered crop is an original-pixel region with a red bbox',
        'instruction':'Assess every event once. Treat explanations as hypotheses; use insufficient_evidence if unsure. '
                      'Suggest only a controlled inspection change; do not invent observations or measurements.'}
    # Original gateway metadata includes decoded dimensions/mime as well as SHA.
    image_meta=[metadata for _,_,metadata in _load_images(images)]
    return observations,task,images,image_meta


def _phase_file(store, folder, phase, key, request_sha, connection, expected_images=None):
    path=_source_path(store,folder,phase+'.json'); before=sha256(path); row=_load_json(path)
    trace=row.get('trace')
    if (row.get('status')!='completed' or row.get('identity')!=key
            or row.get('request_sha256')!=request_sha or row.get('output_sha256')!=v1._digest(row.get('output'))
            or not isinstance(trace,dict) or trace.get('status')!='completed' or trace.get('http_status')!=200
            or trace.get('role')!=phase or trace.get('adapter')!=connection.adapter
            or trace.get('requested_model')!=connection.model or trace.get('actual_model')!=connection.expected_model
            or trace.get('model_identity_verified') is not True):
        raise ValueError('P0-v1 phase identity, request or model trace is invalid')
    if expected_images is not None and trace.get('images')!=expected_images:
        raise ValueError('P0-v1 reflected image metadata changed')
    if expected_images is None and trace.get('images') not in (None,[]):
        raise ValueError('P0-v1 curator unexpectedly received image evidence')
    if sha256(path)!=before:
        raise ValueError('P0-v1 phase changed during verification')
    return row,{'cache_path':str(path),'cache_sha256':before,'request_sha256':request_sha,'role':phase,
                'trace':trace,'usage':trace.get('usage'),'source':'p0-v1-original-cache'}


def _valid_reflection(group, output):
    reflection=CoverageReflection.model_validate(output)
    found=[row.event_id for row in reflection.assessments]
    if len(found)!=len(set(found)) or set(found)!=set(group['event_ids']):
        raise ValueError('P0-v1 reflector did not account for each event')
    return reflection


def _valid_curation(group, reflection, output):
    curation=CoverageCuration.model_validate(output); allowed=set(group['event_ids']); used=set()
    supported={r.event_id for r in reflection.assessments if r.status=='hypothesis'}
    for rule in curation.candidates:
        ids=set(rule.evidence_ids)
        if not ids<=allowed or not ids<=supported:
            raise ValueError('P0-v1 curator cites missing or insufficient evidence')
        used.update(ids)
    omitted=curation.no_rule_event_ids
    if len(omitted)!=len(set(omitted)) or not set(omitted)<=allowed or set(omitted)&used or used|set(omitted)!=allowed:
        raise ValueError('P0-v1 curator did not account for each event')
    return curation


def _curator_task(group,observations,reflection):
    return {'stage':'baseline_rule_curation','batch_id':group['batch_id'],'events':observations,
        'reflection':reflection.model_dump(mode='json'),
        'instruction':'Propose only evidence-cited reusable inspection rules. List every event that does not '
                      'support a rule in no_rule_event_ids. Rules remain unverified candidates.'}


def _usage(traces):
    return v1._usage_summary(traces)


def _history(store, folder, phase, key, request_sha, parent_id, group, reflection, connection):
    """Preserve original failed/rejected records, never promote them as outputs."""
    records=[]; traces=[]
    rejected=sorted(folder.glob(phase+'.semantic-rejected-*.json'))
    plans=sorted(folder.glob(phase+'.semantic-plan-*.json'))
    commits=sorted(folder.glob(phase+'.semantic-commit-*.json'))
    expected=lambda kind:[phase+'.semantic-'+kind+f'-{n:04d}.json' for n in range(1,len(rejected)+1)]
    if ([p.name for p in rejected]!=expected('rejected') or [p.name for p in plans]!=expected('plan')
            or [p.name for p in commits]!=expected('commit')):
        raise ValueError('P0-v1 semantic rejection history is incomplete')
    if len(rejected)>16:
        raise ValueError('P0-v1 semantic rejection history exceeds audit limit')
    for archive,plan_file,commit_file in zip(rejected,plans,commits):
        archive=_source_path(store,folder,archive.name); plan_file=_source_path(store,folder,plan_file.name)
        commit_file=_source_path(store,folder,commit_file.name)
        row=_load_json(archive); trace=row.get('trace'); plan=_load_json(plan_file); commit=_load_json(commit_file)
        if (row.get('status')!='completed' or row.get('identity')!=key or row.get('request_sha256')!=request_sha
                or row.get('output_sha256')!=v1._digest(row.get('output')) or not isinstance(trace,dict)
                or trace.get('status')!='completed' or trace.get('http_status')!=200 or trace.get('role')!=phase
                or trace.get('adapter')!=connection.adapter or trace.get('requested_model')!=connection.model
                or trace.get('actual_model')!=connection.expected_model or trace.get('model_identity_verified') is not True
                or plan.get('manifest_sha256')!=parent_id or plan.get('source_file_sha256')!=sha256(archive)
                or plan.get('request_sha256',plan.get(phase+'_request_sha256'))!=request_sha
                or plan.get('phase',phase)!=phase or plan.get('batch_id',group['batch_id'])!=group['batch_id']
                or plan.get('action') not in ('preserve_rejected_phase_only','preserve_rejected_curator_cache_only')
                or (plan.get('action')=='preserve_rejected_curator_cache_only' and phase!='curator')
                or plan.get('automatic_api_calls')!=0
                or commit.get('status')!=({'preserve_rejected_phase_only':'rejected_phase_preserved',
                    'preserve_rejected_curator_cache_only':'rejected_curator_preserved'}).get(plan.get('action'))
                or commit.get('plan_sha256')!=sha256(plan_file)
                or commit.get('archive_sha256')!=sha256(archive) or not commit.get('committed_utc')
                or not _HASH.fullmatch(str(plan.get('recovery_script_sha256','')))
                or commit.get('recovery_script_sha256')!=plan['recovery_script_sha256']):
            raise ValueError('P0-v1 semantic rejection provenance is invalid')
        (CoverageReflection if phase=='reflector' else CoverageCuration).model_validate(row['output'])
        try:
            if phase=='reflector': _valid_reflection(group,row['output'])
            else: _valid_curation(group,reflection,row['output'])
        except ValueError:
            pass
        else:
            raise ValueError('P0-v1 semantic rejection is actually valid')
        records.append({'kind':'semantic_rejected','phase':phase,'batch_id':group['batch_id'],
            'request_sha256':request_sha,'files':[{'path':str(p),'sha256':sha256(p)}
            for p in (archive,plan_file,commit_file)],'trace':trace,'usage':trace.get('usage')})
        traces.append(trace)
    if list(folder.glob(phase+'.unknown-*.json')):
        raise ValueError('P0-v1 has an unreconciled unknown request')
    other=sorted(folder.glob(phase+'.failed-retry-*.json'))
    if len(other)>3 or [p.name for p in other]!=[phase+f'.failed-retry-{i:03d}.json' for i in range(1,len(other)+1)]:
        raise ValueError('P0-v1 HTTP retry history is incomplete or exceeds its limit')
    for file in other:
        path=_source_path(store,folder,file.name); row=_load_json(path); trace=row.get('trace')
        known_http=(isinstance(trace,dict) and trace.get('http_status') in (408,429,500,502,503,504)
                    and trace.get('error_code') in ('api_http_error','api_rate_limited'))
        invalid_output=(isinstance(trace,dict) and trace.get('http_status')==200
                        and trace.get('error_code')=='model_output_invalid'
                        and trace.get('model_identity_verified') is True
                        and trace.get('actual_model')==connection.expected_model)
        if (row.get('identity')!=key or row.get('request_sha256')!=request_sha
                or row.get('status')!='request_failed_or_unknown' or not isinstance(trace,dict)
                or trace.get('status')!='failed' or trace.get('role')!=phase
                or trace.get('adapter')!=connection.adapter or trace.get('requested_model')!=connection.model
                or not (known_http or invalid_output)):
            raise ValueError('P0-v1 failed or unknown request history differs')
        records.append({'kind':'failed_or_unknown','phase':phase,'batch_id':group['batch_id'],
            'request_sha256':request_sha,'files':[{'path':str(path),'sha256':sha256(path)}],
            'trace':trace,'usage':trace.get('usage') if isinstance(trace,dict) else None})
        traces.append(trace if isinstance(trace,dict) else {})
    return records,traces


def verify_parent_prefix(store: ArtifactStore, parent_manifest_path: str, settings: ModelConfig, *,
                         expected_parent_file_sha256: str, expected_prefix: int=112,
                         expected_count: int=203) -> dict:
    """Verify 112 original full batches plus the next original reflector.

    This is read-only and performs no API calls. `expected_prefix` is exposed
    for small synthetic tests; a production caller must explicitly pin 112/203
    and pass a separately recorded parent manifest file SHA-256.
    """
    if (not isinstance(store,ArtifactStore) or not isinstance(settings,ModelConfig)
            or type(expected_prefix) is not int or expected_prefix<0
            or type(expected_count) is not int or expected_count<1
            or not isinstance(expected_parent_file_sha256,str) or not _HASH.fullmatch(expected_parent_file_sha256)):
        raise ValueError('Explicit P0-v1 identities and counts are required')
    manifest_path=store.resolve_input(parent_manifest_path)
    if sha256(manifest_path)!=expected_parent_file_sha256:
        raise ValueError('P0-v1 parent manifest file hash differs')
    parent=_load_json(manifest_path); unsigned=dict(parent); identity=unsigned.pop('manifest_sha256',None)
    full_root=(store.root/'baseline_p0_full').resolve()
    if (parent.get('schema')!='facade-baseline-p0-full/v1' or parent.get('simulation') is not False
            or not isinstance(identity,str) or not _HASH.fullmatch(identity) or v1._digest(unsigned)!=identity
            or manifest_path.name!='manifest.json' or manifest_path.parent.name!=identity
            or not full_root.is_relative_to(store.root.resolve()) or not manifest_path.is_relative_to(full_root)
            or parent.get('case_count')!=expected_count or parent.get('chunk_size')!=6
            or parent.get('allow_incomplete') is not True):
        raise ValueError('P0-v1 parent manifest identity is invalid')
    parent_dir=manifest_path.parent; shared=(store.root/'baseline_p0_full'/'shared-cache').resolve()
    if not shared.is_relative_to(full_root):
        raise ValueError('P0-v1 cache escaped its workspace')
    if list(shared.glob('*/*.pending.json')) or list(shared.glob('*/*.unknown-*.json')):
        raise ValueError('A P0-v1 request is still pending or has an unknown result')
    if sha256(Path(v1.__file__))!=parent.get('implementation_sha256'):
        raise ValueError('P0-v1 coverage implementation changed')
    if v1._digest(settings.model_dump(mode='json'))!=parent.get('model_config_sha256'):
        raise ValueError('P0-v1 model configuration changed')
    endpoints={}
    for role in ('reflector','curator'):
        connection=settings.connections[settings.roles[role]]
        endpoint=os.environ.get(connection.endpoint_env,'') if connection.endpoint_env else ''
        if not connection.expected_model or connection.expected_model!=connection.model or not endpoint:
            raise ValueError('P0-v1 expected model or endpoint is not pinned')
        endpoints[role]=hashlib.sha256(endpoint.encode()).hexdigest()
    if endpoints!=parent.get('endpoint_hashes'):
        raise ValueError('P0-v1 endpoint identity changed')
    source=parent.get('source_files_sha256'); match_hash=parent.get('match_batch_sha256')
    if not isinstance(source,dict) or not isinstance(match_hash,str) or not _HASH.fullmatch(match_hash):
        raise ValueError('P0-v1 source manifest is invalid')
    matches=[]
    for path,digest in source.items():
        if digest==match_hash:
            candidate=store.resolve_input(path)
            if candidate.is_file() and sha256(candidate)==digest and _load_json(candidate).get('schema')=='facade-baseline-match-batch/v1':
                matches.append(candidate)
    if len(matches)!=1:
        raise ValueError('P0-v1 match batch is missing or ambiguous')
    match_path=matches[0]; db_path=store.resolve_input(str(store.root/'failure_memory.sqlite3'))
    db_hash=sha256(db_path)
    doc,cases,batches,files,events=_readonly_prepare(store,match_path,expected_count)
    if (len(batches)<=expected_prefix or parent.get('cases')!=cases or parent.get('batches')!=batches
            or parent.get('source_files_sha256')!=files or parent.get('experiment_sha256')!=doc.get('experiment_sha256')
            or parent.get('reference_assumption')!=doc.get('reference_assumption')):
        raise ValueError('P0-v1 frozen batch or reference evidence changed')
    progress_path=_source_path(store,parent_dir,'progress.json'); progress=_load_json(progress_path)
    if (progress.get('schema')!='facade-baseline-p0-progress/v1' or progress.get('manifest_sha256')!=identity
            or progress.get('status')!='running'
            or progress.get('completed_batches')!=expected_prefix
            or progress.get('accounted_event_count')!=sum(len(g['event_ids']) for g in batches[:expected_prefix])
            or progress.get('total_batches')!=len(batches)
            or progress.get('total_event_count')!=sum(len(g['event_ids']) for g in batches)):
        raise ValueError('P0-v1 progress does not match the selected prefix')
    # The selected stopped prefix must not hide extra model outcomes from a
    # later batch.  Such histories require an explicit reconciliation policy.
    for group in batches[expected_prefix+1:]:
        later=(shared/_group_key(parent,group)).resolve()
        if not later.is_relative_to(shared):
            raise ValueError('P0-v1 later cache escaped its directory')
        if later.is_dir() and any(later.glob('*.json')):
            raise ValueError('P0-v1 has unexplained cache beyond the stopped batch')
    selected=[]; histories=[]; traces=[]; pinned={str(manifest_path):expected_parent_file_sha256,
                                                  str(progress_path):sha256(progress_path),str(db_path):db_hash}
    for position,group in enumerate(batches[:expected_prefix+1]):
        group_events=[events[event_id] for event_id in group['event_ids']]
        key=_group_key(parent,group); folder=(shared/key).resolve()
        if not folder.is_relative_to(shared) or not folder.is_dir():
            raise ValueError('P0-v1 phase directory is missing or outside its cache')
        observations,reflect_task,images,image_meta=_tasks_and_images(store,parent_dir,group,group_events)
        reflect_sha=v1._digest({'task':reflect_task,'images':[x['sha256'] for x in image_meta]})
        conn=settings.connections[settings.roles['reflector']]
        reflect_row,reflect_prov=_phase_file(store,folder,'reflector',key,reflect_sha,conn,image_meta)
        reflected=_valid_reflection(group,reflect_row['output'])
        reflect_prov['images']=[{'path':str(path),**metadata} for path,metadata in zip(images,image_meta)]
        ctask=_curator_task(group,observations,reflected); curator_sha=v1._digest({'task':ctask,'images':[]})
        records,old_traces=_history(store,folder,'reflector',key,reflect_sha,identity,group,None,conn)
        histories.extend(records); traces.extend(old_traces)
        curator_connection=settings.connections[settings.roles['curator']]
        records,old_traces=_history(store,folder,'curator',key,curator_sha,identity,group,reflected,curator_connection)
        histories.extend(records); traces.extend(old_traces)
        item={'batch_id':group['batch_id'],'observations':observations,
              'reflection':reflected.model_dump(mode='json'),'provenance':{'reflector':reflect_prov}}
        traces.append(reflect_prov['trace']); pinned[reflect_prov['cache_path']]=reflect_prov['cache_sha256']
        for image in reflect_prov['images']: pinned[image['path']]=image['sha256']
        if position<expected_prefix:
            conn=settings.connections[settings.roles['curator']]
            curator_row,curator_prov=_phase_file(store,folder,'curator',key,curator_sha,conn)
            curated=_valid_curation(group,reflected,curator_row['output']); curator_prov['images']=[]
            item['curation']=curated.model_dump(mode='json');item['provenance']['curator']=curator_prov
            traces.append(curator_prov['trace']);pinned[curator_prov['cache_path']]=curator_prov['cache_sha256']
        else:
            # Preserve the stopped curator, but never treat its invalid output
            # as a reusable v1 result or as a v2 response.
            conn=settings.connections[settings.roles['curator']]
            invalid,provenance=_phase_file(store,folder,'curator',key,curator_sha,conn)
            try: _valid_curation(group,reflected,invalid['output'])
            except ValueError: pass
            else: raise ValueError('P0-v1 stopped curator was unexpectedly valid')
            histories.append({'kind':'stopped_invalid_curator','phase':'curator','batch_id':group['batch_id'],
                'request_sha256':curator_sha,'files':[{'path':provenance['cache_path'],
                'sha256':provenance['cache_sha256']}],'trace':provenance['trace'],'usage':provenance['usage']})
            traces.append(provenance['trace']);pinned[provenance['cache_path']]=provenance['cache_sha256']
        selected.append(item)
    for entry in histories:
        for item in entry['files']:
            prior=pinned.setdefault(item['path'],item['sha256'])
            if prior!=item['sha256']:
                raise ValueError('Conflicting P0-v1 history identities')
    response_paths=[phase['cache_path'] for item in selected for phase in item['provenance'].values()]
    response_paths.extend(entry['files'][0]['path'] for entry in histories)
    seen_inodes=set()
    for path in response_paths:
        stat=Path(path).stat(); file_identity=(stat.st_dev,stat.st_ino)
        if file_identity in seen_inodes:
            raise ValueError('P0-v1 response files share an inode')
        seen_inodes.add(file_identity)
    seen_requests=set()
    for trace in traces:
        request_id=trace.get('request_id')
        if request_id is not None:
            if not isinstance(request_id,str) or not request_id or request_id in seen_requests:
                raise ValueError('P0-v1 provider request identity is invalid or repeated')
            seen_requests.add(request_id)
    if (list(shared.glob('*/*.pending.json')) or list(shared.glob('*/*.unknown-*.json'))
            or Path(str(db_path)+'-wal').exists()):
        raise ValueError('P0-v1 evidence changed or a request started during verification')
    if any(path in files and files[path]!=digest for path,digest in pinned.items()):
        raise ValueError('Conflicting P0-v1 source and cache identities')
    for path,digest in {**files,**pinned}.items():
        if not Path(path).is_file() or sha256(Path(path))!=digest:
            raise ValueError('P0-v1 evidence changed during prefix verification')
    return {'manifest':parent,'manifest_path':str(manifest_path),'manifest_file_sha256':expected_parent_file_sha256,
        'match_batch_path':str(match_path),'document':doc,'cases':cases,'batches':batches,'events':events,
        'files':{**files,**pinned},'prefix':selected[:expected_prefix],
        'next_reflection':selected[expected_prefix], 'rejected_history':histories,'usage':_usage(traces),
        'limitations':['P0-v1 cache files were not independently signed by the original v1 manifest; '
                       'v2 binds their presently verified bytes and reconstructed requests.',
                       'Reported usage excludes any unknown or unreported provider-side charges.']}
