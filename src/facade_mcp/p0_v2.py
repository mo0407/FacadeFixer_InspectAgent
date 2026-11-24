"""Versioned P0 candidate with auditable v1-prefix reuse and constrained curation.

This is a separate experiment: it never changes the v1 manifest or phase cache,
never registers/publishes a playbook, and never presents a v1 response as a v2
request. A completed candidate still requires independent rule validation and
reference review. API access is disabled unless explicitly enabled by caller.
"""
from __future__ import annotations

import asyncio
from collections import Counter
import datetime
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Literal

from PIL import Image, ImageDraw
from pydantic import Field, create_model

from .baseline_p0_coverage import (CoverageCuration, CoverageReflection, EvidenceRule,
    _assert_files, _digest, _file_map, _related_baseline_evidence, _usage_summary)
from .model_gateway import GatewayError, ModelGateway
from .playbook import Playbook, Rule, fingerprint
from .storage import ArtifactStore, sha256


CURATOR_INSTRUCTION = (
    'Propose only evidence-cited reusable inspection rules. '
    'A rule may cite only event IDs explicitly listed in eligible_evidence_ids. '
    'Never cite an event marked insufficient_evidence in any rule. '
    'Every insufficient_evidence event must appear in no_rule_event_ids. '
    'Each other event not supporting a rule must appear in no_rule_event_ids. '
    'Account for every supplied event; do not invent evidence or publish rules.'
)
BACKOFF_SECONDS = (30, 90, 180)
RETRYABLE_HTTP = frozenset((408, 429, 500, 502, 503, 504))
HASH = re.compile(r'^[a-f0-9]{64}$')


class StopV2(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


def _literal(values: list[str]):
    if not values or len(values) != len(set(values)) or any(not isinstance(x, str) or not x for x in values):
        raise ValueError('Expected distinct nonempty event IDs')
    return Literal.__getitem__(tuple(values))


def constrained_curation_schema(event_ids: list[str], eligible_ids: list[str]):
    """Expose eligible evidence IDs in the schema as well as in the prompt.

    The current gateway supplies the schema as text; provider-side enforcement
    is not assumed. Pydantic and the semantic coverage check remain mandatory.
    """
    all_ids = _literal(event_ids)
    if not eligible_ids or not set(eligible_ids) <= set(event_ids):
        raise ValueError('Constrained model curation requires eligible evidence')
    eligible = _literal(eligible_ids)
    rule = create_model('P0V2EvidenceRule', __base__=EvidenceRule,
        evidence_ids=(list[eligible], Field(min_length=1, max_length=6)))
    return create_model('P0V2ConstrainedCuration', __base__=CoverageCuration,
        candidates=(list[rule], Field(default_factory=list, max_length=30)),
        no_rule_event_ids=(list[all_ids], Field(default_factory=list, max_length=6)))


def validate_reflection(event_ids: list[str], value):
    result = CoverageReflection.model_validate(value)
    actual = [x.event_id for x in result.assessments]
    if len(actual) != len(set(actual)) or set(actual) != set(event_ids):
        raise ValueError('Reflector must assess every event exactly once')
    return result


def validate_curation(event_ids: list[str], reflection, value):
    """Reject any rule supported by insufficient evidence or incomplete coverage."""
    reflected = validate_reflection(event_ids, reflection)
    result = CoverageCuration.model_validate(value)
    allowed = set(event_ids)
    eligible = {x.event_id for x in reflected.assessments if x.status == 'hypothesis'}
    used = set()
    for rule in result.candidates:
        cited = set(rule.evidence_ids)
        if not cited <= allowed:
            raise ValueError('Curator cited an event outside the batch')
        if not cited <= eligible:
            raise ValueError('Curator cited insufficient evidence')
        used.update(cited)
    omitted = result.no_rule_event_ids
    if (len(omitted) != len(set(omitted)) or not set(omitted) <= allowed
            or set(omitted) & used or used | set(omitted) != allowed):
        raise ValueError('Curator must account for every event exactly once')
    return result


def curator_request(group: dict, observations: list[dict], reflection):
    event_ids = list(group['event_ids'])
    reflected = validate_reflection(event_ids, reflection)
    by_id = {x.event_id: x for x in reflected.assessments}
    eligible = [event_id for event_id in event_ids if by_id[event_id].status == 'hypothesis']
    insufficient = [event_id for event_id in event_ids if by_id[event_id].status == 'insufficient_evidence']
    task = {'stage':'baseline_rule_curation_v2', 'batch_id':group['batch_id'],
            'events':observations, 'reflection':reflected.model_dump(mode='json'),
            'eligible_evidence_ids':eligible, 'insufficient_evidence_ids':insufficient,
            'instruction':CURATOR_INSTRUCTION}
    return task, eligible, insufficient


def _fsync_dir(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive_json(path: Path, value: dict):
    """Durably create, never overwrite, an auditable request or result."""
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        _fsync_dir(path.parent)
    except BaseException:
        # A partial durable marker is treated as unknown on the next run.
        raise


def _replace_json(path: Path, value: dict):
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()
    fd, name = tempfile.mkstemp(prefix='.p0v2-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path); _fsync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def _phase_fingerprint(role: str, task: dict, schema, images: list[Path], revision: str):
    schema_sha = _digest(schema.model_json_schema())
    task_sha = _digest(task)
    image_sha = [sha256(Path(path)) for path in images]
    return _digest({'role':role,'task_sha256':task_sha,'schema_sha256':schema_sha,
                    'image_sha256':image_sha,'revision_sha256':revision}), task_sha, schema_sha, image_sha


def _valid_trace(trace: dict, role: str, expected_model: str | None, *, simulation: bool):
    if not isinstance(trace, dict) or trace.get('role') != role or trace.get('status') != 'completed':
        return False
    if simulation:
        return True
    return (trace.get('model_identity_verified') is True and bool(expected_model)
            and type(trace.get('http_status')) is int and 200 <= trace['http_status'] < 300
            and trace.get('adapter') in ('responses','chat_completions')
            and trace.get('requested_model') == expected_model and trace.get('actual_model') == expected_model)


def _retry_kind(trace: dict, expected_model: str | None, role: str):
    if trace.get('role') != role or trace.get('adapter') != 'responses':
        return 'permanent'
    status, code, http = trace.get('status'), trace.get('error_code'), trace.get('http_status')
    if status in ('started', 'request_outcome_unknown') or code in ('api_timeout', 'api_transport_error'):
        return 'unknown'
    if (status == 'failed' and type(http) is int and http in RETRYABLE_HTTP
            and code == ('api_rate_limited' if http == 429 else 'api_http_error')
            and bool(expected_model) and trace.get('requested_model') == expected_model):
        return 'http'
    if (status == 'failed' and http == 200 and code == 'model_output_invalid'
            and trace.get('model_identity_verified') is True and expected_model
            and trace.get('actual_model') == expected_model and trace.get('requested_model') == expected_model):
        return 'schema'
    return 'permanent'


async def run_phase(folder: Path, role: str, task: dict, schema, images: list[Path],
                    revision: str, gateway, validator, expected_model: str | None,
                    budget: dict, *, simulation=False, sleep=asyncio.sleep, mock_output=None):
    """A single versioned phase with bounded auditable retries and safe resume.

    The caller must hold the revision's global lock. A pending marker from any
    previous process is never retried automatically, even if it may not have
    reached the provider. Completed results are semantically revalidated.
    """
    folder.mkdir(parents=True, exist_ok=True)
    done, pending, sidecar = folder/(role+'.json'), folder/(role+'.pending.json'), folder/(role+'.sha256')
    request_sha, task_sha, schema_sha, image_sha = _phase_fingerprint(role,task,schema,images,revision)
    if pending.exists():
        raise StopV2('halted_unknown_outcome','A previous request has an unknown outcome; inspect before retry')
    rejected = sorted(folder.glob(role+'.rejected-*.json'))
    if [p.name for p in rejected] != [role+f'.rejected-{i:04d}.json' for i in range(1,len(rejected)+1)]:
        raise StopV2('halted_integrity_error','Rejected P0 v2 attempt history is incomplete')
    for expected_attempt,path in enumerate(rejected,1):
        old = json.loads(path.read_text(encoding='utf-8-sig'))
        if not simulation:
            raw = folder / (role + '.raw-' + str(old.get('attempt', '')).zfill(4) + '.response')
            if not raw.is_file() or sha256(raw) != old.get('raw_response_sha256'):
                raise StopV2('halted_integrity_error','Original rejected response bytes are missing or changed')
        signature = path.with_name(path.name+'.sha256')
        if (not signature.is_file() or json.loads(signature.read_text(encoding='utf-8-sig')) != {'sha256':sha256(path)}
                or old.get('attempt') != expected_attempt or old.get('request_sha256') != request_sha or old.get('status') != 'rejected'
                or old.get('role') != role or old.get('trace_sha256') != _digest(old.get('trace'))
                or (not simulation and old.get('raw_response_sha256') != old.get('trace',{}).get('response_body_sha256'))
                or old.get('response_sha256') != (_digest(old['response']) if old.get('response') is not None else None)
                or old.get('kind') not in ('semantic','schema','http','permanent')):
            raise StopV2('halted_integrity_error','Rejected P0 v2 attempt belongs to another request')
        if old['kind'] in ('schema','http') and _retry_kind(old['trace'],expected_model,role) != old['kind']:
            raise StopV2('halted_integrity_error','Rejected P0 v2 attempt has invalid retry evidence')
        if old['kind'] == 'semantic':
            if not _valid_trace(old['trace'],role,expected_model,simulation=simulation):
                raise StopV2('halted_integrity_error','Semantic rejection model identity changed')
            try:
                validator(old['response'])
            except ValueError:
                pass
            else:
                raise StopV2('halted_integrity_error','Stored semantic rejection is actually valid')
        if old.get('kind') == 'permanent':
            raise StopV2('halted_provider_error','Previous request had a non-retryable error')
    if done.exists():
        if not simulation:
            existing = json.loads(done.read_text(encoding='utf-8-sig'))
            raw = folder / (role + '.raw-' + str(existing.get('attempt', '')).zfill(4) + '.response')
            if not raw.is_file() or sha256(raw) != existing.get('raw_response_sha256'):
                raise StopV2('halted_integrity_error','Original response bytes are missing or changed')
        if not sidecar.is_file() or json.loads(sidecar.read_text(encoding='utf-8-sig')) != {'sha256':sha256(done)}:
            raise StopV2('halted_integrity_error','Saved P0 v2 phase sidecar is missing or changed')
        row = json.loads(done.read_text(encoding='utf-8-sig'))
        if (row.get('attempt') != len(rejected)+1 or row.get('raw_response_sha256') != row.get('trace',{}).get('response_body_sha256')
                or row.get('status') != 'completed' or row.get('execution') != 'model'
                or row.get('request_sha256') != request_sha or row.get('task_sha256') != task_sha
                or row.get('schema_sha256') != schema_sha or row.get('image_sha256') != image_sha
                or row.get('output_sha256') != _digest(row.get('output'))
                or not _valid_trace(row.get('trace'),role,expected_model,simulation=simulation)):
            raise StopV2('halted_integrity_error','Saved P0 v2 phase differs from frozen evidence')
        validator(row['output'])
        return row
    while True:
        n = len(rejected)+1
        if n > len(BACKOFF_SECONDS)+1:
            raise StopV2('halted_retry_limit','This phase exhausted its three retries')
        if budget['calls'] >= budget['limit']:
            raise StopV2('paused_call_budget','Per-process new-call limit reached; cached phases remain reusable')
        if n > 1:
            await sleep(BACKOFF_SECONDS[n-2])
        marker = {'schema':'facade-p0-v2-phase-pending/v1','status':'request_started',
                  'role':role,'request_sha256':request_sha,'attempt':n,
                  'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        _exclusive_json(pending, marker)
        budget['calls'] += 1
        before = len(getattr(gateway, 'trace', []))
        response = None; kind = None; detail = None
        try:
            invoke_options={'mock_output':mock_output,'image_paths':images}
            if not simulation:
                invoke_options['capture_path']=folder/(role+f'.raw-{n:04d}.response')
            answer = await gateway.invoke(role, task, schema, **invoke_options)
            response = answer.model_dump(mode='json')
            traces = list(getattr(gateway,'trace',[]))[before:]
            trace = traces[-1] if len(traces) == 1 else {}
            if not _valid_trace(trace,role,expected_model,simulation=simulation):
                raise StopV2('halted_unknown_outcome','Model response identity or trace is not verifiable')
            try:
                validator(response)
            except ValueError as exc:
                kind, detail = 'semantic', str(exc)
        except GatewayError:
            traces = list(getattr(gateway,'trace',[]))[before:]
            trace = traces[-1] if len(traces) == 1 else {}
            kind = _retry_kind(trace,expected_model,role)
            if kind == 'unknown':
                raise StopV2('halted_unknown_outcome','Request outcome or billing may be unknown; no automatic retry') from None
        raw_sha=None
        if not simulation:
            raw_path=folder/(role+f'.raw-{n:04d}.response')
            if not raw_path.is_file() or sha256(raw_path)!=trace.get('response_body_sha256'):
                raise StopV2('halted_unknown_outcome','Original response capture is missing or incomplete')
            raw_sha=sha256(raw_path)
        if kind is None:
            record = {'schema':'facade-p0-v2-phase/v1','status':'completed','execution':'model',
                'role':role,'request_sha256':request_sha,'task_sha256':task_sha,'schema_sha256':schema_sha,
                'attempt':n,'raw_response_sha256':raw_sha,'image_sha256':image_sha,'output':response,'output_sha256':_digest(response),'trace':trace,
                'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            _exclusive_json(done,record)
            _exclusive_json(sidecar,{'sha256':sha256(done)})
            pending.unlink(); _fsync_dir(folder)
            return record
        record = {'schema':'facade-p0-v2-rejected/v1','status':'rejected','kind':kind,'role':role,
            'request_sha256':request_sha,'attempt':n,'raw_response_sha256':raw_sha,'response':response,
            'response_sha256':_digest(response) if response is not None else None,
            'validation_error':detail,'trace':trace,'trace_sha256':_digest(trace),
            'usage_may_be_missing':not isinstance(trace.get('usage'),dict),
            'note':'Raw provider response is retained separately.' if not simulation else 'simulation'}
        target = folder/(role+f'.rejected-{n:04d}.json')
        _exclusive_json(target,record)
        _exclusive_json(target.with_name(target.name+'.sha256'),{'sha256':sha256(target)})
        pending.unlink(); _fsync_dir(folder)
        rejected.append(target)
        if kind == 'permanent':
            raise StopV2('halted_provider_error','Provider or model identity error is not retryable')


def _immutable_bytes(path: Path, data: bytes):
    """Create crop evidence once; a changed existing crop is never overwritten."""
    expected=hashlib.sha256(data).hexdigest()
    if path.exists():
        if not path.is_file() or sha256(path)!=expected:
            raise StopV2('halted_integrity_error','An existing P0 v2 crop changed')
        return expected
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    _fsync_dir(path.parent)
    return expected


def group_evidence(store: ArtifactStore, group: dict, events: list[dict], revision_dir: Path):
    """Produce immutable original-pixel crops and P0 observation context."""
    report_path=store.resolve_input(group['match_path'])
    if sha256(report_path)!=group.get('match_sha256'):
        raise ValueError('Matching report changed')
    report=json.loads(report_path.read_text(encoding='utf-8-sig'))
    source=store.resolve_input(report['image']['path'])
    if sha256(source)!=report['image']['sha256']:
        raise ValueError('Reference image changed')
    baseline=store.resolve_input(report['baseline']['path'])
    if sha256(baseline)!=report['baseline']['sha256']:
        raise ValueError('Figure-1 baseline changed')
    baseline_data=json.loads(baseline.read_text(encoding='utf-8-sig'))
    name=baseline_data.get('overview_file')
    if not isinstance(name,str) or Path(name).name!=name:
        raise ValueError('Invalid Figure-1 overview identity')
    overview=store.resolve_input(str(baseline.parent/name))
    if not overview.is_relative_to(baseline.parent.resolve()) or sha256(overview)!=baseline_data.get('artifacts_sha256',{}).get(name):
        raise ValueError('Figure-1 overview changed or escaped its folder')
    batch_id=group['batch_id']
    if not re.fullmatch(r'\d{4}-\d{4}',batch_id):
        raise ValueError('Invalid P0 group identity')
    folder=revision_dir/'crops'/batch_id;folder.mkdir(parents=True,exist_ok=True)
    images=[overview];observations=[]
    with Image.open(source) as opened:
        opened.load();original=opened.convert('RGB');width,height=original.size
    for index,event in enumerate(events,1):
        if not re.fullmatch(r'[a-f0-9]{32}',str(event.get('event_id',''))):
            raise ValueError('Invalid P0 event identity')
        ctx=event.get('context',{});box=ctx.get('bbox_xyxy')
        if (not isinstance(box,list) or len(box)!=4
                or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box)
                or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height)):
            raise ValueError('Invalid P0 event coordinates')
        px,py=max(16.,(box[2]-box[0])*.25),max(16.,(box[3]-box[1])*.25)
        bounds=[max(0,math.floor(box[0]-px)),max(0,math.floor(box[1]-py)),
                min(width,math.ceil(box[2]+px)),min(height,math.ceil(box[3]+py))]
        crop=original.crop(tuple(bounds));draw=ImageDraw.Draw(crop)
        draw.rectangle([box[0]-bounds[0],box[1]-bounds[1],box[2]-bounds[0],box[3]-bounds[1]],
                       outline=(255,0,0),width=max(1,min(crop.size)//200))
        crop.thumbnail((1280,1280));buffer=io.BytesIO();crop.save(buffer,format='JPEG',quality=90)
        target=folder/(f'{index:02d}-{event["event_id"]}.jpg')
        _immutable_bytes(target,buffer.getvalue());images.append(target)
        observations.append({'event_id':event['event_id'],'reason_code':event['reason_code'],
            'class_code':ctx.get('class_code'),'bbox_xyxy':box,'candidate_id':ctx.get('candidate_id'),
            'reference_id':ctx.get('reference_id'),'match_diagnostic':ctx.get('match_diagnostic'),
            'competing_reference_id':ctx.get('competing_reference_id'),
            'crop_image_index':index,'crop_bbox_original':bounds,
            'related_baseline':_related_baseline_evidence(event,baseline_data),
            'note':'A matching disagreement is conditional on the provided reference, not a proven cause.'})
    if sha256(source)!=report['image']['sha256']:
        raise ValueError('Reference image changed while preparing evidence')
    return images,observations


def _deterministic_curation(folder: Path, event_ids: list[str], reflection, revision: str):
    """When no event supports a rule, record that fact without a paid call."""
    reflected=validate_reflection(event_ids,reflection)
    if any(x.status=='hypothesis' for x in reflected.assessments):
        raise ValueError('Deterministic no-rule decision requires zero eligible events')
    folder.mkdir(parents=True,exist_ok=True)
    result={'candidates':[],'no_rule_event_ids':list(event_ids)}
    validate_curation(event_ids,reflection,result)
    record={'schema':'facade-p0-v2-phase/v1','status':'completed','execution':'deterministic_no_hypothesis',
            'role':'curator','revision_sha256':revision,'reflection_sha256':_digest(reflection),
            'output':result,'output_sha256':_digest(result),'trace':None}
    path=folder/'curator.json';sidecar=folder/'curator.sha256'
    if path.exists():
        if (not sidecar.is_file() or json.loads(sidecar.read_text())!={'sha256':sha256(path)}
                or json.loads(path.read_text())!=record):
            raise StopV2('halted_integrity_error','Deterministic P0 v2 result changed')
    else:
        if (folder/'curator.pending.json').exists():
            raise StopV2('halted_unknown_outcome','An earlier curator request has an unknown outcome')
        _exclusive_json(path,record);_exclusive_json(sidecar,{'sha256':sha256(path)})
    return record


def _usage_with_missing(traces):
    result=_usage_summary(traces)
    def missing(row):
        usage=row.get('usage') if isinstance(row,dict) else None
        if not isinstance(usage,dict): return True
        inp=usage.get('input_tokens',usage.get('prompt_tokens'))
        out=usage.get('output_tokens',usage.get('completion_tokens'))
        return not (type(inp) is int and inp>=0 and type(out) is int and out>=0)
    result['calls_missing_input_or_output']=sum(missing(row) for row in traces)
    return result


def _new_usage(revision_dir: Path):
    traces=[];completed=rejected=deterministic=pending=0
    phases=revision_dir/'phases'
    if phases.exists():
        for folder in phases.iterdir():
            if not folder.is_dir(): continue
            for role in ('reflector','curator'):
                path=folder/(role+'.json')
                if path.is_file():
                    row=json.loads(path.read_text(encoding='utf-8-sig'))
                    if row.get('execution')=='model':
                        completed+=1;traces.append(row.get('trace'))
                    elif row.get('execution')=='deterministic_no_hypothesis': deterministic+=1
                for path in folder.glob(role+'.rejected-*.json'):
                    row=json.loads(path.read_text(encoding='utf-8-sig'))
                    rejected+=1;traces.append(row.get('trace'))
                pending+=int((folder/(role+'.pending.json')).exists())
    return {'new_completed_calls':completed,'new_rejected_calls':rejected,
            'deterministic_no_hypothesis_count':deterministic,'pending_count':pending,
            'reported':_usage_with_missing(traces)}


def prepare_revision(store, settings, parent_path, parent_sha, *, expected_count=203, allow_incomplete=False, expected_revision_id=None):
    """Bind all currently verified evidence; this function never contacts a model."""
    from .p0_v2_reuse import verify_parent_prefix
    if expected_count != 203 or not allow_incomplete:
        raise ValueError('This revision requires 203 source cases and explicit partial-reference acceptance')
    verified=verify_parent_prefix(store,str(parent_path),settings,expected_parent_file_sha256=parent_sha,
                                  expected_prefix=112,expected_count=expected_count)
    parent=verified['manifest']; cases=verified['cases']; groups=verified['batches']
    excluded=[case for case in cases if case.get('excluded_reason')]
    if bool(excluded)!=(verified['document']['status']=='needs_reference_review'):
        raise ValueError('Reference-review status differs from verified cases')
    package=Path(__file__).resolve().parent
    code={str(path):sha256(path) for path in sorted(package.glob('*.py'))}
    reused_traces=[prov['trace'] for item in [*verified['prefix'],verified['next_reflection']]
                   for prov in item['provenance'].values()]
    manifest={'schema':'facade-p0-revision/v2',
        'parent_v1_manifest_path':verified['manifest_path'],
        'parent_v1_manifest_file_sha256':verified['manifest_file_sha256'],
        'parent_v1_manifest_sha256':parent['manifest_sha256'],
        'baseline_experiment_sha256':parent['experiment_sha256'],
        'source_files_sha256':verified['files'],'package_sources_sha256':code,
        'runtime':{'python':platform.python_version(),'pillow':__import__('PIL').__version__},
        'model_config_sha256':_digest(settings.model_dump(mode='json')),
        'endpoint_hashes':parent['endpoint_hashes'],
        'case_count':len(cases),'cases':cases,'batches':groups,
        'reference_assumption':parent.get('reference_assumption'),
        'excluded_cases':excluded,'reuse_v1_phase_count':len(reused_traces),
        'v1_prefix':[{'batch_id':x['batch_id'],'provenance':x['provenance']} for x in verified['prefix']],
        'v1_next_reflection':{'batch_id':verified['next_reflection']['batch_id'],
                               'provenance':verified['next_reflection']['provenance']},
        'v1_reused_usage':_usage_with_missing(reused_traces),
        'v1_historical_usage':{**verified['usage'], 'calls_missing_input_or_output':
            _usage_with_missing(reused_traces+[x.get('trace') for x in verified['rejected_history']])['calls_missing_input_or_output']},'v1_rejected_history':verified['rejected_history'],
        'curator_instruction':CURATOR_INSTRUCTION,'curator_instruction_sha256':_digest(CURATOR_INSTRUCTION),
        'protocol':'mixed_v1_prefix_v2_curator','candidate_only':True,'limitations':verified['limitations']}
    identity=_digest(manifest)
    if expected_revision_id is not None and expected_revision_id != identity:
        raise ValueError('Expected revision does not match the frozen input')
    folder=store.root/'baseline_p0_v2'/identity
    folder.mkdir(parents=True,exist_ok=True); path=folder/'manifest.json'
    frozen={**manifest,'revision_sha256':identity}
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8-sig'))!=frozen:
            raise ValueError('Existing P0-v2 manifest does not match its identity')
    else:
        _exclusive_json(path,frozen)
    return frozen,folder,verified


def _progress(folder, manifest, status, batches, event_count, calls, reason=None):
    usage=_new_usage(folder)
    progress_path=folder/'progress.json'
    previous=json.loads(progress_path.read_text()) if progress_path.exists() else None
    if previous is not None:
        if (previous.get('revision_id')!=manifest['revision_sha256']
                or previous.get('revision_sha256')!=manifest['revision_sha256']):
            raise StopV2('halted_integrity_error','P0-v2 progress identity changed')
        for key in ('new_completed_calls','new_rejected_calls'):
            if type(previous.get(key)) is not int or usage[key] < previous[key]:
                raise StopV2('halted_integrity_error','P0-v2 recorded call history regressed')
        if type(previous.get('completed_batches')) is not int or type(previous.get('accounted_event_count')) is not int:
            raise StopV2('halted_integrity_error','P0-v2 progress high-water mark is invalid')
        batches=max(batches,previous['completed_batches'])
        event_count=max(event_count,previous['accounted_event_count'])
    state={'schema':'facade-p0-v2-progress/v1','revision_id':manifest['revision_sha256'],
           'revision_sha256':manifest['revision_sha256'],'status':status,'reason':reason,
           'completed_batches':batches,'total_batches':len(manifest['batches']),
           'accounted_event_count':event_count,
           'total_event_count':sum(len(g['event_ids']) for g in manifest['batches']),
           'reused_v1_phase_count':manifest['reuse_v1_phase_count'],
           'new_calls_this_run':calls,**usage,'usage':usage['reported'],
           'excluded_case_count':len(manifest['excluded_cases']),'excluded_cases':manifest['excluded_cases'],
           'revision_dir':str(folder),'progress_path':str(folder/'progress.json')}
    _replace_json(folder/'progress.json',state)
    return state


def _audit_phase_layout(folder, groups, prefix_count, prior_progress):
    """Check every existing phase before any new dispatch, including incomplete tails."""
    phases=folder/'phases'
    if phases.exists() and list(phases.rglob('*.pending.json')):
        raise StopV2('halted_unknown_outcome','A prior P0-v2 phase has an unknown outcome')
    expected={f'{i+1:04d}-{group["batch_id"]}':(i,group) for i,group in enumerate(groups) if i>=prefix_count}
    if phases.exists():
        for entry in phases.iterdir():
            if entry.is_symlink() or not entry.is_dir() or entry.name not in expected:
                raise StopV2('halted_integrity_error','Unexpected P0-v2 phase directory')
    open_tail=False; completed_groups=prefix_count; completed_calls=rejected_calls=0
    for name,(index,group) in expected.items():
        location=phases/name
        if location.exists() and not location.is_dir():
            raise StopV2('halted_integrity_error','Invalid P0-v2 phase directory')
        filenames={p.name for p in location.iterdir()} if location.exists() else set()
        if any((location/n).is_symlink() or not (location/n).is_file() for n in filenames):
            raise StopV2('halted_integrity_error','Invalid P0-v2 phase evidence')
        consumed=set(); all_done=True
        for role in (('curator',) if index==prefix_count else ('reflector','curator')):
            done=role+'.json'; side=role+'.sha256'; pending=role+'.pending.json'
            if pending in filenames:
                raise StopV2('halted_unknown_outcome','A prior P0-v2 phase has an unknown outcome')
            rejects=sorted(n for n in filenames if re.fullmatch(re.escape(role)+r'\.rejected-\d{4}\.json',n))
            raw=sorted(n for n in filenames if re.fullmatch(re.escape(role)+r'\.raw-\d{4}\.response',n))
            present=done in filenames or side in filenames or bool(rejects) or bool(raw)
            if open_tail and present:
                raise StopV2('halted_integrity_error','P0-v2 phase history has a missing earlier stage')
            if rejects != [role+f'.rejected-{n:04d}.json' for n in range(1,len(rejects)+1)]:
                raise StopV2('halted_integrity_error','P0-v2 rejection history has a gap')
            for attempt,name_reject in enumerate(rejects,1):
                path=location/name_reject; signature=name_reject+'.sha256'; consumed.add(name_reject)
                if signature not in filenames or json.loads((location/signature).read_text())!={'sha256':sha256(path)}:
                    raise StopV2('halted_integrity_error','P0-v2 rejection sidecar is missing or changed')
                consumed.add(signature);row=json.loads(path.read_text())
                raw_name=role+f'.raw-{attempt:04d}.response'
                if (row.get('attempt')!=attempt or row.get('role')!=role or row.get('status')!='rejected'
                        or raw_name not in filenames or sha256(location/raw_name)!=row.get('raw_response_sha256')
                        or row.get('raw_response_sha256')!=row.get('trace',{}).get('response_body_sha256')):
                    raise StopV2('halted_integrity_error','P0-v2 rejected response evidence changed')
                consumed.add(raw_name);rejected_calls+=1
            if done in filenames:
                if side not in filenames or json.loads((location/side).read_text())!={'sha256':sha256(location/done)}:
                    raise StopV2('halted_integrity_error','P0-v2 completed sidecar is missing or changed')
                consumed.update((done,side)); row=json.loads((location/done).read_text())
                if row.get('status')!='completed' or row.get('role')!=role:
                    raise StopV2('halted_integrity_error','P0-v2 completed phase identity changed')
                if row.get('execution')=='model':
                    attempt=len(rejects)+1; raw_name=role+f'.raw-{attempt:04d}.response'
                    if (row.get('attempt')!=attempt or raw_name not in filenames
                            or sha256(location/raw_name)!=row.get('raw_response_sha256')
                            or row.get('raw_response_sha256')!=row.get('trace',{}).get('response_body_sha256')):
                        raise StopV2('halted_integrity_error','P0-v2 completed response evidence changed')
                    consumed.add(raw_name);completed_calls+=1
                elif row.get('execution')!='deterministic_no_hypothesis' or role!='curator' or rejects:
                    raise StopV2('halted_integrity_error','Invalid deterministic P0-v2 phase')
            else:
                all_done=False;open_tail=True
        if filenames!=consumed:
            raise StopV2('halted_integrity_error','Unexpected or incomplete P0-v2 phase evidence')
        if all_done:
            completed_groups+=1
    if prior_progress is not None:
        prior_count=prior_progress.get('completed_batches')
        prior_events=prior_progress.get('accounted_event_count')
        if (type(prior_count) is not int or not 0<=prior_count<=completed_groups
                or type(prior_events) is not int
                or prior_events!=sum(len(g['event_ids']) for g in groups[:prior_count])
                or type(prior_progress.get('new_completed_calls')) is not int
                or type(prior_progress.get('new_rejected_calls')) is not int
                or prior_progress['new_completed_calls']>completed_calls
                or prior_progress['new_rejected_calls']>rejected_calls):
            raise StopV2('halted_integrity_error','P0-v2 history regressed since the recorded progress')



async def run_revision(store, settings, manifest, folder, verified, *, max_new_calls=1024, gateway=None):
    if type(max_new_calls) is not int or not 1<=max_new_calls<=1024:
        raise ValueError('max-new-calls must be between 1 and 1024')
    if gateway is None:
        from .p0_v2_gateway import RawCaptureGateway
        if not settings.api_enabled:
            raise ValueError('Configured API must be enabled')
        gateway=RawCaptureGateway(settings,allow_api=True)
    identity=manifest['revision_sha256']; groups=verified['batches']; events=verified['events']
    rules={}; provenance={}; accounted=[]; completed=0; budget={'calls':0,'limit':max_new_calls}
    previous_path=folder/'progress.json'
    prior=json.loads(previous_path.read_text()) if previous_path.exists() else None
    audited=False
    try:
        if prior is not None and (prior.get('revision_id')!=identity or prior.get('revision_sha256')!=identity):
            raise StopV2('halted_integrity_error','Recorded P0-v2 progress identity changed')
        _audit_phase_layout(folder,groups,len(verified['prefix']),prior)
        audited=True
        _progress(folder,manifest,'running',completed,len(accounted),0)
        for index,group in enumerate(groups):
            group_events=[events[key] for key in group['event_ids']]
            for event in group_events:
                if _digest(event)!=group['event_sha256'][event['event_id']]:
                    raise StopV2('halted_integrity_error','A frozen event changed')
            if index < len(verified['prefix']):
                original=verified['prefix'][index]
                if original['batch_id']!=group['batch_id']: raise ValueError('Reused group mismatch')
                reflection=original['reflection']; curated=validate_curation(group['event_ids'],reflection,original['curation'])
                phase_provenance=original['provenance']['curator']
                protocol='v1_original'
            else:
                phase_folder=folder/'phases'/f'{index+1:04d}-{group["batch_id"]}'
                if index==len(verified['prefix']):
                    original=verified['next_reflection']; observations=original['observations']; reflection=original['reflection']
                    if original['batch_id']!=group['batch_id']: raise ValueError('Reused reflection group mismatch')
                else:
                    images,observations=group_evidence(store,group,group_events,folder)
                    task={'stage':'baseline_failure_reflection','batch_id':group['batch_id'],'events':observations,
                        'image_order':'image 0 is the Figure-1 overview; each numbered crop is an original-pixel region with a red bbox',
                        'instruction':'Assess every event once. Treat explanations as hypotheses; use insufficient_evidence if unsure. Suggest only a controlled inspection change; do not invent observations or measurements.'}
                    con=settings.connections[settings.roles['reflector']]
                    row=await run_phase(phase_folder,'reflector',task,CoverageReflection,images,identity,gateway,
                        lambda x:validate_reflection(group['event_ids'],x),con.expected_model,budget)
                    reflection=row['output']
                task,eligible,_=curator_request(group,observations,reflection)
                if eligible:
                    con=settings.connections[settings.roles['curator']]
                    schema=constrained_curation_schema(group['event_ids'],eligible)
                    row=await run_phase(phase_folder,'curator',task,schema,[],identity,gateway,
                        lambda x:validate_curation(group['event_ids'],reflection,x),con.expected_model,budget)
                else:
                    row=_deterministic_curation(phase_folder,group['event_ids'],reflection,identity)
                curated=validate_curation(group['event_ids'],reflection,row['output'])
                phase_provenance={'cache_path':str(phase_folder/'curator.json'),
                                  'cache_sha256':sha256(phase_folder/'curator.json'),
                                  'request_sha256':row.get('request_sha256'),'execution':row['execution']}
                protocol='v2_constrained'
            for item in curated.candidates:
                key=fingerprint(item.condition,item.recommendation)
                if key not in rules:
                    rules[key]=Rule(id='rule-'+key[:24],condition=item.condition,recommendation=item.recommendation,
                                    limitations=item.limitations,evidence_ids=item.evidence_ids)
                else:
                    old=rules[key];old.evidence_ids=list(dict.fromkeys(old.evidence_ids+item.evidence_ids))
                    if item.limitations not in old.limitations: old.limitations+='\n'+item.limitations
                provenance.setdefault(key,[]).append({'batch_id':group['batch_id'],'phase':'curator',
                    'protocol':protocol,'evidence_ids':item.evidence_ids,'source':phase_provenance})
            accounted.extend(group['event_ids']);completed+=1
            _progress(folder,manifest,'running',completed,len(accounted),budget['calls'])
        _assert_files(manifest['source_files_sha256']);_assert_files(manifest['package_sources_sha256'])
        all_ids=[event for case in verified['cases'] for event in case['event_ids']]
        if len(accounted)!=len(set(accounted)) or set(accounted)!=set(all_ids):
            raise StopV2('halted_integrity_error','Complete event coverage could not be verified')
        book=Playbook(version='P0-v2-candidate-'+identity[:24],rules=list(rules.values()))
        book_path=folder/'candidate-playbook.json';_replace_json(book_path,book.model_dump(mode='json'))
        status='partial_reference_review' if manifest['excluded_cases'] else 'candidate_only'
        metadata={'schema':'facade-p0-v2-candidate/v1','revision_sha256':identity,'status':status,
            'coverage_status':'partial_reference_review' if manifest['excluded_cases'] else 'conditional_reference_assumption',
            'promoted':False,'registered_with_rule_lifecycle':False,'completed_batches':completed,'total_batches':len(groups),
            'accounted_event_count':len(accounted),'total_event_count':len(all_ids),
            'excluded_case_count':len(manifest['excluded_cases']),'excluded_cases':manifest['excluded_cases'],
            'manifest_file_sha256':sha256(folder/'manifest.json'),'candidate_playbook_path':str(book_path),
            'candidate_playbook_file_sha256':sha256(book_path),'rule_provenance':provenance,
            'reference_assumption':manifest['reference_assumption'],'limitations':manifest['limitations']}
        _replace_json(folder/'candidate-metadata.json',metadata)
        return _progress(folder,manifest,status,completed,len(accounted),budget['calls'])
    except StopV2 as exc:
        if exc.status=='halted_integrity_error' and previous_path.exists():
            preserved=json.loads(previous_path.read_text())
            return {**preserved,'status':exc.status,'reason':exc.reason,'new_calls_this_run':budget['calls']}
        if not audited and prior is not None:
            return {**prior,'status':exc.status,'reason':exc.reason,'new_calls_this_run':0}
        try:
            return _progress(folder,manifest,exc.status,completed,len(accounted),budget['calls'],exc.reason)
        except StopV2 as integrity:
            preserved=json.loads(previous_path.read_text())
            return {**preserved,'status':integrity.status,'reason':integrity.reason,'new_calls_this_run':budget['calls']}
