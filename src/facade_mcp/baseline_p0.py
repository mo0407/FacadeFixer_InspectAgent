"""Match Figure-1 results to explicit polygon references for candidate P0.

This module does not infer on images, train models, invent explanations, or
publish a playbook. Only declared annotated classes are scored. Each observed
disagreement cites the frozen baseline result, input image, reference labels,
and match report; all conclusions remain conditional on the reference labels.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from .failure_memory import FailureMemory
from .playbook import Playbook
from .rule_lifecycle import CLASSES, ReferenceDefect, RuleLifecycle, _iou, _match
from .storage import ArtifactStore, sha256


def _immutable_copy(source: Path, target: Path, expected: str):
    """Keep the exact reference used in this match, without overwriting it."""
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha256(target)!=expected:
            raise ValueError('Existing evidence snapshot differs from its recorded identity')
        return target
    fd,name=tempfile.mkstemp(prefix='.snapshot-',dir=target.parent)
    try:
        with os.fdopen(fd,'wb') as out, source.open('rb') as inp:
            shutil.copyfileobj(inp,out)
        if sha256(Path(name))!=expected or sha256(source)!=expected:
            raise ValueError('Source changed while creating an evidence snapshot')
        try:
            os.link(name,target)
        except FileExistsError:
            if sha256(target)!=expected:
                raise ValueError('Concurrent evidence snapshot differs') from None
    finally:
        Path(name).unlink(missing_ok=True)
    return target


def _diagnostic_evidence(store: ArtifactStore, baseline_file: Path, result: dict):
    """Follow promoted diagnostic results to their original, immutable inputs.

    The current recorded format is intentionally explicit; unknown formats or
    ambiguous/missing source files fail closed instead of silently dropping
    provenance from P0 evidence.
    """
    records=[]
    for name in sorted(result.get('artifacts_sha256',{})):
        match=re.fullmatch(r'vlm-call-(\d{4})\.json',name)
        if not match:
            continue
        call_path=store.resolve_input(str(baseline_file.parent/name))
        call=json.loads(call_path.read_text(encoding='utf-8-sig'))
        origin=call.get('diagnostic_origin')
        if origin is None:
            continue
        if not isinstance(origin,dict) or any(not re.fullmatch(r'[a-f0-9]{64}',str(origin.get(key,'')))
                for key in ('attempt_sha256','outcome_sha256','diagnostic_sha256')):
            raise ValueError('Invalid promoted diagnostic provenance')
        if (call.get('status')!='completed' or not re.fullmatch(r'[a-f0-9]{64}',str(call.get('request_sha256','')))
                or not isinstance(call.get('response'),dict) or not isinstance(call.get('model_trace'),list)
                or not call['model_trace'] or any(not isinstance(origin.get(key),str) or not origin[key]
                                                  for key in ('started_utc','completed_utc'))):
            raise ValueError('Incomplete promoted diagnostic provenance')
        pairs=[]
        for folder in store.root.glob('figure1-diagnostic-*'):
            attempt=folder/'attempt.json'; outcome=folder/'outcome.json'
            if (attempt.is_file() and outcome.is_file() and sha256(attempt)==origin['attempt_sha256']
                    and sha256(outcome)==origin['outcome_sha256']):
                resolved=(store.resolve_input(str(attempt)),store.resolve_input(str(outcome)))
                if not all(path.is_relative_to(store.root.resolve()) for path in resolved):
                    raise ValueError('Promoted diagnostic provenance escaped the workspace')
                pairs.append(resolved)
        scripts=[store.resolve_input(str(path)) for path in store.root.glob('figure1_diagnose*.py')
                 if path.is_file() and sha256(path)==origin['diagnostic_sha256']]
        if len(pairs)!=1 or len(scripts)!=1:
            raise ValueError('Promoted diagnostic provenance is missing or ambiguous')
        if not scripts[0].is_relative_to(store.root.resolve()):
            raise ValueError('Promoted diagnostic script escaped the workspace')
        attempt_path,outcome_path=pairs[0]
        attempt=json.loads(attempt_path.read_text(encoding='utf-8-sig'))
        outcome=json.loads(outcome_path.read_text(encoding='utf-8-sig'))
        if (attempt.get('status')!='request_may_be_sent' or outcome.get('status')!='protocol_valid'
                or attempt.get('experiment_sha256')!=result.get('experiment_sha256')
                or attempt.get('request_sha256')!=call.get('request_sha256')
                or outcome.get('request_sha256')!=call.get('request_sha256')
                or attempt.get('diagnostic_sha256')!=origin['diagnostic_sha256']
                or attempt.get('started_utc')!=origin.get('started_utc')
                or outcome.get('completed_utc')!=origin.get('completed_utc')
                or outcome.get('response')!=call.get('response')
                or outcome.get('model_trace')!=call.get('model_trace')):
            raise ValueError('Promoted diagnostic provenance does not match its call')
        source=attempt.get('source_hashes')
        if not isinstance(source,dict) or set(source)!={'manifest','stage','image','model_config',
                'prior_call_1','prior_call_2','failure_1','failure_2'} or match.group(1)!='0003':
            raise ValueError('Unsupported promoted diagnostic source manifest')
        inputs={'manifest':baseline_file.parent.parent/'baseline-manifest.json',
                'stage':baseline_file.parent/'baseline-expert-result.json',
                'image':baseline_file.parent/'vlm-call-0003-grid.jpg',
                'model_config':store.root/'figure1-apinebula.local.json',
                'prior_call_1':baseline_file.parent/'vlm-call-0001.json',
                'prior_call_2':baseline_file.parent/'vlm-call-0002.json',
                'failure_1':baseline_file.parent/'vlm-call-0003.failed-1.json',
                'failure_2':baseline_file.parent/'vlm-call-0003.failed-2.json'}
        files=[(attempt_path,origin['attempt_sha256']),(outcome_path,origin['outcome_sha256']),
               (scripts[0],origin['diagnostic_sha256'])]
        for key,path in sorted(inputs.items()):
            resolved=store.resolve_input(str(path))
            if (not resolved.is_relative_to(path.parent.resolve())
                    or not re.fullmatch(r'[a-f0-9]{64}',str(source[key])) or sha256(resolved)!=source[key]):
                raise ValueError('Promoted diagnostic source changed')
            files.append((resolved,source[key]))
        records.append({'call_artifact':name,'request_sha256':call['request_sha256'],
                        'files':[{'path':str(path),'sha256':digest} for path,digest in files]})
    return records


def _read_polygons(path: Path, width: int, height: int, class_map: dict[str, str]):
    """Read YOLO polygons without silently dropping invalid source annotations."""
    rows, errors = [], []
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 7 or len(fields) % 2 != 1 or fields[0] not in class_map:
            errors.append({'line': line_no, 'reason': 'invalid_polygon_or_unmapped_class'})
            continue
        try:
            values = [float(v) for v in fields[1:]]
        except ValueError:
            errors.append({'line': line_no, 'reason': 'invalid_polygon_coordinate'})
            continue
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            errors.append({'line': line_no, 'reason': 'polygon_coordinate_out_of_range'})
            continue
        points = list(zip(values[::2], values[1::2]))
        area2 = abs(sum(x1*y2-x2*y1 for (x1,y1), (x2,y2) in zip(points, points[1:]+points[:1])))
        xs, ys = values[::2], values[1::2]
        if area2 <= 1e-12 or min(xs) >= max(xs) or min(ys) >= max(ys):
            errors.append({'line': line_no, 'reason': 'degenerate_polygon'})
            continue
        rows.append({'reference_id': f'gt-{line_no:04d}', 'source_line': line_no,
                     'class_code': class_map[fields[0]], 'source_class_id': fields[0],
                     'bbox_xyxy': [float(min(xs)*width), float(min(ys)*height),
                                   float(max(xs)*width), float(max(ys)*height)]})
    return rows, errors


def _observations(result: dict, width: int, height: int, annotated: set[str]):
    candidates = result.get('candidates')
    if not isinstance(candidates, list) or not isinstance(result.get('trajectory'), list):
        raise ValueError('Baseline result must include candidates and trajectory lists')
    accepted, deferred, excluded = [], [], []
    seen = set()
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError('Candidate must be an object')
        identity = item.get('candidate_id')
        if not isinstance(identity, str) or not identity.strip() or len(identity) > 200 or identity in seen:
            raise ValueError('Candidate IDs must be nonempty and unique')
        seen.add(identity)
        decision, code = item.get('decision'), item.get('class_code')
        if decision not in ('accept', 'reject', 'defer'):
            raise ValueError('Candidate has missing or invalid final decision')
        box = item.get('bbox_xyxy')
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError('Candidate must have an original-pixel bbox')
        if any(isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) for x in box):
            raise ValueError('Candidate bbox must be finite')
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError('Candidate bbox lies outside original image')
        clean = {key: item.get(key) for key in ('candidate_id','class_code','class_name','bbox_xyxy',
                 'confidence','model_profile','kind','decision','reason') if key in item}
        if code not in annotated or decision == 'reject':
            excluded.append(clean)
        elif decision == 'accept':
            accepted.append(clean)
        else:
            deferred.append(clean)
    return accepted, deferred, excluded


def _real_model_evidence(result: dict) -> bool:
    """Fail closed for mock, injected, missing provenance or unverified VLM IDs."""
    if result.get('simulation') is not False:
        return False
    trace = result.get('model_trace')
    experts = result.get('experts')
    artifacts = result.get('artifacts_sha256')
    if not re.fullmatch(r'[a-f0-9]{64}', str(result.get('experiment_sha256', ''))):
        return False
    # Figure 1 requires all three independent experts, not a partial job.
    if not isinstance(trace, list) or not trace or not isinstance(experts, dict) or len(experts) != 3 or not isinstance(artifacts,dict):
        return False
    if any(not isinstance(row, dict) or row.get('status') != 'completed' or row.get('adapter') == 'mock'
           or row.get('model_identity_verified') is not True or not row.get('actual_model') for row in trace):
        return False
    for row in experts.values():
        if not isinstance(row, dict):
            return False
        if row.get('response_file') not in artifacts:
            return False
        identity = row.get('weight_identity')
        values = list(identity.values()) if isinstance(identity, dict) else [identity]
        if not values or any(not isinstance(value,str) or not re.fullmatch(r'[a-f0-9]{64}', value) for value in values):
            return False
    return True


def match_baseline_case(store: ArtifactStore, baseline_path: str, labels_path: str,
                        class_map: dict[str, str], *, iou_threshold: float = .5,
                        reference_complete: bool = False) -> dict:
    """Create an auditable per-image match report and observed FP/FN evidence.

    ``class_map`` maps the reference file's actual YOLO numeric IDs to canonical
    classes; it is never guessed from image names. Unmatched predictions are only
    called FP if completeness of those classes has been explicitly declared.
    An invalid polygon suspends all matching for that image, preventing a
    defective reference from being silently counted as an FP or FN.
    """
    if not isinstance(class_map, dict) or not class_map or any(
            not isinstance(k,str) or not re.fullmatch(r'\d+', k) or v not in CLASSES for k,v in class_map.items()):
        raise ValueError('Provide an explicit reference class-ID to canonical-class map')
    if type(reference_complete) is not bool or isinstance(iou_threshold, bool) or not isinstance(iou_threshold,(float,int)) or not .5 <= iou_threshold <= 1:
        raise ValueError('Invalid IoU threshold or reference completeness declaration')
    baseline_file, label_file = store.resolve_input(baseline_path), store.resolve_input(labels_path)
    baseline_hash, label_hash = sha256(baseline_file), sha256(label_file)
    result = json.loads(baseline_file.read_text(encoding='utf-8-sig'))
    if not isinstance(result, dict) or result.get('schema') != 'facade-baseline-image/v1' or result.get('status') != 'completed':
        raise ValueError('Only a completed Figure-1 baseline result can be matched')
    sidecar=baseline_file.with_name('baseline-result.sha256.json')
    sidecar_hash=None
    if sidecar.exists():
        sidecar=store.resolve_input(str(sidecar))
        if json.loads(sidecar.read_text(encoding='utf-8-sig')) != {'sha256':baseline_hash}:
            raise ValueError('Baseline result sidecar hash differs from the result')
        sidecar_hash=sha256(sidecar)
    elif result.get('simulation') is False:
        raise ValueError('Real baseline result requires its separate SHA-256 sidecar')
    image = result.get('image')
    if not isinstance(image,dict) or not isinstance(image.get('path'),str):
        raise ValueError('Baseline image provenance is missing')
    image_path = store.resolve_input(image['path'])
    image_hash = sha256(image_path)
    if image.get('sha256') != image_hash:
        raise ValueError('Baseline image bytes differ from the stored image hash')
    with Image.open(image_path) as opened:
        width,height = opened.size
        if opened.format not in ('JPEG','PNG') or getattr(opened,'n_frames',1) != 1 or width*height > 80_000_000:
            raise ValueError('Baseline image must be a single JPEG/PNG within the image size budget')
        opened.verify()
    if (image.get('width'),image.get('height')) != (width,height):
        raise ValueError('Baseline image dimensions do not match the source')
    artifacts = result.get('artifacts_sha256', {})
    # Dense images can produce many per-call evidence files. Validate all of
    # them, but bind them transitively through the immutable match report when
    # listing them individually would exceed FailureMemory's 64-ref limit.
    if not isinstance(artifacts,dict) or len(artifacts) > 4096:
        raise ValueError('Invalid baseline artifact manifest')
    artifact_paths = []
    for name, expected in artifacts.items():
        if (not isinstance(name,str) or Path(name).name != name or name in ('.','..')
                or not isinstance(expected,str) or not re.fullmatch(r'[a-f0-9]{64}',expected)):
            raise ValueError('Invalid baseline artifact identity')
        artifact = store.resolve_input(str(baseline_file.parent/name))
        if not artifact.is_relative_to(baseline_file.parent.resolve()) or sha256(artifact) != expected:
            raise ValueError('Baseline artifact changed or escaped its result directory')
        artifact_paths.append(artifact)
    diagnostic_provenance=_diagnostic_evidence(store,baseline_file,result)
    references, issues = _read_polygons(label_file,width,height,class_map)
    accepted,deferred,excluded = _observations(result,width,height,set(class_map.values()))
    matcher_sha=sha256(Path(__file__))
    digest = hashlib.sha256(json.dumps({'image':image_hash,'baseline':baseline_hash,'labels':label_hash,
        'class_map':class_map,'iou_threshold':float(iou_threshold),'reference_complete':reference_complete,
        'matcher_sha256':matcher_sha},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    job_id = 'baseline_match_'+digest[:32]
    target = store.root/'baseline_matches'/digest/'match.json'
    label_snapshot=_immutable_copy(label_file,target.parent/('reference-'+label_file.name),label_hash)
    candidate_snapshot=next((p for p in artifact_paths if p.name=='source'+image_path.suffix.lower()
                             and artifacts[p.name]==image_hash),None)
    image_snapshot=candidate_snapshot or _immutable_copy(image_path,target.parent/('reference-image'+image_path.suffix.lower()),image_hash)
    direct_files=list(dict.fromkeys([image_snapshot,label_snapshot,baseline_file,target,
                                    *([sidecar] if sidecar_hash else []),*artifact_paths]))
    compact_evidence=len(direct_files)>64
    evidence_files=(list(dict.fromkeys([image_snapshot,label_snapshot,baseline_file,target,
                                       *([sidecar] if sidecar_hash else [])]))
                    if compact_evidence else direct_files)
    if len({p.name for p in evidence_files}) != len(evidence_files):
        raise ValueError('Evidence file names must be distinct for hash binding')
    report = {'schema':'facade-baseline-match/v1','status':'needs_reference_review' if issues else 'completed',
        'job_id':job_id,'image':{'path':str(image_snapshot),'original_path':str(image_path),
                               'sha256':image_hash,'width':width,'height':height},
        'baseline':{'path':str(baseline_file),'sha256':baseline_hash,
                    'sidecar_path':str(sidecar) if sidecar_hash else None,'sidecar_sha256':sidecar_hash},
        'artifacts':{p.name:artifacts[p.name] for p in artifact_paths},
        'diagnostic_provenance':diagnostic_provenance,
        'labels':{'path':str(label_snapshot),'original_path':str(label_file),'sha256':label_hash,'class_map':class_map,'issues':issues},
        'simulation':not _real_model_evidence(result),'reference_complete':reference_complete,'matcher_sha256':matcher_sha,
        'iou_threshold':float(iou_threshold),'reference_count':len(references),
        'references':references,'accepted':accepted,'deferred':deferred,'excluded':excluded,
        'matches':[],'false_positives':[],'unverified_predictions':[],'false_negatives':[],
        'unresolved_references':[],'unresolved_predictions':[],
        'limitations':['Observed disagreements are conditional on annotation quality and the declared class map.',
                       'Non-annotated classes and rejected candidates are excluded from FP scoring.',
                       'Deferred candidates are not automatically false positives or false negatives.']}
    if not issues:
        models = [ReferenceDefect(class_code=r['class_code'],bbox_xyxy=r['bbox_xyxy']) for r in references]
        matched = _match(accepted, models, float(iou_threshold))['matches']
        used_predictions = {p['prediction_index'] for p in matched}
        used_references = {p['reference_index'] for p in matched}
        remaining = [i for i in range(len(references)) if i not in used_references]
        deferred_pairs = _match(deferred,[models[i] for i in remaining],float(iou_threshold))['matches']
        unresolved = {remaining[p['reference_index']] for p in deferred_pairs}
        report['matches'] = [{'candidate_id':accepted[p['prediction_index']]['candidate_id'],
            'reference_id':references[p['reference_index']]['reference_id'],'iou':p['iou']} for p in matched]
        report['unresolved_references'] = [references[i] for i in sorted(unresolved)]
        report['unresolved_predictions'] = list(deferred)
        unmatched = []
        for i, prediction in enumerate(accepted):
            if i in used_predictions:
                continue
            # In standard one-to-one evaluation, an extra prediction of an
            # already matched GT is an FP. Keep its duplicate explanation so
            # rule reflection does not confuse it with an unrelated object.
            duplicates = [(j,_iou(prediction['bbox_xyxy'],references[j]['bbox_xyxy']))
                          for j in used_references if prediction['class_code'] == references[j]['class_code']]
            duplicates = [(j,overlap) for j,overlap in duplicates if overlap >= iou_threshold]
            row = dict(prediction)
            if duplicates:
                index,overlap = max(duplicates,key=lambda pair:pair[1])
                row.update(match_diagnostic='duplicate_of_matched_reference',
                           competing_reference_id=references[index]['reference_id'],competing_iou=overlap)
            else:
                row['match_diagnostic'] = 'no_matched_reference'
            unmatched.append(row)
        report['false_positives' if reference_complete else 'unverified_predictions'] = unmatched
        report['false_negatives'] = [r for i,r in enumerate(references) if i not in used_references and i not in unresolved]
    if (sha256(image_path) != image_hash or sha256(label_file) != label_hash or sha256(baseline_file) != baseline_hash
            or (sidecar_hash and sha256(sidecar)!=sidecar_hash)
            or any(sha256(p) != artifacts[p.name] for p in artifact_paths)
            or any(sha256(Path(item['path']))!=item['sha256']
                   for record in diagnostic_provenance for item in record['files'])):
        raise ValueError('Evidence changed while the match was being constructed')
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8-sig'))!=report:
            raise ValueError('Existing match report differs from its immutable identity')
    else:
        store.write_json(target,report)
    report_hash = sha256(target)
    memory = FailureMemory(store.root)
    event_ids = []
    for kind, rows, field in [('false_positive',report['false_positives'],'candidate_id'),
                              ('missed_defect',report['false_negatives'],'reference_id')]:
        for item in rows:
            identity = item[field]
            occurrence = kind+'-'+hashlib.sha256(identity.encode()).hexdigest()[:32]
            event = memory.record(job_id=job_id,stage='baseline_match',reason_code=kind,occurrence_key=occurrence,
                evidence_refs=[str(p) for p in evidence_files],
                context={'source_sha256':image_hash,'baseline_sha256':baseline_hash,'annotation_sha256':label_hash,
                         'match_sha256':report_hash,'candidate_id':item.get('candidate_id'),
                         'reference_id':item.get('reference_id'),'class_code':item.get('class_code'),
                         'bbox_xyxy':item.get('bbox_xyxy'),'simulation':report['simulation'],
                         'reference_complete':reference_complete,
                         'match_diagnostic':item.get('match_diagnostic'),
                         'competing_reference_id':item.get('competing_reference_id'),
                         'evidence_reference_mode':'manifest_chain' if compact_evidence else 'direct',
                         'evidence_sha256':{image_snapshot.name:image_hash,label_snapshot.name:label_hash,
                                            baseline_file.name:baseline_hash,target.name:report_hash,
                                            **({sidecar.name:sidecar_hash} if sidecar_hash else {}),
                                            **({} if compact_evidence else {p.name:artifacts[p.name] for p in artifact_paths})}},
                source='mock' if report['simulation'] else 'baseline_match',cause_status='observed')
            event_ids.append(event['event_id'])
    return {'status':report['status'],'job_id':job_id,'match_path':str(target),'match_sha256':report_hash,
            'source_sha256':image_hash,
            'event_ids':event_ids,'counts':{'tp':len(report['matches']),
            'fp':len(report['false_positives']),'fn':len(report['false_negatives']),
            'unverified_predictions':len(report['unverified_predictions']),
            'unresolved_references':len(report['unresolved_references']),
            'reference_issues':len(issues)},'simulation':report['simulation']}


def match_baseline_batch(store: ArtifactStore, summary_path: str, labels_directory: str,
                         class_map: dict[str, str], *, expected_count: int = 203,
                         iou_threshold: float = .5, reference_complete: bool = False,
                         reference_assumption: str | None = None) -> dict:
    """Match a completed frozen Figure-1 batch, without hiding unresolved cases.

    Requires every expected baseline result; partial expert-only or failed
    batches cannot be mistaken for Figure 1.  An image with malformed reference
    is explicitly flagged for review while other images remain traceable.
    """
    if type(expected_count) is not int or not 1 <= expected_count <= 10000:
        raise ValueError('Expected image count must be a positive integer')
    if reference_assumption is not None and (not reference_complete or not isinstance(reference_assumption,str)
                                               or not reference_assumption.strip()):
        raise ValueError('A reference assumption requires explicit complete-reference scoring and a nonempty explanation')
    source = store.resolve_input(summary_path)
    summary_hash = sha256(source)
    summary = json.loads(source.read_text(encoding='utf-8-sig'))
    if not isinstance(summary,dict) or summary.get('schema') != 'facade-baseline-summary/v1':
        raise ValueError('Expected a completed Figure-1 baseline batch summary')
    completed,failed = summary.get('completed'),summary.get('failed')
    experiment = summary.get('experiment_sha256')
    if (not isinstance(completed,list) or len(completed) != expected_count or not isinstance(failed,list)
            or failed or summary.get('halted_reason') or not re.fullmatch(r'[a-f0-9]{64}',str(experiment))):
        raise ValueError('Baseline batch is incomplete, failed or lacks a frozen experiment identity')
    manifest_file=store.resolve_input(str(source.parent/'baseline-manifest.json'))
    manifest_hash=sha256(manifest_file)
    manifest=json.loads(manifest_file.read_text(encoding='utf-8-sig'))
    if not isinstance(manifest,dict) or manifest.get('schema') != 'facade-baseline-batch/v1':
        raise ValueError('Expected a frozen Figure-1 baseline manifest')
    declared=manifest.pop('experiment_sha256',None)
    calculated=hashlib.sha256(json.dumps(manifest,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    inputs=manifest.get('image_inputs')
    if (declared != experiment or calculated != experiment or not isinstance(inputs,list) or len(inputs) != expected_count
            or any(not isinstance(x,dict) or not isinstance(x.get('path'),str)
                   or not re.fullmatch(r'[a-f0-9]{64}',str(x.get('sha256',''))) for x in inputs)):
        raise ValueError('Baseline frozen experiment manifest has changed')
    frozen_inputs={x['path']:x['sha256'] for x in inputs}
    if len(frozen_inputs) != expected_count or len(set(frozen_inputs.values())) != expected_count:
        raise ValueError('Frozen experiment contains repeated image paths or bytes')
    label_root = Path(labels_directory).expanduser().resolve()
    if not label_root.is_dir() or not any(label_root.is_relative_to(root) for root in store.allowed_input_roots):
        raise ValueError('Reference label directory is missing or outside allowed roots')
    cases,seen_inputs,seen_results,seen_stems=[],set(),set(),set()
    for entry in completed:
        if not isinstance(entry,dict) or not isinstance(entry.get('result_file'),str) or not isinstance(entry.get('path'),str):
            raise ValueError('Invalid baseline summary case')
        result_file = store.resolve_input(entry['result_file'])
        input_path = store.resolve_input(entry['path'])
        data=json.loads(result_file.read_text(encoding='utf-8-sig'))
        if (data.get('experiment_sha256') != experiment or data.get('image',{}).get('path') != str(input_path)
                or frozen_inputs.get(str(input_path)) != data.get('image',{}).get('sha256')
                or result_file in seen_results or input_path in seen_inputs or input_path.stem in seen_stems):
            raise ValueError('Batch result differs from frozen experiment or repeats an image')
        seen_results.add(result_file);seen_inputs.add(input_path);seen_stems.add(input_path.stem)
        label_file = (label_root/(input_path.stem+'.txt')).resolve()
        if not label_file.is_relative_to(label_root):
            raise ValueError('Reference label escaped the configured directory')
        cases.append((input_path,result_file,label_file))
    # Preflight all labels before making any event records; missing files are
    # not equivalent to an image with an explicitly empty label file.
    if any(not label.is_file() for _,_,label in cases):
        raise ValueError('A corresponding reference label file is missing')
    rows=[]
    for image,result_file,label in cases:
        row=match_baseline_case(store,str(result_file),str(label),class_map,
                                iou_threshold=iou_threshold,reference_complete=reference_complete)
        rows.append({'image_path':str(image),**row})
    if sha256(source)!=summary_hash or sha256(manifest_file)!=manifest_hash:
        raise ValueError('Baseline batch summary or manifest changed during matching')
    totals={key:sum(row['counts'][key] for row in rows) for key in
            ('tp','fp','fn','unverified_predictions','unresolved_references','reference_issues')}
    result={'schema':'facade-baseline-match-batch/v1',
            'status':'needs_reference_review' if any(row['status']!='completed' for row in rows) else 'completed',
            'experiment_sha256':experiment,'baseline_summary':str(source),'baseline_summary_sha256':summary_hash,
            'baseline_manifest':str(manifest_file),'baseline_manifest_sha256':manifest_hash,
            'expected_count':expected_count,'matched_count':len(rows),'class_map':class_map,
            'reference_complete':reference_complete,'reference_assumption':reference_assumption,
            'counts':totals,'cases':rows,
            'simulation':any(row['simulation'] for row in rows),
            'limitations':['Counts exclude any case with invalid reference annotations.',
                           'A P0 proposal must explicitly select eligible real case job IDs and remains a candidate.',
                           'Images used for P0 development are not independent P0 test data.']}
    content_hash=hashlib.sha256(json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False,
        separators=(',',':')).encode('utf-8')).hexdigest()
    target=store.root/'baseline_matches'/('batch-'+experiment+'-'+content_hash[:20]+'.json')
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8-sig'))!=result:
            raise ValueError('Existing content-addressed match batch is inconsistent')
    else:
        store.write_json(target,result)
    return {**result,'match_batch_path':str(target)}


async def propose_initial_p0(store: ArtifactStore, settings, job_ids: list[str], *, allow_api=False, transport=None):
    """Reflect and curate registered baseline discrepancies as a candidate only.

    Existing lifecycle bounds apply: select at most 200 eligible events and
    256 MiB evidence per call. Mock/simulated baseline events are ineligible.
    Candidate publication remains a separate independent validation step.
    """
    if not job_ids or len(job_ids) != len(set(job_ids)):
        raise ValueError('Choose nonempty distinct baseline match job IDs')
    if any(not re.fullmatch(r'baseline_match_[a-f0-9]{32}', x) for x in job_ids):
        raise ValueError('P0 bootstrap accepts baseline match jobs only')
    # This older lifecycle entry point checks direct references only. The full
    # coverage path additionally follows and verifies the complete report chain.
    memory=FailureMemory(store.root)
    if any(event.get('context',{}).get('evidence_reference_mode')=='manifest_chain'
           for job_id in job_ids for event in memory.summary(job_id)['events']):
        raise ValueError('Compact baseline evidence requires full-coverage P0 validation')
    lifecycle=RuleLifecycle(store)
    if lifecycle.latest() is not None:
        raise ValueError('Initial P0 cannot replace an already published playbook')
    return await lifecycle.propose(settings,parent=Playbook(version='P0-empty'),job_ids=job_ids,
                                   allow_api=allow_api,transport=transport)
