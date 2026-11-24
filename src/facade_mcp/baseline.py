"""Figure-1 baseline: fixed square expert inputs, bounded VLM review and audit.

No network or model access occurs at import time. The caller supplies the MCP
expert gateway and the image-capable ModelGateway; output is resumable only
under the same frozen input/configuration identity. A mock run is explicitly
marked as simulation and must not be promoted to an operational rule.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import Field, model_validator

from .image_inspection import IMAGE_CLASSES, _canonical_code, _lock_expert, crop_box, validate_box
from .model_gateway import GatewayError, ModelGateway, Strict
from .storage import sha256


class BaselineConfig(Strict):
    profiles: list[str] = Field(default_factory=lambda: ['yolo11m', 'cubit_yolov9e', 'cubit_efficientvit_m5'], min_length=3, max_length=3)
    confidence_floor: float = Field(default=.001, ge=0, le=.25, allow_inf_nan=False)
    expected_count: int | None = Field(default=None, ge=1)
    # Keep each annotated view sparse enough for the VLM to identify boxes.
    decision_batch_size: int = Field(default=8, ge=1, le=8)
    # Both configured source and registered Ultralytics workers use this NMS.
    nms_iou: Literal[0.5] = .5
    max_detections_per_expert: Literal[300] = 300
    expert_input_size: Literal[640] = 640
    normal_boundary: Literal[0.25] = .25
    grid_columns: Literal[8] = 8
    grid_rows: Literal[5] = 5
    vlm_width: Literal[1024] = 1024
    vlm_height: Literal[680] = 680

    @model_validator(mode='after')
    def validate_profiles(self):
        if len(set(self.profiles)) != 3 or any(not re.fullmatch('[a-z][a-z0-9_-]{0,63}', x) for x in self.profiles):
            raise ValueError('Exactly three unique safe expert profile names are required')
        return self


class BaselineVerdict(Strict):
    candidate_id: str = Field(min_length=1, max_length=100)
    decision: Literal['accept', 'reject', 'review']
    reason: str = Field(min_length=1, max_length=1000)
    review_profile: str | None = None
    review_scale: Literal[1, 2, 3] | None = None

    @model_validator(mode='after')
    def review_fields(self):
        if self.decision == 'review':
            if not self.review_profile or self.review_scale is None:
                raise ValueError('Review requires one eligible expert and crop scale')
        elif self.review_profile is not None or self.review_scale is not None:
            raise ValueError('Review settings are only valid for review decisions')
        return self


class BaselineVerdicts(Strict):
    decisions: list[BaselineVerdict]


class UncertainGatewayOutcome(GatewayError):
    """A previous VLM call was dispatched but has no durable result."""


class ExpertImplementationDrift(ValueError):
    """A batch attempted to mix different expert runtime implementations."""


def _json(path: Path, value):
    """Write complete JSON atomically, without exposing a partial success file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _letterbox(source: Image.Image, path: Path, size=640):
    """Produce an actual square raster, retaining exact invertible axis scales."""
    w, h = source.size
    if w < 1 or h < 1:
        raise ValueError('Empty input raster')
    ratio = min(size / w, size / h)
    resized_w, resized_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (size-resized_w)//2, (size-resized_h)//2
    square = Image.new('RGB', (size, size), (114, 114, 114))
    square.paste(source.convert('RGB').resize((resized_w, resized_h), Image.Resampling.BILINEAR), (left, top))
    square.save(path, format='PNG')
    return {'size': [size, size], 'source_size': [w, h], 'resized_size': [resized_w, resized_h],
            'pad_left': left, 'pad_top': top, 'scale_x': resized_w/w, 'scale_y': resized_h/h,
            'padding_rgb': [114, 114, 114], 'resample': 'PIL bilinear'}


def _backproject(box, transform, origin, source_shape):
    """Clip detections to actual resized content and map into source pixels."""
    raw = validate_box(box, *transform['size'])
    left, top = transform['pad_left'], transform['pad_top']
    rw, rh = transform['resized_size']
    x1, y1 = max(raw[0], left), max(raw[1], top)
    x2, y2 = min(raw[2], left+rw), min(raw[3], top+rh)
    if x1 >= x2 or y1 >= y2:
        return None
    xoff, yoff = origin
    out = [(x1-left)/transform['scale_x']+xoff, (y1-top)/transform['scale_y']+yoff,
           (x2-left)/transform['scale_x']+xoff, (y2-top)/transform['scale_y']+yoff]
    w, h = source_shape
    out = [max(0., min(float(v), edge)) for v, edge in zip(out, (w, h, w, h))]
    return validate_box(out, w, h)


def _kind(code, confidence):
    # C/S are target defects. Other recognized defects are figure-1 Env.
    # An unmapped label is explicitly retained, not invented as an Env class.
    if code in ('C', 'S'):
        return 'Normal' if confidence > .25 else 'Ghost'
    return 'Env' if code in IMAGE_CLASSES else 'Unmapped'


_PALETTE = {'Normal':'#ff2222', 'Ghost':'#1684ff', 'Env':'#18a52c', 'Unmapped':'#ffc400'}


def _compact_boxes(draw, candidates, transform, image_size):
    """Render only this request's boxes, with distinct, collision-avoiding badges."""
    width,height=image_size
    try:
        font=ImageFont.truetype('DejaVuSans-Bold.ttf', 14)
    except OSError:
        font=ImageFont.load_default()
    rects=[]
    for candidate in candidates:
        rect=transform(candidate['bbox_xyxy'])
        color=_PALETTE[candidate['kind']]
        draw.rectangle(rect,outline=color,width=2)
        rects.append((candidate,rect,color))
    placed=[]
    for candidate,rect,color in rects:
        label=candidate['overlay_label']
        bounds=draw.textbbox((0,0),label,font=font,stroke_width=0)
        bw,bh=bounds[2]-bounds[0]+8,bounds[3]-bounds[1]+8
        anchor_x,anchor_y=rect[0],rect[1]
        # Prefer the box's upper-left corner, then nearby grid cells.  A leader
        # identifies the original corner if the short badge must be displaced.
        offsets=[(0,0)]
        for radius in range(1,18):
            offsets.extend((dx*40,dy*24) for dx in range(-radius,radius+1)
                           for dy in range(-radius,radius+1) if max(abs(dx),abs(dy))==radius)
        x=y=0
        for dx,dy in offsets:
            x=max(0,min(width-bw,round(anchor_x+dx)))
            y=max(0,min(height-bh,round(anchor_y+dy)))
            if all(x+bw+2<=a or b+2<=x or y+bh+2<=c or d+2<=y for a,c,b,d in placed):
                break
        placed.append((x,y,x+bw,y+bh))
        if abs(x-anchor_x)>3 or abs(y-anchor_y)>3:
            draw.line((anchor_x,anchor_y,x+bw/2,y+bh/2),fill=color,width=1)
        draw.rectangle((x,y,x+bw,y+bh),fill=color,outline='white',width=1)
        draw.text((x+4-bounds[0],y+4-bounds[1]),label,font=font,fill='white',stroke_width=0)


def _overview(source, candidates, target: Path, *, compact=False):
    """Fixed 1024×680 raster, original 8×5 grid, with type and score labels."""
    w, h = source.size
    ratio = min(1024/w, 680/h)
    rw, rh = max(1, round(w*ratio)), max(1, round(h*ratio))
    xpad, ypad = (1024-rw)//2, (680-rh)//2
    out = Image.new('RGB', (1024, 680), (114, 114, 114))
    out.paste(source.resize((rw,rh), Image.Resampling.BILINEAR), (xpad, ypad))
    draw = ImageDraw.Draw(out)
    for col in range(9):
        x=xpad+round(col*rw/8)
        draw.line((x,ypad,x,ypad+rh), fill='#c0c0c0', width=1)
    for row in range(6):
        y=ypad+round(row*rh/5)
        draw.line((xpad,y,xpad+rw,y), fill='#c0c0c0', width=1)
    for row in range(5):
        for col in range(8):
            draw.text((xpad+round(col*rw/8)+2, ypad+round(row*rh/5)+2),
                      f'G{row+1}-{col+1}', fill='#666666', stroke_width=1, stroke_fill='white')
    def transform(box):
        x1,y1,x2,y2=box
        return [xpad+x1*rw/w, ypad+y1*rh/h, xpad+x2*rw/w, ypad+y2*rh/h]
    if compact:
        _compact_boxes(draw,candidates,transform,out.size)
    else:
        for c in candidates:
            rect=transform(c['bbox_xyxy']); color=_PALETTE[c['kind']]
            draw.rectangle(rect, outline=color, width=2)
            text=f"{c['candidate_id']} {c['kind']} {c['class_code']} {c['confidence']:.3f}"
            draw.text((rect[0]+1, rect[1]+1), text, fill=color, stroke_width=1, stroke_fill='white')
    out.save(target, quality=90, format='JPEG')
    return {'width':1024,'height':680,'source_width':w,'source_height':h,
            'display_size':[rw,rh], 'pad_left':xpad, 'pad_top':ypad,
            'scale_x':rw/w,'scale_y':rh/h}


def _review_evidence(roi, bounds, chosen, observations, path, *, compact=False):
    evidence=roi.copy()
    evidence.thumbnail((1024,1024))
    sx,sy=evidence.width/roi.width,evidence.height/roi.height
    draw=ImageDraw.Draw(evidence)
    rows=([chosen] if chosen is not None else [])+observations
    def transform(box):
        x1,y1,x2,y2=box
        return [(x1-bounds[0])*sx,(y1-bounds[1])*sy,(x2-bounds[0])*sx,(y2-bounds[1])*sy]
    if compact:
        _compact_boxes(draw,rows,transform,evidence.size)
    else:
        for row in rows:
            rect=transform(row['bbox_xyxy']); color=_PALETTE[row['kind']]
            draw.rectangle(rect,outline=color,width=2)
            label=f"{row['candidate_id']} {row['class_code']} {row['confidence']:.3f}"
            draw.text((rect[0]+1,rect[1]+1),label,fill=color,stroke_width=1,stroke_fill='white')
    evidence.save(path,quality=90,format='JPEG')


def _validate_decisions(verdicts, candidates, allow_review, profiles, capabilities):
    by_id={item['candidate_id']:item for item in candidates}
    rows=verdicts.decisions
    if len(rows)!=len(by_id) or {r.candidate_id for r in rows}!=set(by_id):
        raise ValueError('VLM must decide each supplied candidate exactly once')
    reviews=[r for r in rows if r.decision=='review']
    if len(reviews)>int(allow_review):
        raise ValueError('Review budget exceeded')
    for v in reviews:
        cap=capabilities.get(v.review_profile, {})
        if v.review_profile not in profiles or not cap.get('local_eligible', True):
            raise ValueError('Review selected an unavailable expert')
        if by_id[v.candidate_id]['class_code'] not in cap.get('class_codes',[]):
            raise ValueError('Review expert cannot inspect that class')
    return rows


def _mock_decisions(candidates):
    return {'decisions':[{'candidate_id':c['candidate_id'], 'decision':'accept',
                          'reason':'Synthetic mock; not evidence of defect quality.',
                          'review_profile':None,'review_scale':None} for c in candidates]}


async def _ask(vlm, task, images, candidates, allow_review, config, capabilities, journal):
    task = {**task, 'candidates':candidates, 'review_allowed':allow_review,
            'instruction':'Decide every candidate exactly once using accept/reject/review. '
                          'For review choose an eligible expert and scale 1, 2, or 3; at most one review for the entire image. '
                          'Context/Env boxes are not target candidates. Do not invent IDs, objects or evidence.'}
    journal['index']+=1
    number=journal['index']
    record=journal['folder']/f'vlm-call-{number:04d}.json'
    pending=journal['folder']/f'vlm-call-{number:04d}.pending.json'
    fingerprint=_digest({'task':task,'images':[sha256(p) for p in images],'schema':'baseline-verdicts/v1'})
    if record.exists():
        cached=json.loads(record.read_text(encoding='utf-8'))
        if cached.get('request_sha256')!=fingerprint or cached.get('status')!='completed':
            raise GatewayError('Prior VLM call does not match the current evidence')
        answer=BaselineVerdicts.model_validate(cached['response'])
        journal['trace'].extend(cached.get('model_trace',[]))
        try:
            return _validate_decisions(answer,candidates,allow_review,config.profiles,capabilities)
        except ValueError:
            raise GatewayError('Prior VLM decision violates the review protocol') from None
    if pending.exists():
        if not journal['acknowledge_unknown']:
            raise UncertainGatewayOutcome('A previous VLM request may have been billed; inspect before retrying')
        i=1
        while (journal['folder']/f'vlm-call-{number:04d}.pending-ack-{i}.json').exists(): i+=1
        pending.rename(journal['folder']/f'vlm-call-{number:04d}.pending-ack-{i}.json')
    _json(pending,{'request_sha256':fingerprint,'stage':task['stage'],'status':'started'})
    trace_start=len(getattr(vlm,'trace',[]))
    try:
        answer=await vlm.invoke('generator', task, BaselineVerdicts, mock_output=_mock_decisions(candidates), image_paths=images)
        try:
            rows=_validate_decisions(answer,candidates,allow_review,config.profiles,capabilities)
        except ValueError:
            raise GatewayError('VLM decision violates the review protocol') from None
    except Exception:
        trace=list(getattr(vlm,'trace',[]))[trace_start:]
        if trace and trace[-1].get('status') not in ('started','request_outcome_unknown'):
            # A known failed response may be retried only via retry_failed.
            i=1
            while (journal['folder']/f'vlm-call-{number:04d}.failed-{i}.json').exists(): i+=1
            _json(journal['folder']/f'vlm-call-{number:04d}.failed-{i}.json',
                  {'request_sha256':fingerprint,'status':'failed','model_trace':trace})
            pending.unlink()
        # Keep an unresolved marker if process/transport outcome is unknown.
        raise
    trace=list(getattr(vlm,'trace',[]))[trace_start:]
    _json(record,{'request_sha256':fingerprint,'status':'completed',
                  'response':answer.model_dump(mode='json'),'model_trace':trace})
    pending.unlink()
    journal['trace'].extend(trace)
    return rows


async def _detect(experts, path, profile, prefix, config, capabilities, locks, implementation_locks, origin, source_shape, transform, source_sha, folder):
    arguments={'image_paths':[str(path)],'model_profile':profile,'threshold':config.confidence_floor,'inference_size':640}
    response=await experts.call('facade_models_detect', arguments)
    _json(folder / (prefix+'-expert.json'), response)
    images=response.get('images', [])
    if response.get('status')!='completed' or len(images)!=1:
        raise ValueError('Expert did not return one completed image')
    if response.get('coordinate_space','original_pixels')!='original_pixels' or response.get('bbox_convention','zero_based_xyxy_max_exclusive')!='zero_based_xyxy_max_exclusive':
        raise ValueError('Expert did not return pixel XYXY coordinates')
    image=images[0]
    if image.get('source_sha256')!=sha256(path) or [image.get('width'),image.get('height')]!=[640,640]:
        raise ValueError('Expert result does not identify the exact square input')
    provenance=image.get('model_provenance',{})
    _lock_expert(profile,provenance,locks,capabilities[profile],True,implementation_locks)
    out=[]; padding_only=0
    for i,row in enumerate(image.get('defects',[])+image.get('unmapped_defects',[]),1):
        score=row.get('confidence')
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Expert returned invalid confidence')
        if score < config.confidence_floor:
            raise ValueError('Expert returned a score below the configured floor')
        code=_canonical_code(row)
        if code in IMAGE_CLASSES and code not in capabilities[profile].get('class_codes',[]):
            raise ValueError('Expert returned a class outside its declared capabilities')
        mapped=_backproject(row.get('bbox_xyxy'),transform,origin,source_shape)
        if mapped is None:
            padding_only+=1
            continue
        out.append({'candidate_id':f'{prefix}-{i:04d}', 'class_code':code,
                    'class_name':str(row.get('original_class_name',row.get('class_name',code))),
                    'bbox_xyxy':mapped, 'confidence':float(score), 'model_profile':profile,
                    'source_instance_id':row.get('instance_id'), 'source_sha256':source_sha,
                    'kind':_kind(code,float(score))})
    return out, provenance, padding_only


def _eligible(candidates, capabilities):
    out=[]
    for c in candidates:
        out.append({**c, 'eligible_profiles':[n for n,p in capabilities.items()
                    if p.get('local_eligible',True) and c['class_code'] in p.get('class_codes',[])]})
    return out


def _provenance_simulation(experts, vlm, requested):
    # Only the actual MCP gateway plus a real, non-injected model connection can
    # generate a result eligible for the later audited rule lifecycle.
    from .orchestration import ToolGateway
    if requested or not isinstance(experts,ToolGateway) or not isinstance(vlm,ModelGateway):
        return True
    con=vlm.config.connections[vlm.config.roles['generator']]
    return con.adapter=='mock' or not vlm.config.api_enabled or not vlm.allow_api or vlm.transport is not None


def _expert_profile_snapshot(capabilities, config):
    capabilities=capabilities.get('profiles',capabilities)
    if any(p not in capabilities or not capabilities[p].get('ready') or not capabilities[p].get('full_image',True)
           for p in config.profiles):
        raise ValueError('Three configured full-image experts must be ready')
    return {p:dict(capabilities[p]) for p in config.profiles}


def _validate_artifacts(folder, files):
    if not isinstance(files,dict) or not files:
        raise ValueError('Existing result has no artifact identities')
    for name, digest in files.items():
        artifact=folder/name
        if Path(name).name!=name or not artifact.is_file() or sha256(artifact)!=digest:
            raise ValueError('Existing result artifacts changed')


def _stage_hash_path(folder):
    return folder/'baseline-expert-result.sha256.json'


def _validate_expert_stage_file(folder):
    """Bind cached observation metadata to a separate durable digest record.

    This detects incomplete writes and unintended edits; it is not a signature
    and cannot defend against an actor able to alter both files.
    """
    path=folder/'baseline-expert-result.json'
    checksum=_stage_hash_path(folder)
    if not path.is_file() or not checksum.is_file():
        raise ValueError('Existing expert stage has no integrity record')
    expected=json.loads(checksum.read_text(encoding='utf-8')).get('sha256')
    if not isinstance(expected,str) or not re.fullmatch(r'[a-f0-9]{64}',expected) or sha256(path)!=expected:
        raise ValueError('Existing expert stage metadata changed')
    return expected


def _validate_baseline_result_file(folder):
    """Check the saved final candidate/decision record before resuming."""
    path=folder/'baseline-result.json'
    checksum=folder/'baseline-result.sha256.json'
    if not path.is_file() or not checksum.is_file():
        raise ValueError('Existing baseline result has no integrity record')
    expected=json.loads(checksum.read_text(encoding='utf-8')).get('sha256')
    if not isinstance(expected,str) or not re.fullmatch(r'[a-f0-9]{64}',expected) or sha256(path)!=expected:
        raise ValueError('Existing baseline result metadata changed')
    return expected


def _lock_batch_implementations(result, locks, profiles):
    """Freeze every expert implementation identity across all batch images."""
    identities=result.get('implementation_identities')
    if not isinstance(identities,dict) or set(identities)!=set(profiles):
        raise ExpertImplementationDrift('Expert implementation identities are incomplete')
    for profile in profiles:
        identity=identities[profile]
        if not isinstance(identity,str) or not re.fullmatch(r'[a-f0-9]{64}',identity):
            raise ExpertImplementationDrift('Invalid expert implementation identity: '+profile)
        if profile in locks and locks[profile]!=identity:
            raise ExpertImplementationDrift('Expert implementation changed within batch: '+profile)
    # Do not partially mutate locks if a later profile fails validation.
    locks.update(identities)


async def execute_baseline_experts(image_path: Path, folder: Path, experts, capabilities: dict,
                                   config: BaselineConfig):
    """Run or safely reuse only the three expert predictions; no VLM is called.

    This stage is explicitly incomplete for Figure 1 and cannot serve as a
    completed baseline or a validated playbook source. No labels are read.
    """
    source_path=Path(image_path).resolve()
    if not source_path.is_file(): raise ValueError('Input image missing')
    selected=_expert_profile_snapshot(capabilities,config)
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    digest=sha256(source_path)
    stage_key=_digest({'source_path':str(source_path),'source_sha256':digest,
                       'config':config.model_dump(mode='json'),'expert_capabilities':selected,
                       'baseline_implementation_sha256':sha256(Path(__file__))})
    stage_file=folder/'baseline-expert-result.json'
    if stage_file.exists():
        _validate_expert_stage_file(folder)
        cached=json.loads(stage_file.read_text(encoding='utf-8'))
        if cached.get('schema')!='facade-baseline-experts/v1' or cached.get('status')!='experts_completed' or cached.get('expert_stage_sha256')!=stage_key:
            raise ValueError('Existing expert stage does not match the frozen image/configuration')
        _validate_artifacts(folder,cached.get('artifacts_sha256'))
        return cached
    snapshot=folder/('source'+source_path.suffix.lower())
    shutil.copyfile(source_path,snapshot)
    if sha256(snapshot)!=digest: raise ValueError('Source changed while creating snapshot')
    with Image.open(snapshot) as raw:
        if raw.format not in ('JPEG','PNG') or getattr(raw,'n_frames',1)!=1 or raw.width*raw.height>80_000_000:
            raise ValueError('Expected a supported still image under the pixel budget')
        raw.load(); source=raw.convert('RGB')
    width,height=source.size
    if width<8 or height<5: raise ValueError('Input image is smaller than the 8x5 grid')
    square=folder/'full-input-640.png'
    transform=_letterbox(source,square)
    _json(folder/'full-input-transform.json',transform)
    observations=[]; results={}; locks={}; implementation_locks={}
    for index,profile in enumerate(config.profiles,1):
        prefix=f'full-{index:02d}'
        rows,prov,dropped=await _detect(experts,square,profile,prefix,config,selected,locks,implementation_locks,
                                        (0,0),source.size,transform,digest,folder)
        results[profile]={'response_file':prefix+'-expert.json','weight_identity':locks[profile],
                          'implementation_identity':implementation_locks.get(profile),
                          'definition_sha256':selected[profile].get('definition_sha256'),
                          'candidates':rows,'padding_only_detections':dropped}
        observations.extend(rows)
    overview=folder/'baseline-overview.jpg'
    display=_overview(source,observations,overview)
    if sha256(source_path)!=digest or sha256(snapshot)!=digest:
        raise ValueError('Source changed during expert inference')
    artifacts=[snapshot,square,folder/'full-input-transform.json',overview]
    artifacts.extend(folder/(f'full-{i:02d}-expert.json') for i in range(1,len(config.profiles)+1))
    from .orchestration import ToolGateway
    result={'schema':'facade-baseline-experts/v1','status':'experts_completed',
            'simulation':not isinstance(experts,ToolGateway), 'vlm_called':False,
            'image':{'path':str(source_path),'sha256':digest,'width':width,'height':height},
            'experts':results,'observations':observations,'overview_file':overview.name,
            'overview_display':display,'weight_identities':locks,'implementation_identities':implementation_locks,
            'config':config.model_dump(mode='json'),'expert_stage_sha256':stage_key,
            'artifacts_sha256':{p.name:sha256(p) for p in artifacts},
            'limitations':['Expert output only: Figure 1 VLM judgement and review have not run.',
                           'Only detections above the configured floor and after NMS/max_det are included.',
                           'Model output is not human-verified ground truth.']}
    # Record the digest first. If the process stops between the two atomic
    # writes, no completed stage exists and the image can safely be retried.
    encoded=(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf-8')
    _json(_stage_hash_path(folder),{'sha256':hashlib.sha256(encoded).hexdigest()})
    _json(stage_file,result)
    _validate_expert_stage_file(folder)
    return result


async def execute_baseline_image(image_path: Path, folder: Path, experts, vlm, capabilities: dict,
                                 config: BaselineConfig, *, simulation=True, experiment_sha256: str | None=None,
                                 acknowledge_unknown=False, batch_implementation_locks=None):
    """Run one image; the caller owns output folder and MCP access allowlist."""
    source_path=Path(image_path).resolve(); folder=Path(folder)
    is_simulation=_provenance_simulation(experts,vlm,simulation)
    if not simulation and is_simulation:
        raise ValueError('Real baseline requires live MCP experts and a real authorized VLM gateway')
    selected=_expert_profile_snapshot(capabilities,config)
    expert_stage=await execute_baseline_experts(source_path,folder,experts,selected,config)
    if not is_simulation and expert_stage.get('simulation') is not False:
        raise ValueError('Simulated expert outputs cannot be promoted into a real baseline')
    if batch_implementation_locks is not None:
        # Enforce before any VLM call so a changed expert cannot consume a
        # request or contaminate a frozen comparison.
        _lock_batch_implementations(expert_stage,batch_implementation_locks,config.profiles)
    stage_hash=_validate_expert_stage_file(folder)
    stage_record_hash=sha256(_stage_hash_path(folder))
    digest=expert_stage['image']['sha256']
    snapshot=folder/('source'+source_path.suffix.lower())
    with Image.open(snapshot) as raw:
        raw.load();source=raw.convert('RGB')
    width,height=source.size
    square=folder/'full-input-640.png'
    all_candidates=expert_stage['observations'];results=expert_stage['experts']
    locks=dict(expert_stage['weight_identities']);implementation_locks=dict(expert_stage['implementation_identities'])
    target=[c for c in all_candidates if c['kind'] in ('Normal','Ghost')]
    contexts=[c for c in all_candidates if c['kind'] in ('Env','Unmapped')]
    overview=folder/'baseline-overview.jpg'
    display=expert_stage['overview_display']
    common={'workflow':'facade-figure1-baseline/v1','original_size':[width,height],
            'coordinate_space':'zero_based_xyxy_max_exclusive','overview_display':display,
            'context_boxes':contexts,'expert_capabilities':{p:{'class_codes':v.get('class_codes',[]),
                              'local_eligible':v.get('local_eligible',True)} for p,v in selected.items()},
            'thresholds':{'candidate_floor':config.confidence_floor,'normal_gt':.25,'ghost_lte':.25},
            'grid':{'columns':8,'rows':5}, 'local_review_budget':1}
    trajectory=[]; verdicts={}; review=None; request_evidence=[]
    journal={'folder':folder,'index':0,'trace':[], 'acknowledge_unknown':acknowledge_unknown}
    def annotated_chunk(items, stage, *, roi=None, bounds=None):
        chunk=[{**c,'overlay_label':f'#{i}'} for i,c in enumerate(_eligible(items,selected),1)]
        number=journal['index']+1
        grid_path=folder/f'vlm-call-{number:04d}-grid.jpg'
        _overview(source,chunk,grid_path,compact=True)
        paths=[grid_path]
        if roi is not None:
            roi_path=folder/f'vlm-call-{number:04d}-roi.jpg'
            _review_evidence(roi,bounds,None,chunk,roi_path,compact=True)
            paths.append(roi_path)
        request_evidence.append({'call':number,'stage':stage,
            'candidates':[{'candidate_id':c['candidate_id'],'overlay_label':c['overlay_label']} for c in chunk],
            'images':[{'file':p.name,'sha256':sha256(p)} for p in paths]})
        return chunk,paths
    annotation_note=('Image 0 shows only this request\'s candidates on the 8-by-5 grid. '
                     'Each short badge is mapped to its full candidate_id by overlay_label; '
                     'return the full candidate_id, not the badge. '
                     'Badges may have leader lines to a box corner. '
                     'Red boxes are Normal and blue boxes Ghost. Context boxes are described in text only.')
    if not target:
        # A true empty result still requires a real, auditable VLM judgement;
        # otherwise downstream P0 matching would misclassify the case as mock
        # and lose a possible missed-defect observation.
        chunk,images=annotated_chunk([],'baseline_no_candidates')
        await _ask(vlm,{**common,'stage':'baseline_no_candidates', 'image_annotation':annotation_note,
                        'note':'No crack or spalling candidates were returned; return an empty decisions list.'},
                   images,chunk,False,config,selected,journal)
        trajectory.append({'stage':'baseline_no_candidates','decision_count':0})
    for start in range(0,len(target),config.decision_batch_size):
        chunk,images=annotated_chunk(target[start:start+config.decision_batch_size],'baseline_judge')
        rows=await _ask(vlm,{**common,'stage':'baseline_judge','image_annotation':annotation_note,
                             'previous_review_selected':review is not None},
                        images,chunk,review is None,config,selected,journal)
        for row in rows:
            verdicts[row.candidate_id]=row.model_dump(mode='json')
            trajectory.append({'stage':'baseline_judge',**row.model_dump(mode='json')})
            if row.decision=='review': review=row
    if review:
        chosen=next(c for c in target if c['candidate_id']==review.candidate_id)
        bounds=crop_box(chosen['bbox_xyxy'],review.review_scale,width,height)
        roi=source.crop(tuple(bounds))
        roi_source=folder/'review-original.png';roi.save(roi_source)
        roi_input=folder/'review-input-640.png';roi_transform=_letterbox(roi,roi_input)
        _json(folder/'review-input-transform.json',{'crop_bbox_original':bounds,**roi_transform})
        local,prov,dropped=await _detect(experts,roi_input,review.review_profile,'review-01',config,selected,locks,
                                         implementation_locks,(bounds[0],bounds[1]),source.size,roi_transform,digest,folder)
        local_target=[c for c in local if c['kind'] in ('Normal','Ghost')]
        local_context=[c for c in local if c['kind'] in ('Env','Unmapped')]
        # ROI evidence is a view of original pixels, independently compressed for the VLM.
        evidence_path=folder/'review-evidence.jpg'
        _review_evidence(roi,bounds,chosen,local,evidence_path)
        trajectory.append({'stage':'review','candidate_id':review.candidate_id,'model_profile':review.review_profile,
                           'scale':review.review_scale,'crop_bbox_original':bounds,'input_sha256':sha256(roi_input),
                           'expert_response_file':'review-01-expert.json','padding_only_detections':dropped})
        review_items=[chosen,*local_target]
        for start in range(0,len(review_items),config.decision_batch_size):
            chunk,images=annotated_chunk(review_items[start:start+config.decision_batch_size],
                                         'baseline_review',roi=roi,bounds=bounds)
            rows=await _ask(vlm,{**common,'stage':'baseline_review','review_context_boxes':local_context,
                                  'crop_bbox_original':bounds,'previous_review_selected':True,
                                  'image_annotation':annotation_note+' Image 1 shows the same numbered boxes on the original-pixel crop.'},
                            images,chunk,False,config,selected,journal)
            for row in rows:
                verdicts[row.candidate_id]=row.model_dump(mode='json')
                trajectory.append({'stage':'baseline_review',**row.model_dump(mode='json')})
        target.extend(local_target)
    if sha256(source_path)!=digest or sha256(snapshot)!=digest:
        raise ValueError('Source changed during inference')
    if _validate_expert_stage_file(folder)!=stage_hash or sha256(_stage_hash_path(folder))!=stage_record_hash:
        raise ValueError('Expert stage changed during VLM judgement')
    candidates=[]
    for c in target:
        verdict=verdicts.get(c['candidate_id'])
        if verdict is None: raise ValueError('Candidate missing VLM decision')
        # A review cannot survive the bounded second-stage decision.
        if verdict['decision']=='review': raise ValueError('Unresolved review decision')
        candidates.append({**c,'decision':verdict['decision'],'reason':verdict['reason']})
    trace=journal['trace']
    identities=dict(getattr(vlm,'model_identities',{}))
    for row in trace:
        if row.get('model_identity_verified') and row.get('connection') and row.get('actual_model'):
            name,model=row['connection'],row['actual_model']
            if name in identities and identities[name]!=model:
                raise ValueError('VLM model identity changed within the baseline')
            identities[name]=model
    artifacts=[folder/name for name in expert_stage['artifacts_sha256']]
    artifacts.append(folder/'baseline-expert-result.json')
    artifacts.append(_stage_hash_path(folder))
    artifacts.extend(folder/f'vlm-call-{i:04d}.json' for i in range(1,journal['index']+1))
    artifacts.extend(folder/image['file'] for entry in request_evidence for image in entry['images'])
    if review:
        artifacts.extend(folder/name for name in ('review-original.png','review-input-640.png',
                        'review-input-transform.json','review-evidence.jpg','review-01-expert.json'))
    result={'schema':'facade-baseline-image/v1','status':'completed','simulation':is_simulation,
            'image':{'path':str(source_path),'sha256':digest,'width':width,'height':height},
            'experts':results,'candidates':candidates,'context_candidates':contexts,'trajectory':trajectory,
            'review_count':int(review is not None),'overview_file':overview.name,'overview_display':display,
            'model_trace':trace,'vlm_model_identities':identities,'vlm_evidence':request_evidence,
            'weight_identities':locks,'implementation_identities':implementation_locks,
            'config':config.model_dump(mode='json'),'experiment_sha256':experiment_sha256,
            'artifacts_sha256':{p.name:sha256(p) for p in artifacts},
            'limitations':['Only detections above the configured floor and after NMS/max_det are included.',
                           'Unmapped classes are not silently treated as known environmental defects.',
                           'Model output is not human-verified ground truth.']}
    # The result itself contains final decisions; bind it separately from the
    # already-hashed evidence files before advertising a completed result.
    encoded=(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf-8')
    _json(folder/'baseline-result.sha256.json',{'sha256':hashlib.sha256(encoded).hexdigest()})
    _json(folder/'baseline-result.json',result)
    _validate_baseline_result_file(folder)
    return result


async def run_baseline_experts_batch(image_paths, folder, experts, capabilities, config=None, *, retry_failed=False,
                                     max_consecutive_failures=3):
    """Run only the three full-image experts, retaining reusable evidence.

    Requires no VLM configuration and performs no mock VLM calls. The result
    is explicitly an incomplete expert stage, never a completed baseline/P0.
    """
    config=config or BaselineConfig()
    if not isinstance(config,BaselineConfig): config=BaselineConfig.model_validate(config)
    if type(max_consecutive_failures) is not int or not 1<=max_consecutive_failures<=100:
        raise ValueError('Consecutive failure limit must be an integer between 1 and 100')
    paths=[Path(p).resolve() for p in image_paths]
    if not paths or len(set(paths))!=len(paths) or (config.expected_count is not None and len(paths)!=config.expected_count):
        raise ValueError('Wrong image count or duplicate paths')
    inputs=[{'path':str(p),'sha256':sha256(p)} for p in paths]
    if len({x['sha256'] for x in inputs})!=len(inputs):
        raise ValueError('Duplicate image contents are not allowed')
    selected=_expert_profile_snapshot(capabilities,config)
    if any(not selected[p].get('weights_sha256') or not selected[p].get('definition_sha256') for p in config.profiles):
        raise ValueError('Expert batch requires frozen checkpoint and implementation identities')
    from .orchestration import ToolGateway
    frozen={'schema':'facade-baseline-expert-batch/v1','config':config.model_dump(mode='json'),
            'image_inputs':inputs,'expert_capabilities':selected,
            'simulation':not isinstance(experts,ToolGateway), 'vlm_called':False,
            'baseline_implementation_sha256':sha256(Path(__file__))}
    batch_hash=_digest(frozen)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'expert-manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        if old.get('expert_batch_sha256')!=batch_hash:
            raise ValueError('Refusing to resume with different images, configuration or expert identities')
    else:
        _json(manifest,{**frozen,'expert_batch_sha256':batch_hash})
    summary={'schema':'facade-baseline-expert-summary/v1','status':'experts_only',
             'expert_batch_sha256':batch_hash,'simulation':frozen['simulation'],'vlm_called':False,
             'completed':[],'failed':[]}
    batch_implementation_locks={}
    consecutive_failures=0
    for index,entry in enumerate(inputs,1):
        image_folder=folder/f'image-{index:04d}-{entry["sha256"][:12]}'
        result_file=image_folder/'baseline-expert-result.json'
        failure_file=image_folder/'expert-failure.json'
        if failure_file.exists() and not result_file.exists() and not retry_failed:
            summary['failed'].append({**json.loads(failure_file.read_text(encoding='utf-8')),'retry_skipped':True})
            continue
        try:
            if sha256(Path(entry['path']))!=entry['sha256']:
                raise ValueError('Source changed after batch was frozen')
            existed=result_file.exists()
            result=await execute_baseline_experts(Path(entry['path']),image_folder,experts,selected,config)
            if not frozen['simulation'] and result.get('simulation') is not False:
                raise ValueError('Simulated expert output cannot be reused as real output')
            _lock_batch_implementations(result,batch_implementation_locks,config.profiles)
            summary['implementation_identities']=dict(batch_implementation_locks)
            summary['completed'].append({'path':entry['path'],'result_file':str(result_file),'resumed':existed})
            consecutive_failures=0
        except Exception as error:
            failure={'path':entry['path'],'error_type':type(error).__name__}
            summary['failed'].append(failure)
            _json(failure_file,failure)
            if isinstance(error,ExpertImplementationDrift):
                summary['halted_reason']='expert_implementation_changed'
                _json(folder/'expert-summary.json',summary)
                return summary
            consecutive_failures+=1
            if consecutive_failures>=max_consecutive_failures:
                summary['halted_reason']='consecutive_expert_failures'
                _json(folder/'expert-summary.json',summary)
                return summary
        _json(folder/'expert-summary.json',summary)
    _json(folder/'expert-summary.json',summary)
    return summary


async def run_baseline_batch(image_paths, folder, experts, vlm, capabilities, config=None, *, simulation=True,
                             retry_failed=False, acknowledge_unknown=False):
    """Execute each distinct image and preserve per-image success/failure for resume.

    Never combine outputs from changed input, configuration, expert definitions or
    model settings: the frozen manifest identity must match exactly on resume.
    The caller must provide the same real MCP gateway and allowed roots for both
    input images and the output folder. No label data is accessed here.
    """
    config=config or BaselineConfig()
    if not isinstance(config,BaselineConfig): config=BaselineConfig.model_validate(config)
    paths=[Path(p).resolve() for p in image_paths]
    if not paths or len(set(paths))!=len(paths) or (config.expected_count is not None and len(paths)!=config.expected_count):
        raise ValueError('Wrong image count or duplicate paths')
    manifest_inputs=[{'path':str(p),'sha256':sha256(p)} for p in paths]
    if len({x['sha256'] for x in manifest_inputs})!=len(manifest_inputs):
        raise ValueError('Duplicate image contents are not allowed')
    caps=capabilities.get('profiles',capabilities)
    if any(p not in caps for p in config.profiles): raise ValueError('Missing expert capabilities')
    selected={p:caps[p] for p in config.profiles}
    is_simulation=_provenance_simulation(experts,vlm,simulation)
    if not simulation and is_simulation:
        raise ValueError('Real baseline requires live MCP experts and a real authorized VLM gateway')
    if not is_simulation:
        con=vlm.config.connections[vlm.config.roles['generator']]
        if not con.expected_model:
            raise ValueError('Real batch requires expected_model to prevent VLM drift across resume')
        if any(not selected[p].get('weights_sha256') or not selected[p].get('definition_sha256') for p in config.profiles):
            raise ValueError('Real batch requires frozen checkpoint and implementation identities')
    if isinstance(vlm,ModelGateway): vlm_settings=vlm.config.model_dump(mode='json')
    else: vlm_settings={'injected_model':f'{type(vlm).__module__}.{type(vlm).__qualname__}'}
    frozen={'schema':'facade-baseline-batch/v1','config':config.model_dump(mode='json'),
            'image_inputs':manifest_inputs, 'expert_capabilities':selected,'vlm_settings':vlm_settings,
            'simulation':is_simulation,'baseline_implementation_sha256':sha256(Path(__file__))}
    experiment_hash=_digest(frozen)
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    manifest_path=folder/'baseline-manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('experiment_sha256')!=experiment_hash:
            raise ValueError('Refusing to resume with different image, configuration or expert identities')
    else:
        _json(manifest_path,{**frozen,'experiment_sha256':experiment_hash})
    summary={'schema':'facade-baseline-summary/v1','experiment_sha256':experiment_hash,'completed':[], 'failed':[], 'simulation':frozen['simulation']}
    batch_implementation_locks={}
    for index,entry in enumerate(manifest_inputs,1):
        image_folder=folder/f'image-{index:04d}-{entry["sha256"][:12]}'
        result_file=image_folder/'baseline-result.json'
        failure_file=image_folder/'baseline-failure.json'
        if result_file.exists():
            _validate_baseline_result_file(image_folder)
            cached=json.loads(result_file.read_text(encoding='utf-8'))
            if cached.get('experiment_sha256')!=experiment_hash or cached.get('image',{}).get('sha256')!=entry['sha256'] or cached.get('status')!='completed':
                raise ValueError('Existing result does not match the frozen experiment')
            _validate_artifacts(image_folder,cached.get('artifacts_sha256'))
            try:
                _lock_batch_implementations(cached,batch_implementation_locks,config.profiles)
            except ExpertImplementationDrift as error:
                summary['failed'].append({'path':entry['path'],'error_type':type(error).__name__})
                summary['halted_reason']='expert_implementation_changed'
                _json(folder/'baseline-summary.json',summary)
                return summary
            summary['implementation_identities']=dict(batch_implementation_locks)
            summary['completed'].append({'path':entry['path'],'result_file':str(result_file),'resumed':True})
            continue
        if failure_file.exists():
            prior=json.loads(failure_file.read_text(encoding='utf-8'))
            if not retry_failed:
                summary['failed'].append({**prior,'retry_skipped':True})
                if prior.get('outcome_unknown'):
                    summary['halted_reason']='previous_api_outcome_unknown'
                    _json(folder/'baseline-summary.json',summary)
                    return summary
                continue
            if prior.get('outcome_unknown') and not acknowledge_unknown:
                raise ValueError('Unknown provider outcome requires explicit acknowledgement before retry')
        try:
            if sha256(Path(entry['path']))!=entry['sha256']:
                raise ValueError('Source changed after experiment was frozen')
            await execute_baseline_image(Path(entry['path']),image_folder,experts,vlm,caps,config,
                                         simulation=simulation,experiment_sha256=experiment_hash,
                                         acknowledge_unknown=acknowledge_unknown,
                                         batch_implementation_locks=batch_implementation_locks)
            summary['implementation_identities']=dict(batch_implementation_locks)
            summary['completed'].append({'path':entry['path'],'result_file':str(result_file),'resumed':False})
        except Exception as error:
            # Avoid response bodies, API keys and user pixels in failure summaries.
            failure={'path':entry['path'],'error_type':type(error).__name__}
            if isinstance(error,GatewayError):
                trace=getattr(vlm,'trace',[])
                last=trace[-1] if trace else {}
                failure['gateway_error_code']=last.get('error_code','gateway_failed')
                failure['outcome_unknown']=isinstance(error,UncertainGatewayOutcome) or last.get('status')=='request_outcome_unknown'
            summary['failed'].append(failure)
            _json(failure_file,failure)
            if isinstance(error,ExpertImplementationDrift):
                summary['halted_reason']='expert_implementation_changed'
                _json(folder/'baseline-summary.json',summary)
                return summary
            # Provider failures require inspection; do not multiply costs or
            # automatically retry a potentially billed, uncertain request.
            if isinstance(error,GatewayError):
                summary['halted_reason']='vlm_gateway_failure'
                _json(folder/'baseline-summary.json',summary)
                return summary
        _json(folder/'baseline-summary.json',summary)
    _json(folder/'baseline-summary.json',summary)
    return summary
