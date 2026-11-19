"""Bounded, image-only active inspection over immutable pixels and MCP experts."""
from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import re
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field, model_validator

from .model_gateway import ModelConfig, ModelGateway, Strict
from .playbook import Playbook
from .storage import ArtifactStore, sha256


IMAGE_CLASSES = {
    'C': '裂缝/网裂', 'S': '剥落/剥离', 'T': '锈渍', 'R': '锈蚀',
    'D': '污染物', 'V': '植被', 'L': '渗漏/潮湿表征', 'E': '白华', 'B': '鼓包',
}
EXPERT_PROFILES = ('yolo11m', 'yolo12m', 'rtdetr', 'faster_rcnn', 'ensemble', 'intersection')


class ContextRegion(Strict):
    id: str = Field(pattern=r'^env-[A-Za-z0-9_-]{1,40}$')
    bbox_xyxy: tuple[int, int, int, int]
    description: str = Field(min_length=1, max_length=300)


class ImageInspectionRequest(Strict):
    image_path: str = Field(min_length=1)
    scope_mask_path: str | None = None
    detector_profiles: list[str] = Field(default_factory=lambda: ['yolo11m', 'yolo12m'], min_length=1, max_length=6)
    low_threshold: float = Field(default=.1, ge=0, le=1, allow_inf_nan=False)
    high_threshold: float = Field(default=.5, ge=0, le=1, allow_inf_nan=False)
    merge_iou: float = Field(default=.5, gt=0, le=1, allow_inf_nan=False)
    max_inspections: int = Field(default=2, ge=0, le=2)
    grid_rows: int = Field(default=3, ge=1, le=6)
    grid_cols: int = Field(default=3, ge=1, le=6)
    inference_size: int | None = Field(default=960, ge=320, le=1536)
    max_candidates: int = Field(default=200, ge=1, le=1000)
    context_regions: list[ContextRegion] = Field(default_factory=list, max_length=30)
    context_mode: Literal['auto','manual','none'] = 'auto'
    expert_class_filters: dict[str,list[str]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def settings(self):
        if self.low_threshold > self.high_threshold:
            raise ValueError('low_threshold must not exceed high_threshold')
        if len(set(self.detector_profiles)) != len(self.detector_profiles):
            raise ValueError('Detector profiles must be unique')
        if any(not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}',name) for name in self.detector_profiles):
            raise ValueError('Invalid expert profile name')
        if not set(self.expert_class_filters) <= set(self.detector_profiles):
            raise ValueError('Class filters must reference configured experts')
        if any(not codes or len(codes)!=len(set(codes)) or not set(codes)<=set(IMAGE_CLASSES)
               for codes in self.expert_class_filters.values()):
            raise ValueError('Expert filters require unique supported business classes')
        if self.inference_size is not None and self.inference_size % 32:
            raise ValueError('inference_size must be a multiple of 32')
        if len({region.id for region in self.context_regions}) != len(self.context_regions):
            raise ValueError('Context IDs must be unique')
        return self


class InspectionAction(Strict):
    action: Literal['inspect', 'finish']
    anchor_id: str | None = None
    model_profile: str | None = None
    scale: Literal[1, 2, 3] | None = None
    reason: str = Field(min_length=1, max_length=1000)
    used_rule_ids: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def action_fields(self):
        fields = (self.anchor_id, self.model_profile, self.scale)
        if self.action == 'inspect' and any(value is None for value in fields):
            raise ValueError('Inspect requires anchor_id, model_profile and scale')
        if self.action == 'finish' and any(value is not None for value in fields):
            raise ValueError('Finish has no inspection fields')
        return self


class CandidateDecision(Strict):
    candidate_id: str
    decision: Literal['keep', 'reject', 'defer']
    scope_decision: Literal['wall', 'excluded', 'uncertain'] = 'uncertain'
    reason: str = Field(min_length=1, max_length=2000)


class FinalReview(Strict):
    decisions: list[CandidateDecision]
    summary: str = Field(min_length=1, max_length=3000)
    used_rule_ids: list[str] = Field(default_factory=list)


class SceneRegion(Strict):
    bbox_normalized: tuple[float,float,float,float]
    role: Literal['target_wall','window','door','exterior_stair','background','context']
    description: str = Field(min_length=1,max_length=300)

    @model_validator(mode='after')
    def valid_geometry(self):
        x1,y1,x2,y2=self.bbox_normalized
        if not all(math.isfinite(v) for v in self.bbox_normalized) or not (0<=x1<x2<=1 and 0<=y1<y2<=1):
            raise ValueError('Scene regions must use normalized XYXY within [0,1]')
        return self


class SceneContext(Strict):
    regions: list[SceneRegion] = Field(default_factory=list,max_length=30)
    summary: str = Field(min_length=1,max_length=1500)


def _capabilities(request, provided):
    """Server capabilities define eligibility; caller filters can only narrow it."""
    if provided is None:
        if not set(request.detector_profiles)<=set(EXPERT_PROFILES):
            raise ValueError('Registered experts require server capability discovery')
        entries={name:{'class_codes':list(IMAGE_CLASSES),'full_image':True,'local_eligible':True,
                       'ready':True,'identity_verified':False} for name in request.detector_profiles}
    else:
        entries=provided.get('profiles',{})
    selected={}
    for name in request.detector_profiles:
        if name not in entries or not entries[name].get('ready',False):
            raise ValueError('Expert unavailable in server capabilities: '+name)
        item=dict(entries[name])
        codes=set(item.get('class_codes',[]))
        if not codes or not codes<=set(IMAGE_CLASSES):
            raise ValueError('Expert must declare supported business classes: '+name)
        item['declared_class_codes']=sorted(codes)
        if name in request.expert_class_filters:
            codes &= set(request.expert_class_filters[name])
        item['class_codes']=sorted(codes)
        if not codes:
            raise ValueError('Expert filter removed all supported classes: '+name)
        selected[name]=item
    if not any(item.get('full_image',True) for item in selected.values()):
        raise ValueError('At least one expert must support full-image detection')
    return selected


def _eligible(anchor, capabilities, mask):
    if anchor.get('scope_role') in ('window','door','exterior_stair','background'):
        return []
    if mask is not None and _scope(anchor['bbox_xyxy'],mask)['status']=='outside':
        return []
    code=anchor.get('class_code')
    return [name for name,item in capabilities.items() if item.get('local_eligible',True)
            and (code not in IMAGE_CLASSES or code in item['class_codes'])]


def _weight_identity(provenance):
    members=provenance.get('models',[])
    if members:
        return {member.get('profile',str(index)):member['weights_sha256'] for index,member in enumerate(members)
                if member.get('weights_sha256')}
    value=provenance.get('weights_sha256') or provenance.get('checkpoint_sha256')
    if isinstance(value,dict):
        return value
    return {'checkpoint':value} if value else {}


def _implementation_identity(provenance):
    keys=('backend','task','library','library_version','source_files_sha256','implementation_sha256','worker_sha256','source_integration')
    payload={key:provenance[key] for key in keys if key in provenance}
    if provenance.get('models'):
        payload['models']={member.get('profile',str(index)):{key:member[key] for key in keys if key in member}
                           for index,member in enumerate(provenance['models'])}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest() if payload else None


def _definition_identity(capability):
    digest=capability.get('definition_sha256')
    if digest is not None:
        if not isinstance(digest,str) or not re.fullmatch('[a-f0-9]{64}',digest):
            raise ValueError('Invalid expert definition identity')
        return digest
    # Explicitly injected test/demo experts have no server implementation files.
    payload={key:capability.get(key) for key in ('backend','task','class_map','class_codes','full_image','local_eligible')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _lock_expert(profile, provenance, locks, capability, strict, implementation_locks=None):
    identity=_weight_identity(provenance)
    if not identity:
        if strict:
            raise ValueError('Expert checkpoint identity missing: '+profile)
        return
    if any(not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value) for value in identity.values()):
        raise ValueError('Invalid expert checkpoint identity')
    declared=capability.get('weights_sha256') or capability.get('checkpoint_sha256')
    if declared:
        matches=identity==declared if isinstance(declared,dict) else len(identity)==1 and next(iter(identity.values()))==declared
        if not matches:
            raise ValueError('Expert checkpoint changed since capability snapshot: '+profile)
    if profile in locks and locks[profile]!=identity:
        raise ValueError('Expert checkpoint changed during image inspection: '+profile)
    locks[profile]=identity
    definition=capability.get('definition_sha256')
    if definition and provenance.get('definition_sha256')!=definition:
        raise ValueError('Expert checkpoint changed: definition differs from frozen capability: '+profile)
    implementation=_implementation_identity(provenance)
    if strict and not implementation:
        raise ValueError('Expert implementation identity missing: '+profile)
    if implementation_locks is not None and implementation:
        if profile in implementation_locks and implementation_locks[profile]!=implementation:
            raise ValueError('Expert checkpoint changed: implementation drifted during inspection: '+profile)
        implementation_locks[profile]=implementation


def _candidate_anchor(candidate, request, source='expert'):
    return {'id':candidate['observation_id'], 'kind':'Box' if candidate['confidence'] is not None
            and candidate['confidence']>=request.high_threshold else 'Ghost',
            'bbox_xyxy':candidate['bbox_xyxy'],'class_code':candidate['class_code'],
            'confidence':candidate['confidence'],'source':source}


async def _environment(request, source, mask, vlm, p0, folder):
    regions=[]
    if request.context_mode=='none':
        return regions
    if request.context_mode=='auto':
        if mask is not None:
            import cv2
            count,_,stats,_=cv2.connectedComponentsWithStats((mask>0).astype('uint8'),connectivity=8)
            components=sorted(stats[1:],key=lambda item:int(item[4]),reverse=True)[:20]
            for index,item in enumerate(components):
                x,y,w,h,area=map(int,item)
                if area:
                    regions.append({'id':f'env-scope-{index+1:02d}','kind':'Env','bbox_xyxy':[x,y,x+w,y+h],
                        'scope_role':'target_wall','description':'Wall component from the provided scope mask.',
                        'source':'scope_mask'})
            summary='Wall context extracted from authoritative scope-mask components.'
        else:
            path=folder/'scene-input.png'
            preview=source.copy(); preview.thumbnail((1280,1280)); preview.save(path)
            task={'stage':'visual_anchoring','p0':p0,'original_size':list(source.size),
                  'instruction':'Identify visible target-building wall context and relevant windows, doors, exterior stairs '
                    'and background buildings. Return normalized XYXY regions. Regions are context cues, not pixel masks '
                    'or confirmed defects. If target building is ambiguous, use role context and explain uncertainty.'}
            context=await vlm.invoke('generator',task,SceneContext,image_paths=[path],mock_output={
                'regions':[{'bbox_normalized':[0,0,1,1],'role':'context',
                            'description':'Mock full-image reference; no semantic wall judgment.'}],
                'summary':'Mock context only; semantic scene recognition has not been performed.'})
            for index,item in enumerate(context.regions):
                x1,y1,x2,y2=item.bbox_normalized
                box=[math.floor(x1*source.width),math.floor(y1*source.height),math.ceil(x2*source.width),math.ceil(y2*source.height)]
                regions.append({'id':f'env-auto-{index+1:02d}','kind':'Env','bbox_xyxy':box,
                                'scope_role':item.role,'description':item.description,'source':'vlm_scene_context'})
            summary=context.summary
        ArtifactStore.write_json(folder/'scene_context.json',{'summary':summary,'regions':regions,'is_scope_mask':False})
    for region in request.context_regions:
        if any(item['id']==region.id for item in regions):
            raise ValueError('Manual context ID collides with automatic context')
        regions.append({'id':region.id,'kind':'Env','bbox_xyxy':validate_box(region.bbox_xyxy,*source.size),
                        'description':region.description,'source':'manual_context'})
    return regions


def validate_box(box, width, height):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
        raise ValueError('Expected four finite original-pixel box coordinates')
    x1, y1, x2, y2 = box
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError('Box outside source raster or empty')
    return list(box)


def crop_box(box, scale, width, height):
    """Scale expands context, never changes pixels before the expert sees them."""
    x1, y1, x2, y2 = validate_box(box, width, height)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    dx, dy = (x2-x1)*scale/2, (y2-y1)*scale/2
    return [max(0, math.floor(cx-dx)), max(0, math.floor(cy-dy)),
            min(width, math.ceil(cx+dx)), min(height, math.ceil(cy+dy))]


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap/union if union else 0


def merge_candidates(candidates, threshold):
    """Deterministic greedy same-class NMS; retain every contributing observation."""
    merged = []
    for candidate in sorted(candidates, key=lambda c: (-(c['confidence'] if c['confidence'] is not None else -1), c['observation_id'])):
        match = next((m for m in merged if candidate['class_code'] != 'unknown' and m['class_code'] == candidate['class_code']
                      and iou(m['bbox_xyxy'], candidate['bbox_xyxy']) >= threshold), None)
        if match is None:
            merged.append({**candidate, 'candidate_id': f'candidate-{len(merged)+1:04d}', 'members': [candidate]})
        else:
            match['members'].append(candidate)
    return merged


def _canonical_code(row):
    # New image-workflow boundary; legacy model checkpoint IDs remain unchanged.
    code = row.get('class_code')
    if code == 'M':
        return 'C'
    if code == 'P':
        return 'S'
    if code == 'L' and any(term in str(row.get('class_name', '')).lower() for term in ('efflorescence', '白华', '白華', '泛碱')):
        return 'E'
    return code if code in IMAGE_CLASSES else 'unknown'


def _scope(box, mask):
    if mask is None:
        return {'status': 'not_reviewed', 'wall_fraction': None}
    x1, y1, x2, y2 = [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])]
    region = mask[y1:y2, x1:x2]
    fraction = float(np.mean(region > 0))
    return {'status': 'inside' if fraction == 1 else 'outside' if fraction == 0 else 'partial', 'wall_fraction': fraction}


def _rules(ids, playbook):
    if len(ids) != len(set(ids)) or not set(ids) <= {r.id for r in playbook.rules}:
        raise ValueError('VLM cites unknown or duplicate P0 rule IDs')


def _overview(source, anchors, destination):
    image = source.copy().convert('RGB')
    image.thumbnail((1280, 1280))
    sx, sy = image.width/source.width, image.height/source.height
    draw = ImageDraw.Draw(image)
    colors = {'Box': '#1677ff', 'Ghost': '#c27b20', 'Env': '#7e57c2', 'Grid': '#8894a7'}
    for anchor in anchors:
        x1, y1, x2, y2 = anchor['bbox_xyxy']
        rect = [x1*sx, y1*sy, x2*sx, y2*sy]
        color = colors.get(anchor['kind'], '#ed672b')
        draw.rectangle(rect, outline=color, width=2)
        draw.text((rect[0]+2, rect[1]+2), anchor['id'], fill=color, stroke_width=1, stroke_fill='white')
    image.save(destination)
    return {'width': image.width, 'height': image.height, 'scale_x': sx, 'scale_y': sy}


async def _detect(gateway, path, profile, request, origin, source_shape, prefix, scope_mask, folder, source_hash,
                  *, capabilities=None, locks=None, strict_identity=False, implementation_locks=None):
    arguments = {'image_paths': [str(path)], 'model_profile': profile, 'threshold': request.low_threshold}
    backend=(capabilities or {}).get(profile,{}).get('backend')
    if request.inference_size is not None and (profile in ('yolo11m', 'yolo12m') or backend in ('ultralytics','yolov9')):
        arguments['inference_size'] = request.inference_size
    response = await gateway.call('facade_models_detect', arguments)
    ArtifactStore.write_json(folder / f'{prefix}-expert.json', response)
    images = response.get('images', [])
    if response.get('status') != 'completed' or len(images) != 1:
        raise ValueError('Expert did not return one completed image result')
    if (response.get('coordinate_space','original_pixels') != 'original_pixels'
            or response.get('bbox_convention','zero_based_xyxy_max_exclusive') != 'zero_based_xyxy_max_exclusive'):
        raise ValueError('Expert must return original-pixel XYXY boxes')
    image = images[0]
    with Image.open(path) as crop:
        width, height = crop.size
    if image.get('source_sha256') != sha256(path) or (image.get('width'), image.get('height')) != (width, height):
        raise ValueError('Expert result source hash/dimensions mismatch')
    provenance=image.get('model_provenance',{})
    if locks is not None:
        _lock_expert(profile,provenance,locks,(capabilities or {}).get(profile,{}),strict_identity,implementation_locks)
        ArtifactStore.write_json(folder/'expert-lock.json',{'identities':locks,'implementations':implementation_locks,'strict':strict_identity})
    rows = image.get('defects', []) + image.get('unmapped_defects', [])
    candidates = []
    for index, row in enumerate(rows):
        box = validate_box(row.get('bbox_xyxy'), width, height)
        mapped = [box[0]+origin[0], box[1]+origin[1], box[2]+origin[0], box[3]+origin[1]]
        validate_box(mapped, *source_shape)
        score = row.get('confidence')
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Invalid expert confidence')
        code = _canonical_code(row)
        if capabilities is not None and code in IMAGE_CLASSES and code not in capabilities[profile]['declared_class_codes']:
            raise ValueError('Expert returned an undeclared business class: '+profile)
        candidates.append({'observation_id': f'{prefix}-{index+1:04d}', 'class_code': code,
            'class_name': IMAGE_CLASSES.get(code, '待映射类别'), 'source_class_code': row.get('original_class_code', row.get('class_code')),
            'source_class_name': row.get('original_class_name', row.get('class_name')), 'source_instance_id': row.get('instance_id'),
            'bbox_xyxy': mapped, 'local_bbox_xyxy': box, 'crop_origin_xy': list(origin),
            'confidence': score, 'model_profile': profile, 'evidence_id': prefix,
            'evidence_image_sha256': sha256(path), 'source_sha256': source_hash,
            'scope': _scope(mapped, scope_mask),
            'expert_mask_evidence':row.get('mask_artifact'),
            'model_identity':_weight_identity(provenance)})
    return candidates


def _html_report(folder, result):
    esc = lambda value: html.escape(str(value))
    labels = {'keep': '保留候选', 'reject': '已排除', 'defer': '待复核'}
    cards = []
    with Image.open(folder/'source.png') as source:
        source = source.convert('RGB')
        for candidate in result['candidates']:
            cid = candidate['candidate_id']
            image = source.copy()
            draw = ImageDraw.Draw(image)
            draw.rectangle(candidate['bbox_xyxy'], outline='#ed672b', width=max(2, image.width//400))
            draw.text(tuple(candidate['bbox_xyxy'][:2]), cid, fill='#ed672b', stroke_width=1, stroke_fill='white')
            image.thumbnail((1100,1100))
            image.save(folder/f'{cid}.jpg', quality=90)
            box = crop_box(candidate['bbox_xyxy'], 2, source.width, source.height)
            source.crop(box).save(folder/f'{cid}-detail.png')
            cards.append(f'<article id="{cid}"><h3>{cid} · {esc(candidate["class_name"])} · {labels[candidate["decision"]]}</h3>'
                f'<div class="pair"><figure><img src="{cid}.jpg"><figcaption>原图定位</figcaption></figure>'
                f'<figure><img src="{cid}-detail.png"><figcaption>原像素局部证据（2倍上下文）</figcaption></figure></div>'
                f'<p>{esc(candidate["reason"])}</p><p>原图 bbox：{esc(candidate["bbox_xyxy"])}；范围：{esc(candidate["scope"]["status"])}；'
                f'来源：{esc(", ".join(m["observation_id"] for m in candidate["members"]))}</p></article>')
    steps = ''.join(f'<li>{esc(s["action"])} · {esc(s.get("anchor_id"))} · {esc(s["reason"])}</li>' for s in result['trajectory'])
    mode = '演示 / mock 复核 · 不代表真实检测效果' if result['demo'] else '模型候选 · 待人工复核'
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>图像主动检查</title>
<style>body{{font:16px/1.6 "Microsoft YaHei",sans-serif;background:#f4f6fa;color:#17243b;max-width:1200px;margin:32px auto;padding:0 20px}}article,section{{background:white;border:1px solid #dfe4ee;border-radius:12px;padding:24px;margin:20px 0}}h1,h2,h3{{line-height:1.35}}img{{max-width:100%;max-height:680px;object-fit:contain}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:0}}figcaption,small{{color:#64748b}}a{{color:#2563eb}}@media(max-width:700px){{.pair{{display:block}}}}@media print{{article{{break-inside:avoid}}}}</style>
<h1>图像主动检查结果</h1><p>{mode}。本报告不提供未经真值验证的准确率或结构安全结论。</p>
<section><h2>全图锚点 → 局部检查 → 最终核验</h2><img src="overview.png"><p>Box 高分框 / Ghost 低分框 / Env 上下文 / Grid 搜索区域</p>
<p>检查次数 {result['inspections_used']} / {result['request']['max_inspections']}；P0 {esc(result['playbook_version'])}；规则摘要 {esc(result['playbook_sha256'][:16])}</p>
<p>{esc(result['summary'])}</p><p>保留 {result['counts']['keep']} · 排除 {result['counts']['reject']} · 待定 {result['counts']['defer']}</p></section>
<section><h2>检查轨迹</h2><ol>{steps}</ol><p><a href="result.json">完整原图坐标结果</a> · <a href="model_trace.json">模型与调用记录</a> · <a href="playbook.json">本轮冻结规则</a> · <a href="failure_summary.json">本轮问题与失败记录</a></p></section>
{''.join(cards) or '<section>本轮没有候选框。零候选不等于无缺陷，可复查原图覆盖和模型能力。</section>'}</html>'''
    (folder/'report.html').write_text(text, encoding='utf-8')


async def execute_image_inspection(request: ImageInspectionRequest, store: ArtifactStore, experts,
                                   vlm: ModelGateway, playbook: Playbook, folder: Path, *, demo=False, expert_capabilities=None):
    """One still image; no video extraction, temporal fusion, or online rule changes."""
    from .image_failures import progress
    progress(folder,'preflight',playbook_version=playbook.version)
    capabilities=_capabilities(request,expert_capabilities)
    strict_identity=expert_capabilities is not None
    locks={}
    implementation_locks={}
    store.write_json(folder/'expert_capabilities.json',{'profiles':capabilities,'discovered':strict_identity})
    # Fail before expensive expert inference if the VLM connection cannot be used.
    if hasattr(vlm, 'config'):
        connection = vlm.config.connections[vlm.config.roles['generator']]
        if not connection.supports_images:
            raise ValueError('Generator must explicitly support image inputs')
        if connection.adapter != 'mock' and not (vlm.config.api_enabled and vlm.allow_api):
            raise ValueError('Remote image API requires api_enabled and --allow-api')
        demo = demo or connection.adapter == 'mock'
    progress(folder,'input_snapshot',playbook_version=playbook.version)
    original = store.resolve_input(request.image_path)
    original_hash = sha256(original)
    source_bytes = folder / ('input'+original.suffix.lower())
    shutil.copyfile(original, source_bytes)
    if sha256(source_bytes) != original_hash:
        raise ValueError('Source changed while making snapshot')
    with Image.open(source_bytes) as image:
        if image.format not in ('JPEG','PNG') or getattr(image,'n_frames',1) != 1:
            raise ValueError('Supply a still PNG/JPEG image')
        if image.width*image.height > 80_000_000:
            raise ValueError('Image exceeds 80 million pixels')
        # Stored raster coordinates are kept; EXIF is deliberately not auto-rotated.
        orientation = image.getexif().get(274)
        source = image.convert('RGB')
    width, height = source.size
    if width < request.grid_cols or height < request.grid_rows:
        raise ValueError('Image smaller than search grid')
    source.save(folder/'source.png')
    source_hash = sha256(folder/'source.png')
    mask = None
    scope_hash = None
    if request.scope_mask_path:
        progress(folder,'scope_snapshot',source_sha256=source_hash)
        scope_path = store.resolve_input(request.scope_mask_path)
        scope_hash = sha256(scope_path)
        with Image.open(scope_path) as scope:
            if scope.format != 'PNG' or scope.size != source.size or scope.mode not in ('1','L'):
                raise ValueError('Scope mask must be original-sized binary grayscale PNG')
            mask = np.asarray(scope.convert('L')).copy()
        if not set(np.unique(mask)) <= {0,255} or not np.any(mask):
            raise ValueError('Scope mask must have nonempty 255 wall pixels and only 0/255 values')
        Image.fromarray(mask).save(folder/'scope.png')
        if sha256(scope_path) != scope_hash:
            raise ValueError('Scope input changed during snapshot')
    frozen = playbook.model_copy(deep=True)
    p0 = frozen.model_dump(mode='json')
    p0_hash = hashlib.sha256(frozen.model_dump_json().encode()).hexdigest()
    store.write_json(folder/'playbook.json', p0)
    store.write_json(folder/'request.json', request.model_dump(mode='json'))
    candidates, anchors, trajectory, inspections = [], [], [], []
    for index, profile in enumerate(request.detector_profiles):
        if not capabilities[profile].get('full_image',True):
            continue
        progress(folder,'full_detection',model_profile=profile,source_sha256=source_hash,step=index+1)
        found = await _detect(experts, folder/'source.png', profile, request, (0,0), source.size,
                              f'full-{index+1:02d}', mask, folder, source_hash,
                              capabilities=capabilities,locks=locks,strict_identity=strict_identity,implementation_locks=implementation_locks)
        candidates.extend(found)
    progress(folder,'anchor_preparation',source_sha256=source_hash)
    if len(candidates) > request.max_candidates:
        raise ValueError('Candidate budget exceeded; nothing silently discarded')
    for candidate in candidates:
        anchors.append(_candidate_anchor(candidate,request))
    progress(folder,'visual_anchoring',source_sha256=source_hash)
    anchors.extend(await _environment(request,source,mask,vlm,p0,folder))
    for row in range(request.grid_rows):
        for col in range(request.grid_cols):
            anchors.append({'id': f'grid-{row+1}-{col+1}', 'kind': 'Grid',
                'bbox_xyxy': [col*width//request.grid_cols, row*height//request.grid_rows,
                              (col+1)*width//request.grid_cols, (row+1)*height//request.grid_rows]})
    for anchor in anchors:
        anchor['eligible_profiles']=_eligible(anchor,capabilities,mask)
    display = _overview(source, anchors, folder/'overview.png')
    shutil.copyfile(folder/'overview.png',folder/'overview-00.png')
    store.write_json(folder/'anchors.json', {'coordinate_space':'original_pixels','display':display,'anchors':anchors})
    store.write_json(folder/'anchors-initial.json',anchors)
    by_id = {anchor['id']:anchor for anchor in anchors}
    evidence_paths = [folder/'overview-00.png']
    common = {'workflow': 'image-active-inspection/v1', 'p0': p0,
        'original_size': [width,height], 'coordinate_space':'zero_based_xyxy_max_exclusive',
        'scope_policy': 'Only main-building wall surfaces; exclude doors, windows, exterior stairs and background buildings. '
                        'A provided scope mask is authoritative; without it scope remains unverified.',
        'class_schema': IMAGE_CLASSES, 'allowed_profiles': request.detector_profiles,
        'scale_meaning': '1,2,3 expand anchor context; crops use original pixels without resizing.',
        'anchors': anchors, 'overview_display': display,'expert_capabilities':capabilities}
    for step in range(request.max_inspections):
        task = {**common, 'stage':'choose_inspection', 'remaining_inspections': request.max_inspections-step,
            'history': trajectory, 'candidates': candidates,
            'images': [evidence_paths[0].name]+[i['annotated_crop'] for i in inspections],
            'instruction': 'Choose one inspect action by existing anchor ID and allowed expert profile, or finish. '
                           'Inspect unexplored grid areas too when experts may have missed defects. '
                           'Return only the action schema; do not invent new tools, coordinates, or rules.'}
        grids = [a['id'] for a in anchors if a['kind'] == 'Grid' and a['eligible_profiles']]
        if step >= len(grids):
            fixture = {'action':'finish','reason':'Mock exhausted available unique grid actions.','used_rule_ids':[]}
        else:
            fixture = {'action':'inspect','anchor_id':grids[step],
                       'model_profile':by_id[grids[step]]['eligible_profiles'][0], 'scale':1,
                       'reason':'Mock demonstration of a bounded grid inspection; not a model judgment.', 'used_rule_ids':[]}
        progress(folder,'choose_inspection',step=step+1,source_sha256=source_hash)
        action = await vlm.invoke('generator', task, InspectionAction, mock_output=fixture, image_paths=evidence_paths)
        progress(folder,'inspection_action_validation',step=step+1,source_sha256=source_hash)
        _rules(action.used_rule_ids, frozen)
        row = action.model_dump(mode='json')
        trajectory.append(row)
        store.write_json(folder/'trajectory.json', trajectory)
        if action.action == 'finish':
            break
        if action.anchor_id not in by_id or action.model_profile not in request.detector_profiles:
            raise ValueError('Invalid anchor or expert profile from VLM')
        if action.model_profile not in by_id[action.anchor_id]['eligible_profiles']:
            raise ValueError('Expert is not eligible for this anchor class or scope')
        signature = (action.anchor_id, action.model_profile, action.scale)
        if any(tuple(i['signature']) == signature for i in inspections):
            raise ValueError('Duplicate inspection action')
        progress(folder,'local_crop',step=step+1,model_profile=action.model_profile,source_sha256=source_hash)
        bounds = crop_box(by_id[action.anchor_id]['bbox_xyxy'], action.scale, width, height)
        prefix = f'local-{len(inspections)+1:02d}'
        crop_path = folder/f'{prefix}.png'
        crop = source.crop(bounds)
        crop.save(crop_path)
        progress(folder,'local_detection',step=step+1,model_profile=action.model_profile,source_sha256=source_hash)
        found = await _detect(experts,crop_path,action.model_profile,request,bounds[:2],source.size,
                              prefix,mask,folder,source_hash,capabilities=capabilities,locks=locks,strict_identity=strict_identity,
                              implementation_locks=implementation_locks)
        candidates.extend(found)
        for candidate in found:
            anchor=_candidate_anchor(candidate,request,source=prefix)
            anchor['eligible_profiles']=_eligible(anchor,capabilities,mask)
            anchors.append(anchor)
            by_id[anchor['id']]=anchor
        display=_overview(source,anchors,folder/'overview.png')
        snapshot=folder/f'overview-{len(inspections)+1:02d}.png'
        shutil.copyfile(folder/'overview.png',snapshot)
        evidence_paths[0]=snapshot
        common['overview_display']=display
        store.write_json(folder/'anchors.json',{'coordinate_space':'original_pixels','display':display,'anchors':anchors})
        store.write_json(folder/f'anchors-after-{len(inspections)+1:02d}.json',anchors)
        if len(candidates) > request.max_candidates:
            raise ValueError('Candidate budget exceeded; nothing silently discarded')
        annotated = folder/f'{prefix}-annotated.png'
        _overview(crop,[{'id':c['observation_id'],'kind':'Box','bbox_xyxy':c['local_bbox_xyxy']} for c in found],annotated)
        inspections.append({'evidence_id':prefix,'signature':list(signature),'anchor_id':action.anchor_id,
            'crop_bbox_original':bounds,'crop_sha256':sha256(crop_path),'model_profile':action.model_profile,
            'annotated_crop':annotated.name,'candidate_ids':[c['observation_id'] for c in found]})
        evidence_paths.append(annotated)
        # Persist each observation before the next VLM call, including mapped coordinates.
        store.write_json(folder/'inspections.json',inspections)
        store.write_json(folder/'observations.json',candidates)
    progress(folder,'final_evidence',source_sha256=source_hash)
    merged = merge_candidates(candidates, request.merge_iou)
    final_display = _overview(source,[{'id':m['candidate_id'],'kind':'Box','bbox_xyxy':m['bbox_xyxy']} for m in merged],folder/'final-overview.png')
    # Final review supplies each candidate's own pixel evidence, not only a tiny
    # rectangle in a thumbnail. These crops do not rerun experts or consume the
    # two active-inspection actions. Four candidates per request keeps inputs bounded.
    review_evidence = {}
    for candidate in merged:
        cid = candidate['candidate_id']
        bounds = crop_box(candidate['bbox_xyxy'],2,width,height)
        detail = source.crop(bounds)
        native_size = detail.size
        detail.thumbnail((2048,2048))
        draw = ImageDraw.Draw(detail)
        sx,sy = detail.width/native_size[0],detail.height/native_size[1]
        box = candidate['bbox_xyxy']
        draw.rectangle([(box[0]-bounds[0])*sx,(box[1]-bounds[1])*sy,
                        (box[2]-bounds[0])*sx,(box[3]-bounds[1])*sy],outline='#ed672b',width=2)
        draw.text((3,3),cid,fill='#ed672b',stroke_width=1,stroke_fill='white')
        detail.save(folder/f'{cid}-evidence.png')
        review_evidence[cid] = {'file':f'{cid}-evidence.png','crop_bbox_original':bounds,
                               'native_size':list(native_size),'display_size':list(detail.size),
                               'native_resolution':detail.size == native_size}
    decisions, summaries, final_rule_ids = {}, [], []
    batches = [merged[i:i+4] for i in range(0,len(merged),4)] or [[]]
    for batch_index,batch in enumerate(batches):
        files = [folder/'final-overview.png']+[folder/review_evidence[m['candidate_id']]['file'] for m in batch]+evidence_paths[1:]
        task = {**common,'stage':'final_verification','batch':batch_index+1,'batch_count':len(batches),
            'remaining_inspections':0,'history':trajectory,'inspections':inspections,'candidates':batch,
            'candidate_display':final_display,'candidate_evidence':{m['candidate_id']:review_evidence[m['candidate_id']] for m in batch},
            'images':[p.name for p in files],
            'instruction':'Review ONLY the supplied batch: exactly one keep/reject/defer per candidate, with evidence-based reasons. '
                          'Independently set scope_decision to wall/excluded/uncertain; wall means the primary building wall '
                          'rather than doors, windows, exterior stairs or background. Respect a provided wall scope mask. '
                          'Do not change bbox coordinates or class labels. Keep only clearly supported in-scope wall defects. '
                          'Defer unknown classes, ambiguous defects, downsampled candidate evidence, or uncertain scope. '
                          'Empty candidates requires empty decisions. File order matches attached images.'}
        fixture = {'summary':'Mock: candidates need independent visual and scope review.','used_rule_ids':[],
                   'decisions':[{'candidate_id':m['candidate_id'],'decision':'defer','scope_decision':'uncertain',
                                 'reason':'Synthetic/mock policy; no visual verification.'} for m in batch]}
        progress(folder,'final_verification',batch=batch_index+1,source_sha256=source_hash)
        review = await vlm.invoke('generator',task,FinalReview,mock_output=fixture,image_paths=files)
        progress(folder,'final_review_validation',batch=batch_index+1,source_sha256=source_hash)
        _rules(review.used_rule_ids,frozen)
        batch_decisions = {d.candidate_id:d for d in review.decisions}
        if len(batch_decisions) != len(review.decisions) or set(batch_decisions) != {m['candidate_id'] for m in batch}:
            raise ValueError('Final review must decide every batch candidate exactly once, with no invented IDs')
        store.write_json(folder/f'final-review-{batch_index+1:03d}.json',review.model_dump(mode='json'))
        decisions.update(batch_decisions)
        summaries.append(review.summary)
        final_rule_ids.extend(review.used_rule_ids)
    for candidate in merged:
        decision = decisions[candidate['candidate_id']]
        candidate.update(decision=decision.decision, reason=decision.reason,visual_scope_decision=decision.scope_decision,
                         review_evidence=review_evidence[candidate['candidate_id']])
        if decision.decision == 'keep' and (candidate['class_code'] == 'unknown'
                or candidate['scope']['status'] in ('partial','outside') or decision.scope_decision != 'wall'
                or not candidate['review_evidence']['native_resolution']):
            candidate.update(decision='defer', proposed_decision='keep',
                reason='Harness scope/class guard: '+decision.reason)
    progress(folder,'integrity_check',source_sha256=source_hash)
    if sha256(original) != original_hash or sha256(source_bytes) != original_hash or sha256(folder/'source.png') != source_hash:
        raise ValueError('Source pixels changed during inspection')
    if hashlib.sha256(frozen.model_dump_json().encode()).hexdigest() != p0_hash:
        raise ValueError('Frozen P0 changed during inspection')
    result = {'schema':'image-active-inspection/v1','status':'draft_completed','demo':demo,
        'request':request.model_dump(mode='json'),'original_input_sha256':original_hash,'source_sha256':source_hash,
        'source_size':[width,height],'exif_orientation':orientation,'exif_rotation_applied':False,
        'coordinate_space':'original_pixels','bbox_convention':'zero_based_xyxy_max_exclusive',
        'scope_mask_sha256':scope_hash,'scope_verified_by_mask':mask is not None,
        'expert_identities':locks,'expert_identity_verified':strict_identity,
        'expert_config_identities':{name:item.get('weights_sha256') for name,item in capabilities.items()},
        'expert_definition_identities':{name:_definition_identity(item) for name,item in capabilities.items()},
        'expert_implementation_identities':implementation_locks,
        'vlm_model_identities':getattr(vlm,'model_identities',{}),
        'playbook_version':frozen.version,'playbook_sha256':p0_hash,'playbook_updated':False,
        'inspections_used':len(inspections),'trajectory':trajectory,'inspections':inspections,
        'merge_method':'greedy same-class IoU; highest-score representative; all member observations retained',
        'candidates':merged,'summary':'\n'.join(summaries),'used_rule_ids':sorted(set(final_rule_ids+[r for a in trajectory for r in a['used_rule_ids']])),
        'counts':{name:sum(c['decision']==name for c in merged) for name in ('keep','reject','defer')},
        'acceptance_status':'pending_review','expert_validated':False,'metrics':None,
        'report_path':str(folder/'report.html')}
    progress(folder,'report_generation',source_sha256=source_hash,playbook_version=frozen.version)
    store.write_json(folder/'observations.json',candidates)
    store.write_json(folder/'trajectory.json',trajectory)
    store.write_json(folder/'inspections.json',inspections)
    store.write_json(folder/'result.json',result)
    _html_report(folder,result)
    return result


async def run_image_inspection(request, store, settings: ModelConfig, playbook=None, *, allow_api=False, experts=None):
    """Real runs use MCP stdio experts; tests/demo can inject deterministic experts."""
    from .orchestration import ToolGateway, server_config
    from .image_failures import progress, record_observations, record_runtime_failure
    from .rule_lifecycle import RuleLifecycle
    job,folder = store.create_job('image_inspection',request.model_dump(mode='json'))
    vlm = ModelGateway(settings,allow_api=allow_api)
    # Contains model parameters and environment VARIABLE NAMES, never credentials.
    try:
        progress(folder,'playbook_load')
        playbook = playbook or RuleLifecycle(store).latest() or Playbook(version='P0-empty')
        store.write_json(folder/'model-config.json',settings.model_dump(mode='json'))
        store.write_json(folder/'request.json',request.model_dump(mode='json'))
        if experts is None:
            progress(folder,'expert_startup',playbook_version=playbook.version)
            from fastmcp.client.transports import MCPConfigTransport
            from langchain.mcp import MCPAdapter
            config = server_config(store.root,store.allowed_input_roots)
            config['mcpServers'] = {'model':config['mcpServers']['model']}
            async with MCPAdapter(MCPConfigTransport(config,name_as_prefix=False)) as adapter:
                gateway=ToolGateway(await adapter.list_tools(),folder/'tool_calls.jsonl')
                discovered=await gateway.call('facade_models_capabilities',{})
                result = await execute_image_inspection(request,store,gateway,vlm,playbook,folder,expert_capabilities=discovered)
        else:
            result = await execute_image_inspection(request,store,experts,vlm,playbook,folder,demo=True)
        store.write_json(folder/'model_trace.json',vlm.trace)
        progress(folder,'observation_recording',source_sha256=result['source_sha256'],playbook_version=playbook.version)
        record_observations(store,folder,result)
        # The latest summary changes after feedback. Registered artifacts remain
        # immutable snapshots so their existing content hashes stay valid.
        shutil.copyfile(folder/'failure_summary.json',folder/'failure_summary-at-completion.json')
        result['failure_summary_path'] = str(folder/'failure_summary.json')
        store.write_json(folder/'result.json',result)
        result['artifacts'] = {name:store.artifact(folder/file, name,job) for name,file in
            [('report','report.html'),('result','result.json'),('anchors','anchors.json'),('trace','model_trace.json'),
             ('playbook','playbook.json'),('model_config','model-config.json'),('failure_summary','failure_summary-at-completion.json')]}
        progress(folder,'completed',source_sha256=result['source_sha256'],playbook_version=playbook.version)
        store.complete_job(job,result)
        return {**result,'job_id':job}
    except Exception as error:
        try:
            store.write_json(folder/'model_trace.json',vlm.trace)
        except Exception:
            error.add_note('Model trace snapshot could not be written.')
        job_error={'type':type(error).__name__,'message':'Image inspection failed; failure ledger could not be updated.',
                   'failure_recorded':False}
        try:
            failure = record_runtime_failure(store,folder,error,vlm.trace)
            job_error.update(reason_code=failure['reason_code'],message=failure['summary'],stage=failure['stage'],
                             failure_summary_path=failure['failure_summary_path'],failure_recorded=True,
                             failure_snapshots_written=failure['snapshots_written'])
        except Exception as recording_error:
            job_error['recording_error_type']=type(recording_error).__name__
            error.add_note('Failure memory could not be updated; inspect local job status.')
        try:
            store.fail_job(job,job_error)
        except Exception:
            error.add_note('Failed job status could not be written; the original exception is preserved.')
        raise
