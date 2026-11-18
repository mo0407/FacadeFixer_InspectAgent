"""Reviewed-wall inputs to local model segmentation and Report MCP drafts."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field

from .model_gateway import Strict
from .orchestration import ToolGateway, server_config
from .storage import ArtifactStore, sha256


class VideoDetectionRequest(Strict):
    wall_inputs: str
    project: dict
    detector_profile: str = 'yolo12m'
    segmenter_profile: str = 'mobile_sam'
    prompt_mode: Literal['bbox', 'point', 'text'] = 'bbox'
    threshold: float = Field(default=.25, ge=0, le=1)
    keyframe_inference_size: int | None = Field(default=1024, ge=320, le=1536, multiple_of=32)
    max_frames: int = Field(default=100, ge=1, le=2000)


def box_scope(box, mask):
    """Conservative gate; boundary candidates stay in the audit, not discarded."""
    if not isinstance(box, list) or len(box) != 4 or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in box):
        raise ValueError('Invalid detector bbox')
    x1,y1,x2,y2=box
    h,w=mask.shape
    if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
        raise ValueError('Detector bbox outside original frame')
    region=mask[math.floor(y1):math.ceil(y2),math.floor(x1):math.ceil(x2)]
    return 'inside' if region.all() else 'boundary_review' if region.any() else 'outside'


async def execute_video_detection(request, store, gateway, folder):
    wall_path=store.resolve_input(request.wall_inputs)
    walls=json.loads(wall_path.read_text(encoding='utf-8'))
    if walls.get('schema')!='facade-wall-inputs/v1' or walls.get('status')!='ready_for_detection':
        raise ValueError('Need ready_for_detection reviewed wall-input manifest')
    frames=walls['frames']
    if not frames or len(frames)>request.max_frames:
        raise ValueError('No reviewed frames or frame budget exceeded')
    if len({r['frame_id'] for r in frames})!=len(frames):
        raise ValueError('Repeated frame IDs')
    # Validate artifact integrity before any expensive model calls.
    prepared=[]
    for row in frames:
        paths={key:store.resolve_input(row[key]['uri']) for key in ('original_image','detection_image','wall_mask')}
        with Image.open(paths['wall_mask']) as im:mask=np.asarray(im).copy()
        with Image.open(paths['original_image']) as im:size=im.size
        with Image.open(paths['detection_image']) as im:det_size=im.size
        if mask.shape!=(size[1],size[0]) or size!=det_size or not np.isin(mask,[0,255]).all() or not mask.any():
            raise ValueError('Invalid reviewed wall input dimensions or mask')
        prepared.append((row, paths))
    groups=[]; detections=[]; audit=[]
    prepared.sort(key=lambda entry: not entry[0].get('is_keyframe', False))
    if any(row.get('is_keyframe') for row, _ in prepared) and request.keyframe_inference_size is not None and request.detector_profile not in {'yolo12m','yolo11m'}:
        raise ValueError('Keyframe high-resolution detection requires YOLO or keyframe_inference_size=null')
    for row, paths in prepared:
        with Image.open(paths['wall_mask']) as image_mask:
            mask = np.asarray(image_mask) > 0
        size = request.keyframe_inference_size if row.get('is_keyframe') else None
        detector_args = dict(image_paths=[str(paths['detection_image'])],
                             model_profile=request.detector_profile, threshold=request.threshold)
        if size is not None:
            detector_args['inference_size'] = size
        raw=await gateway.call('facade_models_detect',detector_args)
        if len(raw.get('images',[]))!=1:
            raise ValueError('Detector must return exactly one image')
        image=raw['images'][0]
        if image.get('source_sha256',image.get('sha256'))!=sha256(paths['detection_image']):
            raise ValueError('Detector source hash mismatch')
        accepted=[]
        for item in image.get('defects',[]):
            decision=box_scope(item['bbox_xyxy'],mask)
            audit.append(dict(frame_id=row['frame_id'],decision=decision,defect=item))
            if decision=='inside':accepted.append(item)
        for item in image.get('unmapped_defects',[]):
            audit.append(dict(frame_id=row['frame_id'],decision='unsupported_class_review',defect=item))
        segmented=await gateway.call('facade_models_segment',dict(image_path=str(paths['original_image']),
            defects=accepted,model_profile=request.segmenter_profile,prompt_mode=request.prompt_mode,allow_paid_api=False))
        if segmented.get('source_sha256')!=sha256(paths['original_image']):
            raise ValueError('Segmentation source hash mismatch')
        expected_ids = {item['instance_id'] for item in accepted}
        returned_ids = [item['instance_id'] for item in segmented['instances']]
        if len(returned_ids) != len(accepted) or set(returned_ids) != expected_ids:
            raise ValueError('Segmentation instance IDs differ from accepted detections')
        # One image ID must link detection records to the original-photo masks.
        segmented['image_id']=row['frame_id']
        segmented.setdefault('image', {}).update(video_seconds=row.get('seconds'),
            is_keyframe=row.get('is_keyframe', False), inspection_priority=row.get('inspection_priority', 1),
            detector_inference_size=size)
        groups.append(segmented)
        detections.append(dict(image_id=row['frame_id'],defects=accepted))
        store.write_json(folder/'progress.json',dict(frames_completed=len(groups),total_frames=len(frames),audit=audit))
    store.write_json(folder/'candidate-audit.json',audit)
    project={**request.project,'building_id':walls['building_id'],
             'video_workflow_status':'frame_candidates_pending_cross_frame_review',
             'physical_defect_count_verified':False,
             'scope_candidate_counts':{k:sum(r['decision']==k for r in audit)
                 for k in ('inside','boundary_review','outside','unsupported_class_review')}}
    project.setdefault('name', '视频外立面检测草稿')
    report=await gateway.call('report_build_draft',dict(project=project,
        segmentation_result={'images':groups},detection_result={'images':detections}))
    ref=report['report_document']['uri']
    validation=await gateway.call('report_validate',dict(document_path=ref))
    if not validation.get('schema_valid'):
        raise ValueError('Report schema validation failed')
    rendered=await gateway.call('report_render_html',dict(document_path=ref))
    return dict(status='draft_completed',report=report,rendered=rendered,validation=validation,
        physical_defect_count_verified=False,acceptance_status='pending_review',
        cross_frame_association='not_performed',frames_processed=len(groups),
        candidate_audit=store.artifact(folder/'candidate-audit.json','candidate_audit',folder.name))


async def run_video_detection(request: VideoDetectionRequest, store: ArtifactStore):
    from langchain.mcp import MCPAdapter
    from fastmcp.client.transports import MCPConfigTransport
    job,folder=store.create_job('video_detection_workflow',request.model_dump())
    try:
        config=server_config(store.root,store.allowed_input_roots)
        config['mcpServers']={k:v for k,v in config['mcpServers'].items() if k in ('model','report')}
        async with MCPAdapter(MCPConfigTransport(config,name_as_prefix=False)) as adapter:
            result=await execute_video_detection(request,store,ToolGateway(await adapter.list_tools(),folder/'tool_calls.jsonl'),folder)
        store.write_json(folder/'result.json',result)
        store.complete_job(job,result)
        return {**result,'job_id':job}
    except Exception as error:
        store.fail_job(job,str(error))
        raise
