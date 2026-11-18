import asyncio

import pytest

from facade_mcp.report_service import ReportService
from facade_mcp.storage import sha256
from facade_mcp.video_detection_workflow import VideoDetectionRequest, execute_video_detection
from test_video_preparation import reviewed_scope, setup_video


@pytest.mark.parametrize('mismatched_mask_id', [False, True])
def test_video_detection_retains_original_and_audits_boundary(tmp_path, mismatched_mask_id):
    service, video = setup_video(tmp_path)
    scope, _ = reviewed_scope(service, video, tmp_path, keyframe_ranges=[[0, 1]])
    walls = service.apply_scope(str(scope))
    store = service.store
    report_service = ReportService(store)
    job, folder = store.create_job('video_fixture', {})
    row = walls['frames'][0]
    box = dict(instance_id='crack', class_code='C', bbox_xyxy=[10, 10, 12, 16])
    boundary = dict(box, instance_id='boundary', bbox_xyxy=[7, 10, 12, 16])

    class Gateway:
        async def call(self, name, args):
            if name == 'facade_models_detect':
                assert args['image_paths'] == [row['detection_image']['path']]
                assert args['inference_size'] == 1024
                return {'images': [dict(source_sha256=row['detection_image']['sha256'],
                                        defects=[box, boundary])]}
            if name == 'facade_models_segment':
                assert args['image_path'] == row['original_image']['path']
                assert args['defects'] == [box]
                instance = dict(box, instance_id='wrong' if mismatched_mask_id else 'crack',
                                mask=row['wall_mask'], area_pixels=256)
                return dict(source_sha256=sha256(store.resolve_input(row['original_image']['uri'])),
                            original_image=row['original_image'], width=32, height=32, instances=[instance])
            handlers = {'report_build_draft': report_service.build_draft,
                        'report_validate': report_service.validate,
                        'report_render_html': report_service.render}
            return handlers[name](**args)

    request = VideoDetectionRequest(wall_inputs=walls['manifest']['uri'], project={})
    task = execute_video_detection(request, store, Gateway(), folder)
    if mismatched_mask_id:
        with pytest.raises(ValueError, match='instance IDs'):
            asyncio.run(task)
        return
    result = asyncio.run(task)
    assert result['validation']['schema_valid'] and result['acceptance_status'] == 'pending_review'
    html = store.resolve_input(result['rendered']['html']['uri']).read_text(encoding='utf-8')
    assert '待跨帧归并' in html and 'data:image/png;base64,' in html
    audit = store.resolve_input(result['candidate_audit']['uri']).read_text(encoding='utf-8')
    assert 'boundary_review' in audit
