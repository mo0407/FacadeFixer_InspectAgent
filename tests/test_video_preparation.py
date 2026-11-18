import json

import cv2
import numpy as np
import pytest
from PIL import Image

from facade_mcp.storage import ArtifactStore
from facade_mcp.video_service import VideoService


def setup_video(tmp_path, count=3, fps=1):
    path = tmp_path / 'fixture.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (32,32))
    assert writer.isOpened()
    for i in range(count):
        writer.write(np.full((32,32,3), 60+i%150, np.uint8))
    writer.release()
    store = ArtifactStore(tmp_path / 'work', [tmp_path])
    return VideoService(store), path


def test_batches_and_pending_scope(tmp_path):
    service, path = setup_video(tmp_path,201)
    result = service.prepare(str(path),'building-A',max_frames=201)
    assert result['status']=='completed'
    assert result['frame_count']==201 and len(result['batch_manifests'])==2
    assert result['workflow_status']=='pending_scope_review'
    refused=service.apply_scope(result['scope_review']['uri'])
    assert refused['status']=='failed'
    assert 'Every frame' in refused['error']


def test_budget_before_extraction(tmp_path):
    service,path=setup_video(tmp_path)
    result=service.prepare(str(path),'A',max_frames=2)
    assert result['status']=='failed' and 'max_frames' in result['error']
    assert not list((tmp_path/'work'/'jobs').glob('extract_*'))


def test_keyframe_sampling_and_combined_budget(tmp_path):
    service, path = setup_video(tmp_path, count=16, fps=8)
    failed = service.prepare(str(path), 'A', max_frames=5, keyframe_ranges=[[0.5, 1.5]])
    assert failed['status'] == 'failed' and 'max_frames' in failed['error']
    result = service.prepare(str(path), 'A', max_frames=6, keyframe_ranges=[[0.5, 1.5]])
    manifest = json.loads(service.store.resolve_input(result['manifest']['uri']).read_text())
    assert [(f['seconds'], f['inspection_priority']) for f in manifest['frames']] == [
        (0, 1), (0.5, 2), (0.75, 2), (1, 2), (1.25, 2), (1.5, 1),
    ]


def reviewed_scope(service,path,tmp_path,**sampling):
    result=service.prepare(str(path),'A',**sampling)
    scope=json.loads(service.store.resolve_input(result['scope_review']['uri']).read_text())
    mask=np.zeros((32,32),np.uint8);mask[8:24,8:24]=255
    Image.fromarray(mask).save(tmp_path/'wall.png')
    for i,row in enumerate(scope['frames']):
        row.update(status='included' if i==0 else 'excluded',reviewer='fixture',reason='synthetic scope',
                   wall_mask=str(tmp_path/'wall.png') if i==0 else None)
    output=tmp_path/'review.json'
    output.write_text(json.dumps(scope))
    return output,scope


def test_scope_preserves_coordinates_and_snapshots(tmp_path):
    service,path=setup_video(tmp_path)
    review,_=reviewed_scope(service,path,tmp_path)
    result=service.apply_scope(str(review))
    assert result['status']=='ready_for_detection'
    assert len(result['frames'])==1 and len(result['audit'])==3
    row=result['frames'][0]
    with Image.open(service.store.resolve_input(row['detection_image']['uri'])) as im:
        pixels=np.asarray(im)
    assert pixels.shape==(32,32,3) and pixels[0,0].sum()==0 and pixels[10,10].sum()>0
    Image.fromarray(np.zeros((32,32),np.uint8)).save(tmp_path/'wall.png')
    with Image.open(service.store.resolve_input(row['wall_mask']['uri'])) as im:
        assert np.asarray(im).sum()>0


@pytest.mark.parametrize('problem',['missing_frame','duplicate_frame','empty_mask','missing_reviewer'])
def test_bad_scope_rejected(tmp_path,problem):
    service,path=setup_video(tmp_path)
    review,scope=reviewed_scope(service,path,tmp_path)
    if problem=='missing_frame':scope['frames'].pop()
    if problem=='duplicate_frame':scope['frames'][1]=scope['frames'][0]
    if problem=='empty_mask':Image.fromarray(np.zeros((32,32),np.uint8)).save(tmp_path/'wall.png')
    if problem=='missing_reviewer':scope['frames'][0]['reviewer']=''
    review.write_text(json.dumps(scope))
    assert service.apply_scope(str(review))['status']=='failed'
