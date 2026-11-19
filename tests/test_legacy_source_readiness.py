"""Required source files are checked before a model worker is started."""

import sys
from pathlib import Path

from PIL import Image
import pytest

from facade_mcp.model_service import ModelService, PROFILES
from facade_mcp.storage import ArtifactStore


@pytest.fixture
def weights_only(tmp_path):
    repo = tmp_path / 'models'
    for profile in PROFILES.values():
        for relative in profile['weights']:
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic checkpoint')
    image = tmp_path / 'image.png'
    Image.new('RGB', (16, 12)).save(image)
    service = ModelService(ArtifactStore(tmp_path / 'workspace', [tmp_path]),
                           repo_path=repo, python_path=Path(sys.executable))
    try:
        yield service, repo, image
    finally:
        service.masks.close()


@pytest.mark.parametrize(('name', 'relative'), [
    ('faster_rcnn', 'backend/detection_agent.py'),
    ('ensemble', 'backend/detection_agent.py'),
    ('intersection', 'backend/detection_agent.py'),
    ('unet_crack', 'training/crack_segmentation/unet/unet_transfer.py'),
])
def test_required_source_controls_readiness_and_fails_before_worker(weights_only, monkeypatch, name, relative):
    service, repo, image = weights_only
    monkeypatch.setattr(service, '_run', lambda *a, **k: pytest.fail('worker must not start'))
    capability = service.capabilities()['profiles'][name]
    assert not capability['ready']
    assert capability['reason'] == 'SOURCE_REPO_NOT_FOUND'
    assert capability['missing_source_files'] == [relative]
    if name == 'unet_crack':
        result = service.segment(str(image), [{'instance_id': '1', 'class_code': 'C',
                                               'bbox_xyxy': [1, 1, 5, 5]}], name)
    else:
        result = service.detect([str(image)], name)
    assert result['status'] == 'failed'
    assert result['error']['code'] == 'SOURCE_REPO_NOT_FOUND'
    assert result['error']['details']['missing'] == [relative]

    source = repo / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('# synthetic source')
    assert service.capabilities()['profiles'][name]['ready']
    service._config(name, PROFILES[name]['operation'])


@pytest.mark.parametrize('name', ['yolo11m', 'yolo12m', 'rtdetr', 'mobile_sam', 'sam3'])
def test_supported_weights_only_profiles_remain_ready(weights_only, name):
    service, _, _ = weights_only
    capability = service.capabilities()['profiles'][name]
    assert capability['ready']
    assert capability['missing_source_files'] == []
    service._config(name, PROFILES[name]['operation'])


def test_unet_definition_tracks_required_architecture(weights_only):
    service, repo, _ = weights_only
    source = repo / 'training/crack_segmentation/unet/unet_transfer.py'
    source.parent.mkdir(parents=True)
    source.write_text('# original')
    before = service.capabilities()['profiles']['unet_crack']['definition_sha256']
    source.write_text('# updated')
    after = service.capabilities()['profiles']['unet_crack']['definition_sha256']
    assert before != after
