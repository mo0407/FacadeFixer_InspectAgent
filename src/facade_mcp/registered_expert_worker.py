"""Fixed offline worker for server-registered Ultralytics and YOLOv9 experts.

YOLOv9 segment tensor processing follows the official segment/predict.py API:
https://github.com/WongKinYiu/yolov9/blob/main/segment/predict.py
No model asset, source checkout or Python executable is downloaded here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

try:
    from .source_detection_worker import _offline, _prediction, _sha256
except ImportError:  # Direct execution in the separately configured model runtime.
    from source_detection_worker import _offline, _prediction, _sha256


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    import numpy as np
    return np.asarray(value)


def _ultralytics(profile, raster, threshold, device, size):
    from ultralytics import YOLO
    model = YOLO(profile['weights_path'], task=profile['task'])
    if model.task != profile['task']:
        raise ValueError('Checkpoint task does not match registered expert task')
    results = model.predict(source=raster, conf=threshold, iou=.5, max_det=300, device=device,
                            imgsz=size, retina_masks=True, verbose=False, save=False,
                            save_txt=False, save_crop=False)
    rows = []
    for result in results:
        if result.boxes is None:
            continue
        names = result.names
        masks = None if result.masks is None else _numpy(result.masks.data)
        if profile['task'] == 'segment' and len(result.boxes) and (masks is None or len(masks) != len(result.boxes)):
            raise ValueError('Segmentation checkpoint did not return one mask per box')
        for index, box in enumerate(result.boxes):
            cid = int(box.cls[0].item())
            label = names[cid]
            mask = masks[index] if masks is not None else None
            rows.append((str(label), cid, box.conf[0], box.xyxy[0], mask))
    return rows


def _yolov9(profile, raster, threshold, device, size):
    """Official DetectMultiBackend + nm=32 NMS + process_mask, one still image."""
    import numpy as np
    import torch
    repo = Path(profile['repo_path']).resolve()
    sys.path.insert(0, str(repo))
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.segment.general import process_mask
    from utils.torch_utils import select_device
    # Refuse similarly named installed modules resolving outside configured repo.
    for name in ('models.common', 'utils.augmentations', 'utils.general', 'utils.segment.general', 'utils.torch_utils'):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(repo):
            raise ValueError('YOLOv9 import escaped configured source repository')
    target = select_device('0' if device == 'cuda' else device.replace('cuda:', ''))
    model = DetectMultiBackend(profile['weights_path'], device=target, dnn=False, fp16=False)
    shape = check_img_size((size, size), s=model.stride)
    padded, ratio, pad = letterbox(raster, shape, stride=model.stride, auto=False)
    tensor = torch.from_numpy(np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))).to(model.device).float() / 255
    tensor = tensor[None]
    with torch.inference_mode():
        prediction, proto = model(tensor, augment=False, visualize=False)[:2]
        detections = non_max_suppression(prediction, threshold, .5, max_det=300, nm=32)[0]
        if len(detections) == 0:
            return []
        masks = process_mask(proto[0], detections[:, 6:], detections[:, :4], tensor.shape[2:], upsample=True)
        # Remove exact letterbox border before nearest-neighbor raster mapping.
        masks = _numpy(masks)
        left, top = int(round(pad[0]-.1)), int(round(pad[1]-.1))
        inner_w, inner_h = int(round(raster.shape[1]*ratio[0])), int(round(raster.shape[0]*ratio[1]))
        boxes = scale_boxes(tensor.shape[2:], detections[:, :4], raster.shape)
        from PIL import Image
        rows = []
        for index, detection in enumerate(detections):
            cid = int(detection[5].item())
            small = masks[index, top:top+inner_h, left:left+inner_w]
            if small.size == 0:
                raise ValueError('Invalid letterbox mask geometry')
            mask = np.asarray(Image.fromarray((small > 0).astype('uint8')*255).resize(
                (raster.shape[1], raster.shape[0]), Image.Resampling.NEAREST)) > 0
            rows.append((str(model.names[cid]), cid, detection[4], boxes[index], mask))
        return rows


def run_registered(request):
    import numpy as np
    from PIL import Image
    profile = request['expert']
    weights, image_path = Path(profile['weights_path']), Path(request['image_path'])
    if not weights.is_file():
        raise FileNotFoundError('Registered expert checkpoint is not installed')
    digest = _sha256(weights)
    if digest != request['expected_weights_sha256']:
        raise ValueError('Registered expert checkpoint changed before inference')
    source_hash = _sha256(image_path)
    output = Path(request['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        raster = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
    height, width = raster.shape[:2]
    size = 960 if request.get('inference_size') is None else request['inference_size']
    threshold = request['threshold']
    if type(size) is not int or not 320 <= size <= 1536 or size % 32:
        raise ValueError('inference_size must be a multiple of 32 in [320,1536]')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid confidence threshold')
    with _offline():
        rows = (_ultralytics if profile['backend'] == 'ultralytics' else _yolov9)(
            profile, raster, threshold, request['device'], size)
    defects = []
    for index, (label, cid, score, box, mask) in enumerate(rows, 1):
        prediction = _prediction(label, cid, score, box, width, height, request['model_profile'])
        if prediction is None or prediction['confidence'] < threshold:
            continue
        prediction.update(instance_id=str(index), class_code=profile['class_map'].get(label, profile['class_map'].get(str(cid), 'unknown')))
        if mask is not None:
            binary = _numpy(mask) > 0
            if binary.shape != (height, width):
                raise ValueError('Expert mask is not in original stored-pixel dimensions')
            path = output / f'mask_{index:04d}.png'
            Image.fromarray(binary.astype('uint8')*255).save(path)
            prediction.update(mask_path=str(path), mask_sha256=_sha256(path))
        defects.append(prediction)
    if _sha256(weights) != digest or _sha256(image_path) != source_hash:
        raise ValueError('Checkpoint or source image changed during inference')
    from importlib.metadata import PackageNotFoundError, version
    try:
        library_version = version('ultralytics' if profile['backend'] == 'ultralytics' else 'torch')
    except PackageNotFoundError:
        library_version = 'unknown'
    sources = {}
    if profile['backend'] == 'yolov9':
        repo = Path(profile['repo_path'])
        for relative in ('models/common.py','utils/augmentations.py','utils/general.py','utils/segment/general.py','utils/torch_utils.py'):
            if (repo/relative).is_file():
                sources[relative] = _sha256(repo/relative)
    provenance = dict(profile=request['model_profile'], weights_sha256=digest, source_image_sha256=source_hash,
                      task=profile['task'], backend=profile['backend'], class_map=profile['class_map'],
                      class_codes=profile['class_codes'], reference_name=profile.get('reference_name'),
                      library_version=library_version, source_files_sha256=sources,
                      automatic_weight_download=False, network_access=False, expert_validated=False,
                      worker_sha256=_sha256(Path(__file__)))
    return dict(status='completed', defects=defects, width=width, height=height,
                coordinate_space='original_pixels', model_provenance=provenance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_registered(json.loads(args.request.read_text(encoding='utf-8')))
    except Exception as error:
        result = {'status':'failed','error':{'code':'REGISTERED_EXPERT_FAILED','type':type(error).__name__}}
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
