"""Allow the paper YOLOv9e checkpoint through PyTorch's weights-only loader.

Loaded only by this runtime's launcher. Aliases are compatible candidates, not
verified copies of the paper authors' fork. No unrestricted pickle loading.
"""
from pathlib import Path
import sys

import dill._dill
import torch
import ultralytics
import ultralytics.nn

if ultralytics.__version__ != '8.2.50':
    raise RuntimeError('The YOLOv9e compatibility runtime requires ultralytics==8.2.50')
ultralytics.nn.__path__.append(str(Path(__file__).parent / 'yolo_compat'))
from ultralytics.nn.extra_modules import block as compat
from ultralytics.nn.modules import block, conv, head
from ultralytics.nn.tasks import SegmentationModel

allowed = [
    dill._dill._load_type, SegmentationModel,
    torch.nn.SiLU, torch.nn.BatchNorm2d, torch.nn.ModuleList, torch.nn.Sequential,
    torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Identity,
    torch.nn.MaxPool2d, torch.nn.Upsample,
    block.DFL, block.Proto, block.SPPELAN, conv.Concat, conv.Conv, head.Segment,
]
allowed += [
    (getattr(compat, name), 'ultralytics.nn.extra_modules.block.' + name)
    for name in ('RepNCSPELAN4', 'RepNCSP', 'RepNBottleneck', 'RepConvN', 'ADown', 'CBLinear', 'CBFuse')
]
torch.serialization.add_safe_globals(allowed)
