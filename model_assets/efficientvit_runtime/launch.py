"""Launch only the pinned CUBIT EfficientViT checkpoint with a compatible worker.

The multi-output routing follows the checkpoint's serialized layer indices. It is
an independently adapted implementation; it has not been verified against the
paper authors' unreleased forward implementation or ground truth.
"""
import argparse
import hashlib
import importlib.util
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import types

RUNTIME = Path(__file__).resolve().parent
PROJECT = RUNTIME.parents[1]
WORKER = (PROJECT / 'src/facade_mcp/registered_expert_worker.py').resolve()
EXPECTED = 'cfbc562cdca0766d06a2c7d328977ceda0a78f25b38bca0ccc67e11baeb6be00'
ALLOWED_GLOBALS = (
    'torch.nn.modules.container.ModuleList', 'torch.nn.modules.container.Sequential',
    'torch.nn.modules.activation.ReLU', 'torch.nn.modules.activation.SiLU',
    'torch.nn.modules.batchnorm.BatchNorm2d', 'torch.nn.modules.linear.Identity',
    'torch.nn.modules.upsampling.Upsample', 'torch.nn.modules.pooling.MaxPool2d',
    'torch.nn.modules.conv.Conv2d', 'torch.nn.modules.conv.ConvTranspose2d',
    'timm.layers.squeeze_excite.SEModule', 'timm.layers.activations.Sigmoid',
    'ultralytics.nn.tasks.SegmentationModel', 'ultralytics.nn.modules.head.Segment',
    'ultralytics.nn.modules.conv.Conv', 'ultralytics.nn.modules.conv.DWConv',
    'ultralytics.nn.modules.conv.Concat', 'ultralytics.nn.modules.block.DFL',
    'ultralytics.nn.modules.block.C3k2', 'ultralytics.nn.modules.block.C2PSA',
    'ultralytics.nn.modules.block.Attention', 'ultralytics.nn.modules.block.Proto',
    'ultralytics.nn.modules.block.PSABlock', 'ultralytics.nn.modules.block.C3k',
    'ultralytics.nn.modules.block.SPPF', 'ultralytics.nn.modules.block.Bottleneck',
    'ultralytics.nn.backbone.efficientViT.LocalWindowAttention',
    'ultralytics.nn.backbone.efficientViT.PatchMerging',
    'ultralytics.nn.backbone.efficientViT.EfficientViT',
    'ultralytics.nn.backbone.efficientViT.CascadedGroupAttention',
    'ultralytics.nn.backbone.efficientViT.FFN',
    'ultralytics.nn.backbone.efficientViT.EfficientViTBlock',
    'ultralytics.nn.backbone.efficientViT.Conv2d_BN',
    'ultralytics.nn.backbone.efficientViT.Residual',
)


def load_builtin_set(name):
    # The checkpoint's dill helper is used to restore a built-in set. Do not
    # allow dill's generic type resolver to execute arbitrary resolutions.
    if name != 'set':
        raise ValueError('Unsupported serialized built-in type')
    return set


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('worker', type=Path)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.worker.resolve() != WORKER:
        raise ValueError('This launcher only executes the registered expert worker')
    request = json.loads(args.request.read_text(encoding='utf-8'))
    profile = request['expert']
    weight = Path(profile['weights_path']).resolve()
    if profile['backend'] != 'ultralytics' or profile['task'] != 'segment':
        raise ValueError('This launcher only supports the specified segmentation expert')
    if request.get('expected_weights_sha256') != EXPECTED or file_sha256(weight) != EXPECTED:
        raise ValueError('Checkpoint must match the pinned author weight')

    # Register the pinned compatibility module at the exact pickled class path.
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    import ultralytics.nn
    import ultralytics.nn.tasks as tasks
    pkgname = 'ultralytics.nn.backbone'
    package = types.ModuleType(pkgname)
    package.__path__ = []
    sys.modules[pkgname] = package
    setattr(ultralytics.nn, 'backbone', package)
    name = pkgname + '.efficientViT'
    spec = importlib.util.spec_from_file_location(name, RUNTIME / 'efficientvit_compat.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    setattr(package, 'efficientViT', module)
    spec.loader.exec_module(module)

    # The author checkpoint stores model objects. Use the restricted PyTorch
    # unpickler with exact vetted classes, and a restricted dill replacement.
    original_load = torch.load
    def pinned_load(path, *load_args, **kwargs):
        if not isinstance(path, (str, os.PathLike)) or Path(path).resolve() != weight:
            raise ValueError('Attempted to load an unapproved checkpoint')
        actual = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        expected = set(ALLOWED_GLOBALS) | {'dill._dill._load_type'}
        if actual != expected:
            raise ValueError('Unexpected serialized global in pinned checkpoint')
        safe = []
        for qualified in ALLOWED_GLOBALS:
            mod, attr = qualified.rsplit('.', 1)
            safe.append((getattr(importlib.import_module(mod), attr), qualified))
        safe.append((load_builtin_set, 'dill._dill._load_type'))
        kwargs['weights_only'] = True
        with torch.serialization.safe_globals(safe):
            return original_load(path, *load_args, **kwargs)
    torch.load = pinned_load

    # The author's model flattens three backbone outputs into logical slots
    # 2, 3 and 4. Current stock Ultralytics instead expects a single output.
    def predict_once(self, x, profile=False, visualize=False, embed=None):
        if profile or visualize or embed:
            raise ValueError('Profiling, visualisation and embeddings are unsupported')
        saved = []
        for layer in self.model:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            if getattr(layer, 'backbone', False):
                if layer.i != 4 or saved:
                    raise ValueError('Unexpected EfficientViT backbone layout')
                features = layer(x)
                if not isinstance(features, (tuple, list)) or len(features) != 3:
                    raise ValueError('Expected exactly three backbone features')
                logical = [None, None, *features]
                saved.extend(value if i in self.save else None for i, value in enumerate(logical))
                x = logical[-1]
            else:
                if layer.i != len(saved):
                    raise ValueError('Unexpected model layer numbering')
                x = layer(x)
                saved.append(x if layer.i in self.save else None)
        return x
    tasks.BaseModel._predict_once = predict_once
    sys.path.insert(0, str(WORKER.parent))
    sys.argv = [str(WORKER), '--request', str(args.request), '--output', str(args.output)]
    runpy.run_path(str(WORKER), run_name='__main__')


if __name__ == '__main__':
    main()
