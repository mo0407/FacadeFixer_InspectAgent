# YOLOv9-e checkpoint compatibility runtime

This is an experimental compatibility runtime for the paper's YOLOv9-e trained segmentation checkpoint; it is **not** the paper authors' fork. The source checkpoint refers to `ultralytics.nn.extra_modules.block`, absent from the official Ultralytics package. The adapter resolves the seven legacy class names to matching-looking classes already included in official Ultralytics 8.2.50. Forward behaviour on one image was checked, but equivalence to the paper's original fork and model accuracy have not been established.

The dedicated `python` launcher uses this runtime's `.venv`, adds its `vendor` and `bootstrap` directories to `PYTHONPATH` and enables PyTorch's `weights_only` loader with an explicit allowlist. On a different host, recreate the runtime as below. The base environment is not modified.

## Local dependencies

The `.venv` and `vendor` directories are intentionally ignored by Git. Recreate them with a Python 3.10 base environment already providing PyTorch and the project model runtime dependencies:

```bash
FACADE_MODEL_BASE_PYTHON=/path/to/base/python sh model_assets/yolov9e_runtime/setup.sh
```

The model runtime tested here uses Python 3.10, PyTorch 2.12.0+cu130, CUDA and the separately installed official `ultralytics==8.2.50` (AGPL-3.0) and `dill==0.3.8` (BSD-3-Clause). Installed distribution licenses are included in `vendor/*dist-info/`. Dependencies such as NumPy, Pillow, OpenCV and torchvision come from the existing base runtime. See https://github.com/ultralytics/ultralytics/tree/v8.2.50 and https://github.com/uqfoundation/dill/tree/dill-0.3.8 . No code from the unverified third-party custom model repository was copied.

## Validation scope

On 2026-11-20, the project `registered_expert_worker.py` with `backend=ultralytics`, `task=segment`, and `crack→C`, `spalling→S` processed one 4500×3000 image at 640 inference size on an RTX 3090 (confidence 0.6): 17 outputs, all masks verified as 4500×3000. This is a functionality smoke test, not a quality assessment, comparison with the authors' original implementation, or a complete Figure 1 batch. No training or GT labels were used. The checkpoint is loaded using the weights-only restricted unpickler, not arbitrary unrestricted pickle.
