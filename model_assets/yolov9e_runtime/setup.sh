#!/bin/sh
# Create a separate runtime and inherit heavy dependencies from the chosen base.
set -eu
runtime_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
base_python=${FACADE_MODEL_BASE_PYTHON:-python3}
if ! command -v "$base_python" >/dev/null 2>&1; then
    printf '%s\n' 'Set FACADE_MODEL_BASE_PYTHON to a Python with CUDA PyTorch and model dependencies.' >&2
    exit 1
fi
if [ -e "$runtime_dir/.venv" ] || [ -e "$runtime_dir/vendor" ]; then
    printf '%s\n' 'Runtime .venv or vendor already exists; refusing to overwrite it.' >&2
    exit 1
fi
"$base_python" -m venv --system-site-packages "$runtime_dir/.venv"
"$runtime_dir/.venv/bin/python" -m pip install --no-deps --target "$runtime_dir/vendor" -r "$runtime_dir/requirements.txt"
YOLO_CONFIG_DIR="$runtime_dir/.venv/ultralytics-config" "$runtime_dir/python" - <<'PY'
import cv2, numpy, PIL, torch, ultralytics
assert ultralytics.__version__ == '8.2.50'
assert hasattr(torch.serialization, 'add_safe_globals')
print('runtime ready:', 'torch', torch.__version__, 'ultralytics', ultralytics.__version__)
PY
