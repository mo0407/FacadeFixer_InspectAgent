#!/bin/sh
# Build the runtime without changing the base environment or downloading torch.
set -eu
runtime_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
base_python=${FACADE_MODEL_BASE_PYTHON:-python3}
if ! command -v "$base_python" >/dev/null 2>&1; then
    printf '%s\n' "Set FACADE_MODEL_BASE_PYTHON to a Python with CUDA PyTorch, timm and required dependencies." >&2
    exit 1
fi
if [ -e "$runtime_dir/.venv" ]; then
    printf '%s\n' "Runtime .venv already exists; refusing to overwrite it." >&2
    exit 1
fi
"$base_python" -m venv --system-site-packages "$runtime_dir/.venv"
"$runtime_dir/.venv/bin/python" -m pip install --no-deps 'ultralytics==8.3.9' 'dill==0.3.9'
YOLO_CONFIG_DIR="$runtime_dir/.venv/ultralytics-config" "$runtime_dir/.venv/bin/python" - <<'PY'
import torch, ultralytics, timm, PIL, cv2, numpy
assert ultralytics.__version__ == '8.3.9'
assert hasattr(torch.serialization, 'get_unsafe_globals_in_checkpoint')
assert hasattr(torch.serialization, 'safe_globals')
print('runtime ready:', 'torch', torch.__version__, 'ultralytics', ultralytics.__version__, 'timm', timm.__version__)
PY
