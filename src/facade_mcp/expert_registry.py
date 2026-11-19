"""Server-owned named expert definitions; no caller-supplied executable paths."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .storage import sha256


BUSINESS_CLASSES = ('C', 'S', 'T', 'R', 'D', 'V', 'L', 'E', 'B')


def local_implementations() -> dict[str, Path]:
    """Shared code that owns inference, normalization and evidence semantics."""
    directory = Path(__file__).resolve().parent
    return {'facade_mcp/'+name:directory/name for name in (
        'expert_registry.py', 'model_service.py', 'registered_expert_worker.py',
        'source_detection_worker.py', 'detection_service.py', 'taxonomy.py',
        'mask_service.py', 'storage.py')}


def definition_fingerprint(profile: dict, implementations: dict[str, Path]) -> dict:
    """Hash semantic configuration and implementation bytes, independent of paths.

    Checkpoint bytes have their own identity and are deliberately not part of
    this hash. Missing source is explicit rather than silently dropping a key.
    """
    files = {key:sha256(path) if path.is_file() else None for key,path in sorted(implementations.items())}
    definition = {key:profile[key] for key in ('backend','task','class_map','full_image','local_eligible')}
    definition['class_codes'] = sorted(profile['class_codes'])
    definition['implementation_sha256'] = files
    encoded = json.dumps(definition, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {'definition_sha256':hashlib.sha256(encoded.encode('utf-8')).hexdigest(),
            'implementation_sha256':files}


def registered_definition(profile: dict) -> dict:
    files = local_implementations()
    if profile['backend'] == 'yolov9':
        repo = Path(profile['repo_path'])
        for relative in ('models/common.py', 'models/yolo.py', 'models/experimental.py',
                         'utils/augmentations.py', 'utils/general.py', 'utils/segment/general.py',
                         'utils/torch_utils.py'):
            files['yolov9/'+relative] = repo/relative
    return definition_fingerprint(profile, files)


class ExpertProfile(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    backend: str
    weights_path: str = Field(min_length=1)
    task: str
    class_map: dict[str, str] = Field(min_length=1, max_length=500)
    class_codes: list[str] = Field(min_length=1, max_length=9)
    local_eligible: bool
    full_image: bool
    repo_path: str | None = None
    python_path: str | None = None
    reference_name: str | None = Field(default=None, max_length=200)
    checkpoint_sha256: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')

    @model_validator(mode='after')
    def constraints(self):
        if self.backend not in ('ultralytics', 'yolov9') or self.task not in ('detect', 'segment'):
            raise ValueError('Select backend ultralytics/yolov9 and task detect/segment')
        if self.backend == 'yolov9' and (self.task != 'segment' or not self.repo_path):
            raise ValueError('YOLOv9 requires task segment and an explicit local repo_path')
        if len(set(self.class_codes)) != len(self.class_codes) or not set(self.class_codes) <= set(BUSINESS_CLASSES):
            raise ValueError('class_codes must be unique current business codes')
        if set(self.class_map.values()) != set(self.class_codes):
            raise ValueError('class_codes must equal the classes produced by class_map')
        if any(not key.strip() or len(key) > 200 for key in self.class_map):
            raise ValueError('Checkpoint labels must be nonempty and at most 200 characters')
        return self


def load_registry(config_path: Path | None = None, *, default_python: Path | None = None,
                  reserved=()) -> dict[str, dict]:
    """Resolve configured paths only under config parent or explicit server roots.

    Missing files remain representable so capabilities can report not ready.
    The fallback interpreter is already server-owned FACADE_SOURCE_PYTHON.
    """
    value = config_path or os.getenv('FACADE_EXPERTS_CONFIG')
    if not value:
        return {}
    path = Path(value).expanduser().resolve()
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Expert registry exceeds 1 MiB')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or set(data) != {'profiles'} or not isinstance(data['profiles'], dict):
        raise ValueError('Expert registry must contain exactly a profiles object')
    if len(data['profiles']) > 50:
        raise ValueError('At most 50 registered experts are allowed')
    roots = [path.parent, *[Path(p).expanduser().resolve() for p in os.getenv('FACADE_EXPERTS_ALLOWED_ROOTS', '').split(';') if p.strip()]]

    def local(value):
        candidate = (path.parent / Path(value).expanduser()).resolve()
        if not any(candidate.is_relative_to(root) for root in roots):
            raise ValueError('Expert paths must remain under registry parent or FACADE_EXPERTS_ALLOWED_ROOTS')
        return str(candidate)

    result = {}
    for name, raw in data['profiles'].items():
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name) or name in reserved:
            raise ValueError('Expert profile names must be safe, unique and not replace builtins')
        profile = ExpertProfile.model_validate(raw).model_dump()
        for key in ('weights_path', 'repo_path', 'python_path'):
            if profile.get(key):
                profile[key] = local(profile[key])
        if not profile['python_path']:
            if default_python is None:
                raise ValueError('Configure a server-owned model Python runtime')
            profile['python_path'] = str(Path(default_python).resolve())
        result[name] = profile
    return result


def registration_capability(name: str, profile: dict) -> dict:
    weights = Path(profile['weights_path'])
    digest = sha256(weights) if weights.is_file() else None
    reason = None
    if digest is None:
        reason = 'WEIGHTS_NOT_FOUND'
    elif profile.get('checkpoint_sha256') and digest != profile['checkpoint_sha256']:
        reason = 'CHECKPOINT_HASH_MISMATCH'
    elif not Path(profile['python_path']).is_file():
        reason = 'SOURCE_PYTHON_NOT_FOUND'
    elif profile['backend'] == 'yolov9' and not all((Path(profile['repo_path']) / f).is_file() for f in
                                                  ('models/common.py', 'utils/augmentations.py', 'utils/general.py',
                                                   'utils/segment/general.py', 'utils/torch_utils.py')):
        reason = 'EXPERT_REPO_NOT_FOUND'
    return dict(profile=name, registered=True, backend=profile['backend'], task=profile['task'],
                class_codes=list(profile['class_codes']), class_map=dict(profile['class_map']),
                local_eligible=profile['local_eligible'], full_image=profile['full_image'],
                reference_name=profile.get('reference_name'), weights_sha256=digest,
                checkpoint_identity=digest, ready=reason is None, reason=reason,
                model_load_verified=False, capability_source='server_declared_class_map',
                **registered_definition(profile))


def composite_identity(members: dict) -> str:
    return hashlib.sha256(json.dumps(members, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
