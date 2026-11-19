"""Evidence-backed, offline P0 proposals and independent replay publication.

Only the existing model gateway can call a network service. A candidate is never
adopted by proposing it: publication requires real, paired image workflow runs
against an independently declared bbox reference dataset. SQLite is the atomic
publication authority; version files and proposal metadata are hash checked.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import Field, StrictBool, StrictFloat, model_validator

from .failure_memory import FailureMemory
from .model_gateway import ModelConfig, ModelGateway, Strict
from .playbook import Playbook, Rule, RuleFeedback, build_delta
from .storage import ArtifactStore, sha256, utc_now


CLASSES = {'C', 'S', 'T', 'R', 'D', 'V', 'L', 'E', 'B'}
IDENTITY_SCOPE = ('Observed local checkpoint bytes, expert definitions and implementation fingerprints; '
                  'provider-reported VLM model IDs. These do not verify a cloud provider\'s physical weights.')


def playbook_hash(book: Playbook) -> str:
    return hashlib.sha256(book.model_dump_json().encode('utf-8')).hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


class FailureInsight(Strict):
    evidence_ids: list[str] = Field(min_length=1, max_length=64)
    hypothesis: str = Field(min_length=1, max_length=2000)
    controlled_change: str = Field(min_length=1, max_length=2000)


class LifecycleReflection(Strict):
    insights: list[FailureInsight] = Field(default_factory=list, max_length=100)
    used_rule_ids: list[str] = Field(default_factory=list, max_length=100)
    rule_feedback: list[RuleFeedback] = Field(default_factory=list, max_length=100)


class RuleChange(Strict):
    action: Literal['add', 'revise', 'deprecate']
    condition: str = Field(min_length=1, max_length=2000)
    recommendation: str = Field(min_length=1, max_length=3000)
    evidence_ids: list[str] = Field(min_length=1, max_length=64)
    limitations: str = Field(min_length=1, max_length=2000)
    target_rule_id: str | None = None


class LifecycleCuration(Strict):
    candidates: list[RuleChange] = Field(default_factory=list, max_length=100)


class ReferenceDefect(Strict):
    class_code: str
    bbox_xyxy: list[StrictFloat] = Field(min_length=4, max_length=4)

    @model_validator(mode='after')
    def geometry(self):
        if self.class_code not in CLASSES:
            raise ValueError('Reference class must use the canonical C/S/T/R/D/V/L/E/B taxonomy')
        if (any(not math.isfinite(v) for v in self.bbox_xyxy)
                or not 0 <= self.bbox_xyxy[0] < self.bbox_xyxy[2]
                or not 0 <= self.bbox_xyxy[1] < self.bbox_xyxy[3]):
            raise ValueError('Reference bbox must have positive extent in original pixels')
        return self


class ReferenceCase(Strict):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,80}$')
    image_path: str = Field(min_length=1)
    scope_mask_path: str | None = None
    defects: list[ReferenceDefect] = Field(default_factory=list, max_length=2000)

    @model_validator(mode='after')
    def distinct_annotations(self):
        keys = [(d.class_code, tuple(d.bbox_xyxy)) for d in self.defects]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate reference boxes are not independent defects')
        return self


class QualityTargets(Strict):
    min_precision: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)
    min_recall: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)
    min_f1: float = Field(default=0, ge=0, le=1, allow_inf_nan=False, strict=True)
    max_regression: float = Field(default=0, ge=0, le=1, allow_inf_nan=False, strict=True)
    min_improvement: float = Field(default=.01, gt=0, le=1, allow_inf_nan=False, strict=True)
    min_cases: int = Field(default=3, ge=2, le=100, strict=True)
    min_reference_defects: int = Field(default=5, ge=2, le=200000, strict=True)


class RuleValidationDataset(Strict):
    version: str = Field(min_length=1, max_length=128)
    independent_reference_confirmed: StrictBool = False
    cases: list[ReferenceCase] = Field(min_length=1, max_length=100)
    targets: QualityTargets
    iou_threshold: float = Field(default=.5, ge=.5, le=1, allow_inf_nan=False, strict=True)
    # These are image request settings only, never executable runner/plugin paths.
    inspection: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def independent_cases(self):
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError('Dataset case IDs must be unique')
        if not any(case.defects for case in self.cases):
            raise ValueError('At least one independently annotated defect is required')
        if {'image_path', 'scope_mask_path'} & self.inspection.keys():
            raise ValueError('Dataset inspection settings must not override case input paths')
        return self


def _eligible_event(event):
    context = event.get('context', {})
    return (event.get('kind') in ('quality', 'runtime')
            and event.get('cause_status') in ('observed', 'confirmed')
            and event.get('hypothesis') is None
            and event.get('source') not in ('mock', 'reflector', 'curator')
            and not any(context.get(key) for key in ('demo', 'mock', 'mock_generated', 'simulation'))
            and context.get('vlm_adapter', context.get('adapter')) != 'mock'
            and not str(context.get('vlm_model', '')).lower().startswith('mock')
            and bool(event.get('evidence_refs')))


def _source_hashes(events):
    values = set()
    for event in events:
        for key in ('source_sha256', 'original_input_sha256', 'source_image_sha256'):
            value = event.get('context', {}).get(key)
            if isinstance(value, str) and re.fullmatch(r'[a-fA-F0-9]{64}', value):
                values.add(value.lower())
        for value in event.get('context', {}).get('source_sha256s', []):
            if isinstance(value, str) and re.fullmatch(r'[a-fA-F0-9]{64}', value):
                values.add(value.lower())
    return values


def _image_hashes(path):
    file_hash = sha256(path)
    with Image.open(path) as image:
        if image.format not in ('JPEG', 'PNG') or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Reference inputs must be still PNG/JPEG images')
        if image.width * image.height > 80_000_000:
            raise ValueError('Reference image exceeds the image workflow pixel budget')
        rgb = image.convert('RGB')
        size = rgb.size
        encoded = io.BytesIO()
        rgb.save(encoded, format='PNG')
    return file_hash, hashlib.sha256(encoded.getvalue()).hexdigest(), size


def _checkpoint_values(value):
    values = list(value.values()) if isinstance(value, dict) else [value]
    if not values or any(not isinstance(item, str) or not re.fullmatch(r'[a-f0-9]{64}', item) for item in values):
        raise ValueError('Invalid configured expert checkpoint identity')
    return set(values)


def _iou(a, b):
    overlap = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / union if union else 0.


def _match(predictions, references, threshold):
    """Maximum-cardinality, one-to-one same-class matching, not reusable GT hits."""
    edges = {i: sorted([j for j, ref in enumerate(references)
                       if pred['class_code'] == ref.class_code and _iou(pred['bbox_xyxy'], ref.bbox_xyxy) >= threshold],
                      key=lambda j: -_iou(pred['bbox_xyxy'], references[j].bbox_xyxy))
             for i, pred in enumerate(predictions)}
    matched, prediction_match = {}, {}
    for index in range(len(predictions)):
        queue, seen_predictions, parent_reference = [index], {index}, {}
        found = False
        for current in queue:
            for target in edges[current]:
                if target in parent_reference:
                    continue
                parent_reference[target] = current
                if target not in matched:
                    # Reverse one alternating path. Iteration avoids recursion
                    # limits for a densely overlapping reference dataset.
                    while True:
                        source = parent_reference[target]
                        previous = prediction_match.get(source)
                        matched[target], prediction_match[source] = source, target
                        if previous is None:
                            break
                        target = previous
                    found = True
                    break
                previous_source = matched[target]
                if previous_source not in seen_predictions:
                    queue.append(previous_source)
                    seen_predictions.add(previous_source)
            if found:
                break
    pairs = [{'prediction_index': i, 'reference_index': j,
              'iou': _iou(predictions[i]['bbox_xyxy'], references[j].bbox_xyxy)}
             for j, i in sorted(matched.items())]
    return {'tp': len(pairs), 'fp': len(predictions)-len(pairs),
            'fn': len(references)-len(pairs), 'matches': pairs}


def _aggregate(rows):
    tp, fp, fn = (sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn'))
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.}


class RuleLifecycle:
    def __init__(self, store: ArtifactStore):
        self.store = store
        self.root = store.root / 'rule_lifecycle'
        (self.root / 'versions').mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / 'catalog.sqlite3'
        with closing(self._connect()) as db, db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS proposals (candidate_rel TEXT PRIMARY KEY, metadata_sha256 TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS versions (version TEXT PRIMARY KEY, playbook_sha256 TEXT NOT NULL UNIQUE, manifest_sha256 TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS current (id INTEGER PRIMARY KEY CHECK(id=1), version TEXT NOT NULL, playbook_sha256 TEXT NOT NULL)')

    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _current_hash(self, db=None):
        if db is not None:
            row = db.execute('SELECT playbook_sha256 FROM current WHERE id=1').fetchone()
            return row[0] if row else None
        with closing(self._connect()) as connection:
            return self._current_hash(connection)

    def latest(self) -> Playbook | None:
        with closing(self._connect()) as db:
            row = db.execute('SELECT v.* FROM current c JOIN versions v ON c.version=v.version WHERE c.id=1').fetchone()
        if row is None:
            return None
        if not re.fullmatch(r'P0-[a-f0-9]{32}', row['version']):
            raise ValueError('Invalid published P0 version')
        directory = self.root / 'versions' / row['version']
        manifest_path = directory / 'manifest.json'
        if sha256(manifest_path) != row['manifest_sha256']:
            raise ValueError('Published P0 manifest changed')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name, digest in manifest['files'].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or sha256(path) != digest:
                raise ValueError('Published P0 files changed')
        book = Playbook.model_validate_json((directory / 'playbook.json').read_text(encoding='utf-8'))
        if playbook_hash(book) != row['playbook_sha256'] or book.version != row['version']:
            raise ValueError('Published P0 hash/version mismatch')
        return book

    def _history(self, job_ids):
        events = FailureMemory(self.store.root).summary()['events']
        if job_ids is not None:
            if not job_ids or not set(job_ids) <= {event['job_id'] for event in events}:
                raise ValueError('Select existing failure-memory job IDs')
            events = [event for event in events if event['job_id'] in set(job_ids)]
        events = [event for event in events if _eligible_event(event)]
        if not events:
            raise ValueError('No evidence-backed, non-mock observed failure events selected')
        if len(events) > 200:
            raise ValueError('Select at most 200 failure events using job_ids; history is not silently truncated')
        return events

    def _evidence_path(self, reference):
        """Resolve declared local evidence only; never search directories or execute references."""
        if not isinstance(reference, str) or reference.startswith('evidence:'):
            return None
        value = reference.split('#', 1)[0]
        if value.startswith('artifact:'):
            try:
                return self.store.resolve_input(value)
            except (ValueError, OSError):
                return None
        path = Path(value)
        for candidate in ([path] if path.is_absolute() else [self.store.root/path, self.store.root/'jobs'/path]):
            try:
                return self.store.resolve_input(str(candidate))
            except (ValueError, OSError):
                continue
        return None

    def _snapshot_evidence(self, events, folder):
        bindings, historical, blocked, total_bytes = [], set(), [], 0
        for event in events:
            expected = event.get('context', {}).get('evidence_sha256', {})
            if not isinstance(expected, dict):
                raise ValueError('Historical evidence hashes must be a filename-to-SHA256 mapping')
            paths = {path for reference in event['evidence_refs'] if (path := self._evidence_path(reference)) is not None}
            # Image workflow failures have an exact job-local source snapshot.
            try:
                source = self.store.job_dir(event['job_id'])/'source.png'
                if source.is_file():
                    paths.add(source.resolve())
            except (ValueError, OSError):
                pass
            # Review metric evidence IDs can name local images or reference JSON.
            # Only explicitly declared fields are followed, with a finite depth.
            for _ in range(2):
                extra = set()
                for path in paths:
                    if path.suffix.lower() != '.json' or path.stat().st_size > 10_000_000:
                        continue
                    try:
                        document = json.loads(path.read_text(encoding='utf-8-sig'))
                    except (ValueError, OSError):
                        continue
                    values = []
                    if isinstance(document, dict):
                        values.extend(document.get('evidence', {}).values() if isinstance(document.get('evidence'), dict) else [])
                        for data in (document, document.get('request', {}), document.get('original_image', {})):
                            if isinstance(data, dict):
                                values.extend(data.get(key) for key in ('image_path', 'source_path', 'path', 'uri'))
                        # Aggregate rule validation metrics bind their actual
                        # replay input artifacts, not an invented aggregate hash.
                        for case in document.get('cases', []) if isinstance(document.get('cases'), list) else []:
                            if isinstance(case, dict) and isinstance(case.get('input_artifact'), dict):
                                values.append(case['input_artifact'].get('uri'))
                    for value in values:
                        if not isinstance(value, str):
                            continue
                        if value.startswith('artifact:'):
                            found = self._evidence_path(value)
                        else:
                            try:
                                found = self.store.resolve_input(str(path.parent/value))
                            except (ValueError, OSError):
                                found = self._evidence_path(value)
                        if found:
                            extra.add(found)
                if extra <= paths:
                    break
                paths |= extra
            checked_expected = set()
            source_verified = False
            known_sources = _source_hashes([event])
            historical |= known_sources
            files = []
            for index, path in enumerate(sorted(paths, key=str)):
                digest = sha256(path)
                matched_expected = [name for name in expected if path.as_posix().endswith('/'+name.replace('\\', '/'))]
                for name in matched_expected:
                    if expected[name] != digest:
                        raise ValueError('Historical reviewer evidence hash changed')
                    checked_expected.add(name)
                total_bytes += path.stat().st_size
                if total_bytes > 256*1024*1024:
                    raise ValueError('Select a smaller failure history; evidence snapshots exceed 256 MiB')
                relative = f'evidence/{event["event_id"]}/{index:03d}{path.suffix.lower()}'
                target = folder/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if sha256(target) != digest or sha256(path) != digest:
                    raise ValueError('Historical evidence changed during snapshot')
                if path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    original, normalized, _ = _image_hashes(target)
                    historical.update((original, normalized))
                    if {original, normalized} & known_sources or matched_expected:
                        source_verified = True
                files.append({'origin_path': str(path), 'snapshot': relative, 'sha256': digest})
            if set(expected) != checked_expected:
                raise ValueError('Historical reviewer evidence files are missing')
            if not files:
                blocked.append({'event_id': event['event_id'], 'reason': 'historical_evidence_files_unavailable'})
            if event['kind'] == 'quality' and not source_verified:
                blocked.append({'event_id': event['event_id'], 'reason': 'quality_source_hash_not_verifiable'})
            bindings.append({'event_id': event['event_id'], 'source_verified': source_verified,
                             'image_source_requirement': 'required' if event['kind'] == 'quality' else 'not_applicable_to_nonimage_runtime',
                             'files': files})
        return bindings, historical, blocked

    @staticmethod
    def _check_bindings(bindings, folder):
        for event in bindings:
            for file in event['files']:
                if sha256(Path(file['origin_path'])) != file['sha256'] or sha256(folder/file['snapshot']) != file['sha256']:
                    raise ValueError('Frozen historical source evidence or snapshot changed')

    async def propose(self, settings: ModelConfig, parent: Playbook | None = None,
                      job_ids: list[str] | None = None, allow_api=False, transport=None) -> dict:
        published = self.latest()
        expected_latest = playbook_hash(published) if published else None
        if parent is not None and published is not None and playbook_hash(parent) != expected_latest:
            raise ValueError('Proposal parent differs from the published latest P0')
        parent = (parent or published or Playbook(version='P0-empty')).model_copy(deep=True)
        parent_hash = playbook_hash(parent)
        events = self._history(job_ids)
        event_ids = {event['event_id'] for event in events}
        job, folder = self.store.create_job('rule_proposal', {'job_ids': job_ids, 'parent_sha256': parent_hash})
        gateway = ModelGateway(settings, allow_api=allow_api, transport=transport)
        adapter_mock = any(settings.connections[settings.roles[role]].adapter == 'mock' for role in ('reflector', 'curator'))
        simulation = adapter_mock or transport is not None
        save = lambda name, value: self.store.write_json(folder / name, value)
        try:
            from .image_failures import progress
            progress(folder, 'rule_evidence_snapshot', simulation=simulation, playbook_version=parent.version)
            save('parent-playbook.json', parent.model_dump(mode='json'))
            save('failure-evidence.json', events)
            save('model-config.json', settings.model_dump(mode='json'))
            bindings, historical_sources, evidence_blocks = self._snapshot_evidence(events, folder)
            save('evidence-bindings.json', bindings)
            task = {'stage': 'offline_rule_proposal', 'parent': parent.model_dump(mode='json'),
                    'failure_events': events, 'allowed_evidence_ids': sorted(event_ids),
                    'instruction': 'Use only the observed failure events as evidence. Explanations are unconfirmed hypotheses. '
                                   'Do not invent measurements or evidence. Propose controlled, reusable changes; '
                                   'preserve unrelated rules. Cite event_id values in evidence_ids. No rule is adopted here.'}
            first = events[0]
            progress(folder, 'review_reflector', simulation=simulation, playbook_version=parent.version)
            reflection = await gateway.invoke('reflector', task, LifecycleReflection, mock_output={
                'insights': [{'evidence_ids': [first['event_id']],
                              'hypothesis': 'Mock proposal; failure causes require independent validation.',
                              'controlled_change': first['recommended_action']}],
                'used_rule_ids': [], 'rule_feedback': []})
            for item in reflection.insights:
                if len(set(item.evidence_ids)) != len(item.evidence_ids) or not set(item.evidence_ids) <= event_ids:
                    raise ValueError('Reflection cites unknown or duplicate failure event IDs')
            build_delta(parent, [], reflection.rule_feedback, reflection.used_rule_ids, event_ids)
            save('reflection.json', {'cause_status': 'hypothesis', 'confirmed': False, **reflection.model_dump(mode='json')})
            progress(folder, 'review_curator', simulation=simulation, playbook_version=parent.version)
            curation = await gateway.invoke('curator', {**task, 'reflection': reflection.model_dump(mode='json')},
                LifecycleCuration, mock_output={'candidates': [{
                    'action': 'add', 'condition': 'A prior observed failure requires review: '+first['reason_code'],
                    'recommendation': first['recommended_action'], 'evidence_ids': [first['event_id']],
                    'limitations': 'Mock-generated candidate; never eligible for automatic publication.'}]})
            delta = build_delta(parent, curation.candidates, reflection.rule_feedback,
                                reflection.used_rule_ids, event_ids)
            rules = {rule.id: rule.model_copy(deep=True) for rule in parent.rules}
            touched = set()
            for item in delta['updates']:
                if len(item['evidence_ids']) != len(set(item['evidence_ids'])):
                    raise ValueError('Candidate rule repeats evidence IDs')
                target = item['target_rule_id']
                if target is not None:
                    if target in touched:
                        raise ValueError('A candidate must not revise/deprecate one rule twice')
                    touched.add(target)
                if item['action'] == 'deprecate':
                    del rules[target]
                    continue
                identity = target if item['action'] == 'revise' else 'rule-'+item['content_fingerprint'][:24]
                old = rules.get(identity)
                if old is not None and item['action'] == 'add':
                    raise ValueError('Generated stable rule ID collides with an existing rule')
                rules[identity] = Rule(id=identity, condition=item['condition'], recommendation=item['recommendation'],
                    limitations=item['limitations'], evidence_ids=list(dict.fromkeys((old.evidence_ids if old else [])+item['evidence_ids'])),
                    helpful=old.helpful if old else 0, harmful=old.harmful if old else 0)
            candidate = Playbook(version='P0-'+job.rsplit('_', 1)[1], rules=list(rules.values()))
            if playbook_hash(parent) != parent_hash:
                raise ValueError('Frozen proposal parent changed')
            save('context-delta.json', delta)
            save('candidate-playbook.json', candidate.model_dump(mode='json'))
            self._check_bindings(bindings, folder)
            save('model_trace.json', gateway.trace)
            metadata = {'schema': 'rule-proposal/v1', 'proposal_id': job, 'created_at': utc_now(),
                'status': 'candidate_only', 'promoted': False, 'mock_generated': simulation,
                'mock_adapter': adapter_mock, 'simulation': simulation, 'evidence_blocks': evidence_blocks,
                'identity_scope': IDENTITY_SCOPE, 'vlm_model_identities': getattr(gateway, 'model_identities', {}),
                'expected_latest_sha256': expected_latest, 'parent_sha256': parent_hash,
                'candidate_sha256': playbook_hash(candidate), 'changed_rules': len(delta['updates']),
                'evidence_event_ids': sorted(event_ids), 'historical_source_sha256': sorted(historical_sources),
                'files': {name: sha256(folder/name) for name in ('parent-playbook.json', 'failure-evidence.json',
                    'model-config.json', 'reflection.json', 'context-delta.json', 'candidate-playbook.json', 'model_trace.json',
                    'evidence-bindings.json')}}
            metadata['files'].update({file['snapshot']: file['sha256'] for item in bindings for file in item['files']})
            save('proposal.json', metadata)
            relative = (folder/'candidate-playbook.json').relative_to(self.store.root).as_posix()
            with closing(self._connect()) as db, db:
                db.execute('INSERT INTO proposals VALUES (?,?)', (relative, sha256(folder/'proposal.json')))
            result = {'status': 'candidate_only', 'promoted': False, 'mock_generated': simulation,
                      'simulation': simulation, 'evidence_blocks': evidence_blocks,
                      'job_id': job, 'candidate_playbook_path': str(folder/'candidate-playbook.json'),
                      'candidate_metadata_path': str(folder/'proposal.json'), 'parent_sha256': parent_hash,
                      'candidate_sha256': playbook_hash(candidate), 'evidence_count': len(events),
                      'candidate_artifact': self.store.artifact(folder/'candidate-playbook.json', 'candidate_playbook', job),
                      'metadata_artifact': self.store.artifact(folder/'proposal.json', 'rule_proposal_metadata', job)}
            self.store.complete_job(job, result)
            return result
        except Exception as error:
            try:
                save('model_trace.json', gateway.trace)
                from .image_failures import record_runtime_failure
                record_runtime_failure(self.store, folder, error, gateway.trace)
            except Exception:
                pass  # Failure telemetry must not replace the proposal error.
            try:
                self.store.fail_job(job, {'type': type(error).__name__, 'message': 'Rule proposal rejected; no rules adopted.'})
            except Exception:
                pass
            raise

    def _load_proposal(self, candidate_path):
        path = self.store.resolve_input(candidate_path)
        if not path.is_relative_to(self.store.root) or path.name != 'candidate-playbook.json':
            raise ValueError('Validate only a registered lifecycle candidate')
        relative = path.relative_to(self.store.root).as_posix()
        with closing(self._connect()) as db:
            registered = db.execute('SELECT metadata_sha256 FROM proposals WHERE candidate_rel=?', (relative,)).fetchone()
        if registered is None or sha256(path.parent/'proposal.json') != registered[0]:
            raise ValueError('Candidate metadata is unregistered or changed')
        metadata = json.loads((path.parent/'proposal.json').read_text(encoding='utf-8'))
        job = self.store.job_status(metadata['proposal_id'])
        if job.get('kind') != 'rule_proposal' or job.get('status') != 'completed':
            raise ValueError('Only completed lifecycle proposals may be validated')
        for name, digest in metadata['files'].items():
            file = (path.parent/name).resolve()
            if not file.is_relative_to(path.parent.resolve()) or sha256(file) != digest:
                raise ValueError('Immutable proposal evidence/playbook changed')
        parent = Playbook.model_validate_json((path.parent/'parent-playbook.json').read_text(encoding='utf-8'))
        candidate = Playbook.model_validate_json(path.read_text(encoding='utf-8'))
        if playbook_hash(parent) != metadata['parent_sha256'] or playbook_hash(candidate) != metadata['candidate_sha256']:
            raise ValueError('Candidate/parent semantic hash mismatch')
        evidence = json.loads((path.parent/'failure-evidence.json').read_text(encoding='utf-8'))
        current_events = {event['event_id']: event for event in FailureMemory(self.store.root).summary()['events']}
        for event in evidence:
            if not _eligible_event(event) or _json_hash(current_events.get(event['event_id'])) != _json_hash(event):
                raise ValueError('Proposal no longer matches actual observed failure-memory evidence')
        if {event['event_id'] for event in evidence} != set(metadata['evidence_event_ids']):
            raise ValueError('Proposal evidence IDs changed')
        self._check_bindings(json.loads((path.parent/'evidence-bindings.json').read_text(encoding='utf-8')), path.parent)
        return metadata, parent, candidate, path.parent

    def _prepare_cases(self, dataset, dataset_file, folder, historical):
        from .image_inspection import ImageInspectionRequest
        prepared, seen = [], set()
        for case in dataset.cases:
            value = case.image_path if case.image_path.startswith('artifact:') else str(dataset_file.parent/case.image_path)
            path = self.store.resolve_input(value)
            digest, normalized, size = _image_hashes(path)
            if {digest, normalized} & historical:
                raise ValueError('Independent replay source overlaps proposal failure history')
            if normalized in seen:
                raise ValueError('Duplicate source image in independent dataset')
            seen.add(normalized)
            for defect in case.defects:
                if defect.bbox_xyxy[2] > size[0] or defect.bbox_xyxy[3] > size[1]:
                    raise ValueError('Reference bbox exceeds its original image')
            inputs = folder/'inputs'/case.id
            inputs.mkdir(parents=True)
            snapshot = inputs/('source'+path.suffix.lower())
            shutil.copyfile(path, snapshot)
            if sha256(snapshot) != digest or sha256(path) != digest:
                raise ValueError('Independent source changed while making snapshot')
            scope_snapshot = None
            scope_digest = None
            if case.scope_mask_path:
                value = case.scope_mask_path if case.scope_mask_path.startswith('artifact:') else str(dataset_file.parent/case.scope_mask_path)
                scope = self.store.resolve_input(value)
                scope_digest = sha256(scope)
                scope_snapshot = inputs/'scope.png'
                shutil.copyfile(scope, scope_snapshot)
                if sha256(scope_snapshot) != scope_digest or sha256(scope) != scope_digest:
                    raise ValueError('Independent scope mask changed while making snapshot')
                with Image.open(scope_snapshot) as scope_image:
                    if scope_image.size != size or scope_image.mode not in ('1', 'L'):
                        raise ValueError('Reference scope must be an original-sized binary wall mask')
                    colors = set(scope_image.convert('L').getdata())
                    if not colors <= {0, 255} or 255 not in colors:
                        raise ValueError('Reference scope must contain only 0/255 and some wall pixels')
            request = ImageInspectionRequest.model_validate({**dataset.inspection, 'image_path': str(snapshot),
                                      'scope_mask_path': str(scope_snapshot) if scope_snapshot else None})
            prepared.append({'case': case, 'request': request, 'source_sha256': normalized,
                             'original_input_sha256': digest, 'size': size,
                             'scope_sha256': scope_digest,
                             'input_artifact': self.store.artifact(snapshot, 'rule_validation_input', folder.name)})
        return prepared

    def _production_proof(self, answer, expected_source, expected_playbook, identity):
        """A callback's demo=False is not evidence of a production model run."""
        job_id = answer.get('job_id')
        if not isinstance(job_id, str) or not re.fullmatch(r'image_inspection_[a-f0-9]{32}', job_id):
            raise ValueError('Production replay requires an actual completed image workflow job')
        job = self.store.job_status(job_id)
        directory = self.store.job_dir(job_id)
        if job.get('kind') != 'image_inspection' or job.get('status') != 'completed':
            raise ValueError('Production image replay job is not completed')
        stored = job.get('result', {})
        keys = ('status', 'demo', 'source_sha256', 'playbook_sha256', 'expert_identities',
                'expert_config_identities', 'expert_definition_identities', 'expert_implementation_identities',
                'vlm_model_identities', 'candidates')
        if (_json_hash({key: stored.get(key) for key in keys}) != _json_hash({key: answer.get(key) for key in keys})
                or stored.get('playbook_sha256') != expected_playbook
                or sha256(directory/'source.png') != expected_source):
            raise ValueError('Production replay job/source/P0 evidence does not match its result')
        trace = json.loads((directory/'model_trace.json').read_text(encoding='utf-8'))
        if not trace or any(row.get('adapter') == 'mock' or row.get('status') != 'completed'
                            or row.get('model_identity_verified') is not True
                            or row.get('actual_model') != identity['vlm_model_identities'].get(row.get('connection'))
                            for row in trace):
            raise ValueError('Production replay requires verified non-mock API model trace')
        tools = [json.loads(line) for line in (directory/'tool_calls.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if not any(row.get('tool') == 'facade_models_detect' and row.get('status') == 'success' for row in tools):
            raise ValueError('Production replay requires actual MCP expert call evidence')
        artifact = stored.get('artifacts', {}).get('result', {})
        if not artifact.get('uri') or sha256(self.store.resolve_input(artifact['uri'])) != artifact.get('sha256'):
            raise ValueError('Production replay result artifact is missing or changed')
        return {'job_id': job_id, 'files': {name: sha256(directory/name)
                for name in ('source.png', 'model_trace.json', 'tool_calls.jsonl', 'result.json')}}

    def _record_quality_failures(self, folder, report, metrics):
        if report['simulation'] or not report['independent_reference_confirmed']:
            return
        for identity, value, threshold, direction in metrics:
            self.store.write_json(folder/'evaluation.json', report)
            FailureMemory(self.store.root).record(job_id=folder.name, stage='rule_validation',
                reason_code='metric_below_target', occurrence_key='metric:'+identity,
                summary=f'Independent rule replay metric {identity}: value={value}, threshold={threshold}, direction={direction}.',
                evidence_refs=[folder.name+'/evaluation.json', folder.name+'/dataset.json'],
                context={'metric_id': identity, 'value': value, 'threshold': threshold, 'direction': direction,
                         'reference_verified': True, 'dataset_version': report['dataset_version'],
                         'parent_sha256': report['parent_sha256'], 'candidate_sha256': report['candidate_sha256'],
                         'source_sha256s': list(dict.fromkeys(case['source_sha256'] for case in report['cases'])),
                         'evidence_sha256': {'evaluation.json': sha256(folder/'evaluation.json'),
                                             'dataset.json': sha256(folder/'dataset.json')},
                         'simulation': False}, source='evaluator', cause_status='observed')

    async def validate_and_promote(self, candidate_path: str, dataset_path: str, settings: ModelConfig,
                                   *, allow_api=False, runner=None) -> dict:
        metadata, parent, candidate, proposal_folder = self._load_proposal(candidate_path)
        dataset_file = self.store.resolve_input(dataset_path)
        dataset_bytes_hash = sha256(dataset_file)
        dataset = RuleValidationDataset.model_validate_json(dataset_file.read_text(encoding='utf-8-sig'))
        job, folder = self.store.create_job('rule_validation', {'proposal_id': metadata['proposal_id'],
                                                           'dataset_version': dataset.version})
        save = lambda name, value: self.store.write_json(folder/name, value)
        report = {'schema': 'rule-validation/v1', 'job_id': job, 'status': 'not_promoted', 'promoted': False,
                  'proposal_id': metadata['proposal_id'], 'parent_sha256': metadata['parent_sha256'],
                  'candidate_sha256': metadata['candidate_sha256'], 'dataset_version': dataset.version,
                  'dataset_sha256': dataset_bytes_hash, 'independent_reference_confirmed': dataset.independent_reference_confirmed,
                  'targets': dataset.targets.model_dump(), 'iou_threshold': dataset.iou_threshold,
                  'runner': 'injected' if runner is not None else 'image_workflow', 'cases': [],
                  'simulation': runner is not None or metadata['mock_generated'],
                  'evidence_blocks': metadata['evidence_blocks'], 'identity_scope': IDENTITY_SCOPE}
        simulation = report['simulation']
        try:
            from .image_failures import progress
            progress(folder, 'rule_validation', simulation=simulation, playbook_version=parent.version)
            save('dataset.json', dataset.model_dump(mode='json'))
            save('model-config.json', settings.model_dump(mode='json'))
            reason = ('independent_reference_required' if not dataset.independent_reference_confirmed else
                      'mock_proposal' if metadata['mock_adapter'] else
                      'mock_validation_model' if settings.connections[settings.roles['generator']].adapter == 'mock' else
                      'simulation_proposal' if metadata['simulation'] and runner is None else
                      'historical_evidence_not_verifiable' if metadata['evidence_blocks'] and runner is None else
                      'no_rule_changes' if not metadata['changed_rules'] else
                      'insufficient_reference_cases' if len(dataset.cases) < dataset.targets.min_cases else
                      'insufficient_reference_defects' if sum(len(case.defects) for case in dataset.cases) < dataset.targets.min_reference_defects else
                      'stale_parent' if self._current_hash() != metadata['expected_latest_sha256'] else None)
            if reason:
                report['reason'] = reason
                if reason in ('insufficient_reference_cases', 'insufficient_reference_defects'):
                    identity = 'reference_cases' if reason.endswith('cases') else 'reference_defects'
                    value = len(dataset.cases) if identity == 'reference_cases' else sum(len(case.defects) for case in dataset.cases)
                    threshold = dataset.targets.min_cases if identity == 'reference_cases' else dataset.targets.min_reference_defects
                    report['reference_count_check'] = {'metric': identity, 'value': value, 'threshold': threshold, 'passed': False}
                    self._record_quality_failures(folder, report, [(identity, value, threshold, 'min')])
            else:
                prepared = self._prepare_cases(dataset, dataset_file, folder, set(metadata['historical_source_sha256']))
                if runner is None:
                    from .image_inspection import run_image_inspection
                    runner = run_image_inspection
                baseline_rows, candidate_rows = [], []
                settings_hash = _json_hash(settings.model_dump(mode='json'))
                replay_identity = None
                implementation_locks = {}
                for item in prepared:
                    case_report = {'case_id': item['case'].id, 'source_sha256': item['source_sha256'],
                                   'original_input_sha256': item['original_input_sha256'], 'input_artifact': item['input_artifact']}
                    for label, book, expected_hash, rows in (
                            ('baseline', parent, metadata['parent_sha256'], baseline_rows),
                            ('candidate', candidate, metadata['candidate_sha256'], candidate_rows)):
                        frozen = book.model_copy(deep=True)
                        config_copy = settings.model_copy(deep=True)
                        request_copy = item['request'].model_copy(deep=True)
                        request_hash = _json_hash(request_copy.model_dump(mode='json'))
                        answer = runner(request_copy, self.store, config_copy,
                                        playbook=frozen, allow_api=allow_api)
                        answer = await answer if inspect.isawaitable(answer) else answer
                        if (playbook_hash(frozen) != expected_hash or _json_hash(config_copy.model_dump(mode='json')) != settings_hash
                                or _json_hash(request_copy.model_dump(mode='json')) != request_hash):
                            raise ValueError('Replay changed its frozen P0 or model configuration')
                        if not isinstance(answer, dict) or answer.get('demo') is not False:
                            raise ValueError('Mock or undeclared replay results cannot validate P0')
                        if answer.get('status') != 'draft_completed' or answer.get('source_sha256') != item['source_sha256']:
                            raise ValueError('Replay is incomplete or belongs to a different source')
                        experts, models = answer.get('expert_identities'), answer.get('vlm_model_identities')
                        configured = answer.get('expert_config_identities')
                        definitions = answer.get('expert_definition_identities')
                        implementations = answer.get('expert_implementation_identities')
                        if (answer.get('expert_identity_verified') is not True or not isinstance(experts, dict) or not experts
                                or not isinstance(configured, dict) or not configured
                                or not isinstance(definitions, dict) or set(definitions) != set(configured)
                                or not isinstance(implementations, dict) or set(implementations) != set(experts)
                                or not isinstance(models, dict) or not models
                                or not set(item['request'].detector_profiles) <= set(configured)
                                or not set(experts) <= set(configured)
                                or any(not isinstance(members, dict) or not members or
                                       any(not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value)
                                           for value in members.values()) for members in experts.values())
                                or any(not isinstance(value, str) or not value.strip() for value in models.values())):
                            raise ValueError('Replay requires verified expert checkpoints and actual VLM model identities')
                        for value in configured.values():
                            _checkpoint_values(value)
                        if any(not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value)
                               for value in [*definitions.values(), *implementations.values()]):
                            raise ValueError('Invalid expert definition/implementation identity')
                        for profile, used in experts.items():
                            expected = configured[profile]
                            # Preserve named ensemble member -> checkpoint
                            # associations; comparing only hash sets loses them.
                            matches = used == expected if isinstance(expected, dict) else len(used) == 1 and next(iter(used.values())) == expected
                            if not matches:
                                raise ValueError('Actual expert checkpoint identity differs from its frozen configuration')
                            implementation = implementations[profile]
                            if profile in implementation_locks and implementation_locks[profile] != implementation:
                                raise ValueError('Actual expert implementation identity drifted across paired replay')
                            implementation_locks[profile] = implementation
                        # The strategy may legitimately route to an additional
                        # local expert. Compare the entire configured inventory,
                        # not the set of experts a particular strategy used.
                        identity = {'expert_config_identities': configured, 'expert_definition_identities': definitions,
                                    'vlm_model_identities': models}
                        if replay_identity is None:
                            replay_identity = identity
                            report['model_identities'] = identity
                        elif identity != replay_identity:
                            raise ValueError('Expert checkpoint or actual VLM model identity drifted across paired replay')
                        report['expert_implementation_identities'] = dict(implementation_locks)
                        if not simulation:
                            case_report[label+'_production_proof'] = self._production_proof(
                                answer, item['source_sha256'], expected_hash,
                                {**identity, 'expert_implementation_identities': implementations})
                        predictions = []
                        for row in answer.get('candidates', []):
                            if row.get('decision') != 'keep':
                                continue
                            box = row.get('bbox_xyxy')
                            if (row.get('class_code') not in CLASSES or not isinstance(box, (list, tuple)) or len(box) != 4
                                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                                    or not 0 <= box[0] < box[2] <= item['size'][0]
                                    or not 0 <= box[1] < box[3] <= item['size'][1]):
                                raise ValueError('Replay returned invalid kept bbox/class')
                            predictions.append(row)
                        measured = _match(predictions, item['case'].defects, dataset.iou_threshold)
                        rows.append(measured)
                        result_name = f'{item["case"].id}-{label}.json'
                        save(result_name, answer)
                        case_report[label] = {**measured, 'result_artifact': self.store.artifact(folder/result_name, 'rule_replay_result', job)}
                        if _image_hashes(Path(item['request'].image_path))[0:2] != (item['original_input_sha256'], item['source_sha256']):
                            raise ValueError('Replay source snapshot changed')
                        if item['request'].scope_mask_path and sha256(Path(item['request'].scope_mask_path)) != item['scope_sha256']:
                            raise ValueError('Replay scope snapshot changed')
                    report['cases'].append(case_report)
                if sha256(dataset_file) != dataset_bytes_hash:
                    raise ValueError('Reference dataset changed during replay')
                report['baseline'], report['candidate'] = _aggregate(baseline_rows), _aggregate(candidate_rows)
                report['improvement'] = {key: report['candidate'][key]-report['baseline'][key] for key in ('precision', 'recall', 'f1')}
                report['target_checks'] = {key: report['candidate'][key] >= getattr(dataset.targets, 'min_'+key)
                                           for key in ('precision', 'recall', 'f1')}
                report['regression_checks'] = {key: value >= -dataset.targets.max_regression-1e-12
                                               for key, value in report['improvement'].items()}
                report['improvement_check'] = report['improvement']['f1'] >= dataset.targets.min_improvement-1e-12
                report['validation_passed'] = (all(report['target_checks'].values()) and all(report['regression_checks'].values())
                                               and report['improvement_check'])
                if report['validation_passed']:
                    # Re-read registered metadata/evidence after all potentially remote calls.
                    self._load_proposal(candidate_path)
                    if simulation:
                        report['reason'] = 'simulation_no_publication'
                    else:
                        save('evaluation.json', report)
                        published = self._publish(metadata, candidate, proposal_folder, folder)
                        report.update(published)
                else:
                    report['reason'] = 'quality_targets_not_met'
                    failures = [(key, report['candidate'][key], getattr(dataset.targets, 'min_'+key), 'min')
                                for key, passed in report['target_checks'].items() if not passed]
                    failures += [(key+'_regression', -report['improvement'][key], dataset.targets.max_regression, 'max')
                                 for key, passed in report['regression_checks'].items() if not passed]
                    if not report['improvement_check']:
                        failures.append(('f1_improvement', report['improvement']['f1'], dataset.targets.min_improvement, 'min'))
                    self._record_quality_failures(folder, report, failures)
            save('evaluation.json', report)
            report['evaluation_path'] = str(folder/'evaluation.json')
            report['evaluation_artifact'] = self.store.artifact(folder/'evaluation.json', 'rule_validation', job)
            self.store.complete_job(job, report)
            return report
        except Exception as error:
            # Publication commits before optional final output/job bookkeeping.
            # A subsequent IO failure must never claim that latest stayed old.
            try:
                committed = self._current_hash() == metadata['candidate_sha256']
            except Exception:
                committed = report.get('promoted', False)
            report.update(status='published_with_output_failure' if committed else 'validation_failed',
                          promoted=committed, publication_committed=committed,
                          reason='post_publication_output_failure' if committed else 'replay_or_integrity_failure',
                          error_type=type(error).__name__)
            try:
                save('evaluation.json', report)
                from .image_failures import record_runtime_failure
                record_runtime_failure(self.store, folder, error, [])
            except Exception:
                pass
            try:
                self.store.fail_job(job, {'type': type(error).__name__, 'publication_committed': committed,
                                         'message': 'P0 publication committed; output bookkeeping failed.' if committed
                                                    else 'Rule validation failed; P0 was not adopted.'})
            except Exception:
                pass
            raise

    def _publish(self, metadata, candidate, proposal_folder, validation_folder):
        # The transaction serializes the parent check and publication. A failed
        # competitor can never overwrite latest, even after a long remote replay.
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            if self._current_hash(db) != metadata['expected_latest_sha256']:
                return {'status': 'not_promoted', 'promoted': False, 'reason': 'stale_parent'}
            directory = self.root/'versions'/candidate.version
            directory.mkdir(exist_ok=False)
            files = {'playbook.json': proposal_folder/'candidate-playbook.json',
                     'proposal.json': proposal_folder/'proposal.json', 'evaluation.json': validation_folder/'evaluation.json',
                     'dataset.json': validation_folder/'dataset.json', 'model-config.json': validation_folder/'model-config.json'}
            # Preserve proposal provenance alongside the measured comparison.
            # Large image snapshots remain registered in the proposal job;
            # these copies retain their binding manifests and declared hashes.
            for name in ('parent-playbook.json', 'failure-evidence.json', 'evidence-bindings.json',
                         'reflection.json', 'context-delta.json', 'model_trace.json'):
                files['proposal/'+name] = proposal_folder/name
            for path in validation_folder.glob('*-baseline.json'):
                files[path.name] = path
            for path in validation_folder.glob('*-candidate.json'):
                files[path.name] = path
            for name, source in files.items():
                (directory/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, directory/name)
            manifest = {'schema': 'published-p0/v1', 'version': candidate.version, 'created_at': utc_now(),
                        'playbook_sha256': metadata['candidate_sha256'], 'parent_sha256': metadata['parent_sha256'],
                        'proposal_id': metadata['proposal_id'], 'validation_job_id': validation_folder.name,
                        'files': {name: sha256(directory/name) for name in files},
                        'identity_scope': IDENTITY_SCOPE,
                        'publication_authority': 'catalog.sqlite3 transaction; no in-flight image P0 changes'}
            self.store.write_json(directory/'manifest.json', manifest)
            db.execute('INSERT INTO versions VALUES (?,?,?)', (candidate.version, metadata['candidate_sha256'], sha256(directory/'manifest.json')))
            db.execute('INSERT INTO current VALUES (1,?,?) ON CONFLICT(id) DO UPDATE SET version=excluded.version, playbook_sha256=excluded.playbook_sha256',
                       (candidate.version, metadata['candidate_sha256']))
        return {'status': 'promoted', 'promoted': True, 'version': candidate.version,
                'playbook_path': str(directory/'playbook.json'), 'manifest_path': str(directory/'manifest.json')}
