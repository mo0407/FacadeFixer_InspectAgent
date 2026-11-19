import asyncio
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from facade_mcp.failure_memory import FailureMemory
from facade_mcp.model_gateway import ModelConfig
from facade_mcp.playbook import Playbook, Rule
from facade_mcp.rule_lifecycle import (
    ReferenceDefect, RuleLifecycle, RuleValidationDataset, _match, playbook_hash,
)
from facade_mcp.storage import ArtifactStore, sha256


BOX = [10, 10, 30, 30]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('RULE_TEST_ENDPOINT', 'https://fixture.invalid/chat/completions')
    monkeypatch.setenv('RULE_TEST_KEY', 'fixture-key')
    return ArtifactStore(tmp_path/'workspace', [tmp_path])


def config(mock=False):
    return ModelConfig.model_validate({
        'schema_version': 'model-gateway-v1', 'api_enabled': not mock,
        'connections': {'test': {'adapter': 'mock' if mock else 'chat_completions',
            'model': 'mock-rule' if mock else 'fixture-vision', 'supports_images': True,
            **({} if mock else {'endpoint_env': 'RULE_TEST_ENDPOINT', 'api_key_env': 'RULE_TEST_KEY'})}},
        'roles': {role: 'test' for role in ('generator', 'reflector', 'curator')},
    })


def failure(store, *, occurrence='one', context=None, reason='missed_defect', cause='observed', refs=None):
    history = store.root/'history-one'
    history.mkdir(exist_ok=True)
    source = history/'source.png'
    Image.new('RGB', (80, 60), '#887766').save(source)
    (history/'result.json').write_text(json.dumps({'source_sha256': sha256(source),
                                                  'request': {'image_path': str(source)}}), encoding='utf-8')
    return FailureMemory(store.root).record(
        job_id='history-one', stage='evaluation', reason_code=reason, occurrence_key=occurrence,
        evidence_refs=['history-one/result.json', 'history-one/source.png'] if refs is None else refs,
        context={'source_sha256': sha256(source), **(context or {})}, source='evaluator', cause_status=cause,
        hypothesis='Unconfirmed cause' if cause == 'hypothesis' else None)


def transport(*, invented=False, changes=None):
    def handler(request):
        payload = json.loads(request.content)
        task = json.loads(payload['messages'][1]['content'])
        evidence = task['allowed_evidence_ids']
        if 'reflection' not in task:
            output = {'insights': [{'evidence_ids': evidence[:1], 'hypothesis': 'Local evidence may need closer review.',
                                   'controlled_change': 'Inspect uncertain wall regions before finishing.'}],
                      'used_rule_ids': [], 'rule_feedback': []}
        else:
            output = {'candidates': changes if changes is not None else [{
                'action': 'add', 'condition': 'An uncertain wall region is present',
                'recommendation': 'Inspect that region before finishing.',
                'evidence_ids': ['invented-event'] if invented else evidence[:1],
                'limitations': 'Validated only on the declared independent replay dataset.'}]}
        return httpx.Response(200, json={'id': 'fixture-response', 'model': 'fixture-vision',
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(output)}}]})
    return httpx.MockTransport(handler)


def propose(lifecycle, **kwargs):
    return asyncio.run(lifecycle.propose(config(), allow_api=True, transport=transport(), **kwargs))


def dataset(tmp_path, **changes):
    image = tmp_path/'reference.png'
    Image.new('RGB', (80, 60), '#c9b79b').save(image)
    image_two = tmp_path/'reference-two.png'
    Image.new('RGB', (80, 60), '#c9b79c').save(image_two)
    value = {'version': 'human-reference-v1', 'independent_reference_confirmed': True,
             'cases': [{'id': 'case-1', 'image_path': image.name,
                        'defects': [{'class_code': 'C', 'bbox_xyxy': BOX}]},
                       {'id': 'case-2', 'image_path': image_two.name,
                        'defects': [{'class_code': 'C', 'bbox_xyxy': BOX}]}],
             'targets': {'min_precision': .9, 'min_recall': .9, 'max_regression': 0.0,
                         'min_cases': 2, 'min_reference_defects': 2},
             **changes}
    path = tmp_path/'dataset.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


class Replay:
    """Explicit test double; injected runs are recorded as injected in validation."""
    def __init__(self, *, baseline=None, candidate=None, demo=False, decision='keep'):
        self.baseline = [] if baseline is None else baseline
        self.candidate = [BOX] if candidate is None else candidate
        self.demo, self.decision, self.calls = demo, decision, []

    async def __call__(self, request, store, settings, *, playbook, allow_api):
        self.calls.append({'source': request.image_path, 'version': playbook.version,
                           'rules': playbook.model_dump(mode='json'), 'request': request.model_dump(mode='json')})
        boxes = self.candidate if playbook.rules else self.baseline
        return {'status': 'draft_completed', 'demo': self.demo, 'source_sha256': sha256(Path(request.image_path)),
                'expert_identities': {profile: {'checkpoint': 'b'*64} for profile in request.detector_profiles},
                'expert_config_identities': {profile: 'b'*64 for profile in request.detector_profiles},
                'expert_definition_identities': {profile: 'c'*64 for profile in request.detector_profiles},
                'expert_implementation_identities': {profile: 'e'*64 for profile in request.detector_profiles},
                'expert_identity_verified': True, 'vlm_model_identities': {'test': 'fixture-vision-v1'},
                'candidates': [{'candidate_id': f'c-{index}', 'class_code': 'C', 'bbox_xyxy': box,
                                'decision': self.decision} for index, box in enumerate(boxes)]}


def test_proposal_uses_only_observed_nonmock_failure_events_and_freezes_parent(store):
    accepted = failure(store)
    failure(store, occurrence='hypothesis', cause='hypothesis')
    failure(store, occurrence='mock', context={'demo': True})
    failure(store, occurrence='adapter-mock', context={'vlm_adapter': 'mock'})
    failure(store, occurrence='observation', reason='unknown_class')
    failure(store, occurrence='no-evidence', reason='input_missing', refs=[])
    parent = Playbook(version='P0-manual', rules=[Rule(id='old', condition='Keep a prior constraint',
                                                     recommendation='Preserve it')])
    original = parent.model_dump_json()
    lifecycle = RuleLifecycle(store)
    result = propose(lifecycle, parent=parent)
    metadata = json.loads(Path(result['candidate_metadata_path']).read_text(encoding='utf-8'))
    assert metadata['evidence_event_ids'] == [accepted['event_id']]
    assert accepted['context']['source_sha256'] in metadata['historical_source_sha256']
    assert metadata['evidence_blocks'] == []
    candidate = Playbook.model_validate_json(Path(result['candidate_playbook_path']).read_text(encoding='utf-8'))
    assert len(candidate.rules) == 2 and candidate.rules[0].id == 'old'
    assert candidate.rules[1].evidence_ids == [accepted['event_id']]
    reflection = json.loads((Path(result['candidate_playbook_path']).parent/'reflection.json').read_text(encoding='utf-8'))
    assert reflection['cause_status'] == 'hypothesis' and reflection['confirmed'] is False
    assert parent.model_dump_json() == original
    assert result['parent_sha256'] == playbook_hash(parent)
    assert not result['promoted'] and lifecycle.latest() is None


def test_no_eligible_failure_or_invented_evidence_cannot_create_adoptable_rule(store):
    lifecycle = RuleLifecycle(store)
    failure(store, cause='hypothesis')
    with pytest.raises(ValueError, match='No evidence-backed'):
        propose(lifecycle)
    failure(store, occurrence='real')
    with pytest.raises(ValueError, match='unknown evidence'):
        asyncio.run(lifecycle.propose(config(), allow_api=True, transport=transport(invented=True)))
    assert lifecycle.latest() is None


def test_mock_proposal_is_candidate_only_and_cannot_be_promoted(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = asyncio.run(lifecycle.propose(config(mock=True)))
    replay = Replay()
    result = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)),
                                                       config(), runner=replay))
    assert proposal['mock_generated'] is True
    assert result['reason'] == 'mock_proposal' and not result['promoted']
    assert replay.calls == [] and lifecycle.latest() is None


def test_injected_replay_compares_metrics_but_never_automatically_publishes(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    replay = Replay()
    result = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)),
                                                       config(), runner=replay))
    assert result['validation_passed'] and not result['promoted']
    assert result['reason'] == 'simulation_no_publication' and result['simulation'] is True
    assert result['baseline']['recall'] == 0 and result['candidate']['recall'] == 1
    assert result['improvement']['f1'] == 1
    assert len(replay.calls) == 4 and replay.calls[0]['source'] == replay.calls[1]['source']
    assert replay.calls[0]['request'] == replay.calls[1]['request']
    assert result['runner'] == 'injected'
    assert lifecycle.latest() is None
    # The private publication primitive is tested separately from the public
    # simulation gate. This creates only a temporary unit-test catalog fixture.
    metadata, _, candidate, proposal_folder = lifecycle._load_proposal(proposal['candidate_playbook_path'])
    published = lifecycle._publish(metadata, candidate, proposal_folder, Path(result['evaluation_path']).parent)
    current = lifecycle.latest()
    assert current is not None and playbook_hash(current) == proposal['candidate_sha256']
    # Loading latest returns a new frozen snapshot, not a shared mutable object.
    current.rules[0].recommendation = 'Outside mutation'
    assert lifecycle.latest().rules[0].recommendation != 'Outside mutation'
    manifest = json.loads(Path(published['manifest_path']).read_text(encoding='utf-8'))
    assert {'dataset.json', 'evaluation.json', 'playbook.json', 'case-1-baseline.json', 'case-1-candidate.json'} <= set(manifest['files'])
    assert manifest['parent_sha256'] == proposal['parent_sha256']


@pytest.mark.parametrize('change,reason', [
    ({'independent_reference_confirmed': False}, 'independent_reference_required'),
])
def test_independent_ground_truth_is_required_before_any_replay(store, tmp_path, change, reason):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    replay = Replay()
    result = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path, **change)),
                                                       config(), runner=replay))
    assert result['reason'] == reason and not result['promoted'] and replay.calls == []


def test_mock_validation_model_and_mock_replay_output_cannot_pass(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    data = str(dataset(tmp_path))
    replay = Replay()
    result = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], data,
                                                       config(mock=True), runner=replay))
    assert result['reason'] == 'mock_validation_model' and replay.calls == []
    with pytest.raises(ValueError, match='Mock or undeclared replay'):
        asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], data,
                                                   config(), runner=Replay(demo=True)))
    assert lifecycle.latest() is None


def test_independent_source_cannot_reuse_failure_history_image(store, tmp_path):
    data = dataset(tmp_path)
    failure(store, context={'source_sha256': sha256(tmp_path/'reference.png')})
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    replay = Replay()
    with pytest.raises(ValueError, match='overlaps proposal failure history'):
        asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(data), config(), runner=replay))
    assert replay.calls == [] and lifecycle.latest() is None


def test_only_keep_results_count_and_duplicate_predictions_cannot_reuse_gt(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    data = str(dataset(tmp_path))
    deferred = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], data,
                                                         config(), runner=Replay(decision='defer')))
    assert not deferred['promoted'] and deferred['candidate']['recall'] == 0
    duplicated = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], data,
                                                          config(), runner=Replay(candidate=[BOX, BOX])))
    assert not duplicated['promoted']
    assert duplicated['candidate']['tp'] == 2 and duplicated['candidate']['fp'] == 2
    assert duplicated['candidate']['precision'] == .5


def test_regression_gate_blocks_even_when_minimum_targets_are_met(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    data = dataset(tmp_path, targets={'min_precision': .4, 'min_recall': .9, 'max_regression': 0.0,
                                      'min_cases': 2, 'min_reference_defects': 2})
    result = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(data), config(),
                                                       runner=Replay(baseline=[BOX], candidate=[BOX, BOX])))
    assert all(result['target_checks'].values())
    assert not result['regression_checks']['precision']
    assert not result['promoted'] and lifecycle.latest() is None


def test_class_matching_is_one_to_one_and_can_find_non_greedy_assignment():
    references = [ReferenceDefect(class_code='C', bbox_xyxy=[0, 0, 20, 20]),
                  ReferenceDefect(class_code='C', bbox_xyxy=[5, 0, 25, 20])]
    predictions = [{'class_code': 'C', 'bbox_xyxy': [0, 0, 20, 20]},
                   {'class_code': 'C', 'bbox_xyxy': [0, 0, 13, 20]},
                   {'class_code': 'S', 'bbox_xyxy': [0, 0, 20, 20]}]
    measured = _match(predictions, references, .5)
    assert measured['tp'] == 2 and measured['fp'] == 1 and measured['fn'] == 0
    assert len({pair['reference_index'] for pair in measured['matches']}) == 2


@pytest.mark.parametrize('name', ['candidate-playbook.json', 'proposal.json', 'failure-evidence.json'])
def test_candidate_and_metadata_tampering_rejected_before_replay(store, tmp_path, name):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    path = Path(proposal['candidate_playbook_path']).parent/name
    path.write_text(path.read_text(encoding='utf-8')+' ', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)),
                                                   config(), runner=Replay()))
    assert lifecycle.latest() is None


def test_unregistered_external_candidate_and_changed_published_files_are_rejected(store, tmp_path):
    external = tmp_path/'candidate-playbook.json'
    external.write_text(Playbook(version='fake').model_dump_json(), encoding='utf-8')
    lifecycle = RuleLifecycle(store)
    with pytest.raises(ValueError, match='registered lifecycle'):
        asyncio.run(lifecycle.validate_and_promote(str(external), str(dataset(tmp_path)), config(), runner=Replay()))
    failure(store)
    proposal = propose(lifecycle)
    result = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)), config(), runner=Replay()))
    metadata, _, candidate, proposal_folder = lifecycle._load_proposal(proposal['candidate_playbook_path'])
    published = lifecycle._publish(metadata, candidate, proposal_folder, Path(result['evaluation_path']).parent)
    Path(published['playbook_path']).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Published P0 files changed'):
        lifecycle.latest()


def test_replay_cannot_mutate_frozen_parent_or_candidate(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    base = Replay()

    async def mutating(request, store, settings, *, playbook, allow_api):
        answer = await base(request, store, settings, playbook=playbook, allow_api=allow_api)
        playbook.version = 'changed-during-run'
        return answer

    with pytest.raises(ValueError, match='frozen P0'):
        asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)),
                                                   config(), runner=mutating))
    assert lifecycle.latest() is None


@pytest.mark.parametrize('drift', ['expert', 'vlm', 'missing'])
def test_entire_paired_replay_requires_stable_expert_and_vlm_identities(store, tmp_path, drift):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    base = Replay()

    async def changing(request, store, settings, *, playbook, allow_api):
        answer = await base(request, store, settings, playbook=playbook, allow_api=allow_api)
        if playbook.rules:
            if drift == 'expert':
                answer['expert_identities'][request.detector_profiles[0]]['checkpoint'] = 'c'*64
            elif drift == 'vlm':
                answer['vlm_model_identities']['test'] = 'different-provider-model'
            else:
                del answer['vlm_model_identities']
        return answer

    with pytest.raises(ValueError, match='identit'):
        asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)),
                                                   config(), runner=changing))
    assert lifecycle.latest() is None


def test_local_expert_routing_may_change_when_configured_checkpoint_inventory_is_fixed(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    base = Replay()

    async def local_strategy(request, store, settings, *, playbook, allow_api):
        answer = await base(request, store, settings, playbook=playbook, allow_api=allow_api)
        answer['expert_config_identities']['local-specialist'] = 'd'*64
        answer['expert_definition_identities']['local-specialist'] = 'f'*64
        if playbook.rules:
            answer['expert_identities']['local-specialist'] = {'checkpoint': 'd'*64}
            answer['expert_implementation_identities']['local-specialist'] = '9'*64
        return answer

    result = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)),
                                                       config(), runner=local_strategy))
    assert result['validation_passed'] and not result['promoted']
    assert result['model_identities']['expert_config_identities']['local-specialist'] == 'd'*64


def test_native_single_checkpoint_can_report_its_member_profile_name(store, tmp_path):
    failure(store)
    lifecycle=RuleLifecycle(store)
    proposal=propose(lifecycle)
    base=Replay()
    async def native_single(request, store, settings, *, playbook, allow_api):
        answer=await base(request,store,settings,playbook=playbook,allow_api=allow_api)
        answer['expert_identities']={profile:{profile:'b'*64} for profile in request.detector_profiles}
        return answer
    result=asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'],str(dataset(tmp_path)),
                                                     config(),runner=native_single))
    assert result['validation_passed'] and result['simulation'] and not result['promoted']


def test_named_ensemble_members_cannot_swap_checkpoint_associations(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    data = dataset(tmp_path, inspection={'detector_profiles': ['ensemble']})
    base = Replay()

    async def swapped(request, store, settings, *, playbook, allow_api):
        answer = await base(request, store, settings, playbook=playbook, allow_api=allow_api)
        answer['expert_config_identities']['ensemble'] = {'member-a': 'b'*64, 'member-b': 'c'*64}
        answer['expert_identities']['ensemble'] = ({'member-a': 'c'*64, 'member-b': 'b'*64} if playbook.rules
                                                  else {'member-a': 'b'*64, 'member-b': 'c'*64})
        return answer

    with pytest.raises(ValueError, match='checkpoint identity differs'):
        asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(data),
                                                   config(), runner=swapped))
    assert lifecycle.latest() is None


@pytest.mark.parametrize('drift', ['definition', 'implementation', 'missing-definition', 'missing-implementation',
                                  'invalid-definition', 'invalid-implementation'])
def test_definition_and_implementation_identity_are_required_and_locked(store, tmp_path, drift):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    base = Replay()

    async def changed(request, store, settings, *, playbook, allow_api):
        answer = await base(request, store, settings, playbook=playbook, allow_api=allow_api)
        # Include an unused configured specialist to ensure definitions are
        # checked for the whole inventory, not only an arm's selected experts.
        answer['expert_config_identities']['unused-specialist'] = 'd'*64
        answer['expert_definition_identities']['unused-specialist'] = 'f'*64
        if playbook.rules:
            if drift == 'definition':
                answer['expert_definition_identities']['unused-specialist'] = '1'*64
            elif drift == 'implementation':
                answer['expert_implementation_identities'][request.detector_profiles[0]] = '2'*64
            elif drift == 'missing-definition':
                del answer['expert_definition_identities']
            elif drift == 'missing-implementation':
                del answer['expert_implementation_identities']
            elif drift == 'invalid-definition':
                answer['expert_definition_identities']['unused-specialist'] = 'not-a-hash'
            else:
                answer['expert_implementation_identities'][request.detector_profiles[0]] = 'not-a-hash'
        return answer

    with pytest.raises(ValueError, match='identit'):
        asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)),
                                                   config(), runner=changed))
    assert lifecycle.latest() is None


def test_implementation_is_locked_across_cases_as_well_as_arms(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    base = Replay()

    async def changed_on_next_case(request, store, settings, *, playbook, allow_api):
        answer = await base(request, store, settings, playbook=playbook, allow_api=allow_api)
        if len(base.calls) == 3:
            answer['expert_implementation_identities'][request.detector_profiles[0]] = '2'*64
        return answer

    with pytest.raises(ValueError, match='implementation identity drifted'):
        asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)),
                                                   config(), runner=changed_on_next_case))


def test_transport_is_explicit_simulation_and_cannot_launch_production_promotion(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    assert proposal['mock_generated'] is True and proposal['simulation'] is True
    result = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)), config()))
    assert result['reason'] == 'simulation_proposal' and result['cases'] == []
    assert lifecycle.latest() is None


def test_no_improvement_and_insufficient_references_cannot_qualify(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    no_gain = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)),
                                                       config(), runner=Replay(baseline=[BOX], candidate=[BOX])))
    assert all(no_gain['target_checks'].values()) and no_gain['improvement']['f1'] == 0
    assert no_gain['improvement_check'] is False and no_gain['validation_passed'] is False
    data = json.loads(dataset(tmp_path).read_text(encoding='utf-8'))
    data['targets']['min_cases'] = 3
    path = tmp_path/'too-small.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    replay = Replay()
    too_small = asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(path),
                                                          config(), runner=replay))
    assert too_small['reason'] == 'insufficient_reference_cases' and replay.calls == []
    # A simulation never invents real quality-failure observations.
    assert len([event for event in FailureMemory(store.root).summary()['events'] if event['kind'] == 'quality']) == 1


def test_quality_history_without_verifiable_image_hash_stays_blocked_candidate(store):
    failure(store, context={'source_sha256': None})
    proposal = propose(RuleLifecycle(store))
    assert any(item['reason'] == 'quality_source_hash_not_verifiable' for item in proposal['evidence_blocks'])
    assert proposal['promoted'] is False


@pytest.mark.parametrize('where', ['original', 'snapshot'])
def test_historical_image_sources_and_evidence_snapshots_are_rechecked_before_replay(store, tmp_path, where):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposal = propose(lifecycle)
    directory = Path(proposal['candidate_playbook_path']).parent
    bindings = json.loads((directory/'evidence-bindings.json').read_text(encoding='utf-8'))
    file = next(file for file in bindings[0]['files'] if file['origin_path'].endswith('.png'))
    target = Path(file['origin_path']) if where == 'original' else directory/file['snapshot']
    Image.new('RGB', (80, 60), '#123456').save(target)
    with pytest.raises(ValueError, match='changed'):
        asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], str(dataset(tmp_path)),
                                                   config(), runner=Replay()))
    assert lifecycle.latest() is None


def test_historical_reviewer_hash_mismatch_rejected_and_runtime_error_counted(store):
    failure(store, context={'evidence_sha256': {'source.png': '0'*64}})
    with pytest.raises(ValueError, match='reviewer evidence hash changed'):
        propose(RuleLifecycle(store))
    runtime = [event for event in FailureMemory(store.root).summary()['events'] if event['kind'] == 'runtime']
    assert len(runtime) == 1 and runtime[0]['context']['simulation'] is True
    assert runtime[0]['reason_code'] == 'unclassified_runtime_error'


def test_production_proof_rejects_callback_result_without_actual_job(store, tmp_path):
    lifecycle = RuleLifecycle(store)
    with pytest.raises(ValueError, match='actual completed image workflow job'):
        lifecycle._production_proof({'demo': False}, 'a'*64, 'b'*64, {'vlm_model_identities': {'test': 'fixture'}})


def test_evaluator_quality_failure_counts_are_idempotent_and_simulations_are_excluded(store):
    lifecycle = RuleLifecycle(store)
    job, folder = store.create_job('rule_validation', {})
    store.write_json(folder/'dataset.json', {'version': 'unit-reference-fixture'})
    report = {'simulation': True, 'independent_reference_confirmed': True, 'dataset_version': 'unit-reference-fixture',
              'parent_sha256': 'a'*64, 'candidate_sha256': 'b'*64, 'cases': []}
    metrics = [('recall', .5, .9, 'min')]
    lifecycle._record_quality_failures(folder, report, metrics)
    assert FailureMemory(store.root).summary(job)['events'] == []
    # Exercise just the ledger adapter using declared unit-test measurements.
    report['simulation'] = False
    lifecycle._record_quality_failures(folder, report, metrics)
    lifecycle._record_quality_failures(folder, report, metrics)
    events = FailureMemory(store.root).summary(job)['events']
    assert len(events) == 1 and events[0]['reason_code'] == 'metric_below_target'
    assert events[0]['context']['value'] == .5 and events[0]['context']['threshold'] == .9


def test_atomic_publication_primitive_cannot_overwrite_newer_parent(store, tmp_path):
    failure(store)
    lifecycle = RuleLifecycle(store)
    proposals = [propose(lifecycle), propose(lifecycle)]
    data = str(dataset(tmp_path))

    validations = [asyncio.run(lifecycle.validate_and_promote(proposal['candidate_playbook_path'], data,
                              config(), runner=Replay())) for proposal in proposals]
    assert all(result['simulation'] and not result['promoted'] for result in validations)
    from concurrent.futures import ThreadPoolExecutor

    def publish(index):
        metadata, _, candidate, proposal_folder = lifecycle._load_proposal(proposals[index]['candidate_playbook_path'])
        return lifecycle._publish(metadata, candidate, proposal_folder, Path(validations[index]['evaluation_path']).parent)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, (0, 1)))
    assert sum(result['promoted'] for result in results) == 1
    loser = next(result for result in results if not result['promoted'])
    assert loser['reason'] == 'stale_parent'
    winner = next(result for result in results if result['promoted'])
    assert lifecycle.latest().version == winner['version']


def test_dataset_is_strict_and_uses_canonical_business_classes():
    base = {'version': 'v1', 'independent_reference_confirmed': True,
            'cases': [{'id': 'one', 'image_path': 'one.png', 'defects': [{'class_code': 'B', 'bbox_xyxy': BOX}]}],
            'targets': {'min_precision': .9, 'min_recall': .9}}
    assert RuleValidationDataset.model_validate(base).cases[0].defects[0].class_code == 'B'
    with pytest.raises(ValidationError):
        RuleValidationDataset.model_validate({**base, 'independent_reference_confirmed': 'yes'})
    with pytest.raises(ValidationError):
        RuleValidationDataset.model_validate({**base, 'arbitrary_runner_path': 'plugin.py'})
    with pytest.raises(ValidationError):
        RuleValidationDataset.model_validate({**base, 'inspection': {'image_path': 'override.png'}})
