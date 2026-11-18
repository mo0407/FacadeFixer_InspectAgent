import asyncio
import json
from pathlib import Path

import httpx
import pytest

from facade_mcp.playbook import Playbook, Rule, RuleFeedback, build_delta
from facade_mcp.review_cycle import Experience, run_review
from test_model_gateway import config, request


def snapshot():
    return Playbook(version='empty', rules=[Rule(id='sampling-00001',
        condition='Near wall', recommendation='Use dense frames')])


def candidate(action='add', **kwargs):
    return Experience(action=action, condition='Near wall', recommendation='Use dense frames',
                      evidence_ids=['e1'], limitations='Needs independent validation', **kwargs)


def test_delta_is_deduplicated_and_snapshot_immutable():
    book = snapshot()
    original = book.model_dump_json()
    feedback = RuleFeedback(rule_id='sampling-00001', tag='helpful', evidence_ids=['e1'], reason='verified measurement')
    delta = build_delta(book, [candidate(), candidate('revise', target_rule_id='sampling-00001')],
                        [feedback], ['sampling-00001'], {'e1'})
    assert delta['suppressed_duplicate_count'] == 1
    assert delta['updates'][0]['target_rule_id'] == 'sampling-00001'
    assert not delta['promoted'] and original == book.model_dump_json()
    assert book.rules[0].helpful == 0


@pytest.mark.parametrize('action,target', [('revise',None),('deprecate','missing'),('add','sampling-00001')])
def test_invalid_delta_targets(action,target):
    with pytest.raises(ValueError):
        build_delta(snapshot(),[candidate(action,target_rule_id=target)],[],[],{'e1'})


def test_feedback_must_cite_a_used_rule_and_valid_evidence():
    item=RuleFeedback(rule_id='sampling-00001',tag='harmful',evidence_ids=['unknown'],reason='test')
    with pytest.raises(ValueError):
        build_delta(snapshot(),[],[item],['sampling-00001'],{'e1'})
    item.evidence_ids=['e1']
    with pytest.raises(ValueError):
        build_delta(snapshot(),[],[item],[],{'e1'})


def test_context_reaches_all_roles_and_refs_are_structured(tmp_path,monkeypatch):
    monkeypatch.setenv('TEST_FACADE_ENDPOINT','https://fixture.invalid/chat/completions')
    monkeypatch.setenv('TEST_FACADE_KEY','fixture-key')
    seen=[]
    def handler(req):
        data=json.loads(req.content)
        task=json.loads(data['messages'][1]['content'])
        assert task['playbook']['rules'][0]['id']=='sampling-00001'
        seen.append(task)
        outputs=[
            {'stages':['sampling'],'rationale':'use context','used_rule_ids':['sampling-00001']},
            {'insights':[], 'rule_feedback':[{'rule_id':'sampling-00001','tag':'neutral','evidence_ids':['e1'],'reason':'pending evidence'}]},
            {'candidates':[]},
        ]
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(outputs[len(seen)-1])}}]})
    result=asyncio.run(run_review(config(True),request(),tmp_path,playbook=snapshot(),
                                 allow_api=True,transport=httpx.MockTransport(handler)))
    folder=Path(result['output'])
    assert len(seen)==3 and (folder/'playbook_snapshot.json').is_file()
    delta=json.loads((folder/'context_delta.json').read_text())
    assert delta['rule_feedback'][0]['tag']=='neutral'
    assert result['acceptance_status']=='pending_review'


def test_version_mismatch_fails_before_run_creation(tmp_path):
    book=snapshot()
    book.version='other'
    with pytest.raises(ValueError,match='version'):
        asyncio.run(run_review(config(),request(),tmp_path,playbook=book))
    assert not list(tmp_path.iterdir())
