"""Immutable ACE-style context snapshots and validated candidate deltas."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Literal

from pydantic import Field, model_validator

from .model_gateway import Strict


class Rule(Strict):
    id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)
    limitations: str = ''
    evidence_ids: list[str] = Field(default_factory=list)
    helpful: int = Field(default=0, ge=0)
    harmful: int = Field(default=0, ge=0)


class Playbook(Strict):
    version: str = Field(min_length=1)
    rules: list[Rule] = Field(default_factory=list)

    @model_validator(mode='after')
    def unique_ids(self):
        if len({rule.id for rule in self.rules}) != len(self.rules):
            raise ValueError('Duplicate playbook rule IDs')
        return self


class RuleFeedback(Strict):
    rule_id: str
    tag: Literal['helpful', 'harmful', 'neutral']
    evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


def fingerprint(condition, recommendation):
    def normalize(value):
        return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())
    content = [normalize(condition), normalize(recommendation)]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode('utf-8')).hexdigest()


def build_delta(playbook, candidates, feedback, used_ids, evidence_ids):
    """Validate references and retain proposals without mutating adopted rules."""
    rules = {rule.id: rule for rule in playbook.rules}
    if len(set(used_ids)) != len(used_ids) or not set(used_ids) <= rules.keys():
        raise ValueError('Plan cites unknown or duplicate rule IDs')
    seen_feedback = set()
    for item in feedback:
        if item.rule_id not in used_ids or item.rule_id in seen_feedback:
            raise ValueError('Feedback must reference a unique rule used by the generator')
        if not set(item.evidence_ids) <= evidence_ids:
            raise ValueError('Rule feedback cites unknown evidence')
        seen_feedback.add(item.rule_id)
    existing = {fingerprint(rule.condition, rule.recommendation) for rule in playbook.rules}
    updates = []
    duplicates = []
    for item in candidates:
        if not item.evidence_ids or not set(item.evidence_ids) <= evidence_ids:
            raise ValueError('Candidate cites missing or unknown evidence')
        target = item.target_rule_id
        if item.action == 'add' and target is not None:
            raise ValueError('Add cannot target an existing rule')
        if item.action != 'add' and target not in rules:
            raise ValueError('Revise/deprecate requires an existing target rule')
        digest = fingerprint(item.condition, item.recommendation)
        if item.action == 'add' and digest in existing:
            duplicates.append(digest)
            continue
        if item.action == 'add':
            existing.add(digest)
        updates.append({'proposal_id': f'delta-{len(updates)+1:05d}',
                        'content_fingerprint': digest, **item.model_dump()})
    return {'schema': 'facade-context-delta/v1', 'parent_version': playbook.version,
            'parent_sha256': hashlib.sha256(playbook.model_dump_json().encode()).hexdigest(),
            'status': 'candidate_only', 'promoted': False, 'updates': updates,
            'suppressed_duplicate_count': len(duplicates),
            'rule_feedback': [item.model_dump() for item in feedback],
            'feedback_scope': 'model judgment backed by cited evidence; adoption requires independent validation'}
