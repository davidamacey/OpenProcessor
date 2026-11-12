"""A class name matches an item by its class; the detector's proposal name only names an item that
has no class yet. Otherwise an item relabelled away from the detector's guess is listed under the
guess and counted under its class, and the two numbers drift (``GET /classes`` vs the item list)."""

from __future__ import annotations

from src.services.curation.name_match import name_clause, name_matches


def test_a_relabelled_item_is_not_named_by_the_detectors_guess() -> None:
    assert not name_matches(['person'], class_name='cup', proposal_name='person')


def test_an_item_with_no_class_is_named_by_the_proposal() -> None:
    assert name_matches(['traffic light'], class_name=None, proposal_name='traffic light')
    assert name_matches(['person'], class_name='person', proposal_name='dog')


def _terms(clause: dict) -> list[dict]:
    return clause['bool']['should']


def test_the_stored_query_names_a_proposal_only_for_an_unlabelled_item() -> None:
    clause = name_clause(['person'])
    assert clause is not None
    proposal = [c for c in _terms(clause) if 'proposal_name' in str(c)]
    assert proposal, clause
    for c in proposal:
        assert c['bool']['must_not'] == [{'exists': {'field': 'class_name'}}], c
    assert all('must_not' not in c.get('bool', {}) for c in _terms(clause) if c not in proposal)
