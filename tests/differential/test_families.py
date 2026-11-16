"""families.toml gives every operation of every contract file present exactly one family."""

from __future__ import annotations

import random

import pytest

from . import families as fam


def op(method: str, path: str, contract: str = 'curation') -> fam.Operation:
    return fam.Operation(contract, method, path, f'{method.lower()}_op')


def family(name: str, *paths: str, contract: str = 'curation') -> fam.Family:
    return fam.Family(name, contract, 'R0', tuple(fam.Pattern.parse(p) for p in paths), ())


def test_every_operation_of_every_contract_file_present_has_one_family() -> None:
    operations = fam.load_operations()
    assert {o.contract for o in operations} >= {'curation'}, (
        'the curation contract is not being read'
    )
    assert len(operations) >= 388
    assert fam.coverage_problems(operations, fam.load_families()) == []


def test_the_committed_mapping_assigns_each_operation_to_a_family_of_its_own_contract() -> None:
    families = fam.load_families()
    grouped = fam.by_family(fam.load_operations(), families)
    assert sum(len(v) for v in grouped.values()) == len(fam.load_operations())
    contract_of = {f.name: f.contract for f in families}
    for name, operations in grouped.items():
        assert {o.contract for o in operations} <= {contract_of[name]}, name


def test_an_operation_in_no_family_is_reported_with_its_label() -> None:
    problems = fam.coverage_problems([op('GET', '/curation/zzz')], [family('a', '/curation/other')])
    assert any('GET /curation/zzz' in p and 'no family' in p for p in problems)


def test_an_equally_specific_pair_of_families_is_reported_whatever_the_order() -> None:
    operations = [op('GET', '/curation/x')]
    families = [family('a', '/curation/x'), family('b', '/curation/x')]
    for _ in range(3):
        random.shuffle(families)
        assert any('equally specific' in p for p in fam.coverage_problems(operations, families))


@pytest.mark.parametrize(
    ('method', 'path', 'winner'),
    [
        ('GET', '/curation/p/crops', 'reads'),
        ('POST', '/curation/p/crops', 'writes'),
        ('GET', '/curation/p/crops/x/reprocess', 'exact'),
        ('GET', '/curation/p/crops/x/deeper', 'longer'),
        ('GET', '/curation/p', 'whole'),
    ],
)
def test_the_most_specific_pattern_wins(method: str, path: str, winner: str) -> None:
    families = [
        family('whole', '/curation/p/*'),
        family('reads', 'GET /curation/p/crops/*'),
        family('writes', 'POST,PUT /curation/p/crops/*'),
        family('longer', '/curation/p/crops/x/deeper/*', '/curation/p/crops/x/deeper'),
        family('exact', '/curation/p/crops/x/reprocess'),
    ]
    assert fam.assign(op(method, path), families).name == winner


def test_a_prefix_matches_at_a_segment_boundary_only() -> None:
    pattern = fam.Pattern.parse('/curation/p/crops/*')
    assert pattern.matches(op('GET', '/curation/p/crops'))
    assert pattern.matches(op('GET', '/curation/p/crops/1'))
    assert not pattern.matches(op('GET', '/curation/p/cropsy'))


@pytest.mark.parametrize('text', ['curation/x', 'FETCH /x', '/x/*/y', '/x*'])
def test_a_bad_pattern_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match=r'families\.toml'):
        fam.Pattern.parse(text)


def test_a_dead_pattern_and_a_family_of_the_wrong_contract_are_reported() -> None:
    operations = [op('GET', '/curation/x')]
    problems = fam.coverage_problems(operations, [family('a', '/curation/x', '/curation/never')])
    assert any("'/curation/never' owns no operation" in p for p in problems)
    problems = fam.coverage_problems(operations, [family('a', '/curation/x', contract='core')])
    assert any('a core family' in p for p in problems)


def test_a_duplicate_family_name_is_reported() -> None:
    problems = fam.coverage_problems([op('GET', '/a')], [family('a', '/a'), family('a', '/b')])
    assert any('listed twice' in p for p in problems)


def test_a_family_of_a_contract_that_is_not_present_is_not_called_dead() -> None:
    problems = fam.coverage_problems(
        [op('GET', '/curation/x')],
        [family('a', '/curation/x'), family('c', '/detect/*', contract='core')],
    )
    assert problems == []


# core.json is written by another lane and was not in the tree when these families were built. The
# paths below are the "Inference API" prefixes of CLAUDE.md (the /v1 twins exist for every one but two,
# which `app.openapi()` shows); when core.json is present the first test above covers its real
# operations, and this one stays as a guard for the prefixes themselves.
NO_V1_TWIN = {'/jobs', '/health'}
CORE_PREFIXES = {
    '/detect': 'core_inference',
    '/faces': 'core_inference',
    '/embed': 'core_inference',
    '/ocr': 'core_inference',
    '/analyze': 'core_inference',
    '/search': 'core_store',
    '/ingest': 'core_store',
    '/clusters': 'core_store',
    '/query': 'core_store',
    '/persons': 'core_store',
    '/jobs': 'core_store',
    '/models': 'core_models',
    '/health': 'core_health',
}


@pytest.mark.parametrize('prefix', sorted(CORE_PREFIXES))
def test_the_core_route_prefixes_and_their_v1_twins_have_a_family(prefix: str) -> None:
    families = fam.load_families()
    twins = () if prefix in NO_V1_TWIN else (f'/v1{prefix}', f'/v1{prefix}/x')
    for path in (prefix, f'{prefix}/x', *twins):
        assert fam.assign(op('POST', path, 'core'), families).name == CORE_PREFIXES[prefix], path
