"""One by-name class resolver: a deprecated class and an active class may
share a name (the name was reused after deprecation); every by-name caller
must land on the ACTIVE one, never silently on the retired one."""

from __future__ import annotations

from itertools import permutations

import pytest
from _class_registry_fake import InMemoryRegistry

from src.services.classes.registry import ClassRegistryError
from src.services.classes.resolve import ensure_class_by_name
from src.services.curation.dataset_import.mapping import RegistryClassView, suggest_mapping
from src.services.curation.detector_vocabulary import DetectorLabel, plan_seed
from src.utils.class_names import class_name_deprecation_index, resolve_class_by_name


def _view(class_id: int, name: str, deprecated: bool) -> RegistryClassView:
    return RegistryClassView(class_id=class_id, class_name=name, deprecated=deprecated)


OLD = _view(0, 'wheel', True)
NEW = _view(5, 'wheel', False)


@pytest.mark.parametrize('order', list(permutations([OLD, NEW, _view(9, 'car', False)])))
def test_active_wins_whatever_the_registry_order(order) -> None:
    match = resolve_class_by_name(order, 'Wheel')
    assert match.active is not None
    assert (match.status, match.active.class_id) == ('active', 5)


def test_deprecated_only_is_an_explicit_result_not_a_match() -> None:
    match = resolve_class_by_name([OLD], 'wheel')
    assert match.status == 'deprecated'
    assert resolve_class_by_name([OLD], 'nope').status == 'none'
    assert resolve_class_by_name([OLD], '  ').status == 'none'


def test_deprecation_index_gives_active_precedence() -> None:
    assert class_name_deprecation_index([NEW, OLD]) == {'wheel': False}
    assert class_name_deprecation_index([OLD, NEW]) == {'wheel': False}
    assert class_name_deprecation_index([OLD]) == {'wheel': True}


def test_mapping_suggestion_targets_the_active_class() -> None:
    for order in ([OLD, NEW], [NEW, OLD]):
        s = suggest_mapping('wheel', registry_classes=order)
        assert (s.action, s.class_id) == ('map', 5)
    assert suggest_mapping('wheel', registry_classes=[OLD]).action == 'create'


def test_seed_reports_exists_not_deprecated_when_an_active_twin_exists() -> None:
    label = DetectorLabel(class_id=3, name='wheel', slug='wheel')
    plan = plan_seed([label], class_name_deprecation_index([OLD, NEW]), None)
    assert [s.reason for s in plan.skipped] == ['exists']
    plan = plan_seed([label], class_name_deprecation_index([OLD]), None)
    assert [s.reason for s in plan.skipped] == ['deprecated']


def _registry(rows: list[dict]) -> InMemoryRegistry:
    reg = InMemoryRegistry()
    reg.seed_rows(rows)
    return reg


def _row(class_id: int, name: str, deprecated: bool = False) -> dict:
    return {'class_id': class_id, 'class_name': name, 'group': 'g', 'deprecated': deprecated}


@pytest.mark.asyncio
async def test_ensure_resolves_the_active_class_and_never_the_retired_one() -> None:
    reg = _registry([_row(0, 'wheel', True), _row(5, 'wheel')])
    assert (await ensure_class_by_name(reg, 'Wheel', group='g')).class_id == 5


@pytest.mark.asyncio
async def test_ensure_does_not_reuse_a_deprecated_only_class() -> None:
    reg = _registry([_row(0, 'wheel', True)])
    resolved = await ensure_class_by_name(reg, 'wheel', group='g')
    assert resolved.class_id == 1


def test_creating_an_active_twin_of_a_deprecated_name_is_allowed() -> None:
    reg = _registry([_row(0, 'wheel', True)])
    assert reg.add('wheel') == 1


def test_two_active_classes_never_share_a_name_even_by_spelling() -> None:
    reg = _registry([_row(5, 'wheel')])
    with pytest.raises(ClassRegistryError):
        reg.add('Wheel')
    reg2 = _registry([_row(1, 'Car'), _row(2, 'bus')])
    with pytest.raises(ClassRegistryError):
        reg2.rename_class(2, 'car')


def test_restoring_the_retired_twin_is_refused_while_the_active_one_lives() -> None:
    reg = _registry([_row(0, 'wheel', True), _row(5, 'wheel')])
    with pytest.raises(ClassRegistryError):
        reg.deprecate(0, False)
