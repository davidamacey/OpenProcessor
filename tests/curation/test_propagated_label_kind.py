"""The ``propagated`` label kind: a machine decision carried from a sampled class to
the rest of a node (auto-label loop, #286).

It is written like any class label, but as ``class_source: assist_propagated``,
unvalidated: never locked, never exported before review, never relabeled by the
automatic VLM paths, and never put back into the residual clustering pool.
"""

from __future__ import annotations

from typing import Any

import pytest

from curation.query_fakes import matches
from scripts.curation.vlm_pending import _build_pending_query
from src.clients.occ import is_locked_class
from src.domain.history import find_undo_snapshot, restore_class_state
from src.domain.ingest_class_sources import PROPAGATED_CLASS_SOURCE, confident_class_sources
from src.services.curation.autolabel.selection import vlm_selection_query
from src.services.curation.class_label import ItemLabel, class_label_fields, class_label_update
from src.services.curation.class_sources import CLASS_SOURCE_ROLES, class_source_catalog
from src.services.curation.class_write_guard import class_write_locked
from src.services.curation.clustering.residual_fetch import _residual_pool_query
from src.services.curation.export_readiness import validated_export_query


RUN = 'alp_20261028T000000Z_abc123'


def _label() -> ItemLabel:
    return ItemLabel.propagated(
        class_id=4, class_name='widget', run_id=RUN, writer=f'assist_loop:{RUN}'
    )


def _vlm_item() -> dict[str, Any]:
    return {
        'crop_id': 'c1',
        'class_id': 7,
        'class_name': 'gadget',
        'class_source': 'vlm',
        'label_source': 'vlm',
        'class_validated': False,
        'cluster_id': 10004,
        'cluster_subid': '10004a',
        'embedding_state': 'ready',
        'pe_embedding': [0.1],
    }


def test_the_source_constant_is_the_documented_value() -> None:
    assert PROPAGATED_CLASS_SOURCE == 'assist_propagated'


def test_fields_for_a_new_item() -> None:
    fields = class_label_fields(_label())
    stamp = fields.pop('class_labeled_at')
    assert isinstance(stamp, str)
    assert fields == {
        'class_id': 4,
        'class_name': 'widget',
        'class_source': 'assist_propagated',
        'class_validated': False,
        'label_source': 'assist_loop',
        'cluster_id': 4,
        'cluster_subid': None,
        'class_detector': 'assist_loop',
        'class_detector_version': RUN,
        'class_labeler': 'assist_loop',
    }


def test_update_snapshots_the_state_it_replaces_and_audits_nothing() -> None:
    current = _vlm_item()
    update = class_label_update(current, _label())

    entry = update['class_id_history'][-1]
    assert (entry['writer'], entry['restorable']) == (f'assist_loop:{RUN}', True)
    assert (entry['class_name'], entry['class_source']) == ('gadget', 'vlm')
    assert update['class_source'] == 'assist_propagated'
    assert update['class_validated'] is False
    assert update['updated_at']
    assert not any(key.startswith('audit') for key in update)
    restored = restore_class_state(entry)
    assert (restored['class_name'], restored['class_source']) == ('gadget', 'vlm')
    # a person's Undo walks human label writes only; the loop's write is not one
    assert find_undo_snapshot(update['class_id_history']) is None


def test_a_propagated_item_stays_open_to_later_fixes() -> None:
    written = {**_vlm_item(), **class_label_update(_vlm_item(), _label())}
    assert not is_locked_class(written)
    assert not class_write_locked(written)


def test_it_counts_as_confident_so_the_normalizer_and_residual_pool_leave_it() -> None:
    assert PROPAGATED_CLASS_SOURCE in confident_class_sources()
    item = {**_vlm_item(), **class_label_update(_vlm_item(), _label())}
    pool = _residual_pool_query(
        include_candidate_clusters=True, candidate_cluster_id_min=None, extra_must=None
    )
    assert matches(_vlm_item() | {'class_source': 'unlabeled_proposal'}, pool)
    assert not matches(item, pool)


def test_it_is_not_exportable_until_a_human_validates_it() -> None:
    item = {**_vlm_item(), **class_label_update(_vlm_item(), _label())}
    assert not matches(item, validated_export_query())
    assert matches({**item, 'class_validated': True}, validated_export_query())


def test_the_automatic_vlm_paths_do_not_relabel_it() -> None:
    item = {
        **_vlm_item(),
        **class_label_update(_vlm_item(), _label()),
        'confidence': 0.1,
        'embedding_state': 'ready',
    }
    plain = {**item, 'class_source': 'unlabeled_proposal'}
    sweep = vlm_selection_query(
        class_id=None, cluster_id=None, classifier_confidence_skip_vlm=0.9, now=None
    )
    worker = _build_pending_query(0.9)
    for query in (sweep, worker):
        assert matches(plain, query), 'control: an unlabeled item is selected'
        assert not matches(item, query)


def test_the_catalog_lists_it_with_its_own_role() -> None:
    by_id = {entry['id']: entry for entry in class_source_catalog()}
    assert by_id['assist_propagated']['role'] == 'propagated'
    assert 'propagated' in CLASS_SOURCE_ROLES


def test_a_non_propagated_machine_kind_is_not_confused_with_it() -> None:
    with pytest.raises(TypeError):
        ItemLabel.propagated(class_id=1, class_name='x', writer='w')  # type: ignore[call-arg]
