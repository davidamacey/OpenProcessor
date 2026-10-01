"""The pure dedup rules: duplicate detection and per-box merge decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.projects.combine.dedup import (
    Fingerprint,
    Probe,
    decide_merges,
    find_duplicates,
    trust_rank,
)


if TYPE_CHECKING:
    from pathlib import Path


def _file(tmp: Path, name: str, data: bytes) -> str:
    path = tmp / name
    path.write_bytes(data)
    return str(path)


def test_identical_bytes_are_duplicates_and_the_first_source_is_the_priority(
    tmp_path: Path,
) -> None:
    a = _file(tmp_path, 'a.jpg', b'same')
    b = _file(tmp_path, 'b.jpg', b'same')
    c = _file(tmp_path, 'c.jpg', b'same')
    dups = find_duplicates(
        [Fingerprint(2, 'c', 'h', c), Fingerprint(0, 'a', 'h', a), Fingerprint(1, 'b', 'h', b)]
    )
    assert dups == {(1, 'b'): (0, 'a'), (2, 'c'): (0, 'a')}


def test_equal_imohash_with_different_bytes_is_not_a_duplicate(tmp_path: Path) -> None:
    # imohash samples the file: a collision must be confirmed by a full hash.
    a = _file(tmp_path, 'a.jpg', b'one')
    b = _file(tmp_path, 'b.jpg', b'two')
    assert find_duplicates([Fingerprint(0, 'a', 'h', a), Fingerprint(1, 'b', 'h', b)]) == {}


def test_an_unreadable_file_is_never_called_a_duplicate(tmp_path: Path) -> None:
    a = _file(tmp_path, 'a.jpg', b'x')
    gone = str(tmp_path / 'gone.jpg')
    assert find_duplicates([Fingerprint(0, 'a', 'h', a), Fingerprint(1, 'b', 'h', gone)]) == {}


def test_the_same_source_twice_is_not_a_cross_source_duplicate(tmp_path: Path) -> None:
    a = _file(tmp_path, 'a.jpg', b'x')
    b = _file(tmp_path, 'b.jpg', b'x')
    assert find_duplicates([Fingerprint(0, 'a', 'h', a), Fingerprint(0, 'b', 'h', b)]) == {}


BOX = (0.1, 0.1, 0.5, 0.5)


def test_merge_conflict_and_union_decisions() -> None:
    existing = [Probe(BOX, 'car', 3), Probe((0.6, 0.6, 0.9, 0.9), 'car', 3)]
    incoming = [
        Probe(BOX, 'car', 0),  # same class, more trusted
        Probe((0.6, 0.6, 0.9, 0.9), 'truck', 0),  # different class
        Probe((0.0, 0.9, 0.1, 1.0), 'car', 3),  # matches nothing
    ]
    got = decide_merges(existing, incoming, iou_min=0.9)
    assert [(d.kind, d.existing, d.take_incoming_label) for d in got] == [
        ('merge', 0, True),
        ('conflict', 1, False),
        ('union', None, False),
    ]


def test_a_tie_keeps_the_existing_label_and_a_box_matches_once() -> None:
    existing = [Probe(BOX, 'car', 1)]
    incoming = [Probe(BOX, 'car', 1), Probe(BOX, 'car', 1)]
    got = decide_merges(existing, incoming, iou_min=0.9)
    assert [(d.kind, d.take_incoming_label) for d in got] == [('merge', False), ('union', False)]


def test_the_iou_threshold_gates_a_match() -> None:
    existing = [Probe((0.0, 0.0, 1.0, 1.0), 'car', 3)]
    shifted = [Probe((0.0, 0.0, 1.0, 0.8), 'car', 3)]  # IoU 0.8
    assert decide_merges(existing, shifted, iou_min=0.9)[0].kind == 'union'
    assert decide_merges(existing, shifted, iou_min=0.7)[0].kind == 'merge'


def test_an_unclassed_box_merges_and_never_wins() -> None:
    existing = [Probe(BOX, 'car', 3)]
    got = decide_merges(existing, [Probe(BOX, None, 9)], iou_min=0.9)
    assert (got[0].kind, got[0].take_incoming_label) == ('merge', False)


def test_trust_order_is_human_import_vlm_model() -> None:
    def rank(label: str, cls: str = 'x') -> int:
        return trust_rank({'class_id': 1, 'label_source': label, 'class_source': cls})

    assert rank('human') < rank('import', 'external_label') < rank('vlm', 'vlm') < rank('detector')
    assert trust_rank({}) > rank('detector')  # no class: nothing to trust
