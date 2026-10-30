"""Selection and disjointness checks of the Z2 vehicle set (no network, no pixels)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scripts.datasets import fetch_vehicle_eval as fve, vehicle_eval_selection as ves


if TYPE_CHECKING:
    from pathlib import Path


def _row(i: int, primary: str, classes: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        'source': 'openimages',
        'image_id': f'{i:04d}',
        'file_name': f'oi_{i:04d}.jpg',
        'split': 'validation',
        'license': 'l',
        'author': '',
        'attribution_url': '',
        'classes': classes or {primary: 1},
        'primary': primary,
        'small': 0,
    }


def _key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [dict(r, key=key) for r in rows]


def test_a_short_key_hands_its_places_to_the_others() -> None:
    pool = _key([_row(i, 'a') for i in range(5)], 'a') + _key([_row(10, 'b')], 'b')
    got = ves.balanced_draw(pool, 4, 's', set())
    # round-robin a, b, a, a: b has one row, so a supplies the other three
    assert sorted(r['key'] for r in got) == ['a', 'a', 'a', 'b']


def test_balanced_draw_is_even_when_keys_are_big_enough() -> None:
    pool = _key([_row(i, 'a') for i in range(10)], 'a') + _key(
        [_row(100 + i, 'b') for i in range(10)], 'b'
    )
    got = ves.balanced_draw(pool, 6, 's', set())
    assert sorted(r['key'] for r in got) == ['a'] * 3 + ['b'] * 3


def test_balanced_draw_skips_taken_rows_and_is_seeded() -> None:
    pool = _key([_row(i, 'a') for i in range(6)], 'a')
    taken = {'oi_0000.jpg', 'oi_0001.jpg'}
    got = ves.balanced_draw(pool, 10, 's', taken)
    assert {r['file_name'] for r in got} == {f'oi_{i:04d}.jpg' for i in range(2, 6)}
    again = ves.balanced_draw(pool, 3, 's', set())
    assert [r['file_name'] for r in again] == [
        r['file_name'] for r in ves.balanced_draw(pool, 3, 's', set())
    ]
    other = ves.balanced_draw(pool, 3, 'different', set())
    assert [r['file_name'] for r in other] != [r['file_name'] for r in again] or len(pool) < 4


def _rows() -> list[dict[str, Any]]:
    rows = [_row(i, 'car') for i in range(40)]
    rows += [_row(100 + i, 'dog') for i in range(10)]
    rows += [_row(200 + i, 'person') for i in range(30)]
    rows += [_row(300 + i, 'alpaca') for i in range(3)]
    rows += [_row(400, 'person', {'person': 2, 'camel': 1, 'car': 1})]
    return rows


def test_select_base_puts_each_image_in_one_group_and_honours_the_exclusion() -> None:
    excluded = frozenset({'oi_0000.jpg', 'oi_0100.jpg'})
    # scale 0.01: target 80, quotas 12 open-set, 12 animal, 52 vehicle
    base = ves.select_base(_rows(), seed=1, scale=0.01, exclude=excluded)
    names = [r['file_name'] for r in base]
    assert len(names) == len(set(names))
    assert not excluded & set(names)
    by = {g: sum(r['group'] == g for r in base) for g in ves.GROUP_ORDER}
    # 84 rows - 2 excluded. Open set: 3 alpaca + the camel scene = 4. Animals: 9 dogs left.
    # Vehicles: 39 cars left (the camel scene is taken). Rest: 80 - 52 = 28 of the 30 persons.
    assert by == {'openset': 4, 'animal': 9, 'vehicle': 39, 'rest': 28}
    assert sum(by.values()) == 80


def test_the_open_set_key_is_the_scarcest_class_of_the_image() -> None:
    rows = [
        _row(1, 'alpaca'),
        _row(2, 'alpaca'),
        _row(3, 'person', {'person': 1, 'alpaca': 1, 'camel': 1}),
    ]
    pool = ves._group_pool('openset', rows)
    assert {r['file_name']: r['key'] for r in pool}['oi_0003.jpg'] == 'camel'  # 1 image against 3


def test_the_rest_group_fills_to_the_target() -> None:
    rows = [_row(i, 'person') for i in range(500)]
    base = ves.select_base(rows, seed=1, scale=0.02)  # target 160
    assert len(base) == 160
    assert {r['group'] for r in base} == {'rest'}


def test_pin_carries_what_the_labeller_and_the_splitter_read() -> None:
    base = [
        {**_row(1, 'car'), 'group': 'vehicle', 'width': 1, 'height': 1, 'bytes': 1, 'sha256': 'x'}
    ]
    pin = ves.build_pin(base, 5, 1.0, 'ann')
    assert pin['rare_classes'] == sorted(ves.OPENSET)
    assert pin['near_duplicates'] == []
    assert pin['images'][0]['group'] == 'vehicle'
    assert len(pin['manifest_sha256']) == 64


def test_overlap_reports_shared_names_and_hashes() -> None:
    got = ves.overlap({'a', 'b'}, {'h1', 'h2'}, {'b', 'c'}, {'h2', 'h3'})
    assert got == {'names': ['b'], 'hashes': ['h2']}
    assert ves.overlap({'a'}, {'h'}, {'b'}, {'g'}) == {'names': [], 'hashes': []}


def test_disjointness_report_finds_a_same_bytes_image_under_another_name() -> None:
    base = [
        {'file_name': 'oi_1.jpg', 'sha256': 'aa'},
        {'file_name': 'oi_2.jpg', 'sha256': 'bb'},
    ]
    clean = fve.disjointness_report(base, frozenset({'oi_9.jpg'}), {'cc'})
    assert clean['shared_names'] == []
    assert clean['shared_hashes'] == []
    dirty = fve.disjointness_report(base, frozenset({'oi_2.jpg'}), {'aa'})
    assert dirty['shared_names'] == ['oi_2.jpg']
    assert dirty['shared_hashes'] == ['aa']
    dup = fve.disjointness_report(
        [*base, {'file_name': 'oi_3.jpg', 'sha256': 'aa'}], frozenset(), set()
    )
    assert dup['duplicate_hashes_inside_the_set'] == 1


def test_excluded_hashes_reads_pins_and_variants(tmp_path: Path) -> None:
    pin = tmp_path / 'pin.json'
    pin.write_text(
        json.dumps(
            {
                'images': [{'file_name': 'a.jpg', 'sha256': 'h1'}],
                'near_duplicates': [
                    {'parent': 'a.jpg', 'variants': [{'file_name': 'v', 'sha256': 'h2'}]}
                ],
            }
        )
    )
    listed = tmp_path / 'list.json'
    listed.write_text(json.dumps([{'file_name': 'c.jpg', 'sha256': 'h3'}]))
    assert fve.excluded_hashes([pin, listed], tmp_path) == {'h1', 'h2', 'h3'}


def test_select_base_takes_its_own_quotas_and_target() -> None:
    rows = [_row(i, 'dog') for i in range(40)] + [_row(100 + i, 'car') for i in range(40)]
    base = ves.select_base(rows, quotas={'openset': 0, 'animal': 5, 'vehicle': 7}, target=20)
    by = {g: sum(r['group'] == g for r in base) for g in ves.GROUP_ORDER}
    assert by == {'openset': 0, 'animal': 5, 'vehicle': 7, 'rest': 8}
    # the default quotas are untouched by a call with its own
    assert ves.QUOTAS == {'openset': 1200, 'animal': 1200, 'vehicle': 5200}


def test_the_pin_records_the_quotas_and_target_it_was_drawn_with() -> None:
    row = _row(1, 'car') | {'group': 'vehicle', 'width': 1, 'height': 1, 'bytes': 1, 'sha256': 'x'}
    pin = ves.build_pin([row], 1, 1.0, 'h', quotas={'vehicle': 3}, target=9)
    assert (pin['quotas'], pin['target_images']) == ({'vehicle': 3}, 9)
    assert ves.build_pin([row], 1, 1.0, 'h')['target_images'] == ves.TARGET_IMAGES


def test_quota_options_are_parsed_and_bad_ones_refused() -> None:
    assert fve.parse_quotas(['openset=10', 'animal=3']) == {'openset': 10, 'animal': 3}
    for bad in ('nogroup=3', 'animal=x', 'animal'):
        try:
            fve.parse_quotas([bad])
        except fve.FetchError:
            continue
        raise AssertionError(f'{bad!r} was accepted')
