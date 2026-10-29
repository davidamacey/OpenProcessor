"""contact_sheet: deterministic selection, layout and rendering, manifest round trip, rebuild guard."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src.services.labeling import contact_sheet as cs


def _crop(k: int, size: tuple[int, int] = (40, 30)) -> bytes:
    img = Image.new('RGB', size, (k * 7 % 256, k * 13 % 256, k * 29 % 256))
    img.putpixel((k % size[0], k % size[1]), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    return buf.getvalue()


def _members(n: int = 40, protos: int = 3) -> list[cs.SheetMember]:
    return [
        cs.SheetMember(
            item_id=f'i{k:03d}',
            box_id='b0',
            bbox_norm=(0.1, 0.1, 0.5, 0.6),
            image=_crop(k),
            is_prototype=k < protos,
            distance=0.0 if k < protos else k / 100,
        )
        for k in range(n)
    ]


def _build(members: list[cs.SheetMember], **kw: object) -> cs.Sheet:
    args: dict[str, object] = {'sheet_id': 's1', 'seed': 5, 'kind': 'object'}
    args.update(kw)
    return cs.build_cluster_sheet(members, **args)  # type: ignore[arg-type]


def test_building_twice_gives_byte_identical_png_and_manifest() -> None:
    a, b = _build(_members()), _build(_members())
    assert a.png == b.png
    assert a.manifest == b.manifest
    assert a.manifest['png_sha256'] == cs.sha256_hex(a.png)
    assert a.png.startswith(b'\x89PNG')


def test_selection_does_not_depend_on_input_order() -> None:
    members = _members()
    a = _build(members)
    b = _build(list(reversed(members)))
    assert a.png == b.png
    assert a.manifest == b.manifest


def test_selection_roles_prototypes_outliers_and_cap() -> None:
    sheet = _build(_members(60, protos=3))
    cells = sheet.manifest['cells']
    assert len(cells) == 24
    roles = [c['role'] for c in cells]
    assert roles.count('prototype') == 3
    assert roles.count('outlier') == 3
    outliers = {c['item_id'] for c in cells if c['role'] == 'outlier'}
    assert outliers == {'i059', 'i058', 'i057'}  # the largest distances
    assert len({c['item_id'] for c in cells}) == 24


def test_more_than_eight_prototypes_are_capped_at_eight() -> None:
    sheet = _build(_members(50, protos=12))
    roles = [c['role'] for c in sheet.manifest['cells']]
    assert roles.count('prototype') == 8


def test_cell_order_is_shuffled_and_seed_dependent_but_roles_are_not_drawn() -> None:
    a, b = _build(_members(), seed=1), _build(_members(), seed=2)
    ids_a = [c['item_id'] for c in a.manifest['cells']]
    assert ids_a != sorted(ids_a)
    assert ids_a != [c['item_id'] for c in b.manifest['cells']]
    assert [c['cell'] for c in a.manifest['cells']] == list(range(1, 25))
    assert a.png != b.png


def test_order_variant_keeps_the_members_and_changes_the_order() -> None:
    first = _build(_members(), sheet_id='x')
    second = _build(_members(), sheet_id='x-r', selection_id='x', order_variant=1)
    ids1 = [c['item_id'] for c in first.manifest['cells']]
    ids2 = [c['item_id'] for c in second.manifest['cells']]
    assert set(ids1) == set(ids2)
    assert ids1 != ids2
    assert second.manifest['order_variant'] == 1


def test_small_cluster_gets_a_smaller_grid() -> None:
    sheet = _build(_members(5, protos=1))
    assert len(sheet.manifest['cells']) == 5
    assert sheet.manifest['grid'] == [5, 1]
    big = _build(_members(10, protos=1))
    assert big.manifest['grid'] == [6, 2]
    img = Image.open(io.BytesIO(big.png))
    assert img.size == (6 * 160 + 7 * 4, 2 * 160 + 3 * 4)


def test_sheet_geometry_follows_grid_and_thumb_size() -> None:
    sheet = _build(_members(), grid=(4, 4), thumb_px=128)
    assert sheet.manifest['grid'] == [4, 4]
    assert sheet.manifest['thumb_px'] == 128
    assert Image.open(io.BytesIO(sheet.png)).size == (4 * 128 + 5 * 4, 4 * 128 + 5 * 4)


def test_letterbox_geometry_of_an_elongated_crop() -> None:
    assert cs.letterbox_geometry(400, 100, 160) == (160, 40, 0, 60)
    assert cs.letterbox_geometry(100, 400, 160) == (40, 160, 60, 0)
    assert cs.letterbox_geometry(50, 50, 160) == (160, 160, 0, 0)


def test_letterbox_bars_are_neutral_grey_and_pixels_are_not_stretched() -> None:
    wide = cs.SheetMember(item_id='w', image=_solid((255, 0, 0), (400, 100)))
    sheet = cs.build_cluster_sheet([wide], sheet_id='w', seed=0, kind='object')
    img = Image.open(io.BytesIO(sheet.png)).convert('RGB')
    x0 = 4  # gutter
    assert img.getpixel((x0 + 80, 4 + 80)) == (255, 0, 0)  # picture band is rows 60..99
    assert img.getpixel((x0 + 80, 4 + 150)) == cs.GREY  # letterbox bar below
    assert img.getpixel((x0 + 80, 4 + 55)) == cs.GREY  # and above
    assert img.getpixel((x0 + 80, 4 + 62)) == (255, 0, 0)  # band starts where computed


def _solid(color: tuple[int, int, int], size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, 'PNG')
    return buf.getvalue()


def test_manifest_maps_cells_back_to_items_and_hashes() -> None:
    members = _members()
    by_id = {m.item_id: m for m in members}
    sheet = _build(members)
    for c in sheet.manifest['cells']:
        m = by_id[c['item_id']]
        assert c['content_hash'] == cs.sha256_hex(m.image)
        assert c['box_id'] == 'b0'
        assert c['bbox_norm'] == [0.1, 0.1, 0.5, 0.6]
    meta = sheet.manifest
    assert meta['builder_version'] == cs.BUILDER_VERSION
    assert (meta['sheet_id'], meta['seed'], meta['kind']) == ('s1', 5, 'object')
    assert cs.cell_to_item(meta, 3) == (meta['cells'][2]['item_id'], 'b0')


def test_rebuild_from_manifest_reproduces_the_png_hash() -> None:
    members = _members()
    sheet = _build(members)
    crops = {cs.member_key(m.item_id, m.box_id): m.image for m in members}
    again = cs.rebuild_sheet(sheet.manifest, crops)
    assert again.png == sheet.png
    assert again.manifest['png_sha256'] == sheet.manifest['png_sha256']


def test_rebuild_refuses_a_changed_source_crop() -> None:
    members = _members()
    sheet = _build(members)
    crops = {cs.member_key(m.item_id, m.box_id): m.image for m in members}
    victim = sheet.manifest['cells'][4]
    crops[cs.member_key(victim['item_id'], victim['box_id'])] = _crop(999)
    with pytest.raises(cs.SheetRebuildError, match='content_hash'):
        cs.rebuild_sheet(sheet.manifest, crops)


def test_rebuild_refuses_a_missing_crop_and_a_tampered_sheet_hash() -> None:
    members = _members()
    sheet = _build(members)
    crops = {cs.member_key(m.item_id, m.box_id): m.image for m in members}
    first = sheet.manifest['cells'][0]
    missing = dict(crops)
    del missing[cs.member_key(first['item_id'], first['box_id'])]
    with pytest.raises(cs.SheetRebuildError, match='missing'):
        cs.rebuild_sheet(sheet.manifest, missing)
    bad = {**sheet.manifest, 'png_sha256': '0' * 64}
    with pytest.raises(cs.SheetRebuildError, match='png_sha256'):
        cs.rebuild_sheet(bad, crops)


def test_pair_sheet_has_one_pair_per_row_with_random_side_swaps() -> None:
    members = _members(24, protos=0)
    pairs = [(members[2 * k], members[2 * k + 1]) for k in range(12)]
    sheet = cs.build_pair_sheet(pairs, sheet_id='p1', seed=3, kind='object')
    cells = sheet.manifest['cells']
    assert len(cells) == 24
    assert [c['label'] for c in cells[:4]] == ['A1', 'B1', 'A2', 'B2']
    assert Image.open(io.BytesIO(sheet.png)).size == (2 * 160 + 3 * 4, 12 * 160 + 13 * 4)
    swapped = [c['swapped'] for c in cells[::2]]
    assert any(swapped)
    assert not all(swapped)
    for k, (first, second) in enumerate(pairs):
        a, b = cells[2 * k], cells[2 * k + 1]
        got = {a['item_id'], b['item_id']}
        assert got == {first.item_id, second.item_id}
        assert (a['item_id'] == second.item_id) == a['swapped']
    again = cs.build_pair_sheet(pairs, sheet_id='p1', seed=3, kind='object')
    assert again.png == sheet.png
    crops = {cs.member_key(m.item_id, m.box_id): m.image for m in members}
    assert cs.rebuild_sheet(sheet.manifest, crops).png == sheet.png


def test_pair_sheet_refuses_more_than_twelve_pairs() -> None:
    members = _members(28, protos=0)
    with pytest.raises(ValueError, match='12'):
        cs.build_pair_sheet(
            [(members[2 * k], members[2 * k + 1]) for k in range(13)],
            sheet_id='p',
            seed=0,
            kind='object',
        )


def test_search_sheet_puts_the_query_in_cell_zero() -> None:
    members = _members(30, protos=0)
    sheet = cs.build_search_sheet(members[0], members[1:24], sheet_id='q1', seed=1, kind='object')
    cells = sheet.manifest['cells']
    assert cells[0]['label'] == 'Q'
    assert cells[0]['role'] == 'query'
    assert cells[0]['item_id'] == 'i000'
    assert len(cells) == 24
    assert {c['role'] for c in cells[1:]} == {'result'}
    assert sorted(c['rank'] for c in cells[1:]) == list(range(1, 24))
    crops = {cs.member_key(m.item_id, m.box_id): m.image for m in members}
    assert cs.rebuild_sheet(sheet.manifest, crops).png == sheet.png


def test_glyphs_cover_every_label_character() -> None:
    for ch in '0123456789ABQ':
        assert ch in cs.GLYPHS
