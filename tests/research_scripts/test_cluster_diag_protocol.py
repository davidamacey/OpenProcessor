"""The held-out protocol, the export script and the runners (scripts/research/cluster_diag)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from scripts.research.cluster_diag import fetch_vectors as fv, heldout as ho
from scripts.research.cluster_diag.dataset import load_bundle


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------- split and protocol


def test_split_groups_is_a_disjoint_cover_independent_of_input_order() -> None:
    groups = [f'img{i}' for i in range(11)]
    first, second = ho.split_groups(groups, seed=5)
    assert sorted(first + second) == sorted(groups)
    assert not set(first) & set(second)
    assert (len(first), len(second)) == (5, 6)  # the odd group goes to the second half
    assert ho.split_groups(reversed(groups * 2), seed=5) == (first, second)


def test_split_groups_changes_with_the_seed() -> None:
    groups = [f'img{i}' for i in range(40)]
    halves = {tuple(ho.split_groups(groups, seed=s)[0]) for s in range(6)}
    assert len(halves) > 1


def test_half_mask_marks_items_of_the_listed_groups() -> None:
    assert ho.half_mask(['a', 'b', 'a', 'c'], ['a', 'c']).tolist() == [True, False, True, True]


TRUTH = np.array(list('aabbaabb'))
FIRST = np.array([True] * 4 + [False] * 4)


def test_choose_and_score_picks_on_calibration_and_reports_on_heldout() -> None:
    perfect_then_wrong = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    # calibration (rows 0-3) is perfect: precision 1, recall 1. Held-out (rows 4-7) pairs
    # (4,6) and (5,7) are a/b: no true pair, precision 0, recall 0.
    one_blob = np.zeros(8, dtype=np.int64)  # calibration: 2 of 6 pairs right = 1/3, recall 1
    singletons = np.arange(8)  # precision undefined: never chosen
    got = ho.choose_and_score(
        [('p1', perfect_then_wrong), ('blob', one_blob), ('single', singletons)],
        TRUTH,
        FIRST,
        ~FIRST,
        min_recall=0.5,
    )
    assert got is not None
    assert got['parameter'] == 'p1'
    assert got['calibration'] == {'precision': 1.0, 'recall': 1.0}
    assert got['heldout']['precision'] == 0.0
    assert got['heldout']['recall'] == 0.0


def test_choose_and_score_prefers_the_precise_cut_only_among_those_reaching_recall() -> None:
    pure_but_low_recall = np.array([0, 0, 1, 2, 3, 3, 4, 5])  # cal: (0,1) right -> P 1, R 1/2
    blob = np.zeros(8, dtype=np.int64)  # cal recall 1, precision 1/3
    got = ho.choose_and_score(
        [('pure', pure_but_low_recall), ('blob', blob)], TRUTH, FIRST, ~FIRST, 0.8
    )
    assert got is not None
    assert got['parameter'] == 'blob'
    assert ho.choose_and_score([('pure', pure_but_low_recall)], TRUTH, FIRST, ~FIRST, 0.8) is None


def test_two_fold_swaps_the_halves() -> None:
    part = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    fold1, fold2 = ho.two_fold([('p', part)], TRUTH, FIRST, min_recall=0.0)
    assert fold1 is not None
    assert fold2 is not None
    assert fold1['heldout']['precision'] == 0.0  # chosen on rows 0-3, scored on rows 4-7
    assert fold2['heldout']['precision'] == 1.0  # chosen on rows 4-7, scored on rows 0-3


# ---------------------------------------------------------------- export (read only)


class _ReadOnlyClient:
    """Serves crop-id-sorted pages by ``search_after``; anything but ``search`` is an error."""

    def __init__(self, n: int) -> None:
        self.docs = [
            {
                'crop_id': f'c{i:04d}',
                'image_id': f'im{i // 3}',
                'pe_embedding': [float(i), 1.0],
            }
            for i in range(n)
        ]
        self.bodies: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append({'index': index, **body})
        start = 0 if 'search_after' not in body else int(body['search_after'][0][1:]) + 1
        page = self.docs[start : start + body['size']]
        return {
            'hits': {
                'hits': [{'_source': dict(d), 'sort': [d['crop_id']]} for d in page],
            }
        }


@pytest.mark.asyncio
async def test_fetch_pages_by_crop_id_and_splits_vectors_from_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fv, 'PAGE', 4)
    client = _ReadOnlyClient(10)
    vectors, meta = await fv.fetch(client, 'op_prj_x__items', None)
    assert vectors.shape == (10, 2)
    assert vectors[:, 0].tolist() == [float(i) for i in range(10)]
    assert [m['crop_id'] for m in meta] == [f'c{i:04d}' for i in range(10)]
    assert 'pe_embedding' not in meta[0]
    assert all(b['sort'] == [{'crop_id': 'asc'}] for b in client.bodies)
    assert all(b['index'] == 'op_prj_x__items' for b in client.bodies)


@pytest.mark.asyncio
async def test_fetch_honours_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fv, 'PAGE', 4)
    vectors, meta = await fv.fetch(_ReadOnlyClient(10), 'op_prj_x__items', 6)
    assert len(vectors) == len(meta) == 6


def test_write_export_round_trips(tmp_path: Path) -> None:
    vectors = np.arange(6, dtype=np.float32).reshape(3, 2)
    fv.write_export(
        tmp_path / 'sub' / 'proj', vectors, [{'crop_id': 'a'}, {'crop_id': 'b'}, {'crop_id': 'c'}]
    )
    assert np.array_equal(np.load(tmp_path / 'sub' / 'proj.npy'), vectors)
    rows = (tmp_path / 'sub' / 'proj.meta.jsonl').read_text().splitlines()
    assert [json.loads(r)['crop_id'] for r in rows] == ['a', 'b', 'c']


# ---------------------------------------------------------------- dataset bundle


def _write_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    stem = tmp_path / 'proj'
    meta = [
        {
            'crop_id': 'c0',
            'image_id': 'i0',
            'image_path': '/x/a.jpg',
            'bbox_norm': [0, 0, 0.5, 0.25],
        },
        {
            'crop_id': 'c1',
            'image_id': 'i0',
            'image_path': '/x/a.jpg',
            'bbox_norm': [0, 0, 0.5, 0.5],
        },
        {
            'crop_id': 'c2',
            'image_id': 'i1',
            'image_path': '/y/b.jpg',
            'bbox_norm': [0, 0, 1.0, 1.0],
        },
    ]
    np.save(stem.with_suffix('.npy'), np.array([[3, 4], [1, 0], [0, 2]], dtype=np.float32))
    stem.with_suffix('.meta.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in meta))
    labels = tmp_path / 'labels.jsonl'
    labels.write_text(
        json.dumps({'item_id': 'c0', 'label': 'car', 'stratum': 'low_res'})
        + '\n'
        + json.dumps({'item_id': 'c2', 'label': 'bus', 'stratum': 'rare'})
        + '\n'
    )
    gt = tmp_path / 'gt.json'
    gt.write_text(
        json.dumps(
            {
                'images': [
                    {'file_name': 'a.jpg', 'width': 200, 'height': 100},
                    {'file_name': 'b.jpg', 'width': 100, 'height': 100},
                ]
            }
        )
    )
    return stem, labels, gt


def test_load_bundle_aligns_labels_geometry_and_strata(tmp_path: Path) -> None:
    b = load_bundle(*_write_project(tmp_path))
    assert np.allclose(np.linalg.norm(b.vectors, axis=1), 1.0)
    assert b.labelled.tolist() == [0, 2]  # c1 has no label
    assert b.truth.tolist() == ['car', 'bus']
    assert b.group == ['i0', 'i1']
    # c0: 0.5 x 200 = 100 px wide, 0.25 x 100 = 25 px tall -> aspect 4, short side 25
    assert b.aspect.tolist() == [4.0, 1.0]
    assert b.side.tolist() == [25.0, 100.0]
    masks = b.strata_masks()
    assert masks['elongated>2:1'].tolist() == [True, False]
    assert masks['side<32px'].tolist() == [True, False]
    assert masks['side>=96px'].tolist() == [False, True]
    assert masks['low_res'].tolist() == [True, False]


# ---------------------------------------------------------------- runners end to end


def _synthetic_project(tmp_path: Path, per_class: int = 40) -> tuple[Path, Path, Path]:
    """Three well-separated classes, detector class = label, a third of the crops low-confidence."""
    rng = np.random.default_rng(7)
    classes = ['car', 'bus', 'dog']
    centres = np.eye(16)[:3]
    meta, vecs, labels = [], [], []
    for ci, name in enumerate(classes):
        for j in range(per_class):
            idx = ci * per_class + j
            meta.append(
                {
                    'crop_id': f'c{idx:04d}',
                    'image_id': f'im{idx // 2}',
                    'image_path': '/x/a.jpg',
                    'bbox_norm': [0, 0, 0.5, 0.5],
                    'detector_class_name': name,
                    'detector_confidence': 0.2 if j % 3 == 0 else 0.9,
                }
            )
            vecs.append(centres[ci] + 0.08 * rng.standard_normal(16))
            labels.append({'item_id': f'c{idx:04d}', 'label': name, 'stratum': 'rare'})
    stem = tmp_path / 'syn'
    np.save(stem.with_suffix('.npy'), np.asarray(vecs, dtype=np.float32))
    stem.with_suffix('.meta.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in meta))
    lab = tmp_path / 'labels.jsonl'
    lab.write_text(''.join(json.dumps(r) + '\n' for r in labels))
    gt = tmp_path / 'gt.json'
    gt.write_text(json.dumps({'images': [{'file_name': 'a.jpg', 'width': 100, 'height': 100}]}))
    return stem, lab, gt


def _args(stem: Path, lab: Path, gt: Path, out: Path, **extra: Any) -> Any:
    from argparse import Namespace

    extra.setdefault('split_manifest', None)
    return Namespace(bundle=stem, labels=lab, coco_gt=gt, out=out, **extra)


def test_e5_recovers_separated_classes_and_reports_the_prior(tmp_path: Path) -> None:
    from scripts.research.cluster_diag import run

    stem, lab, gt = _synthetic_project(tmp_path)
    out = run.run_e5(_args(stem, lab, gt, tmp_path / 'o.json', confidence_cuts=[0.5]))
    entry = out['cuts']['detector_confidence<0.5']
    assert entry['subset_crops'] == 3 * 14  # j % 3 == 0 for j in 0..39
    assert entry['subset_labelled'] == 42
    assert entry['detector_class_agrees_with_label'] == 1.0
    for method in entry['methods'].values():
        for folds in method.values():
            for fold in folds:
                assert fold is not None
                assert fold['heldout']['precision'] == pytest.approx(1.0)


def test_e1_scores_every_arm_and_the_pre_and_post_fix_permutation_ari(tmp_path: Path) -> None:
    pytest.importorskip('leidenalg')
    from scripts.research.cluster_diag import run

    stem, lab, gt = _synthetic_project(tmp_path)
    out = run.run_e1(_args(stem, lab, gt, tmp_path / 'o.json'))
    assert set(out['arms']) == {
        'ivf_current(K=min(512,n//6))',
        'minibatch_kmeans_2b',
        'ahc_average',
        'ahc_complete',
        'leiden_cpm_mknn30',
        'ecg_mknn30(once)',
    }
    avg = out['arms']['ahc_average']['target_recall>=0.2']
    assert all(f is not None and f['heldout']['precision'] == pytest.approx(1.0) for f in avg)
    ari = out['row_permutation_ari']
    assert ari['ivf_current_sorted_by_id(#372 fix)'] == pytest.approx(1.0)
    assert ari['ahc_average_fixed_cut'] == pytest.approx(1.0)


# ---------------------------------------------------------------- uploaded images and the eval split


def test_load_bundle_keys_annotations_by_the_uploader_identifier_and_label_group(
    tmp_path: Path,
) -> None:
    stem, labels, gt = _write_project(tmp_path)
    rows = [json.loads(x) for x in stem.with_suffix('.meta.jsonl').read_text().splitlines()]
    for r, ident in zip(rows, ['/data/a.jpg', '/data/a.jpg', '/data/b.jpg'], strict=True):
        r['image_path'] = (
            f'/uploads/{r["crop_id"]}hash.jpg'  # an upload's hash name matches nothing
        )
        r['source_identifier'] = ident
    stem.with_suffix('.meta.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    labels.write_text(
        json.dumps({'item_id': 'c0', 'label': 'car', 'stratum': 'low_res', 'group': 'a.jpg'})
        + '\n'
        + json.dumps({'item_id': 'c2', 'label': 'bus', 'stratum': 'rare', 'group': 'b.jpg'})
        + '\n'
    )
    b = load_bundle(stem, labels, gt)
    assert b.side.tolist() == [25.0, 100.0]
    assert b.group == ['a.jpg', 'b.jpg']


class _IdentifierClient:
    def __init__(self) -> None:
        self.docs = [{'image_id': f'im{i}', 'source_identifier': f'/d/{i}.jpg'} for i in range(5)]

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        assert index == 'op_prj_x__images'
        assert body['query'] == {'exists': {'field': 'source_identifier'}}
        start = 0 if 'search_after' not in body else int(body['search_after'][0][2:]) + 1
        page = self.docs[start : start + body['size']]
        return {'hits': {'hits': [{'_source': d, 'sort': [d['image_id']]} for d in page]}}


@pytest.mark.asyncio
async def test_fetch_identifiers_pages_through_every_named_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fv, 'PAGE', 2)
    got = await fv.fetch_identifiers(_IdentifierClient(), 'op_prj_x__images')
    assert got == {f'im{i}': f'/d/{i}.jpg' for i in range(5)}


def test_calibration_mask_follows_the_split_manifest_and_refuses_unknown_groups(
    tmp_path: Path,
) -> None:
    from scripts.bench.eval_split import build_manifest, write_manifest
    from scripts.research.cluster_diag import run

    stem, lab, gt = _synthetic_project(tmp_path)
    rows = [json.loads(x) for x in lab.read_text().splitlines()]
    for r in rows:
        r['group'] = 'g0.jpg' if int(r['item_id'][1:]) % 2 else 'g1.jpg'
    lab.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    b = load_bundle(stem, lab, gt)
    manifest = build_manifest('t', 1, 'source_image', ['g0.jpg', 'g1.jpg'])
    path = tmp_path / 'split.json'
    write_manifest(path, manifest)
    calibration = manifest['calibration'][0]
    mask = run.calibration_mask(b, path)
    assert mask.tolist() == [g == calibration for g in b.group]
    write_manifest(path, build_manifest('t', 1, 'source_image', ['g0.jpg']))
    with pytest.raises(SystemExit):
        run.calibration_mask(b, path)


def test_build_labels_matches_uploaded_images_by_identifier(tmp_path: Path) -> None:
    from scripts.research.cluster_diag import build_labels

    stem = tmp_path / 'proj'
    meta = [
        {
            'crop_id': 'c0',
            'image_id': 'i0',
            'image_path': '/uploads/zz/hash0.jpg',
            'source_identifier': '/data/a.jpg',
            'bbox_norm': [0.0, 0.0, 0.5, 0.5],
            'cluster_id': 3,
        }
    ]
    stem.with_suffix('.meta.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in meta))
    gt = tmp_path / 'gt.json'
    gt.write_text(
        json.dumps(
            {
                'images': [{'id': 1, 'file_name': 'a.jpg', 'width': 100, 'height': 100}],
                'categories': [{'id': 1, 'name': 'car'}],
                'annotations': [{'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 50, 50]}],
            }
        )
    )
    pin = tmp_path / 'pin.json'
    pin.write_text(json.dumps({'rare_classes': [], 'manifest_sha256': 'x', 'seed': 1}))
    out = tmp_path / 'out'
    assert (
        build_labels.main(
            [
                '--bundle',
                str(stem),
                '--coco-gt',
                str(gt),
                '--eval-manifest',
                str(pin),
                '--out-dir',
                str(out),
            ]
        )
        == 0
    )
    label = json.loads((out / 'labels.jsonl').read_text())
    assert (label['item_id'], label['label'], label['group']) == ('c0', 'car', 'a.jpg')
    assert json.loads((out / 'partition.jsonl').read_text()) == {'item_id': 'c0', 'cluster_id': 3}


def test_export_partition_cuts_the_dendrogram_at_the_given_height(tmp_path: Path) -> None:
    from scripts.research.cluster_diag import export_partition

    stem, _, _ = _synthetic_project(tmp_path)
    out = tmp_path / 'p.jsonl'
    argv = ['--bundle', str(stem), '--arm', 'ahc_average', '--param', '0.5', '--out', str(out)]
    assert export_partition.main(argv) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    ids = [r['item_id'] for r in rows]
    assert ids == sorted(ids)  # the export's crop-id order
    # three well-separated classes of 40 crops each -> three clusters, one per class
    by_class = {c: {r['cluster_id'] for r in rows[c * 40 : (c + 1) * 40]} for c in range(3)}
    assert all(len(v) == 1 for v in by_class.values())
    assert len(set().union(*by_class.values())) == 3
    with pytest.raises(SystemExit):
        export_partition.main(['--bundle', str(stem), '--arm', 'leiden', '--out', str(out)])
