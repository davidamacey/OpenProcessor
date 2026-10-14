"""Subset rewrites, class_remap propagation and orientation-sensitive class resolution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import dataset_prep
import job_protocol
import pytest
import subset_dataset

from curation.trainer_protocol._helpers import _write_job
from src.services.training.class_remap import build_class_id_to_name, resolve_class_remap


if TYPE_CHECKING:
    from pathlib import Path


def test_subset_rewrite_emits_a_remap_promote_can_read(tmp_path: Path, export_dir: Path) -> None:
    """The remap payload shape is a contract with resolve_class_remap()."""
    out = tmp_path / 'subset'
    result = subset_dataset.build_subset_view(
        export_dir=export_dir, include_classes=[12, 5], single_cls=False, out_dir=out
    )

    # include_classes order drives the new ids, not registry order.
    assert result.class_remap == {12: 0, 5: 1}
    assert result.kept_rows == 6  # 2 kept classes x 3 splits
    assert result.dropped_rows == 3  # the excluded class, once per split

    weights_dir = tmp_path / 'run' / 'weights'
    weights_dir.mkdir(parents=True)
    (weights_dir / 'class_remap.json').write_bytes((out / 'class_remap.json').read_bytes())

    remap = resolve_class_remap(job_id='j', checkpoint_path=weights_dir / 'best.pt', manifest=None)
    assert remap.source == 'weights_dir'
    assert remap.mapping == {12: 0, 5: 1}
    assert remap.names == ['gadget', 'widget']
    assert remap.include_classes == [12, 5]

    # labels.txt for the promoted model must be the subset, renumbered.
    full_registry = {5: 'widget', 9: 'serial_plate', 12: 'gadget'}
    assert build_class_id_to_name(remap=remap, full_registry=full_registry) == {
        0: 'gadget',
        1: 'widget',
    }


def test_single_cls_rewrite_collapses_to_one_class(tmp_path: Path, export_dir: Path) -> None:
    out = tmp_path / 'subset'
    subset_dataset.build_subset_view(
        export_dir=export_dir, include_classes=[9], single_cls=True, out_dir=out
    )
    weights_dir = tmp_path / 'run' / 'weights'
    weights_dir.mkdir(parents=True)
    (weights_dir / 'class_remap.json').write_bytes((out / 'class_remap.json').read_bytes())

    remap = resolve_class_remap(job_id='j', checkpoint_path=weights_dir / 'best.pt', manifest=None)
    assert remap.single_cls is True
    # one source class: its own name, not the trainer's generic 'object'
    assert build_class_id_to_name(remap=remap, full_registry={9: 'serial_plate'}) == {
        0: 'serial_plate'
    }


def test_single_cls_collapsing_several_classes_keeps_the_trainer_name() -> None:
    from src.services.training.class_remap import ClassRemapResult

    remap = ClassRemapResult(
        mapping={1: 0, 2: 0},
        names=['vehicle'],
        single_cls=True,
        include_classes=[1, 2],
        source='manifest',
    )
    assert build_class_id_to_name(remap=remap, full_registry={1: 'car', 2: 'truck'}) == {
        0: 'vehicle'
    }


def test_subset_rewrite_rejects_a_registry_id_outside_the_export(
    tmp_path: Path, export_dir: Path
) -> None:
    with pytest.raises(ValueError, match='outside this export'):
        subset_dataset.build_subset_view(
            export_dir=export_dir,
            include_classes=[5, 999],
            single_cls=False,
            out_dir=tmp_path / 'subset',
        )


def test_class_remap_copy_reports_failure_loudly_for_a_subset_run(
    jobs_dir: Path, export_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A subset run whose remap never reaches the weights dir must say so.

    Silently returning True here is the bug the promote-side fix exists for:
    promote would fall back to the full registry and ship a labels.txt that
    doesn't describe the model it is serving.
    """
    monkeypatch.setattr(job_protocol, 'TMP_ROOT', tmp_path / 'tmp')
    job_id = _write_job(dataset_export_dir=str(export_dir), include_classes=[5], single_cls=False)
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    save_dir = tmp_path / 'run'
    save_dir.mkdir()

    # Subset run, but no class_remap.json was produced -> not a silent pass.
    assert dataset_prep.copy_class_remap_to_weights_dir(spec, save_dir) is False

    # Now produce one; the copy must land next to best.pt.
    spec.subset_dir.mkdir(parents=True)
    (spec.subset_dir / 'class_remap.json').write_text(
        json.dumps({'original_to_new': {'5': 0}, 'single_cls': False, 'names': ['widget']})
    )
    assert dataset_prep.copy_class_remap_to_weights_dir(spec, save_dir) is True
    assert (save_dir / 'weights' / 'class_remap.json').is_file()


def test_class_remap_copy_is_a_no_op_for_a_whole_export_run_missing_the_remap(
    jobs_dir: Path, export_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without ``write_full_class_remap`` having run (this test parses the
    spec directly, skipping ``prepare_dataset``), a full-class run has no
    ``full_class_remap_path`` file yet -- that alone is not a failure; an
    export that predates the dense-id remap hits this same path for real."""
    monkeypatch.setattr(job_protocol, 'TMP_ROOT', tmp_path / 'tmp')
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    save_dir = tmp_path / 'run'
    save_dir.mkdir()

    assert dataset_prep.copy_class_remap_to_weights_dir(spec, save_dir) is True
    assert not (save_dir / 'weights' / 'class_remap.json').exists()


def test_write_full_class_remap_for_a_whole_export_run_with_a_registry_gap(
    jobs_dir: Path, export_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The root-cause fix: a whole-export run's dense export ids (0/1/2) do
    NOT equal its registry ids (5/9/12, the ``export_dir`` fixture's
    deliberately-gapped registry) -- promote's ``labels.txt`` must be built
    from this remap, never straight from the registry.

    Before the fix, ``write_full_class_remap`` did not exist and a
    whole-export run left promote with only the (registry-id-keyed) live
    registry -- ``build_class_id_to_name`` for ``remap.source == 'none'``
    then wrote ``labels.txt`` line 0 as registry class 0's name (nothing,
    here) instead of dense id 0's real class (``widget``, registry id 5).
    """
    monkeypatch.setattr(job_protocol, 'TMP_ROOT', tmp_path / 'tmp')
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    written = dataset_prep.write_full_class_remap(spec)
    assert written == spec.full_class_remap_path
    payload = json.loads(written.read_text())
    assert payload['original_to_new'] == {'5': 0, '9': 1, '12': 2}
    assert payload['names'] == ['widget', 'serial_plate', 'gadget']
    assert payload['single_cls'] is False
    assert payload['include_classes'] is None

    save_dir = tmp_path / 'run'
    save_dir.mkdir()
    assert dataset_prep.copy_class_remap_to_weights_dir(spec, save_dir) is True
    weights_remap = save_dir / 'weights' / 'class_remap.json'
    assert weights_remap.is_file()

    remap = resolve_class_remap(job_id=job_id, checkpoint_path=weights_remap, manifest=None)
    assert remap.source == 'weights_dir'

    # The bug, made concrete: registry ids are NOT 0/1/2, so an identity
    # map (what the pre-fix code effectively used) would mislabel every
    # class after the first. This must fail on today's (pre-fix) code.
    full_registry = {5: 'widget', 9: 'serial_plate', 12: 'gadget'}
    labels = build_class_id_to_name(remap=remap, full_registry=full_registry)
    assert labels == {0: 'widget', 1: 'serial_plate', 2: 'gadget'}
    identity_guess = {i: full_registry.get(i, f'unknown_{i}') for i in range(3)}
    assert labels != identity_guess


def test_text_classes_resolve_by_name_against_the_training_data_yaml(
    jobs_dir: Path, export_dir: Path
) -> None:
    job_id = _write_job(
        dataset_export_dir=str(export_dir),
        augmentation={'enabled': True, 'text_class_names': ['Serial_Plate']},
    )
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    # Matching is case-insensitive and lands in the data.yaml's id space.
    assert dataset_prep.resolve_text_classes(spec, export_dir / 'data.yaml') == {1}


def test_text_classes_follow_the_subset_renumbering(
    tmp_path: Path, jobs_dir: Path, export_dir: Path
) -> None:
    """After a subset rewrite the same name must resolve to its NEW id."""
    out = tmp_path / 'subset'
    subset_dataset.build_subset_view(
        export_dir=export_dir, include_classes=[12, 9], single_cls=False, out_dir=out
    )
    job_id = _write_job(
        dataset_export_dir=str(export_dir),
        include_classes=[12, 9],
        augmentation={'enabled': True, 'text_class_names': ['serial_plate']},
    )
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    assert dataset_prep.resolve_text_classes(spec, out / 'data.yaml') == {1}


def test_text_classes_fall_back_to_the_deployment_default(
    jobs_dir: Path, export_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_TRAIN_TEXT_CLASS_NAMES', 'serial_plate,absent_class')
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    # Unknown names are ignored rather than raising -- a subset run drops
    # classes legitimately.
    assert dataset_prep.resolve_text_classes(spec, export_dir / 'data.yaml') == {1}
