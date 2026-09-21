"""Subset-training dataset rewrite.

Filters YOLO label files to a chosen subset of class IDs and renumbers the
surviving IDs to ``0..N-1`` contiguously, preserving the order of the original
``include_classes`` list. Ultralytics requires contiguous class IDs in
``data.yaml``, so a subset of a large registry can't be trained directly --
this module produces a renumbered view at training time without disturbing the
frozen export.

Outputs:

* ``out_dir/labels/{train,val,test}/*.txt`` -- filtered + renumbered labels.
* ``out_dir/images/{train,val,test}/*`` -- symlinks to the export's images, so
  Ultralytics' ``images/`` -> ``labels/`` path inference resolves against the
  rewritten labels rather than the export's originals.
* ``out_dir/data.yaml`` -- points at the symlinked images dir, the rewritten
  labels dir, and the renumbered ``names`` list.
* ``out_dir/class_remap.json`` -- original -> new mapping, consumed by
  ``POST {api_prefix}/train/promote/{job_id}`` (see
  ``src/services/training/triton_promote.py::resolve_class_remap``) to write
  the right ``labels.txt`` for the promoted Triton model. Its payload shape is
  a contract with that parser: ``{original_to_new, new_to_original, single_cls,
  names, include_classes}``.

The original labels are never modified.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


log = logging.getLogger('subset_dataset')


SPLITS = ('train', 'val', 'test')


@dataclass
class SubsetResult:
    """Return value of :func:`build_subset_view`."""

    data_yaml: Path
    class_remap: dict[int, int]  # original class_id -> new class_id
    dropped_rows: int
    kept_rows: int
    per_class_counts: dict[int, int]  # new_class_id -> row count (train+val+test)


def _read_export_data_yaml(export_dir: Path) -> dict[str, Any]:
    """Load the export's ``data.yaml`` to recover the original class names.

    The exporter writes a single ``data.yaml`` at the export root.
    """
    candidates = [export_dir / 'data.yaml', export_dir / 'dataset.yaml']
    for c in candidates:
        if c.is_file():
            with c.open('r', encoding='utf-8') as fh:
                loaded = yaml.safe_load(fh) or {}
                return dict(loaded)
    msg = f'no data.yaml or dataset.yaml under {export_dir} -- subset rewrite requires it'
    raise FileNotFoundError(msg)


def _normalize_names(names_raw: dict[Any, Any] | list[Any] | None) -> dict[int, str]:
    """Return ``{class_id: name}`` regardless of YOLO's representation flavor."""
    if not names_raw:
        return {}
    if isinstance(names_raw, list):
        return dict(enumerate(str(n) for n in names_raw))
    return {int(k): str(v) for k, v in names_raw.items()}


def _read_export_id_map(export_dir: Path) -> dict[int, int]:
    """Load the ``registry_id -> dense export_id`` map for one export.

    The dataset exporter (``src/services/curation/export.py``) writes every
    label ``.txt`` file using a DENSE export id (``0..nc-1``, contiguous even
    when a lower registry id is deprecated) -- never the raw registry
    ``class_id``. It snapshots the translation it used into
    ``class_registry.json`` next to ``data.yaml`` as ``export_id_map``
    (``{"<registry_id>": export_id}``).

    ``include_classes`` is documented and used everywhere else (``TrainJobSpec``,
    a labeler's class picker, ``POST {api_prefix}/train/preflight``) as raw
    REGISTRY ids -- a *different* id space from what's actually written in this
    export's label files once any lower registry id is deprecated. This map is
    what lets :func:`build_subset_view` translate between the two before it ever
    compares against a label file's class-id token.
    """
    registry_path = export_dir / 'class_registry.json'
    if not registry_path.is_file():
        msg = (
            f'{registry_path} missing -- cannot translate include_classes (registry '
            f"ids) into this export's dense label ids without the exporter's "
            f'export_id_map'
        )
        raise FileNotFoundError(msg)
    payload = json.loads(registry_path.read_text(encoding='utf-8'))
    raw_map = payload.get('export_id_map')
    if not isinstance(raw_map, dict):
        msg = (
            f'{registry_path} has no export_id_map -- was this export produced '
            f'by an older exporter that predates the dense-id remap?'
        )
        raise ValueError(msg)
    return {int(k): int(v) for k, v in raw_map.items()}


def build_subset_view(
    export_dir: Path,
    include_classes: list[int],
    single_cls: bool,
    out_dir: Path,
) -> SubsetResult:
    """Filter labels and renumber class IDs.

    Args:
        export_dir: Path to the frozen export root (contains ``images/`` and
            ``labels/{train,val,test}/`` plus ``data.yaml``).
        include_classes: Original REGISTRY class IDs to keep, in the desired
            output order. ``[7, 12]`` produces the class registered at id 7 as
            ``0`` and id 12 as ``1``, regardless of their original registry ids
            OR this export's dense ids -- translated internally via the export's
            own ``class_registry.json:export_id_map`` before matching against
            the label files, which already carry dense export ids.
        single_cls: When ``True``, all included classes collapse to a single
            ``object`` class with ID ``0``; ``include_classes`` still controls
            which rows survive the filter.
        out_dir: Target dir for the rewritten labels + data.yaml.

    Returns:
        :class:`SubsetResult` describing the rewrite.

    Raises:
        FileNotFoundError: if ``export_dir`` lacks ``data.yaml`` or
            ``class_registry.json``.
        ValueError: if ``include_classes`` is empty, references a registry id
            outside this export's active/dense id space (e.g. deprecated, or
            never exported), or no rows survive the filter.
    """
    if not include_classes:
        msg = 'include_classes must be non-empty for subset rewrite'
        raise ValueError(msg)

    export_dir = Path(export_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_yaml = _read_export_data_yaml(export_dir)
    src_names = _normalize_names(src_yaml.get('names'))

    # registry_id -> this export's dense export_id. include_classes is registry
    # ids; the label .txt files are already dense export ids -- every lookup
    # below MUST go through this map, never compare the two spaces directly.
    export_id_map = _read_export_id_map(export_dir)
    unknown = [c for c in include_classes if c not in export_id_map]
    if unknown:
        msg = (
            f"include_classes contains registry ids outside this export's active "
            f'class set (deprecated, or never exported): {unknown}'
        )
        raise ValueError(msg)

    # Build the original(registry id) -> new map. Order of include_classes
    # drives the new IDs. Kept registry-id-keyed (an API-side contract) since
    # class_remap.json's original_to_new is consumed downstream as registry ids
    # (triton_promote.py::build_class_id_to_name looks names up in the full
    # REGISTRY, not the export's dense id space).
    if single_cls:
        class_remap: dict[int, int] = dict.fromkeys(include_classes, 0)
        new_names: dict[int, str] = {0: 'object'}
    else:
        class_remap = {orig: i for i, orig in enumerate(include_classes)}
        new_names = {
            new_id: src_names.get(export_id_map[orig], f'class_{orig}')
            for orig, new_id in class_remap.items()
        }

    # INTERNAL-only map actually used to filter/relabel the frozen export's
    # label .txt files below, which carry dense export ids -- NOT the registry
    # ids in ``class_remap`` above.
    dense_id_to_new_id = {export_id_map[orig]: new_id for orig, new_id in class_remap.items()}

    dropped = 0
    kept = 0
    per_class_counts: dict[int, int] = dict.fromkeys(new_names, 0)

    for split in SPLITS:
        src_labels = export_dir / 'labels' / split
        if not src_labels.is_dir():
            log.info('subset: split %s has no labels dir, skipping', split)
            continue
        dst_labels = out_dir / 'labels' / split
        dst_labels.mkdir(parents=True, exist_ok=True)
        for txt in src_labels.glob('*.txt'):
            new_lines: list[str] = []
            for line in txt.read_text(encoding='utf-8').splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    log.warning('subset: bad class_id token in %s: %r', txt, parts[0])
                    dropped += 1
                    continue
                # ``cid`` here is the export's DENSE id -- match against
                # ``dense_id_to_new_id``, never ``class_remap`` (registry-id
                # keyed) directly.
                if cid not in dense_id_to_new_id:
                    dropped += 1
                    continue
                new_id = dense_id_to_new_id[cid]
                kept += 1
                per_class_counts[new_id] = per_class_counts.get(new_id, 0) + 1
                new_lines.append(' '.join([str(new_id), *parts[1:]]))
            # Always write the file (even when empty) so the dataset stays
            # paired with the original images. Ultralytics treats empty label
            # files as "background image with no boxes".
            (dst_labels / txt.name).write_text(
                ('\n'.join(new_lines) + ('\n' if new_lines else '')),
                encoding='utf-8',
            )

    if kept == 0:
        msg = (
            f'subset rewrite produced 0 surviving rows for include_classes={include_classes}'
            f' -- every label was dropped'
        )
        raise ValueError(msg)

    # Mirror the images dir into out_dir as symlinks. Ultralytics infers label
    # paths from image paths by swapping ``images/`` -> ``labels/`` -- if we
    # point train: at the original export's images, it will read the original
    # full-registry labels next to them and ignore our subset rewrite entirely.
    # By symlinking images under out_dir, both ``out_dir/images/{split}/x.jpg``
    # and ``out_dir/labels/{split}/x.txt`` live under the same root.
    for split in SPLITS:
        src_images = export_dir / 'images' / split
        if not src_images.is_dir():
            continue
        dst_images = out_dir / 'images' / split
        dst_images.mkdir(parents=True, exist_ok=True)
        for img in src_images.iterdir():
            link = dst_images / img.name
            if link.exists() or link.is_symlink():
                continue
            try:
                link.symlink_to(img.resolve())
            except OSError as exc:
                log.warning('subset: symlink failed for %s: %s', img, exc)

    # Build the new data.yaml. Both images + labels resolve under out_dir so
    # YOLO's images -> labels path inference picks up the subset rewrite.
    dst_yaml = {
        'path': str(out_dir.resolve()),
        'train': str((out_dir / 'images' / 'train').resolve()),
        'val': str((out_dir / 'images' / 'val').resolve()),
        'test': str((out_dir / 'images' / 'test').resolve()),
        'nc': len(new_names),
        'names': [new_names[i] for i in sorted(new_names)],
    }
    data_yaml_path = out_dir / 'data.yaml'
    with data_yaml_path.open('w', encoding='utf-8') as fh:
        yaml.safe_dump(dst_yaml, fh, sort_keys=False)

    # Persist class_remap for promote-to-Triton. Shape is a contract with
    # src/services/training/triton_promote.py::_parse_class_remap_payload.
    remap_path = out_dir / 'class_remap.json'
    remap_payload = {
        'original_to_new': {str(k): v for k, v in class_remap.items()},
        'new_to_original': {str(v): k for k, v in class_remap.items() if not single_cls},
        'single_cls': single_cls,
        'names': [new_names[i] for i in sorted(new_names)],
        'include_classes': include_classes,
    }
    remap_path.write_text(json.dumps(remap_payload, indent=2), encoding='utf-8')

    log.info(
        'subset: kept=%d dropped=%d nc=%d single_cls=%s',
        kept,
        dropped,
        len(new_names),
        single_cls,
    )
    return SubsetResult(
        data_yaml=data_yaml_path,
        class_remap=class_remap,
        dropped_rows=dropped,
        kept_rows=kept,
        per_class_counts=per_class_counts,
    )
