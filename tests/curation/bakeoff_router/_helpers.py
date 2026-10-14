"""Builders for the bake-off router tests: tiny YOLO exports and training-run status/manifest pairs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


def _names_yaml(names: list[str]) -> str:
    return 'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(names))


def make_export(
    root: Path,
    rel: str,
    *,
    nc: int,
    test_labels: dict[str, list[int]],
    export_id_map: dict[int, int] | None = None,
    manifest: dict[str, Any] | None = None,
    names: list[str] | None = None,
    train_labels: dict[str, list[int]] | None = None,
    background: tuple[str, ...] = (),
) -> Path:
    """A minimal YOLO export under ``root/rel``.

    ``test_labels`` / ``train_labels`` map an image stem to the class ids of
    its boxes. ``background`` stems get an image and no label file.
    """
    d = root / rel
    for split, labels in (('test', test_labels), ('train', train_labels or {})):
        (d / 'labels' / split).mkdir(parents=True, exist_ok=True)
        (d / 'images' / split).mkdir(parents=True, exist_ok=True)
        for stem, classes in labels.items():
            (d / 'labels' / split / f'{stem}.txt').write_text(
                ''.join(f'{c} 0.5 0.5 0.2 0.2\n' for c in classes)
            )
            (d / 'images' / split / f'{stem}.jpg').write_bytes(b'img')
    for stem in background:
        (d / 'images' / 'test' / f'{stem}.jpg').write_bytes(b'img')
    names = names or [f'class_{i}' for i in range(nc)]
    (d / 'data.yaml').write_text(f'nc: {nc}\n' + _names_yaml(names))
    (d / 'class_registry.json').write_text(
        json.dumps({'export_id_map': {str(k): v for k, v in (export_id_map or {}).items()}})
    )
    (d / 'manifest.json').write_text(json.dumps(manifest or {}))
    return d


# The live 84-class export's shape: 5 classes present in test.
LIVE_NAMES = [f'class_{i}' for i in range(84)]
for _eid, _n in {37: 'miata', 38: 'minicooper', 43: 'mustang', 51: 'porsche', 78: 'vw'}.items():
    LIVE_NAMES[_eid] = _n
LIVE_EXPORT_ID_MAP = {r: r - 1 for r in range(1, 85)}  # registry r -> export r-1
LIVE_REMAP = {
    'original_to_new': {'38': 0, '39': 1, '44': 2, '52': 3, '79': 4},
    'new_to_original': {'0': 38, '1': 39, '2': 44, '3': 52, '4': 79},
    'single_cls': False,
    'names': ['miata', 'minicooper', 'mustang', 'porsche', 'vw'],
    'include_classes': [38, 39, 44, 52, 79],
}


def _live_test_labels() -> dict[str, list[int]]:
    return {f'img_{eid}_{k}': [eid] for eid in (37, 38, 43, 51, 78) for k in range(5)}


def make_run(
    dirs: dict[str, Path],
    run_id: str,
    *,
    export_dir: Path,
    class_remap: dict[str, Any] | None = None,
    single_cls: bool = False,
    include_classes: list[int] | None = None,
    imgsz: int = 640,
    frozen_test_sha: str | None = None,
    eval_block: dict[str, Any] | None = None,
    state: str = 'finished',
) -> Path:
    """A finished training run as the trainer leaves it: status + manifest + best.pt."""
    ckpt = dirs['runs'] / run_id / 'weights' / 'best.pt'
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b'pt')
    # `dirs['train_jobs']` is already default's fully-nested
    # `OP_TRAIN_JOBS_DIR/projects/default` (see `default_train_jobs_dir`
    # in tests/_project_paths.py) -- do not nest it again here.
    jobs = dirs['train_jobs']
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / f'{run_id}.status.json').write_text(
        json.dumps(
            {
                'job_id': run_id,
                'state': state,
                'finished_at': '2026-09-24T23:51:09Z',
                'checkpoint_path': str(ckpt),
                'eval': eval_block or {'map50': 0.9356, 'split': 'test'},
            }
        )
    )
    (jobs / f'{run_id}.manifest.json').write_text(
        json.dumps(
            {
                'kind': 'train',
                'job_id': run_id,
                'lineage': {
                    'export_dir': str(export_dir),
                    'dataset_sha': None,
                    'frozen_test_sha': frozen_test_sha,
                    'include_classes': include_classes,
                    'single_cls': single_cls,
                    'class_remap': class_remap,
                },
                'spec': {
                    'model_family': 'yolo26',
                    'model_size': 'n',
                    'hyperparameters': {'imgsz': imgsz},
                },
            }
        )
    )
    return ckpt


def _live_export(dirs: dict[str, Path], rel: str = '20260924T233203Z', **kw: Any) -> Path:
    kw.setdefault(
        'manifest',
        {
            'dataset_sha': 'd' * 64,
            'exported_at': '2026-09-24T23:32:11+00:00',
            'unlabeled_items_on_exported_images': 69,
        },
    )
    return make_export(
        dirs['exports'],
        rel,
        nc=84,
        names=LIVE_NAMES,
        test_labels=_live_test_labels(),
        export_id_map=LIVE_EXPORT_ID_MAP,
        **kw,
    )


def _post_run(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post('/curation/projects/default/bakeoff/run', json=body)
