"""YOLO dataset discovery shared by the dataset import and region-eval CLIs.

Resolves a ``data.yaml`` (or an ``images/<split>`` / ``<split>/images``
tree) into per-split image lists, pairs each image with its YOLO label file,
and draws reproducible stratified samples that keep the positive/background
proportion (positive = a non-empty label file).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
SPLIT_KEYS = ('train', 'val', 'valid', 'test')


class DatasetError(RuntimeError):
    """The dataset layout could not be resolved."""


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    n_labels: int
    label_exists: bool

    @property
    def positive(self) -> bool:
        return self.n_labels > 0


def label_path_for(image: Path) -> Path:
    """YOLO convention: the last ``images`` path segment becomes ``labels``, suffix ``.txt``."""
    parts = list(image.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == 'images':
            parts[i] = 'labels'
            return Path(*parts).with_suffix('.txt')
    return image.with_suffix('.txt')


def _count_label_rows(label: Path) -> tuple[int, bool]:
    try:
        text = label.read_text(encoding='utf-8')
    except FileNotFoundError:
        return 0, False
    rows = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    return len(rows), True


def _images_under(entry: Path, base: Path) -> list[Path]:
    if entry.is_dir():
        return sorted(
            p for p in entry.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    # A YOLO split may also be a .txt list of image paths.
    out = []
    for line in entry.read_text(encoding='utf-8').splitlines():
        if line.strip():
            p = Path(line.strip())
            out.append(p if p.is_absolute() else (base / p))
    return sorted(out)


def _resolve_entry(entry: str, yaml_dir: Path, root_field: str | None) -> Path:
    """Resolve a data.yaml split entry; tolerate a stale absolute ``path:``.

    A copied/moved dataset usually keeps its original ``path:`` — so the
    YAML's own directory is tried first, then ``path:``.
    """
    candidates: list[Path] = []
    if Path(entry).is_absolute():
        candidates.append(Path(entry))
    else:
        candidates.append(yaml_dir / entry)
        if root_field:
            root = Path(root_field)
            candidates.append((root if root.is_absolute() else yaml_dir / root) / entry)
    for c in candidates:
        if c.exists():
            return c
    tried = ', '.join(str(c) for c in candidates)
    raise DatasetError(f'split entry {entry!r} not found (tried: {tried})')


def _find_yaml(dataset: Path) -> Path | None:
    if dataset.is_file():
        return dataset
    for name in ('data.yaml', 'data.yml', 'dataset.yaml'):
        if (dataset / name).is_file():
            return dataset / name
    return None


def _names_list(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(n) for n in raw]
    return [str(raw[k]) for k in sorted(raw, key=int)]


def discover(dataset: Path) -> tuple[dict[str, list[Path]], list[str] | None]:
    """Return ``({split: [image paths]}, class names or None)``."""
    yaml_path = _find_yaml(dataset)
    if yaml_path is not None:
        import yaml

        data = yaml.safe_load(yaml_path.read_text(encoding='utf-8')) or {}
        splits: dict[str, list[Path]] = {}
        for key in SPLIT_KEYS:
            value = data.get(key)
            if not value:
                continue
            entries = value if isinstance(value, list) else [value]
            images: list[Path] = []
            for entry in entries:
                resolved = _resolve_entry(str(entry), yaml_path.parent, data.get('path'))
                images.extend(_images_under(resolved, resolved.parent))
            splits[key] = sorted(set(images))
        if not splits:
            raise DatasetError(f'{yaml_path}: no train/val/test entries')
        return splits, _names_list(data.get('names'))

    splits = {}
    if (dataset / 'images').is_dir():
        for d in sorted((dataset / 'images').iterdir()):
            if d.is_dir():
                splits[d.name] = _images_under(d, d)
    else:
        for d in sorted(dataset.iterdir()):
            if (d / 'images').is_dir():
                splits[d.name] = _images_under(d / 'images', d)
    if not splits:
        raise DatasetError(f'{dataset}: no data.yaml, images/<split>/ or <split>/images/ found')
    return splits, None


def load_samples(images: list[Path]) -> list[Sample]:
    out = []
    for image in images:
        label = label_path_for(image)
        n, exists = _count_label_rows(label)
        out.append(Sample(image=image, label=label, n_labels=n, label_exists=exists))
    return out


def stratified_sample(samples: list[Sample], limit: int, seed: int) -> list[Sample]:
    """Deterministic sample keeping the positive/background proportion."""
    if limit >= len(samples):
        return samples
    rng = random.Random(seed)  # nosec B311 - reproducible cohort selection, not crypto
    pos = [s for s in samples if s.positive]
    neg = [s for s in samples if not s.positive]
    n_pos = round(limit * len(pos) / len(samples))
    n_pos = min(len(pos), max(n_pos, 1 if pos else 0))
    n_neg = min(len(neg), limit - n_pos)
    picked = rng.sample(pos, n_pos) + rng.sample(neg, n_neg)
    return sorted(picked, key=lambda s: str(s.image))
