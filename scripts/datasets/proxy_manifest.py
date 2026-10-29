"""Manifest, verification and label rows shared by the face and vehicle proxy fetchers.

A proxy dataset lives in its own directory (``<dir>/images/...``) next to a ``manifest.json``
that lists every file with its byte size and SHA-256, the source URL and the licence. The
manifest digest covers the file list, so a pin of ``manifest_sha256`` is enough to detect
any change. ``verify_manifest`` reports missing files, size and hash mismatches and files the
manifest does not list. Labels are JSONL rows with a fixed key set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scripts.datasets._common import FetchError, sha256_file, write_json


if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = 'proxy-dataset-manifest/1'
MANIFEST_NAME = 'manifest.json'
LABELS_NAME = 'labels.jsonl'
LABEL_KEYS = ('path', 'class_name', 'taxonomy_path', 'view', 'identity')
FILE_KEYS = ('path', 'bytes', 'sha256')
# Names the verifier never reports as unlisted: the dataset's own metadata and partial downloads.
DEFAULT_IGNORE = frozenset({MANIFEST_NAME, LABELS_NAME, 'ATTRIBUTION.csv', 'review_sample.csv'})


@dataclass(frozen=True)
class Licence:
    """The terms a source publishes. ``redistributable`` is False unless the source says so."""

    name: str
    url: str
    redistributable: bool
    note: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'url': self.url,
            'redistributable': self.redistributable,
            'note': self.note,
        }


@dataclass
class VerifyReport:
    missing: list[str] = field(default_factory=list)
    size_mismatch: list[str] = field(default_factory=list)
    hash_mismatch: list[str] = field(default_factory=list)
    unlisted: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.size_mismatch or self.hash_mismatch or self.unlisted)

    def problems(self) -> list[str]:
        out = [f'{p}: missing' for p in self.missing]
        out += [f'{p}: size differs from the manifest' for p in self.size_mismatch]
        out += [f'{p}: sha256 differs from the manifest' for p in self.hash_mismatch]
        out += [f'{p}: present but not in the manifest' for p in self.unlisted]
        return out


def safe_relpath(rel: str) -> str:
    """``rel`` when it is a relative POSIX path with no ``..`` part; manifests come from disk."""
    if (
        not rel
        or '\\' in rel
        or rel.startswith('/')
        or any(p in {'..', '.', ''} for p in rel.split('/'))
    ):
        raise FetchError(f'refusing the unsafe manifest path {rel!r}')
    return rel


def files_digest(files: list[dict[str, Any]]) -> str:
    """SHA-256 over the file list in path order; extra per-file keys are covered too."""
    body = json.dumps(sorted(files, key=lambda f: f['path']), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def file_entry(root: Path, rel: str, **extra: Any) -> dict[str, Any]:
    path = root / safe_relpath(rel)
    return {'path': rel, 'bytes': path.stat().st_size, 'sha256': sha256_file(path), **extra}


def build_manifest(
    dataset: str,
    source_url: str,
    licence: Licence,
    files: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    entries = sorted(files, key=lambda f: f['path'])
    paths = [f['path'] for f in entries]
    if len(set(paths)) != len(paths):
        raise FetchError(f'{dataset}: duplicate paths in the manifest')
    return {
        'schema': SCHEMA,
        'dataset': dataset,
        'source_url': source_url,
        'licence': licence.as_dict(),
        'count': len(entries),
        'total_bytes': sum(f['bytes'] for f in entries),
        'manifest_sha256': files_digest(entries),
        'files': entries,
        **extra,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    """The manifest at ``path`` after checking its schema, paths, count and digest."""
    doc = json.loads(path.read_text(encoding='utf-8'))
    files = doc.get('files') if isinstance(doc, dict) else None
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA or not isinstance(files, list):
        raise FetchError(f'{path}: not a {SCHEMA} manifest')
    for f in files:
        if not isinstance(f, dict) or set(FILE_KEYS) - set(f):
            raise FetchError(f'{path}: a file row lacks one of {FILE_KEYS}')
        safe_relpath(f['path'])
    if len({f['path'] for f in files}) != len(files):
        raise FetchError(f'{path}: duplicate paths')
    if doc.get('count') != len(files) or doc.get('manifest_sha256') != files_digest(files):
        raise FetchError(f'{path}: count or manifest_sha256 does not match the file rows')
    return doc


def _walk(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob('*')
        if p.is_file() and not p.name.endswith('.part')
    }


def verify_manifest(
    manifest: dict[str, Any],
    root: Path,
    *,
    check_unlisted: bool = True,
    ignore: frozenset[str] = DEFAULT_IGNORE,
    check_hashes: bool = True,
) -> VerifyReport:
    report = VerifyReport()
    listed = set()
    for f in manifest['files']:
        listed.add(f['path'])
        path = root / safe_relpath(f['path'])
        if not path.is_file():
            report.missing.append(f['path'])
        elif path.stat().st_size != f['bytes']:
            report.size_mismatch.append(f['path'])
        elif check_hashes and sha256_file(path) != f['sha256']:
            report.hash_mismatch.append(f['path'])
    if check_unlisted:
        report.unlisted = sorted(p for p in _walk(root) - listed if p not in ignore)
    return report


def pin_of(manifest: dict[str, Any]) -> dict[str, Any]:
    """The small committed summary of a manifest kept out of git (digest, count, licence)."""
    return {
        'schema': SCHEMA + '-pin',
        'dataset': manifest['dataset'],
        'source_url': manifest['source_url'],
        'licence': manifest['licence'],
        'count': manifest['count'],
        'total_bytes': manifest['total_bytes'],
        'manifest_sha256': manifest['manifest_sha256'],
    }


def label_row(
    path: str,
    class_name: str,
    taxonomy_path: list[str],
    *,
    view: str | None = None,
    identity: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    if not class_name or not taxonomy_path or taxonomy_path[-1] != class_name:
        raise FetchError(f'{path}: taxonomy_path must end with the class name {class_name!r}')
    return {
        'path': safe_relpath(path),
        'class_name': class_name,
        'taxonomy_path': list(taxonomy_path),
        'view': view,
        'identity': identity,
        **extra,
    }


def write_labels(path: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if set(LABEL_KEYS) - set(row):
            raise FetchError(f'label row lacks one of {LABEL_KEYS}: {row}')
    ordered = sorted(rows, key=lambda r: r['path'])
    path.write_text(
        ''.join(json.dumps(r, sort_keys=True) + '\n' for r in ordered), encoding='utf-8'
    )


def read_labels(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    for row in rows:
        if set(LABEL_KEYS) - set(row):
            raise FetchError(f'{path}: a label row lacks one of {LABEL_KEYS}')
    return rows


def verify_dataset_dir(directory: Path, *, check_hashes: bool = True) -> VerifyReport:
    """Verify ``directory`` against its own manifest and check labels point at listed files."""
    manifest = load_manifest(directory / MANIFEST_NAME)
    report = verify_manifest(manifest, directory, check_hashes=check_hashes)
    labels = directory / LABELS_NAME
    if labels.is_file():
        listed = {f['path'] for f in manifest['files']}
        report.missing += [
            f'{r["path"]} (label without a file)'
            for r in read_labels(labels)
            if r['path'] not in listed
        ]
    return report


def write_dataset(
    directory: Path, manifest: dict[str, Any], labels: list[dict[str, Any]], pin_path: Path | None
) -> None:
    write_json(directory / MANIFEST_NAME, manifest)
    write_labels(directory / LABELS_NAME, labels)
    if pin_path is not None:
        write_json(pin_path, pin_of(manifest))
