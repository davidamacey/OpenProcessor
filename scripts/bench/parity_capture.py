"""Collect the per-stage outputs of one ingest of the pinned set from a running stack.

Public API only, project scoped: images go in through ``POST .../ingest/upload``, items and their
cluster fields come back through ``GET .../crops``, and the core routes (MobileCLIP, faces, OCR)
are probed on a sample. The curation API deliberately never serves embedding vectors, so the PE
vectors are read by document id from the project's own OpenSearch indexes (the datastore of the
stack under test, never an index of another project).
"""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from scripts.bench.parity_store import CaptureError
from scripts.bench.suite_ingest import run_autolabel
from src.config.curation import ITEM_EMBEDDING_FIELD


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import httpx

# The list routes refuse to page past this many rows, so wider result sets are split in bands.
RESULT_WINDOW = 10_000
PAGE_SIZE = 500
MGET_CHUNK = 200
EMBED_ROUTE = '/embed/image'
FACES_ROUTE = '/faces/recognize'
OCR_ROUTE = '/ocr/predict'


def chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def project_indexes(client: httpx.Client, base: str) -> tuple[str, str]:
    """The project's ``(items, images)`` index names, as the API reports them."""
    names = client.get(base).raise_for_status().json()['resources']['indexes'].values()
    items = [n for n in names if n.endswith('__items')]
    images = [n for n in names if n.endswith('__images')]
    if len(items) != 1 or len(images) != 1:
        raise CaptureError(f'expected one items and one images index, got {items} and {images}')
    return items[0], images[0]


def require_embed_all(client: httpx.Client, base: str) -> str:
    """Fail unless the project embeds every crop; a parity set needs a vector per item."""
    policy = client.get(f'{base}/ingest/policy').raise_for_status().json()
    mode = policy['embedding']['mode']
    if mode != 'all':
        raise CaptureError(f'ingest embedding policy is {mode!r}; a parity capture needs "all"')
    return mode


def ingest_images(
    client: httpx.Client, base: str, files: Sequence[Path], *, batch_size: int, concurrency: int
) -> dict[str, str]:
    """Upload every file; returns ``{image_id: file name}``. Any failed image aborts."""

    def post(batch: Sequence[Path]) -> dict[str, Any]:
        upload = [('images', (p.name, p.read_bytes(), 'image/jpeg')) for p in batch]
        return client.post(f'{base}/ingest/upload', files=upload).raise_for_status().json()

    with ThreadPoolExecutor(concurrency) as pool:
        bodies = list(pool.map(post, chunks(list(files), batch_size)))
    names: dict[str, str] = {}
    failed: list[str] = []
    for body in bodies:
        for res in body['results']:
            if res['status'] == 'failed':
                failed.append(
                    f'{res.get("source_identifier") or res["image_path"]}: {res.get("error")}'
                )
            else:
                names[res['image_id']] = (res.get('source_identifier') or res['image_path']).rsplit(
                    '/', 1
                )[-1]
    if failed:
        raise CaptureError(f'{len(failed)} images failed to ingest, first: {failed[0]}')
    return names


def _crops_page(client: httpx.Client, base: str, params: dict[str, Any]) -> dict[str, Any]:
    query = {'include_test': 'true', 'include_excluded': 'true', **params}
    return client.get(f'{base}/crops', params=query).raise_for_status().json()


def fetch_items(
    client: httpx.Client, base: str, *, window: int = RESULT_WINDOW, page_size: int = PAGE_SIZE
) -> list[dict[str, Any]]:
    """Every item of the project through ``GET .../crops``.

    A result set wider than the API's paging window is split by confidence band, recursively;
    items on a band edge appear in both and are de-duplicated by id. The union must equal the
    unfiltered total, otherwise the capture aborts (unstable paging or a missed band).
    """
    total = int(_crops_page(client, base, {'page_size': 1})['total'])
    items: dict[str, dict[str, Any]] = {}
    bands = [(0.0, 1.0)]
    while bands:
        low, high = bands.pop()
        band = {'conf_min': low, 'conf_max': high}
        count = int(_crops_page(client, base, {**band, 'page_size': 1})['total'])
        if count > window and high - low > 1e-6:
            mid = (low + high) / 2
            bands += [(low, mid), (mid, high)]
            continue
        for page in range(1, math.ceil(count / page_size) + 1):
            body = _crops_page(client, base, {**band, 'page': page, 'page_size': page_size})
            items.update({c['id']: c for c in body['crops']})
    if len(items) != total:
        raise CaptureError(f'read {len(items)} items but the project reports {total}')
    return list(items.values())


def mget_vectors(
    client: httpx.Client,
    os_url: str,
    index: str,
    ids: Sequence[str],
    field: str = ITEM_EMBEDDING_FIELD,
) -> dict[str, list[float]]:
    """``{doc id: vector}`` read by id from one index; documents without a vector are absent."""
    out: dict[str, list[float]] = {}
    for part in chunks(list(ids), MGET_CHUNK):
        resp = client.post(
            f'{os_url}/{index}/_mget', params={'_source_includes': field}, json={'ids': list(part)}
        )
        for doc in resp.raise_for_status().json()['docs']:
            vec = (doc.get('_source') or {}).get(field) if doc.get('found') else None
            if vec:
                out[doc['_id']] = vec
    return out


def run_clustering(client: httpx.Client, base: str, timeout: float) -> dict[str, Any]:
    """One cluster training pass (no VLM) through the auto-label job.

    The job is claimed by the auto-label worker (compose profile ``curation``); without it
    the job stays queued until ``timeout``.
    """
    result = run_autolabel(
        client, base, {'train_clusters': 'true', 'run_vlm': 'false'}, timeout=timeout
    )
    if result['status'] != 'completed':
        raise CaptureError(f'cluster training ended {result["status"]}')
    return result


def probe_core(
    client: httpx.Client, api_url: str, files: Sequence[Path], *, concurrency: int
) -> dict[str, list[tuple[Any, ...]]]:
    """MobileCLIP, face (box, score, ArcFace vector) and OCR (box, text) outputs per file.

    A 422 from a route means "nothing usable in this image" and yields no rows; any other
    error aborts.
    """

    def post(route: str, path: Path) -> dict[str, Any] | None:
        resp = client.post(
            f'{api_url}{route}', files={'image': (path.name, path.read_bytes(), 'image/jpeg')}
        )
        if resp.status_code == 422:
            return None
        return resp.raise_for_status().json()

    def one(path: Path) -> dict[str, list[tuple[Any, ...]]]:
        out: dict[str, list[tuple[Any, ...]]] = {'clip': [], 'faces': [], 'ocr': []}
        if clip := post(EMBED_ROUTE, path):
            out['clip'].append((path.name, clip['embedding']))
        if faces := post(FACES_ROUTE, path):
            out['faces'] += [
                (path.name, f['box'], f['score'], vec)
                for f, vec in zip(faces['faces'], faces.get('embeddings') or [], strict=False)
            ]
        if ocr := post(OCR_ROUTE, path):
            out['ocr'] += [(path.name, r['box_normalized'], r['text']) for r in ocr['regions']]
        return out

    merged: dict[str, list[tuple[Any, ...]]] = {'clip': [], 'faces': [], 'ocr': []}
    with ThreadPoolExecutor(concurrency) as pool:
        for part in pool.map(one, files):
            for key, rows in part.items():
                merged[key] += rows
    return merged


def vlm_sample(
    items: Sequence[dict[str, Any]], image_names: dict[str, str], count: int
) -> list[str]:
    """Item ids of about ``count`` crops, whole images chosen by a hash of the file name.

    The choice depends only on file names, so two captures of the same set sample the same images.
    """
    by_file: dict[str, list[str]] = {}
    for item in items:
        by_file.setdefault(image_names[item['image_id']], []).append(item['id'])
    chosen: list[str] = []
    for name in sorted(by_file, key=lambda n: hashlib.sha256(n.encode()).hexdigest()):
        if len(chosen) >= count:
            break
        chosen += sorted(by_file[name])
    return chosen


def label_with_vlm(
    client: httpx.Client, base: str, ids: Sequence[str], classes: Sequence[str], batch: int
) -> dict[str, str]:
    """Run ``vlm/label_batch`` on ``ids`` and return ``{item id: proposed class name}``."""
    for name in classes:
        client.post(f'{base}/classes', json={'name': name, 'group': 'parity'})
    for part in chunks(list(ids), batch):
        client.post(f'{base}/vlm/label_batch', json={'crop_ids': list(part)}).raise_for_status()
    out: dict[str, str] = {}
    for part in chunks(list(ids), 100):
        rows = _crops_page(client, base, {'ids': ','.join(part)})['crops']
        out.update(
            {
                r['id']: r.get('vlm_proposed_class_name') or r.get('vlm_raw_class') or ''
                for r in rows
            }
        )
    return out
