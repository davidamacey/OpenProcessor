"""The local VLM catalog: what the in-compose vLLM *can serve* (W9.7).

One file, two readers. ``examples/vlm/catalog.tsv`` is read by the
installer/CLI's bash (``scripts/lib/vlm_catalog.sh``, which PICKS) and by
this module (which only DISPLAYS and computes ``fits``), so there is no
duplicated policy. ``tests/curation/test_vlm_catalog.py`` pins that both
readers see the same rows.

The catalog is not an endpoint registry: a catalog entry becomes reachable
only through an ordinary endpoint (the ``env`` built-in on an installed
stack) after the host restarts the vlm container with that entry's
arguments.
"""

from __future__ import annotations

import csv
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[3] / 'examples' / 'vlm' / 'catalog.tsv'

#: The first non-comment line names these columns; the bash readers address
#: them by position, so new ones are only ever appended.
COLUMNS = (
    'id',
    'hf_repo',
    'licence',
    'vram_gb',
    'served_context',
    'max_context',
    'max_images',
    'status',
    'rank',
    'gated',
    'vllm_image_key',
    'family',
    'params_b',
    'quantization',
    'disk_gb',
    'reasoning_parser',
    'chat_template',
    'extra_args',
    'multi_box_verified',
    'text_reading_verified',
)


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    hf_repo: str
    family: str
    license: str
    gated: bool
    params_b: float | None
    quantization: str | None
    context_max: int
    max_model_len: int
    max_images: int
    vram_gb: float
    disk_gb: float | None
    rank: int
    status: str
    vllm_image_key: str
    reasoning_parser: str
    chat_template: str
    extra_args: str
    multi_box_verified: bool | None
    text_reading_verified: bool | None

    @property
    def license_url(self) -> str:
        return f'https://huggingface.co/{self.hf_repo}'


def _opt_float(raw: str) -> float | None:
    return float(raw) if raw.strip() else None


def _opt_bool(raw: str) -> bool | None:
    value = raw.strip().lower()
    if value in ('', 'unknown'):
        return None
    return value == 'true'


def _entry(row: dict[str, str]) -> CatalogEntry:
    return CatalogEntry(
        id=row['id'].strip(),
        hf_repo=row['hf_repo'].strip(),
        family=row['family'].strip(),
        license=row['licence'].strip().lower(),
        gated=row['gated'].strip().lower() == 'true',
        params_b=_opt_float(row['params_b']),
        quantization=row['quantization'].strip() or None,
        context_max=int(row['max_context']),
        max_model_len=int(row['served_context']),
        max_images=int(row['max_images']),
        vram_gb=float(row['vram_gb']),
        disk_gb=_opt_float(row['disk_gb']),
        rank=int(row['rank']),
        status=row['status'].strip(),
        vllm_image_key=row['vllm_image_key'].strip(),
        reasoning_parser=row['reasoning_parser'].strip(),
        chat_template=row['chat_template'].strip(),
        extra_args=row['extra_args'].strip(),
        multi_box_verified=_opt_bool(row['multi_box_verified']),
        text_reading_verified=_opt_bool(row['text_reading_verified']),
    )


@functools.lru_cache(maxsize=8)
def _load(path_str: str, mtime_ns: int) -> tuple[CatalogEntry, ...]:  # noqa: ARG001 - cache key
    lines = [
        line
        for line in Path(path_str).read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.startswith('#')
    ]
    reader = csv.DictReader(lines, delimiter='\t', restval='')
    missing = set(COLUMNS) - set(reader.fieldnames or ())
    if missing:
        msg = f'{path_str}: catalog is missing column(s) {sorted(missing)}'
        raise ValueError(msg)
    return tuple(_entry(row) for row in reader)


def load_catalog(path: Path | str | None = None) -> list[CatalogEntry]:
    """Every catalog row, file order."""
    resolved = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    return list(_load(str(resolved), resolved.stat().st_mtime_ns))


def catalog_entry(catalog_id: str | None) -> CatalogEntry | None:
    if not catalog_id:
        return None
    return next((e for e in load_catalog() if e.id == catalog_id), None)


def catalog_entry_for_root(root: str | None) -> CatalogEntry | None:
    """The entry whose ``hf_repo`` is the ``root`` a server reports for its
    served model (vLLM reports the underlying repo even behind an alias)."""
    if not root:
        return None
    return next((e for e in load_catalog() if e.hf_repo == root), None)


def fits(entry: CatalogEntry, gpu_total_gb: float | None) -> bool | None:
    """``vram_gb <= gpu_total_gb``; ``None`` when the card size is unknown."""
    if gpu_total_gb is None:
        return None
    return entry.vram_gb <= gpu_total_gb


def gpu_total_gb_from_env(raw: str | None) -> float | None:
    """``OP_LOCAL_VLM_GPU_TOTAL_MIB`` (MiB) -> GB; ``None`` unset/invalid."""
    if not raw or not raw.strip():
        return None
    try:
        mib = float(raw)
    except ValueError:
        return None
    return mib / 1024 if mib > 0 else None


def desired_command(catalog_id: str) -> str:
    return f'openprocessor vlm use {catalog_id}'


def local_vlm_status(
    *,
    endpoint_name: str,
    served_model: str | None,
    served_root: str | None,
    served_max_model_len: int | None,
    desired: dict[str, Any] | None,
    gpu_total_gb: float | None,
) -> dict[str, Any]:
    """The ``VlmLocalStatus`` wire dict. ``endpoint_name`` empty means there
    is no in-compose vLLM (``configured: false``).

    ``restart_required`` compares the desired catalog id's ``hf_repo`` with
    the probed ``root``, so it flips off only after the host command ran
    AND the re-probe recorded the new root -- never on the API's say-so.
    """
    configured = bool(endpoint_name)
    served_entry = catalog_entry_for_root(served_root)
    desired_id = (desired or {}).get('catalog_id')
    desired_entry = catalog_entry(desired_id) if desired_id else None
    restart_required = bool(
        desired_entry is not None and (served_root is None or served_root != desired_entry.hf_repo)
    )
    return {
        'configured': configured,
        'endpoint': endpoint_name or None,
        'served': (
            {
                'model': served_model,
                'root': served_root,
                'catalog_id': served_entry.id if served_entry else None,
                'max_model_len': served_max_model_len,
            }
            if configured
            else None
        ),
        'desired': (
            {
                'catalog_id': desired_id,
                'requested_at': (desired or {}).get('requested_at'),
                'command': desired_command(str(desired_id)),
            }
            if desired
            else None
        ),
        'restart_required': restart_required,
        'poll_after_s': 10 if restart_required else None,
        'gpu_total_gb': gpu_total_gb,
        'can_restart_from_api': False,
        'reason': (
            'The local VLM serves one model. Switching it restarts the vlm container, '
            'which is done from the host with the command above.'
        ),
    }


__all__ = [
    'COLUMNS',
    'DEFAULT_CATALOG_PATH',
    'CatalogEntry',
    'catalog_entry',
    'catalog_entry_for_root',
    'desired_command',
    'fits',
    'gpu_total_gb_from_env',
    'load_catalog',
    'local_vlm_status',
]
