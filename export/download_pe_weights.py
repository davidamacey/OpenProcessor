#!/usr/bin/env python3
"""
PE-Core Checkpoint Download + Verification
==========================================

Fetches the Perception Encoder (PE-Core) checkpoint both PE exporters load:

* ``export/export_pe_image_encoder.py`` — vision tower -> ``pe_image_encoder``
* ``export/export_pe_text_encoder.py``  — text tower   -> ``pe_text_encoder``

and, at runtime, the in-process PyTorch fallback of
``src/clients/pe_encoder.py`` (``PEEncoder.warm_text_encoder``), which goes
through ``perception_models``' own ``hf_hub_download`` call and so reads the
same HuggingFace cache this script populates.

Source
------
``facebook/PE-Core-L14-336`` on the HuggingFace Hub, file
``PE-Core-L14-336.pt`` (~2.7 GB, both towers in one state dict). The repo is
**not gated** (Apache-2.0) — no ``huggingface-cli login`` needed. The
HF-native repackaging ``facebook/PE-Core-L14-336-hf`` (only used by the image
exporter's ``--method optimum``) *is* gated; that path needs ``HF_TOKEN``.

Pinning
-------
The download is pinned to an exact repo commit and the file is checked
against its SHA-256 (the Hub's LFS object id, which is also the blob name in
the local HF cache), so a silently re-uploaded checkpoint cannot change the
embedding space under an existing index. Other PE variants have no pin on
record; they download from ``main`` with a warning and skip the checksum.

Where it lands
--------------
The standard HF cache (``$HF_HUB_CACHE`` / ``$HF_HOME/hub``, default
``~/.cache/huggingface/hub``). In the API container that is
``/home/appuser/.cache/huggingface`` — bind-mounted from
``./cache/huggingface`` by docker-compose.yml — so one download serves the
exporters and the API.

Usage
-----
    # In the API container (recommended; the cache is mounted there)
    docker compose exec yolo-api python /app/export/download_pe_weights.py

    # Anywhere with huggingface_hub installed; print only the resolved path
    python export/download_pe_weights.py --print-path

    # Verify an already-downloaded / hand-copied checkpoint (offline)
    python export/download_pe_weights.py --verify-file /path/to/PE-Core-L14-336.pt
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


DEFAULT_VARIANT = 'PE-Core-L14-336'


@dataclass(frozen=True)
class PECheckpointPin:
    """A reproducible reference to one PE checkpoint on the HF Hub."""

    repo_id: str
    filename: str
    revision: str | None
    sha256: str | None
    size_bytes: int | None = None


# Recorded from the Hub API (``/api/models/facebook/PE-Core-L14-336``) and the
# file's ``X-Linked-ETag`` header, which is the LFS SHA-256.
PINS: dict[str, PECheckpointPin] = {
    'PE-Core-L14-336': PECheckpointPin(
        repo_id='facebook/PE-Core-L14-336',
        filename='PE-Core-L14-336.pt',
        revision='bafb0f76541d399057e980a25947f67acec76575',
        sha256='0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8',
        size_bytes=2_684_747_432,
    ),
}


class ChecksumMismatchError(RuntimeError):
    """The file on disk is not the pinned checkpoint."""


def pin_for(variant: str) -> PECheckpointPin:
    """The pin for ``variant``, or an unpinned ``main`` reference.

    The unpinned fallback follows ``perception_models``' own naming
    convention (``hf://facebook/<name>:<name>.pt``).
    """
    if variant in PINS:
        return PINS[variant]
    return PECheckpointPin(
        repo_id=f'facebook/{variant}', filename=f'{variant}.pt', revision=None, sha256=None
    )


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Streaming SHA-256 of a (multi-GB) file."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path, pin: PECheckpointPin) -> None:
    """Raise :class:`ChecksumMismatchError` unless ``path`` matches ``pin``.

    A pin without a checksum verifies nothing (and says so).
    """
    if pin.sha256 is None:
        logger.warning(f'No checksum on record for {pin.repo_id}; skipping verification.')
        return
    if pin.size_bytes is not None:
        size = Path(path).stat().st_size
        if size != pin.size_bytes:
            raise ChecksumMismatchError(
                f'{path} is {size} bytes, expected {pin.size_bytes} for {pin.repo_id}@'
                f'{pin.revision} — truncated or wrong file.'
            )
    logger.info(f'Verifying SHA-256 of {path} ...')
    actual = sha256_file(path)
    if actual != pin.sha256:
        raise ChecksumMismatchError(
            f'{path} has SHA-256 {actual}, expected {pin.sha256} ({pin.repo_id}@{pin.revision}).'
        )
    logger.info('  checksum OK')


def download_checkpoint(
    variant: str = DEFAULT_VARIANT,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    verify: bool = True,
) -> Path:
    """Download (or reuse from cache) the PE checkpoint and return its path.

    Args:
        variant: PE checkpoint name, e.g. ``PE-Core-L14-336``.
        revision: Override the pinned repo commit. Overriding also disables
            the checksum, which only describes the pinned revision.
        cache_dir: HF cache root override (default: the HF env defaults).
        verify: Check the pinned SHA-256 after download.
    """
    from huggingface_hub import hf_hub_download

    pin = pin_for(variant)
    if revision is not None and revision != pin.revision:
        pin = PECheckpointPin(pin.repo_id, pin.filename, revision, None)
    if pin.revision is None:
        logger.warning(f'{variant} has no pinned revision; downloading from main.')

    logger.info(f'Resolving {pin.repo_id}:{pin.filename} @ {pin.revision or "main"}')
    path = Path(
        hf_hub_download(
            repo_id=pin.repo_id,
            filename=pin.filename,
            revision=pin.revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    logger.info(f'  -> {path}')
    if verify:
        verify_checkpoint(path, pin)
    return path


def resolve_checkpoint(
    variant: str = DEFAULT_VARIANT,
    checkpoint_path: Path | None = None,
    *,
    verify: bool = True,
) -> Path:
    """The checkpoint an exporter should load.

    An explicit ``checkpoint_path`` (a hand-copied / air-gapped file) wins
    and is verified against the pin when the variant has one; otherwise the
    pinned revision is downloaded into (or found in) the HF cache.
    """
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f'--checkpoint-path {path} does not exist')
        if verify:
            verify_checkpoint(path, pin_for(variant))
        return path
    return download_checkpoint(variant, verify=verify)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Download + verify the PE-Core checkpoint used by both PE exporters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--variant', default=DEFAULT_VARIANT, help=f'default: {DEFAULT_VARIANT}')
    parser.add_argument(
        '--revision', default=None, help='Override the pinned HF commit (disables the checksum)'
    )
    parser.add_argument('--cache-dir', type=Path, default=None, help='HF cache root override')
    parser.add_argument('--no-verify', action='store_true', help='Skip the SHA-256 check')
    parser.add_argument(
        '--verify-file',
        type=Path,
        default=None,
        help='Verify an existing checkpoint file against the pin and exit (no network)',
    )
    parser.add_argument(
        '--print-path', action='store_true', help='Print only the resolved path on stdout'
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_path:
        logging.getLogger().setLevel(logging.WARNING)
    try:
        if args.verify_file is not None:
            verify_checkpoint(args.verify_file, pin_for(args.variant))
            path = args.verify_file
        else:
            path = download_checkpoint(
                args.variant,
                revision=args.revision,
                cache_dir=args.cache_dir,
                verify=not args.no_verify,
            )
    except (ChecksumMismatchError, FileNotFoundError) as exc:
        logger.error(str(exc))
        return 1
    print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
