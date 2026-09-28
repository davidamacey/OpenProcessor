"""W10 fix pass (Opus review 2026-09-28, finding M1):
scripts/curation/import_labeled_dataset.py's default (labeled) mode
posts to two removed surfaces -- IngestBatchRequest.extra='forbid'
rejects the label fields it sends to /ingest/batch (422), and
--relabel-duplicates posts to the deleted /import_labels/batch (404).
Its own test (tests/integration/test_import_labeled_dataset.py) was
deleted in the same diff with no replacement, so nothing caught it.

Rather than resurrecting the removed HTTP surface or wiring an
in-process dataset_import client (no route/job infra exists yet for
this to front), the labeled mode now fails loudly and immediately --
before any dataset discovery or HTTP call -- instead of 422ing deep in
a request with no operator-visible signal. --images-only is
unaffected: this test proves it is NOT caught by the new guard.
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

import pytest

from scripts.curation.import_labeled_dataset import _async_main


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_labeled_mode_fails_loudly_before_any_dataset_work(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """images_only=False must return non-zero and log a clear message,
    without ever reaching dataset discovery (a bogus, nonexistent
    ``dataset`` path would otherwise raise a DIFFERENT (DatasetError)
    message if discovery ran -- it must not)."""
    args = argparse.Namespace(images_only=False, dataset=tmp_path / 'does-not-exist' / 'data.yaml')
    with caplog.at_level(logging.ERROR):
        rc = await _async_main(args)
    assert rc == 1
    messages = ' '.join(r.message for r in caplog.records)
    assert 'not available' in messages
    assert '/datasets/imports' in messages
    assert '--images-only' in messages
    # Proves discovery never ran: discover()'s own DatasetError names the
    # dataset path/data.yaml in a very different phrasing than the
    # guard's own message.
    assert 'does-not-exist' not in messages


@pytest.mark.asyncio
async def test_images_only_mode_is_not_caught_by_the_labeled_mode_guard(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """images_only=True must proceed PAST the new guard -- reaching (and
    failing inside) dataset discovery instead, proving the guard is
    scoped to labeled mode only. A missing dataset directory raises
    FileNotFoundError out of discover() itself (pre-existing, unrelated
    to this fix) -- reaching that call at all is what this test pins."""
    bogus = tmp_path / 'does-not-exist' / 'data.yaml'
    args = argparse.Namespace(images_only=True, dataset=bogus)
    with caplog.at_level(logging.ERROR), pytest.raises(FileNotFoundError):
        await _async_main(args)
    messages = ' '.join(r.message for r in caplog.records)
    # The labeled-mode guard's message must never have fired.
    assert 'not available' not in messages
