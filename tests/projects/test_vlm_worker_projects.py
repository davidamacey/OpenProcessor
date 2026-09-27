"""vlm_worker.py has no index literal and calls project-scoped routes.
The end-to-end isolation proof (reads and label calls per project behind
the guard) is ``test_worker_leak.py``."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.curation import vlm_worker
from scripts.curation._project_worker_utils import scoped_url


pytestmark = pytest.mark.unbound


def test_no_items_index_literal_in_source() -> None:
    text = Path(vlm_worker.__file__).read_text(encoding='utf-8')
    assert 'op_items' not in text
    assert '_items' not in text


def test_label_batch_url_is_project_scoped() -> None:
    url = scoped_url('http://api', '/curation', 'alpha', '/vlm/label_batch')
    assert url == 'http://api/curation/projects/alpha/vlm/label_batch'
