"""The reprocess-specific busy refusal is gone: a second start is the queue's 409 ``job_busy``."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_no_source_or_user_doc_names_the_old_busy_code() -> None:
    hits = [
        str(path.relative_to(ROOT))
        for path in [*(ROOT / 'src').rglob('*.py'), ROOT / 'docs' / 'CURATION.md']
        if 'reprocess_busy' in path.read_text(encoding='utf-8')
    ]
    assert hits == []
