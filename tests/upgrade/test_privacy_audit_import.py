"""The upgrade imports the old purge audit files once and leaves them in place."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.db


@pytest.mark.asyncio
async def test_the_upgrade_imports_the_jsonl_records(make_runner, clean_db, tmp_path: Path) -> None:
    audit = tmp_path / '.install' / 'data' / 'audit'
    audit.mkdir(parents=True)
    line = json.dumps({'lines': 3, 'at': '2026-10-30T08:00:00+00:00'}) + '\n'
    (audit / 'ocr_purge.jsonl').write_text(line)
    await make_runner().run()
    with clean_db.engine('super').begin() as conn:
        rows = conn.execute(text('SELECT kind, counts FROM privacy_purges')).all()
        conn.execute(text('DELETE FROM privacy_purges'))
    assert [(r[0], r[1]) for r in rows] == [('ocr', {'lines': 3})]
    assert (audit / 'ocr_purge.jsonl').read_text() == line
