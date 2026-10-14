"""One stored-timestamp format: UTC, microsecond precision, lexically ordered."""

import re
from datetime import datetime
from pathlib import Path

from src.domain.history import record_class_history, record_class_snapshot
from src.services.curation.edit_history import EditKind, record_edit
from src.utils.timeutil import now_iso


STAMP = re.compile(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00$')
SRC = Path(__file__).resolve().parents[1] / 'src'


def test_now_iso_is_fixed_width_utc_microseconds():
    stamp = now_iso()
    assert STAMP.match(stamp), stamp
    assert datetime.fromisoformat(stamp).utcoffset().total_seconds() == 0


def test_now_iso_stamps_sort_lexically_in_time_order():
    stamps = [now_iso() for _ in range(50)]
    assert stamps == sorted(stamps)


def test_class_history_entries_use_the_shared_format():
    entries = record_class_history({'class_id': 1, 'class_source': 'vlm'}, writer='t')
    assert STAMP.match(entries[-1]['at']), entries[-1]['at']
    snap = record_class_snapshot(
        {'class_id': 1, 'class_source': 'vlm'}, writer='t', restorable=True
    )
    assert STAMP.match(snap[-1]['at']), snap[-1]['at']


def test_edit_history_entries_use_the_shared_format():
    entries = record_edit({}, kind=EditKind.VLM_DISMISS, writer='t')
    assert STAMP.match(entries[-1]['at']), entries[-1]['at']


def test_no_private_now_helper_is_defined_in_src():
    offenders = [
        str(p.relative_to(SRC))
        for p in SRC.rglob('*.py')
        if p.name != 'timeutil.py'
        and re.search(r'^def (_?now_iso|_now|_utc_now)\(', p.read_text(), re.M)
    ]
    assert offenders == []
