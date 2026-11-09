"""The file protocol between the API and the trainer is gone, in both halves."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCANNED = (ROOT / 'docker' / 'trainer', ROOT / 'src' / 'services' / 'training')
# `.job.json`, `.status.json` and the `.cancel` sentinel are file names of the protocol;
# `claim.g<N>` is the O_EXCL claim generation file. `.cancelled(` and `.cancel_requested` are not.
PROTOCOL = re.compile(r'\.job\.json|\.status\.json|claim\.g|\.cancel(?![A-Za-z_])')


def _hits() -> list[str]:
    found = []
    for base in SCANNED:
        for path in sorted(base.rglob('*')):
            if path.suffix not in {'.py', '.txt', '.md', ''}:
                continue
            if not path.is_file():
                continue
            for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if PROTOCOL.search(line):
                    found.append(f'{path.relative_to(ROOT)}:{number}: {line.strip()}')
    return found


def test_no_protocol_file_name_is_left_in_the_trainer_or_the_training_services() -> None:
    assert _hits() == []


def test_the_protocol_modules_are_deleted() -> None:
    for name in ('job_protocol.py', 'job_claim.py'):
        assert not (ROOT / 'docker' / 'trainer' / name).exists()
    assert not (ROOT / 'src' / 'services' / 'training' / 'job_files.py').exists()


def test_the_scan_finds_planted_names() -> None:
    for planted in ('a.job.json', 'b.status.json', 'claim.g3', "x + '.cancel'"):
        assert PROTOCOL.search(planted)
    assert not PROTOCOL.search('ctx.cancelled() and state.cancel_requested')
