"""``service_hostnames`` names every replica of a service, not just one."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

from tests.live import _compose


if TYPE_CHECKING:
    import pytest


def test_all_replicas_of_a_service_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_compose(*args: str, **_: Any) -> subprocess.CompletedProcess[str]:
        assert args == ('ps', '-q', 'jobs-worker')
        return subprocess.CompletedProcess(args, 0, stdout='aaa\nbbb\n', stderr='')

    def fake_run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        assert argv[:2] == ['docker', 'inspect']
        assert argv[-2:] == ['aaa', 'bbb']
        return subprocess.CompletedProcess(argv, 0, stdout='host-a\nhost-b\n', stderr='')

    monkeypatch.setattr(_compose, 'compose', fake_compose)
    monkeypatch.setattr(_compose.subprocess, 'run', fake_run)
    assert _compose.service_hostnames('jobs-worker') == {'host-a', 'host-b'}
