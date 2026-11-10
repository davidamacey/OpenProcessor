"""``docker_ps_snapshot`` must see only the harness's own compose project.

``test_start_queues_a_run_and_stops_no_container`` compares two snapshots around ``POST
/train/start``. On a shared host another stack's containers start and stop at any time, so a
host-wide ``docker ps`` made the check fail for reasons unrelated to the arbiter.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

from tests.live import conftest as live


if TYPE_CHECKING:
    import pytest


def test_the_snapshot_asks_docker_for_this_compose_project_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='b\t\t\na\t\t\n', stderr='')

    monkeypatch.setattr(live.subprocess, 'run', fake_run)

    assert live.docker_ps_snapshot() == 'a\nb'
    assert f'label=com.docker.compose.project={live.COMPOSE_PROJECT}' in seen[0]
