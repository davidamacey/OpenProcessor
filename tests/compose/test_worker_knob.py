"""One knob sets the API worker count: ``API_WORKERS`` feeds ``WEB_CONCURRENCY``, which uvicorn reads.

uvicorn: ``--workers`` defaults to ``$WEB_CONCURRENCY`` (https://www.uvicorn.org/settings/).
"""

from __future__ import annotations

import re

from compose._helpers import REPO_ROOT, _services


def _api_env() -> dict[str, str]:
    env = _services()['api']['environment']
    return {e.split('=', 1)[0]: e.split('=', 1)[1] for e in env if '=' in e}


def test_api_command_has_no_workers_flag() -> None:
    command = [str(a) for a in _services()['api']['command']]
    assert not [a for a in command if a.startswith('--workers')], command


def test_api_web_concurrency_comes_from_api_workers() -> None:
    assert _api_env()['WEB_CONCURRENCY'] == '${API_WORKERS:-8}'


def test_dockerfile_cmd_has_no_workers_flag() -> None:
    text = (REPO_ROOT / 'Dockerfile').read_text()
    cmd = text[text.rindex('CMD ["uvicorn"') :]
    assert '"--workers"' not in cmd


def test_no_stale_worker_count_comments_remain() -> None:
    text = (REPO_ROOT / 'docker-compose.yml').read_text()
    assert not re.search(r'64 workers|Benchmark: 4 workers', text)


def test_control_flag_still_detected_in_a_command_list() -> None:
    assert [a for a in ['uvicorn', '--workers=4'] if a.startswith('--workers')]
