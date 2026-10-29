"""`./openprocessor status` probes the services and ports of THIS install (#363)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import test_cli_project
from test_cli_project import cli


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

inst = test_cli_project.inst


def _plain(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def _probed_urls(shimmed: Shimmed) -> list[str]:
    return [ln for ln in shimmed.log.read_text().splitlines() if ln.startswith('curl ')]


def test_a_refused_probe_prints_one_http_code_not_two(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('health_down')
    done = cli(shimmed, inst, 'status')
    assert done.returncode == 0, done.stderr
    assert 'HTTP 000)' in done.stdout
    assert '000000' not in done.stdout


def test_status_probes_the_ports_in_this_installs_env(shimmed: Shimmed, inst: Path) -> None:
    with (inst / '.env').open('a') as env:
        env.write('API_PORT=5703\nTRITON_HTTP_PORT=5700\nOPENSEARCH_PORT=5707\n')
    done = cli(shimmed, inst, 'status')
    assert done.returncode == 0, done.stderr
    urls = ' '.join(_probed_urls(shimmed))
    assert 'localhost:5700/' in urls
    assert 'localhost:5703/' in urls
    assert 'localhost:4600' not in urls
    assert 'Triton (5700)' in done.stdout


def test_a_service_outside_the_installed_tiers_is_not_probed(shimmed: Shimmed, inst: Path) -> None:
    # compose lists only the services of the active profiles and overlays; a
    # control-plane-only install has no Triton and a core install has no Grafana
    shimmed.flag('services', 'opensearch\napi\n')
    done = cli(shimmed, inst, 'status')
    assert done.returncode == 0, done.stderr
    urls = ' '.join(_probed_urls(shimmed))
    assert ':4600' not in urls
    assert ':4605' not in urls
    out = _plain(done.stdout)
    assert 'Triton (4600):    not part of this install' in out
    assert 'Grafana (4605):   not part of this install' in out
    assert 'API (4603):       healthy' in out
