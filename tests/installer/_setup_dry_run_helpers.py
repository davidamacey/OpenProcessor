"""Shared runners for the setup-openprocessor.sh dry-run tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from installer_harness import PROJECT, RELEASE


if TYPE_CHECKING:
    from installer_harness import Shimmed


def dry(shimmed: Shimmed, *extra: str, tiers: str = 'core', **env: str):
    return shimmed.run(
        [
            '--dry-run',
            '--unattended',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--version',
            RELEASE,
            '--tiers',
            tiers,
            *extra,
        ],
        **env,
    )


def configure(shimmed: Shimmed, *extra: str, tiers: str = 'core', **env: str):
    """A real (non-dry) run that writes the install but starts nothing; the
    docker shim accepts the pulls/probe and records them."""
    shimmed.flag('allow_mutations')
    return shimmed.run(
        [
            '--unattended',
            '--no-start',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--version',
            RELEASE,
            '--tiers',
            tiers,
            *extra,
        ],
        **env,
    )


def env_file(shimmed: Shimmed) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (shimmed.root / 'inst' / '.env').read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            out[k] = v
    return out


def compose_lines(shimmed: Shimmed, result) -> list[str]:
    lines = [
        ln[len('DRY: ') :]
        for ln in result.stdout.splitlines()
        if ln.startswith('DRY: docker compose')
    ]
    lines += list(shimmed.log_lines('docker compose'))
    return lines
