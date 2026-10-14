"""Pin container names, published host ports, the bind address and the compose project name."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from compose._helpers import COMPOSE_PATH, REPO_ROOT, _load_compose, _load_yaml, _services


_NAME_LINE_RE = re.compile(r'^name:\s*(.+)$', re.MULTILINE)


def test_no_duplicate_container_names() -> None:
    services = _services()
    names = [spec['container_name'] for spec in services.values() if spec.get('container_name')]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f'duplicate container_name values: {dupes}'


def _published_ports(spec: dict[str, Any]) -> list[str]:
    """The host-side port identity for each published port.

    Installer plan §7: every entry now also carries a leading
    ``${OP_BIND_ADDRESS:-127.0.0.1}`` publish-address segment (short
    syntax becomes ``ADDR:HOST:CONTAINER``), so a naive
    ``str.split(':')[0]`` collapses every service's host port to the same
    ``${OP_BIND_ADDRESS`` string and hides real duplicates. Extract the
    actual host-port var(s) the same way ``_host_port_vars`` does instead.
    """
    published: list[str] = []
    for entry in spec.get('ports') or []:
        if isinstance(entry, dict):
            published.append(str(entry.get('published')))
            continue
        text = str(entry)
        host_vars = _HOST_PORT_VAR_RE.findall(text) if '${' in text else None
        if host_vars:
            published.append(','.join(v for v in host_vars if v not in _NON_HOST_PORT_VARS))
        else:
            # Short syntax with no vars at all: "HOST:CONTAINER" or a bare port.
            published.append(text.split(':', maxsplit=1)[0] if ':' in text else text)
    return published


def test_no_duplicate_host_ports() -> None:
    services = _services()
    all_ports: list[tuple[str, str]] = []
    for name, spec in services.items():
        all_ports.extend((name, port) for port in _published_ports(spec))
    seen: dict[str, str] = {}
    dupes: list[str] = []
    for name, port in all_ports:
        if port in seen and seen[port] != name:
            dupes.append(f'{port} used by both {seen[port]!r} and {name!r}')
        else:
            seen[port] = name
    assert not dupes, 'duplicate host ports:\n' + '\n'.join(dupes)


# =============================================================================
# Fresh-start gaps batch B — compose/install portability (G-01/G-03/G-06/
# G-07/G-08/G-10/G-15/G-28).
# =============================================================================


def test_no_fixed_compose_project_name() -> None:
    """G-01: a hardcoded top-level `name:` forces every `docker compose`
    invocation onto the same project regardless of directory/-p/env,
    which is exactly how a fresh clone can end up operating on a
    different, already-running stack's containers and volumes."""
    raw = COMPOSE_PATH.read_text(encoding='utf-8')
    parsed = _load_compose()
    match = _NAME_LINE_RE.search(raw)
    assert match, 'expected a top-level `name:` key in docker-compose.yml'
    value = match.group(1).strip()
    assert '${COMPOSE_PROJECT_NAME' in value, (
        f'docker-compose.yml `name:` must be interpolated from COMPOSE_PROJECT_NAME, got: {value!r}'
    )
    assert 'services' in parsed  # sanity: file still parses as valid compose


def test_every_container_name_is_interpolated_from_project_name() -> None:
    """G-01: every container_name must move with COMPOSE_PROJECT_NAME, not
    just the ones someone remembered to update."""
    services = _services()
    bad = {
        name: spec['container_name']
        for name, spec in services.items()
        if spec.get('container_name')
        and '${COMPOSE_PROJECT_NAME' not in str(spec['container_name'])
    }
    assert not bad, f'container_name values not interpolated from COMPOSE_PROJECT_NAME: {bad}'


def test_no_hardcoded_published_host_ports() -> None:
    """G-03: every published host port must come from an env var (with a
    default matching today's value), not a bare literal -- otherwise
    env.template's port vars are decorative and a second stack on the same
    host collides on first `up`."""
    services = _services()
    bad: list[str] = []
    for name, spec in services.items():
        for entry in spec.get('ports') or []:
            if isinstance(entry, dict):
                published = entry.get('published')
                if published is not None and '${' not in str(published):
                    bad.append(f'{name}: {entry!r}')
                continue
            text = str(entry)
            host_part = text.split(':', maxsplit=1)[0] if ':' in text else text
            if host_part and '${' not in host_part:
                bad.append(f'{name}: {text!r}')
    assert not bad, 'published host ports not driven by an env var:\n' + '\n'.join(bad)


def test_every_published_port_uses_op_bind_address() -> None:
    """installer plan §7: the API has no auth and OpenSearch security is
    disabled, so every published port must default to loopback-only via
    ${OP_BIND_ADDRESS:-127.0.0.1} -- a bare ${SOME_PORT:-1234}:container
    entry would silently publish on 0.0.0.0."""
    services = _services()
    bad: list[str] = []
    for name, spec in services.items():
        for entry in spec.get('ports') or []:
            if isinstance(entry, dict):
                continue
            text = str(entry)
            # Cropwright has its own knob so the installer can open only the web UI to a LAN.
            if not text.startswith(
                ('${OP_BIND_ADDRESS', '${OP_UI_BIND_ADDRESS', '${CROPWRIGHT_BIND_ADDRESS')
            ):
                bad.append(f'{name}: {text!r}')
    assert not bad, 'published ports missing ${OP_BIND_ADDRESS prefix:\n' + '\n'.join(bad)


def test_api_carries_op_api_network_alias() -> None:
    """G-28: Cropwright's default API_UPSTREAM is http://op-api:8000; this
    alias lets that default resolve without every deployer overriding it."""
    networks = _services()['api'].get('networks')
    assert isinstance(networks, dict), 'expected api networks: to carry aliases (dict form)'
    aliases = (networks.get('triton_net') or {}).get('aliases') or []
    assert 'op-api' in aliases, f'expected op-api in api triton_net aliases: {aliases}'


# =============================================================================
# a container must never read a *host*-port env var (the left side of
# a `ports:` mapping, e.g. SEGMENTER_PORT) as its own in-container config.
# `env_file: .env` loads the whole file into the container, so a same-named
# Dockerfile ENV default silently gets overridden by the host-port value --
# the segmenter bound uvicorn to the host port number instead of 8000.
# =============================================================================

_HOST_PORT_LEADING_RE = re.compile(r'^(\$\{[A-Z0-9_]+(?::-[^}]*)?\}|[0-9]+)')
_HOST_PORT_VAR_RE = re.compile(r'\$\{([A-Z0-9_]+)(?::-[^}]*)?\}')
_DOCKERFILE_ENV_VAR_RE = re.compile(r'^\s*([A-Z0-9_]+)=', re.MULTILINE)

# installer plan §7: every published port now also carries a leading
# ${OP_BIND_ADDRESS:-127.0.0.1} publish-address segment (`ADDR:HOST:CONTAINER`
# short syntax), which is never the host-port-var-reused-as-container-ENV
# case this check guards against -- exclude it explicitly rather than
# reworking the leading-anchor match for a 3-segment string.
_NON_HOST_PORT_VARS = frozenset({'OP_BIND_ADDRESS', 'OP_UI_BIND_ADDRESS'})


def _host_port_vars(spec: dict[str, Any]) -> set[str]:
    """Extract the env var name(s) backing the *host*-side of each
    published port. Short syntax is ``[ADDR:]HOST:CONTAINER[/proto]``,
    where ADDR/HOST are themselves ``${VAR:-default}`` -- naive
    ``str.split(':')`` breaks because that colon appears inside the
    ``${VAR:-default}`` expression too. The container-side port in this
    repo's compose file is always a literal integer, never a var, so
    every ``${VAR}`` reference found in the whole entry is host-side.
    """
    names: set[str] = set()
    for entry in spec.get('ports') or []:
        if isinstance(entry, dict):
            published = str(entry.get('published') or '')
        else:
            published = str(entry)
        names.update(_HOST_PORT_VAR_RE.findall(published))
    return names - _NON_HOST_PORT_VARS


def _dockerfile_env_var_names(dockerfile_path: Path) -> set[str]:
    if not dockerfile_path.is_file():
        return set()
    names: set[str] = set()
    for block in re.findall(
        r'^ENV\s+(.+?)(?=^\S|\Z)',
        dockerfile_path.read_text(encoding='utf-8'),
        re.MULTILINE | re.DOTALL,
    ):
        names.update(_DOCKERFILE_ENV_VAR_RE.findall(block))
    return names


def test_no_service_reads_a_host_port_var_as_its_own_container_config() -> None:
    """Regression guard (segmenter bound to the host port because
    SEGMENTER_PORT named both the host mapping and the Dockerfile's default
    listen-port ENV).

    Installer plan §1: every `build:` block moved from docker-compose.yml to
    docker-compose.dev.yml (the deploy-safe base carries no `build:` at
    all), so the Dockerfile lookup below reads from the dev overlay by
    service name instead of the base spec.
    """
    services = _services()
    dev_services = _load_yaml(REPO_ROOT / 'docker-compose.dev.yml')['services']
    collisions: list[str] = []
    for name, spec in services.items():
        host_port_vars = _host_port_vars(spec)
        if not host_port_vars:
            continue
        build = dev_services.get(name, {}).get('build') or {}
        context = Path(str(build.get('context', '.')))
        dockerfile = build.get('dockerfile')
        if not dockerfile:
            continue
        dockerfile_path = REPO_ROOT / context / dockerfile
        env_names = _dockerfile_env_var_names(dockerfile_path)
        shared = host_port_vars & env_names
        if shared:
            collisions.append(f'{name}: {sorted(shared)} (host-port var reused as container ENV)')
    assert not collisions, (
        'host-port env var name(s) reused as in-container ENV default(s):\n' + '\n'.join(collisions)
    )
