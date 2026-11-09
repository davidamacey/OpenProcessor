"""The arbiter's container control: the Docker SDK over the mounted socket, by Compose labels.

Only ``gpu-arbiter-worker`` mounts the socket, and only the ``op.periodic.gpu_arbiter`` tick
calls this module. A service is found by the labels Compose puts on every container it creates,
``com.docker.compose.project`` and ``com.docker.compose.service``, never by container name (no
service sets ``container_name``, and a scaled service has several containers). The Docker SDK
reference is https://docker-py.readthedocs.io/en/stable/containers.html ``containers.list``
(``filters={'label': [...]}``), ``Container.stop`` and ``Container.start``.

The functions are blocking; the tick runs them with ``asyncio.to_thread``. A call that cannot be
made raises :class:`ArbiterDockerError`: the tick records it and acknowledges nothing, so a run
never starts next to a service it could not stop.
"""

from __future__ import annotations

import socket
from typing import Any

import docker.errors
import requests

from src.config.env import env_str


PROJECT_LABEL = 'com.docker.compose.project'
SERVICE_LABEL = 'com.docker.compose.service'
STOP_TIMEOUT_S = 30
# What the SDK raises for a daemon that is unreachable or refuses a call: its own errors, and
# the ``requests`` ones it lets through from a dead socket.
DOCKER_ERRORS = (docker.errors.DockerException, requests.exceptions.RequestException, OSError)


class ArbiterDockerError(RuntimeError):
    """The Docker daemon cannot be reached or a stop or start failed."""


def docker_client() -> Any:
    """A client over the mounted socket (``DOCKER_HOST`` or ``/var/run/docker.sock``)."""
    try:
        client = docker.from_env()  # type: ignore[attr-defined]
        client.ping()
    except DOCKER_ERRORS as exc:
        raise ArbiterDockerError(f'the Docker daemon is not reachable: {exc}') from exc
    return client


def compose_project(client: Any) -> str:
    """The Compose project this stack runs under.

    ``COMPOSE_PROJECT_NAME`` when the worker's environment carries it; otherwise the label of the
    worker's own container, found by its hostname (the default container id). Neither known is an
    error: guessing a project could stop another stack's containers.
    """
    named = env_str('COMPOSE_PROJECT_NAME')
    if named:
        return named
    try:
        own = client.containers.get(socket.gethostname())
        project = (own.labels or {}).get(PROJECT_LABEL)
    except Exception as exc:
        raise ArbiterDockerError(f"cannot read this container's Compose labels: {exc}") from exc
    if not project:
        raise ArbiterDockerError(
            'this container has no Compose project label; set COMPOSE_PROJECT_NAME'
        )
    return str(project)


def _containers(client: Any, project: str, service: str) -> list[Any]:
    return list(
        client.containers.list(
            all=True,
            filters={'label': [f'{PROJECT_LABEL}={project}', f'{SERVICE_LABEL}={service}']},
        )
    )


def stop_services(client: Any, project: str, services: tuple[str, ...]) -> list[str]:
    """Stop every running container of each service; returns the services that had one running."""
    stopped: list[str] = []
    for service in services:
        hit = False
        for container in _containers(client, project, service):
            container.reload()
            if container.status == 'running':
                container.stop(timeout=STOP_TIMEOUT_S)
                hit = True
        if hit:
            stopped.append(service)
    return stopped


def start_services(client: Any, project: str, services: tuple[str, ...]) -> list[str]:
    """Start every container of each service that is not running; returns the services started."""
    started: list[str] = []
    for service in services:
        hit = False
        for container in _containers(client, project, service):
            container.reload()
            if container.status != 'running':
                container.start()
                hit = True
        if hit:
            started.append(service)
    return started
