"""The arbiter's container control: Compose labels, the project, and what fails closed.

docker-py: ``containers.list(all=True, filters={'label': [...]})``, ``Container.stop`` and
``Container.start`` (https://docker-py.readthedocs.io/en/stable/containers.html).
"""

from __future__ import annotations

from typing import Any

import docker.errors
import pytest

from src.services.periodic import gpu_arbiter_docker as control


class Container:
    def __init__(self, status: str, labels: dict[str, str] | None = None) -> None:
        self.status = status
        self.labels = labels or {}
        self.calls: list[str] = []

    def reload(self) -> None:
        self.calls.append('reload')

    def stop(self, timeout: int) -> None:
        self.calls.append(f'stop({timeout})')
        self.status = 'exited'

    def start(self) -> None:
        self.calls.append('start')
        self.status = 'running'


class Client:
    def __init__(
        self, by_service: dict[str, list[Container]], own: Container | None = None
    ) -> None:
        self.by_service = by_service
        self.own = own
        self.queries: list[dict[str, Any]] = []
        self.containers = self

    def list(self, **kwargs: Any) -> list[Container]:
        self.queries.append(kwargs)
        service = next(
            item.split('=', 1)[1]
            for item in kwargs['filters']['label']
            if item.startswith(control.SERVICE_LABEL)
        )
        return self.by_service.get(service, [])

    def get(self, name: str) -> Container:
        if self.own is None:
            raise docker.errors.NotFound(f'no container {name}')
        return self.own


def test_a_service_is_found_by_its_compose_labels_never_by_a_container_name() -> None:
    vlm = Container('running')
    client = Client({'vlm': [vlm]})

    assert control.stop_services(client, 'op', ('vlm',)) == ['vlm']

    [query] = client.queries
    assert query['all'] is True
    assert query['filters'] == {
        'label': [f'{control.PROJECT_LABEL}=op', f'{control.SERVICE_LABEL}=vlm']
    }
    assert vlm.calls == ['reload', f'stop({control.STOP_TIMEOUT_S})']


def test_every_container_of_a_scaled_service_is_stopped_and_one_stop_is_reported() -> None:
    a, b, down = Container('running'), Container('running'), Container('exited')
    client = Client({'vlm': [a, b, down]})

    assert control.stop_services(client, 'op', ('vlm',)) == ['vlm']

    assert (a.status, b.status) == ('exited', 'exited')
    assert 'stop(30)' not in down.calls


def test_a_service_with_nothing_running_is_not_reported_stopped() -> None:
    client = Client({'vlm': [Container('exited')], 'ghost': []})

    assert control.stop_services(client, 'op', ('vlm', 'ghost')) == []


def test_start_only_starts_what_is_not_running() -> None:
    down, up = Container('exited'), Container('running')
    client = Client({'vlm': [down], 'triton': [up]})

    assert control.start_services(client, 'op', ('vlm', 'triton')) == ['vlm']

    assert down.status == 'running'
    assert 'start' not in up.calls


def test_the_project_comes_from_the_environment_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('COMPOSE_PROJECT_NAME', 'from-env')
    assert control.compose_project(Client({}, own=Container('running', {}))) == 'from-env'


def test_the_project_falls_back_to_the_workers_own_compose_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('COMPOSE_PROJECT_NAME', raising=False)
    own = Container('running', {control.PROJECT_LABEL: 'stack-a'})
    assert control.compose_project(Client({}, own=own)) == 'stack-a'


def test_an_unknown_project_is_an_error_never_a_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('COMPOSE_PROJECT_NAME', raising=False)
    with pytest.raises(control.ArbiterDockerError, match='no Compose project label'):
        control.compose_project(Client({}, own=Container('running', {})))
    with pytest.raises(control.ArbiterDockerError, match='cannot read this container'):
        control.compose_project(Client({}, own=None))


def test_an_unreachable_daemon_is_an_arbiter_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> None:
        raise docker.errors.DockerException('Error while fetching server API version')

    monkeypatch.setattr(docker, 'from_env', refuse)
    with pytest.raises(control.ArbiterDockerError, match='not reachable'):
        control.docker_client()


def test_a_daemon_that_does_not_answer_ping_is_an_arbiter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Dead:
        def ping(self) -> None:
            raise ConnectionError('socket closed')

    monkeypatch.setattr(docker, 'from_env', Dead)
    with pytest.raises(control.ArbiterDockerError, match='socket closed'):
        control.docker_client()
