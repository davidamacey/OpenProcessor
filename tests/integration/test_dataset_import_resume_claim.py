"""``POST /datasets/imports/{id}/resume`` takes the project start lock before
it rescans, like a start and an undo do: a second resume, or an undo, that
lands while the first resume is rescanning is refused, and a resume that
fails before its worker runs hands the job back as it found it."""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

from integration.test_dataset_import_routes import BASE, _body, _dataset, _wait, client, fake_os
from src.services.curation.dataset_import import runner


if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration
__all__ = ['client', 'fake_os']  # fixtures from the routes module


def _interrupted_import(client: TestClient, tmp_path: Path) -> str:
    started = client.post(f'{BASE}/imports', json=_body(_dataset(tmp_path)))
    import_id = started.json()['import_id']
    _wait(client, import_id, until={'completed'})
    state = next((tmp_path / 'imports').rglob(f'{import_id}/state.json'))
    doc = json.loads(state.read_text())
    doc['status'] = 'interrupted'
    state.write_text(json.dumps(doc))
    return import_id


def _hold_the_rescan(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, threading.Event]:
    """Make the rescan block until released; returns ``(entered, release)``."""
    entered, release = threading.Event(), threading.Event()
    real = runner.rescan_for_resume

    def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, 'rescan_for_resume', held)
    return entered, release


def _resume_in_background(client: TestClient, import_id: str) -> tuple[threading.Thread, list[Any]]:
    out: list[Any] = []

    def post() -> None:
        out.append(client.post(f'{BASE}/imports/{import_id}/resume'))

    thread = threading.Thread(target=post)
    thread.start()
    return thread, out


def _error(resp: Any) -> str:
    return resp.json()['detail']['error']


def test_a_second_resume_during_the_rescan_is_refused_and_one_worker_runs(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_id = _interrupted_import(client, tmp_path)
    spawned: list[str] = []
    real_spawn = runner.spawn

    def counting_spawn(ctx: Any, store: Any, entries: Any) -> None:
        spawned.append(store.import_id)
        real_spawn(ctx, store, entries)

    monkeypatch.setattr(runner, 'spawn', counting_spawn)
    entered, release = _hold_the_rescan(monkeypatch)
    thread, first = _resume_in_background(client, import_id)
    assert entered.wait(10)

    second = client.post(f'{BASE}/imports/{import_id}/resume')
    assert (second.status_code, _error(second)) == (409, 'import_not_resumable')

    release.set()
    thread.join(10)
    assert first[0].status_code == 202
    _wait(client, import_id, until={'completed'})
    assert spawned == [import_id]


def test_an_undo_during_the_rescan_is_refused(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_id = _interrupted_import(client, tmp_path)
    entered, release = _hold_the_rescan(monkeypatch)
    thread, first = _resume_in_background(client, import_id)
    assert entered.wait(10)

    undo = client.post(f'{BASE}/imports/{import_id}/undo', json={'dry_run': False})
    assert (undo.status_code, _error(undo)) == (409, 'import_not_undoable')

    release.set()
    thread.join(10)
    assert first[0].status_code == 202
    assert _wait(client, import_id, until={'completed'})['status'] == 'completed'


def test_a_resume_during_a_claimed_undo_is_refused(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_id = _interrupted_import(client, tmp_path)
    monkeypatch.setattr(runner, 'spawn_undo', lambda *_args, **_kwargs: None)
    undo = client.post(f'{BASE}/imports/{import_id}/undo', json={'dry_run': False})
    assert undo.status_code == 202
    assert client.get(f'{BASE}/imports/{import_id}').json()['status'] == 'undoing'

    resume = client.post(f'{BASE}/imports/{import_id}/resume')
    assert (resume.status_code, _error(resume)) == (409, 'import_not_resumable')
    assert client.get(f'{BASE}/imports/{import_id}').json()['status'] == 'undoing'


def test_a_resume_the_dataset_change_refuses_leaves_the_import_resumable_as_it_was(
    client: TestClient, tmp_path: Path
) -> None:
    import_id = _interrupted_import(client, tmp_path)
    (tmp_path / 'ds' / 'images' / 'train' / 'a.jpg').write_bytes(b'not the same bytes')

    resume = client.post(f'{BASE}/imports/{import_id}/resume')
    assert (resume.status_code, _error(resume)) == (409, 'dataset_changed')
    assert client.get(f'{BASE}/imports/{import_id}').json()['status'] == 'interrupted'


def test_a_resume_that_crashes_before_its_worker_runs_leaves_the_import_as_it_was(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_id = _interrupted_import(client, tmp_path)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError('registry unavailable')

    monkeypatch.setattr(runner, 'prepare_resume', boom)
    with pytest.raises(RuntimeError, match='registry unavailable'):
        client.post(f'{BASE}/imports/{import_id}/resume')
    assert client.get(f'{BASE}/imports/{import_id}').json()['status'] == 'interrupted'


def test_the_claim_stays_live_through_a_rescan_longer_than_the_stale_window(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import file_job

    import_id = _interrupted_import(client, tmp_path)
    monkeypatch.setattr(file_job, 'HEARTBEAT_STALE_S', 0.4)
    monkeypatch.setattr(file_job, 'HEARTBEAT_TICK_S', 0.05)
    entered, release = _hold_the_rescan(monkeypatch)
    thread, first = _resume_in_background(client, import_id)
    assert entered.wait(10)

    time.sleep(1.2)  # three stale windows with the rescan still running
    assert client.get(f'{BASE}/imports/{import_id}').json()['status'] == 'queued'

    release.set()
    thread.join(10)
    assert first[0].status_code == 202


def test_the_served_actions_match_what_the_routes_accept(
    client: TestClient, tmp_path: Path
) -> None:
    import_id = _interrupted_import(client, tmp_path)

    actions = client.get(f'{BASE}/imports/{import_id}').json()['actions']

    assert actions['can_resume'] == {'allowed': True, 'reason': None}
    assert actions['can_undo']['allowed'] is True
    assert actions['can_cancel']['allowed'] is False
    assert actions['can_cancel']['reason']


def test_a_completed_import_cannot_resume_and_says_why(client: TestClient, tmp_path: Path) -> None:
    started = client.post(f'{BASE}/imports', json=_body(_dataset(tmp_path)))
    import_id = started.json()['import_id']
    _wait(client, import_id, until={'completed'})

    actions = client.get(f'{BASE}/imports/{import_id}').json()['actions']

    assert actions['can_resume']['allowed'] is False
    assert 'interrupted' in actions['can_resume']['reason']
    assert client.post(f'{BASE}/imports/{import_id}/resume').status_code == 409
