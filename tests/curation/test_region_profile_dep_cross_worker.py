"""A region route must not answer 409 no_active_profile from a stale worker snapshot.

Two config stores stand for two uvicorn workers over one database: an activation made through
worker A bumps the epoch, and worker B (whose poll has not ticked yet) must still see it when
the very next request lands on it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi import HTTPException

from src.routers.curation._common import _require_region_profile_dep
from src.services.config_store import store as store_mod
from src.services.config_store.snapshot import _EMPTY_SNAPSHOT, ConfigSnapshot, StoredConfig
from src.services.config_store.store import ConfigStore
from src.services.detection import profile_registry


pytestmark = pytest.mark.asyncio


class _Db:
    """The shared database: an epoch and the activated profile."""

    def __init__(self) -> None:
        self.epoch = 1
        self.active: StoredConfig | None = None
        self.down = False

    def snapshot(self) -> ConfigSnapshot:
        if self.active is None:
            return replace(_EMPTY_SNAPSHOT, config_revision=self.epoch, loaded_at=1.0)
        ref = (self.active.name, self.active.revision)
        return replace(
            _EMPTY_SNAPSHOT,
            config_revision=self.epoch,
            loaded_at=1.0,
            profiles={self.active.name: self.active},
            active_profile=ref,
            active_profile_body=self.active,
        )

    def activate(self) -> None:
        self.epoch += 1
        self.active = StoredConfig(
            kind='detection_profile',
            name='objects',
            revision=1,
            body={'region_class_name': 'object', 'detector_model': 'm'},
        )


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> _Db:
    shared = _Db()

    async def epoch(_store: ConfigStore) -> int:
        if shared.down:
            raise OSError('database down')
        return shared.epoch

    async def load(_store: ConfigStore) -> ConfigSnapshot:
        return shared.snapshot()

    monkeypatch.setattr(ConfigStore, '_epoch', epoch)
    monkeypatch.setattr(ConfigStore, '_load', load)
    profile_registry._reset_registry_for_tests()
    return shared


def _serve_from(monkeypatch: pytest.MonkeyPatch, store: ConfigStore) -> None:
    monkeypatch.setattr(store_mod, 'get_config_store', lambda **_: store)
    monkeypatch.setattr('src.services.config_store.get_config_store', lambda **_: store)


async def test_activation_through_another_worker_is_seen_without_waiting(
    db: _Db, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_a, worker_b = ConfigStore(label='p'), ConfigStore(label='p')
    await worker_a.refresh()
    await worker_b.refresh()
    db.activate()
    await worker_a.refresh()  # the activate route ran on worker A
    _serve_from(monkeypatch, worker_b)  # the next request lands on worker B, not yet polled

    profile = await _require_region_profile_dep()

    assert profile.name == 'objects'


async def test_still_refuses_when_the_store_has_no_active_profile(
    db: _Db, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = ConfigStore(label='p')
    await worker.refresh()
    _serve_from(monkeypatch, worker)

    with pytest.raises(HTTPException) as err:
        await _require_region_profile_dep()

    assert err.value.status_code == 409
    assert isinstance(err.value.detail, dict)
    assert err.value.detail['error'] == 'no_active_profile'


async def test_an_unreadable_store_is_503_not_a_guess(
    db: _Db, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = ConfigStore(label='p')
    await worker.refresh()
    _serve_from(monkeypatch, worker)
    db.down = True

    with pytest.raises(HTTPException) as err:
        await _require_region_profile_dep()

    assert err.value.status_code == 503


async def test_assist_segment_check_sees_an_activation_made_on_another_worker(
    db: _Db, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from src.services.curation.assist.prepare import _check_segment
    from src.services.rejections import RequestRejectedError

    worker_a, worker_b = ConfigStore(label='p'), ConfigStore(label='p')
    await worker_a.refresh()
    await worker_b.refresh()
    db.activate()  # the profile has no segmenter prompt, so the next refusal is a different one
    await worker_a.refresh()
    _serve_from(monkeypatch, worker_b)
    request = SimpleNamespace(actions=['segment'], segment=object())

    with pytest.raises(RequestRejectedError) as err:
        await _check_segment(request, object())  # type: ignore[arg-type]

    assert 'no_segmenter_leg' in str(err.value.detail)
