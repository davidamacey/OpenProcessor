"""The re-index encoder: one decode per image, fail closed on a model it cannot serve."""

from __future__ import annotations

import datetime as dt
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.db.repos import embedding_spaces as repo
from src.domain.embedding_spaces import EmbeddingSpaceError, KnnMethodSpec, Role, Status, Target
from src.services.embedding_spaces import reindex_encoder as mod
from src.services.embedding_spaces.reindex import DocRef


pytestmark = pytest.mark.asyncio


def _space(target: Target = Target.ITEM) -> repo.SpaceRow:
    return repo.SpaceRow(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        space_key='k',
        model_version_id=uuid.uuid4(),
        target=target,
        dim=2,
        role=Role.CANDIDATE,
        status=Status.BUILDING,
        index_role='items',
        vector_field='vec_k',
        state_field='vec_k_state',
        nested_path=None,
        knn_method=KnnMethodSpec(),
        revision=1,
        built_by_job_id=None,
        ready_at=None,
        retired_at=None,
        created_at=dt.datetime.now(dt.UTC),
        created_by='t',
    )


class FakeOS:
    async def mget(self, **kw: Any) -> dict[str, Any]:
        return {
            'docs': [
                {'_id': i, 'found': True, '_source': {'image_id': 'img1'}}
                for i in kw['body']['ids']
            ]
        }


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {'decodes': [], 'model': mod.PE_IMAGE_MODEL}

    @asynccontextmanager
    async def fake_session():
        yield object()

    async def fake_version(_s: Any, _id: Any) -> Any:
        return SimpleNamespace(triton_image_model=state['model'])

    async def fake_images(_os: Any, ids: list[str], *, index: str) -> dict[str, Any]:
        return {i: {'image_path': f'/p/{i}.jpg'} for i in ids}

    async def fake_vectors(_pe, _gpu, path, crops, regions, *, frame):
        state['decodes'].append(path)
        return SimpleNamespace(
            frame=np.array([1.0, 0.0]) if frame else None,
            crops=[np.array([float(i), 1.0]) for i, _ in enumerate(crops)],
            regions=[np.array([float(i), 2.0]) for i, _ in enumerate(regions)],
        )

    class _Factory:
        def __call__(self) -> Any:
            return fake_session()

    monkeypatch.setattr(mod, 'async_session_factory', _Factory)
    monkeypatch.setattr(mod, 'project_tx', lambda _s, _p: fake_session())
    monkeypatch.setattr(mod.repo, 'get_model_version_by_id', fake_version)
    monkeypatch.setattr(mod, 'existing_images', fake_images)
    monkeypatch.setattr(mod, '_target_vectors', fake_vectors)
    monkeypatch.setattr(
        mod, 'get_curation_config', lambda: SimpleNamespace(items_index='it', images_index='im')
    )
    return state


async def test_two_items_of_one_image_decode_once(wired: dict[str, Any]) -> None:
    enc = mod.ReindexEncoder(FakeOS(), pe=object())
    got = await enc.encode(
        _space(), [DocRef('a', (0.0, 0.0, 0.5, 0.5)), DocRef('b', (0.5, 0.5, 1.0, 1.0))]
    )
    assert wired['decodes'] == ['/p/img1.jpg']
    assert set(got) == {'a', 'b'}
    assert got['b'] == [1.0, 1.0]


async def test_box_documents_use_the_region_slots(wired: dict[str, Any]) -> None:
    enc = mod.ReindexEncoder(FakeOS(), pe=object())
    got = await enc.encode(_space(Target.BOX), [DocRef('a#b1', (0.0, 0.0, 0.5, 0.5))])
    assert got == {'a#b1': [0.0, 2.0]}


async def test_frame_document_is_its_own_image(wired: dict[str, Any]) -> None:
    got = await mod.ReindexEncoder(FakeOS(), pe=object()).encode(
        _space(Target.FRAME), [DocRef('img1')]
    )
    assert got == {'img1': [1.0, 0.0]}


async def test_another_model_is_refused(wired: dict[str, Any]) -> None:
    wired['model'] = 'some_other_encoder'
    with pytest.raises(EmbeddingSpaceError) as exc:
        await mod.ReindexEncoder(FakeOS(), pe=object()).encode(
            _space(), [DocRef('a', (0, 0, 1, 1))]
        )
    assert exc.value.code == 'embedding_model_unavailable'


async def test_a_document_without_an_image_is_absent(wired: dict[str, Any]) -> None:
    class NoImage(FakeOS):
        async def mget(self, **_kw: Any) -> dict[str, Any]:
            return {'docs': [{'_id': 'a', 'found': False}]}

    got = await mod.ReindexEncoder(NoImage(), pe=object()).encode(
        _space(), [DocRef('a', (0, 0, 1, 1))]
    )
    assert got == {}
