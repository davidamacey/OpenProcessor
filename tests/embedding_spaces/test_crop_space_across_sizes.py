"""A project created under one ``PE_CROP_SIZE`` and served under the other is refused, against a
real PostgreSQL registry.

The project is created the way a real create does (``seed_default_spaces`` runs the product's
``default_space_rows`` with ``crop_encoder()`` of the size in force), then the deployment's size
is switched, which is all a real redeploy with the other ``PE_CROP_SIZE`` changes. Nothing of the
registry is faked: the offline twins (``test_encoder_guard_size.py``) register the spaces by hand.
The request is served through the app's own handlers (``install_project_exception_handlers``)."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from src.clients import pe_encoder
from src.domain.embedding_spaces import EmbeddingSpaceError, Target
from src.routers.curation._project_deps import install_project_exception_handlers
from src.services.embedding_spaces import resolve
from src.services.embedding_spaces.encoder_guard import encoder_mismatch_of, require_encoder_for


pytestmark = [pytest.mark.db, pytest.mark.asyncio]

CROSSED = [(224, 336), (336, 224)]
MATCHED = [(224, 224), (336, 336)]


@pytest.fixture
def created_size(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> int:
    """The deployment's size while the project is created (must run before ``world``)."""
    size: int = request.param[0]
    monkeypatch.setattr(pe_encoder, 'PE_CROP_SIZE', size)
    return size


def _served(monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    monkeypatch.setattr(pe_encoder, 'PE_CROP_SIZE', size)
    resolve.clear_cache()


def _app() -> FastAPI:
    app = FastAPI()
    install_project_exception_handlers(app)

    @app.post('/write')
    async def write() -> dict[str, str]:
        await require_encoder_for(Target.ITEM, Target.BOX)
        return {'status': 'written'}

    return app


@pytest.mark.parametrize('created_size', CROSSED, indirect=True, ids=lambda p: f'{p[0]}to{p[1]}')
async def test_a_project_of_one_size_refuses_a_deployment_of_the_other(
    created_size: int, world, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = next(b for a, b in CROSSED if a == created_size)
    _served(monkeypatch, served)

    found = await encoder_mismatch_of(_slug())
    assert found is not None, f'no mismatch served for a {created_size} project under {served}'
    assert sorted(s.target for s in found.spaces) == ['box', 'item'], found
    for part in (f'{created_size}x{created_size}', f'{served}x{served}', 'new project'):
        assert part in found.message, (part, found.message)

    for target in (Target.ITEM, Target.BOX):
        with pytest.raises(EmbeddingSpaceError) as err:
            await require_encoder_for(target)
        assert err.value.code == 'embedding_model_unavailable', err.value.detail
        assert err.value.status == 409
        assert err.value.detail == next(s.message for s in found.spaces if s.target == target.value)

    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url='http://t') as http:
        resp = await http.post('/write')
    assert resp.status_code == 409, (
        f'a refused write must answer 409: {resp.status_code} {resp.text}'
    )
    detail = resp.json()['detail']
    assert detail['error'] == 'embedding_model_unavailable', resp.text
    assert 'new project' in detail['message'], resp.text


@pytest.mark.parametrize('created_size', MATCHED, indirect=True, ids=lambda p: f'{p[0]}to{p[1]}')
async def test_a_project_of_the_deployments_size_is_not_refused(
    created_size: int, world, monkeypatch: pytest.MonkeyPatch
) -> None:
    _served(monkeypatch, created_size)
    assert await encoder_mismatch_of(_slug()) is None
    await require_encoder_for(Target.ITEM, Target.BOX)


def _slug() -> str:
    from src.config.project_context import current_project

    return current_project().record.slug
