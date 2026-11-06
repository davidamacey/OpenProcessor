"""An in-memory stand-in for ``src.services.training.model_registry`` (the offline tier has no
database; the ``db`` tier runs the real module against PostgreSQL).

Seed it with ``promote(...)``. It keeps the same rules as the repository: one live promotion per
Triton name, a share is a consumer slug, the sharing revision is ``1 + consumers``.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import TYPE_CHECKING

from src.db.repos import models as repo
from src.services.training import model_registry


if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

_NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


class FakeModelRegistry:
    def __init__(self) -> None:
        self.live: dict[str, repo.LiveModel] = {}
        self.shares: dict[str, set[str]] = {}
        self.retired: list[str] = []

    def promote(
        self,
        triton_name: str,
        *,
        owner: str = 'default',
        labels: Sequence[str] = (),
        shared_with: Collection[str] = (),
        run_job_id: str | None = None,
        version: int = 1,
    ) -> None:
        model = repo.ModelRow(
            id=uuid.uuid4(),
            project_id=uuid.uuid4(),
            name=triton_name,
            version=version,
            origin='trained',
            training_run_id=None,
            family='yolo26',
            task='detect',
            labels=tuple(labels),
            artifact_storage_key=f'models/{triton_name}/{version}/model.onnx',
            created_at=_NOW,
            created_by='test',
        )
        promotion = repo.PromotionRow(
            id=uuid.uuid4(),
            project_id=model.project_id or uuid.uuid4(),
            model_id=model.id,
            triton_model_name=triton_name,
            triton_version=version,
            class_remap=None,
            job_id=None,
            promoted_at=_NOW,
            promoted_by='test',
            superseded_at=None,
        )
        self.live[triton_name] = repo.LiveModel(promotion, model, owner, run_job_id)
        self.shares[triton_name] = set(shared_with)

    async def live_model(self, triton_name: str) -> repo.LiveModel | None:
        return self.live.get(triton_name)

    async def owner_slug(self, triton_name: str) -> str | None:
        found = self.live.get(triton_name)
        return None if found is None else found.owner_slug

    async def owned_live(self, slug: str) -> list[repo.LiveModel]:
        return [m for _, m in sorted(self.live.items()) if m.owner_slug == slug]

    async def owned_names(self, slug: str) -> list[str]:
        return [m.promotion.triton_model_name for m in await self.owned_live(slug)]

    async def all_live(self) -> list[repo.LiveModel]:
        return [m for _, m in sorted(self.live.items())]

    async def shared_with(self, slug: str) -> list[repo.LiveModel]:
        return [m for name, m in sorted(self.live.items()) if slug in self.shares.get(name, ())]

    async def consumers(self, triton_name: str) -> list[str]:
        return sorted(self.shares.get(triton_name, ()))

    async def retire(self, triton_name: str) -> bool:
        self.retired.append(triton_name)
        self.shares.pop(triton_name, None)
        return self.live.pop(triton_name, None) is not None

    async def set_sharing(
        self,
        _owner_project_id: uuid.UUID,
        triton_name: str,
        *,
        shared: bool,
        expected_revision: int,
        consumers: Collection[uuid.UUID],
        requested_by: str,  # noqa: ARG002 - the real signature
    ) -> int:
        if triton_name not in self.live:
            raise model_registry.ModelNotOwnedError(triton_name)
        current = repo.sharing_revision(len(self.shares[triton_name]))
        if expected_revision != current:
            raise model_registry.SharingRevisionConflictError(current)
        self.shares[triton_name] = {str(c) for c in consumers} if shared else set()
        return repo.sharing_revision(len(self.shares[triton_name]))


CURRENT: FakeModelRegistry | None = None


def install(monkeypatch: object) -> FakeModelRegistry:
    """Replace the module's functions with the fake's; ``monkeypatch`` is pytest's fixture."""
    global CURRENT  # noqa: PLW0603 - the helpers of a test directory seed the installed fake
    fake = CURRENT = FakeModelRegistry()
    for name in (
        'live_model',
        'owner_slug',
        'owned_live',
        'owned_names',
        'all_live',
        'shared_with',
        'consumers',
        'retire',
        'set_sharing',
    ):
        monkeypatch.setattr(model_registry, name, getattr(fake, name))  # type: ignore[attr-defined]
    return fake
