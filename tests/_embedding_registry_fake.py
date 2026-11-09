"""The embedding registry the offline tier reads: the three default spaces of a fresh project.

``resolve._load`` is PostgreSQL; offline tests get this stand-in instead (a ``db`` test runs the
real one). A test that needs another space calls ``add``; ``reset`` returns to the defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.clients.curation_opensearch.base import default_knn_spec
from src.domain.embedding_spaces import (
    ResolvedSpace,
    Target,
    legacy_names,
    physical_names,
    space_key,
)
from src.services.embedding_spaces import resolve


if TYPE_CHECKING:
    import pytest

DEFAULT_KEY = 'pe_core_l14_336_v1'
DIM = 4


def default_space(target: Target, key: str = DEFAULT_KEY, dim: int = DIM) -> ResolvedSpace:
    names = legacy_names(target)
    return ResolvedSpace(
        space_key=key,
        target=target,
        vector_field=names.vector_field,
        state_field=names.state_field,
        nested_path=names.nested_path,
        dim=dim,
        index_role='images' if target is Target.FRAME else 'items',
        model_key=key.rsplit('_v', 1)[0],
        model_version=1,
        triton_image_model='pe_image_encoder',
        triton_text_model='pe_text_encoder',
        knn_method=default_knn_spec(),
    )


def new_space(
    target: Target, model_key: str, *, text_model: str | None = None, dim: int = DIM
) -> ResolvedSpace:
    """A space created after the 0.6.0 layout: the registry names its fields (``vec_<key>``)."""
    key = space_key(model_key, 1)
    names = physical_names(key, target)
    return ResolvedSpace(
        space_key=key,
        target=target,
        vector_field=names.vector_field,
        state_field=names.state_field,
        nested_path=names.nested_path,
        dim=dim,
        index_role='images' if target is Target.FRAME else 'items',
        model_key=model_key,
        model_version=1,
        triton_image_model=f'{model_key}_image',
        triton_text_model=text_model,
        knn_method=default_knn_spec(),
    )


class EmbeddingRegistryFake:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.spaces: list[ResolvedSpace] = [
            default_space(Target.ITEM),
            default_space(Target.BOX),
            default_space(Target.FRAME),
        ]
        self.primary: dict[Target, str] = {
            Target.ITEM: DEFAULT_KEY,
            Target.BOX: DEFAULT_KEY,
            Target.FRAME: DEFAULT_KEY,
        }

    def add(self, space: ResolvedSpace, *, primary: bool = False) -> ResolvedSpace:
        self.spaces.append(space)
        if primary:
            self.primary[space.target] = space.space_key
        resolve.clear_cache()
        return space

    def entry(self) -> resolve._Entry:
        import uuid

        return resolve._Entry(
            uuid.uuid4(),
            0,
            tuple(self.spaces),
            tuple((t, k) for t, k in self.primary.items()),
        )

    def install(self, monkeypatch: pytest.MonkeyPatch) -> EmbeddingRegistryFake:
        async def load(_slug: str) -> resolve._Entry:
            return self.entry()

        async def epoch(_slug: str) -> int:
            return 0

        resolve.clear_cache()
        monkeypatch.setattr(resolve, '_load', load)
        monkeypatch.setattr(resolve, '_current_epoch', epoch)
        return self
