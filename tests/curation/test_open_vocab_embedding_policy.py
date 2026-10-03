"""Items the full-image SAM 3 pass writes follow the project's embedding policy,
can be selected by origin, and are embedded later like any other detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.open_vocab_fixtures import (
    FakeSegmenter,
    StatefulRegistry,
    cand,
    image_doc,
    ingested_world,
    make_set,
)
from curation.reprocess_fixtures import docs
from src.services.curation.ingest_policy import EmbeddingPolicy, IngestPolicy
from src.services.curation.item_filter import ItemFilter
from src.services.curation.open_vocab_run import run_open_vocab_image
from src.services.curation.reprocess import apply_reprocess
from src.services.curation.reprocess_models import (
    EmbedOptions,
    ReprocessFilter,
    ReprocessRequest,
    ReprocessTargets,
)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> StatefulRegistry:
    reg = StatefulRegistry()
    monkeypatch.setattr('src.services.curation.open_vocab_run.get_class_registry', lambda: reg)
    return reg


async def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: IngestPolicy) -> Any:
    fake, service, (image_id,) = await ingested_world(tmp_path, monkeypatch)
    service.policy = policy
    seg = FakeSegmenter()
    seg.by_prompt['traffic cone'] = [cand()]
    await run_open_vocab_image(
        fake, service, image_id, image_doc(fake, image_id), make_set(), revision=1, segment=seg
    )
    return fake, service


@pytest.mark.asyncio
async def test_an_open_vocab_item_is_embedded_under_the_default_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, _ = await _run(tmp_path, monkeypatch, IngestPolicy())
    (doc,) = docs(fake).values()
    assert doc['embedding_state'] == 'embedded'
    assert 'pe_embedding' in doc


@pytest.mark.asyncio
async def test_a_lazy_project_stores_the_item_without_a_vector_then_embeds_it_on_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service = await _run(
        tmp_path, monkeypatch, IngestPolicy(embedding=EmbeddingPolicy(mode='lazy'))
    )
    (doc,) = docs(fake).values()
    assert doc['embedding_state'] == 'deferred'
    assert 'pe_embedding' not in doc

    async def factory() -> Any:
        return service

    resp = await apply_reprocess(
        fake,
        ReprocessRequest(
            targets=ReprocessTargets(
                filter=ReprocessFilter(origin=['sam3'], embedding_state=['deferred'])
            ),
            scopes=['embed'],
            embed=EmbedOptions(only_missing=True),
            dry_run=False,
        ),
        service_factory=factory,
    )
    assert resp.scopes[0].detail['crop_written'] == 1
    (after,) = docs(fake).values()
    assert after['embedding_state'] == 'embedded'
    assert after['class_source'] == 'open_vocab_target'  # embedding never touches the class


def test_origin_sam3_means_the_item_carries_a_prompt_set() -> None:
    from curation.query_fakes import matches
    from src.services.curation.item_filter import item_filter_clauses

    query = {'bool': {'filter': item_filter_clauses(ItemFilter(origin=['sam3']))}}
    assert matches({'open_vocab_set': 'street'}, query)
    assert not matches({'proposal_name': 'car'}, query)
