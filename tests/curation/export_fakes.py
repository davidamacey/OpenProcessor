"""Fakes and factories shared by the dataset export tests.

``FakeOpenSearch`` serves one scroll page of hits per query and records the
query bodies; the factories build the generic and the single-class export
services over a tmp export root and a tmp class registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.registry import ClassRegistry
from src.config import CurationConfig
from src.config.region_fields import RegionFields
from src.services.curation.export import GenericYoloExportService
from src.services.curation.export_single_class import SingleClassExportService


if TYPE_CHECKING:
    from pathlib import Path

    from src.config.region_state import RegionStatus
    from src.services.curation.export_models import SingleClassExportProfile


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class FakeOpenSearch:
    """Returns one page of hits per distinct query, then exhausts the scroll.

    ``queries`` records every query body it was asked for, so a test can
    assert the exporter pushed the right filter into OpenSearch rather
    than scrolling everything and filtering in Python.
    """

    def __init__(self, docs: list[dict[str, Any]], *, by_status: bool = False) -> None:
        self._docs = docs
        self._by_status = by_status
        self.queries: list[dict[str, Any]] = []
        self._served: set[str] = set()

    async def search(self, index: str, body: dict[str, Any], scroll: str | None = None) -> dict:  # noqa: ARG002
        self.queries.append(body['query'])
        docs = self._docs if not self._by_status else self._match(body['query'])
        scroll_id = f'scroll-{len(self.queries)}'
        self._served.add(scroll_id)
        hits = [{'_id': d.get('crop_id', ''), '_source': d} for d in docs]
        return {'_scroll_id': scroll_id, 'hits': {'hits': hits}}

    def _match(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        """Crude status-terms matcher, enough to split the three region pools.

        The empty-frame sample query is wrapped in ``function_score``
        (random_score) rather than a bare bool — unwrap it first.
        """
        if 'function_score' in query:
            query = query['function_score']['query']
        # status/class_id predicates now live in filter context.
        clauses = query.get('bool', {}).get('filter', [])
        wanted: set[str] = set()
        for clause in clauses:
            for field_name, values in clause.get('terms', {}).items():
                if field_name == FIELDS.status:
                    wanted.update(values)
        if not wanted:
            return self._docs
        return [d for d in self._docs if d.get(FIELDS.status) in wanted]

    async def scroll(self, scroll_id: str, scroll: str) -> dict:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, scroll_id: str) -> None:  # noqa: ARG002
        return None


FIELDS = RegionFields()


def make_registry(base_dir: Path, names: list[str]) -> ClassRegistry:
    reg = ClassRegistry(path=base_dir / 'class_registry.json')
    for name in names:
        reg.add_class(name)
    return reg


def generic_service(
    tmp_path: Path, docs: list[dict[str, Any]], names: list[str]
) -> GenericYoloExportService:
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    registry = make_registry(tmp_path / 'registry', names)
    return GenericYoloExportService(FakeOpenSearch(docs), config=cfg, registry=registry)


def single_class_service(
    tmp_path: Path,
    opensearch: FakeOpenSearch,
    profile: SingleClassExportProfile,
    *,
    names: list[str] | None = None,
    source_root: Path | None = None,
) -> SingleClassExportService:
    cfg = CurationConfig(
        export_root=tmp_path / 'exports',
        source_root=source_root or (tmp_path / 'images'),
    )
    return SingleClassExportService(
        opensearch,
        profile=profile,
        config=cfg,
        registry=make_registry(tmp_path / 'registry', names or ['car', 'truck', 'bus']),
        region_fields=FIELDS,
    )


def make_item(
    idx: int,
    class_id: int,
    *,
    bbox: list[float] | None = None,
    test_holdout: bool = False,
    cluster_id: int | None = None,
) -> dict[str, Any]:
    return {
        'crop_id': f'crop-{idx}',
        'image_id': f'img-{idx}',
        'image_path': f'{idx}.jpg',
        'bbox_norm': bbox or [0.1, 0.1, 0.5, 0.5],
        'class_id': class_id,
        'test_holdout': test_holdout,
        'cluster_id': cluster_id,
    }


def make_region_item(
    idx: int,
    status: RegionStatus,
    *,
    region_bbox: list[float] | None = None,
    class_id: int = 0,
    test_holdout: bool = False,
    region_cluster_id: int | None = None,
) -> dict[str, Any]:
    doc = make_item(idx, class_id, test_holdout=test_holdout)
    doc[FIELDS.status] = status.value
    doc[FIELDS.boxes] = (
        [
            {
                'box_id': 'b1',
                'bbox_norm': list(region_bbox),
                'state': 'accepted',
                'cluster_id': region_cluster_id,
            }
        ]
        if region_bbox is not None
        else []
    )
    return doc
