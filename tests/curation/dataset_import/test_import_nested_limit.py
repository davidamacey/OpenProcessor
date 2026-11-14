"""An import never writes onto an item at the nested-document limit (OpenSearch k-NN issue
3623): the image is reported failed with the reason, nothing is truncated, and an undo does not
put such an item back."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.harness import Harness, map_all, write_yolo
from src.config.region_fields import get_region_fields
from src.domain.nested_limit import ERROR_CODE, NestedLimitError
from src.services.curation.dataset_import.undo import check_reinstate


if TYPE_CHECKING:
    from pathlib import Path

SAME = '0 0.3 0.3 0.4 0.4'
F = get_region_fields()


def _boxes(n: int) -> list[dict[str, str]]:
    return [{'box_id': f'b{i}'} for i in range(n)]


def test_an_item_at_the_limit_is_not_put_back_by_an_undo() -> None:
    with pytest.raises(NestedLimitError):
        check_reinstate('c1', {F.boxes: _boxes(75)}, F)
    check_reinstate('c1', {F.boxes: _boxes(74)}, F)


@pytest.mark.db
@pytest.mark.asyncio
async def test_an_import_onto_an_item_over_the_limit_reports_the_image_failed(
    tmp_path: Path, monkeypatch
) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'a': [SAME]})
    image = root / 'images/train/a.jpg'
    result = await h.service.ingest_one(image.read_bytes(), str(image.resolve()))
    assert result.status == 'success', result
    (cid,) = h.items
    h.items[cid][F.boxes] = _boxes(75)

    store, _ = await h.run(h.request(root, map_all(h.registry, 'car')))

    [row] = store.ledger_rows()
    assert row['status'] == 'failed'
    assert row['error_kind'] == ERROR_CODE
    assert h.items[cid]['class_validated'] is False
    assert len(h.items[cid][F.boxes]) == 75
