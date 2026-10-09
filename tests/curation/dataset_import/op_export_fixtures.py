"""Real OpenProcessor exports for the dataset-import tests: the REAL
exporters run against a query-evaluating in-memory OpenSearch, then the
image files an export would have copied are written next to the labels
(``copy_images=False`` keeps the exporters off the source-image paths)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch.registry import ClassRegistry
from src.config import CurationConfig
from src.config.region_fields import RegionFields
from src.services.curation.export import GenericYoloExportService
from src.services.curation.export_single_class import (
    SingleClassExportProfile,
    SingleClassExportService,
)


if TYPE_CHECKING:
    from src.config.region_state import RegionStatus

ITEMS = 'op_test_items'
IMAGES = 'op_test_images'
F = RegionFields()


def config(tmp_path: Path) -> CurationConfig:
    return CurationConfig(items_index=ITEMS, images_index=IMAGES, export_root=tmp_path / 'exports')


def item(
    crop_id: str,
    image_id: str,
    class_id: int,
    class_name: str,
    bbox: list[float],
    **extra: Any,
) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_id': image_id,
        'image_path': f'/data/{image_id}.jpg',
        'bbox_norm': bbox,
        'class_id': class_id,
        'class_name': class_name,
        'class_validated': True,
        'class_source': 'human',
        **extra,
    }


def write_images(export_dir: Path, *, suffix: str = '.jpg') -> None:
    """One tiny image per label file, as ``copy_images=True`` would write."""
    for label in sorted((export_dir / 'labels').glob('*/*.txt')):
        split = label.parent.name
        target = export_dir / 'images' / split / f'{label.stem}{suffix}'
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (64, 48), color='gray').save(target, format='JPEG')


def multi_registry(
    base: Path, names: list[str], *, deprecate: tuple[str, ...] = ()
) -> ClassRegistry:
    reg = ClassRegistry(path=base / 'class_registry.json')
    ids = {name: reg.add_class(name) for name in names}
    for name in deprecate:
        reg.merge_class(ids[name], ids[names[0]])
    return reg


async def export_multi_class(
    tmp_path: Path,
    docs: dict[str, dict[str, Any]],
    registry: ClassRegistry,
    *,
    images: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> tuple[Path, QueryFakeOpenSearch]:
    fake = QueryFakeOpenSearch({ITEMS: docs, IMAGES: images or {}})
    service = GenericYoloExportService(fake, config=config(tmp_path), registry=registry)
    result = await service.export_dataset(version_tag='fx', copy_images=False, **kwargs)
    export_dir = Path(result.export_dir)
    write_images(export_dir)
    return export_dir, fake


def region_item(
    crop_id: str,
    image_id: str,
    status: RegionStatus,
    *,
    parent_class: int = 0,
    box: list[float] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    doc = item(crop_id, image_id, parent_class, 'car', [0.1, 0.1, 0.9, 0.9], **extra)
    doc[F.status] = status.value
    doc[F.boxes] = (
        [{'box_id': 'b1', 'bbox_norm': box, 'state': 'accepted'}] if box is not None else []
    )
    return doc


async def export_region_whole_frame(
    tmp_path: Path,
    docs: dict[str, dict[str, Any]],
    *,
    images: dict[str, dict[str, Any]] | None = None,
    image_mode: str = 'whole_frame',
    empty_bg_ratio: float = 10.0,
    **kwargs: Any,
) -> tuple[Path, QueryFakeOpenSearch]:
    reg = multi_registry(tmp_path / 'reg', ['car', 'truck'])
    fake = QueryFakeOpenSearch({ITEMS: docs, IMAGES: images or {}})
    service = SingleClassExportService(
        fake,
        profile=SingleClassExportProfile(
            name='plates',
            class_ids=(0,),
            box_source='region',
            region_class_name='plate',
            **kwargs,
        ),
        config=config(tmp_path),
        registry=reg,
        region_fields=F,
    )
    result = await service.export(
        version_tag='fx',
        copy_images=False,
        empty_bg_ratio=empty_bg_ratio,
        image_mode=image_mode,  # type: ignore[arg-type]
    )
    export_dir = Path(result.export_dir)
    write_images(export_dir)
    return export_dir, fake
