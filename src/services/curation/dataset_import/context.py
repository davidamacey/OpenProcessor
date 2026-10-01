"""What one import run carries: pinned, serializable inputs plus the
process-local handles (OpenSearch, the ingest service) a worker rebuilds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.config.region_fields import RegionFields, get_region_fields


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from opensearchpy import AsyncOpenSearch

    from src.services.curation.dataset_import.mapping import ResolvedMapping
    from src.services.curation.dataset_import.options import DatasetImportOptions
    from src.services.curation.dataset_import.prepare import PinnedProfile
    from src.services.curation.ingest import CurationIngestService


@dataclass
class ImportContext:
    import_id: str
    options: DatasetImportOptions
    resolved: ResolvedMapping
    profile: PinnedProfile | None
    parents: str
    source_sha: str
    source_format: str
    source_root: Path
    opensearch: AsyncOpenSearch | Any
    service: CurationIngestService
    images_index: str
    items_index: str
    crop_cache_dir: str | Path | None
    upload_root: Path
    export_root: Path | None
    freeze_test: bool
    region_fields: RegionFields = field(default_factory=get_region_fields)
    stem_resolutions: dict[str, Any] = field(default_factory=dict)
    """OpenProcessor-export stems resolved to existing docs (per chunk)."""
    seen_images: dict[str, dict[str, Any]] = field(default_factory=dict)
    """imohash -> the image this run already indexed (search may not see it yet)."""
    after_chunk: Callable[[int], Awaitable[None]] | None = None
    """Test hook: awaited after each finished chunk (crash injection)."""

    @property
    def negative_for(self) -> list[str]:
        """The class names a reviewed negative frame is negative for: every
        mapped item class and every mapped region class (the profile's name
        for it when one is active)."""
        names: set[str] = set()
        for dataset_class, target in self.resolved.targets.items():
            if target.kind == 'item' and target.class_name:
                names.add(target.class_name)
            elif target.kind == 'region':
                names.add((self.profile.region_class_name if self.profile else '') or dataset_class)
        return sorted(names)

    @property
    def writer(self) -> str:
        return f'import:{self.import_id}'

    @property
    def uses_detector(self) -> bool:
        has_region = any(t.kind == 'region' for t in self.resolved.targets.values())
        return self.options.processing == 'propose' or (has_region and self.parents == 'detect')


__all__ = ['ImportContext']
