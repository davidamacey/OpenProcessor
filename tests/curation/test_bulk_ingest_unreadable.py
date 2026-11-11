"""A folder pass that finds only entries it cannot read must say so, not complete clean."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.bulk_ingest.folder_scan import scan_folder


if TYPE_CHECKING:
    from pathlib import Path


class _Queue:
    def __init__(self) -> None:
        self.meta: dict[str, Any] = {}

    def meta_get(self, name: str) -> Any:
        return self.meta.get(name)

    def meta_set(self, name: str, value: Any) -> None:
        self.meta[name] = value

    def enqueue(self, chunk: list[Any]) -> tuple[int, int]:
        return len(chunk), 0


def test_dangling_symlinks_are_counted_with_relative_samples(tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / f'broken{i}.jpg').symlink_to(tmp_path / 'missing.jpg')
    queue = _Queue()

    result = scan_folder(queue, tmp_path, extensions=['.jpg'], full=True)  # type: ignore[arg-type]

    assert result.listed == 0
    assert result.unreadable == 3
    assert sorted(result.unreadable_samples) == ['broken0.jpg', 'broken1.jpg', 'broken2.jpg']
    assert queue.meta['last_scan']['unreadable'] == 3
