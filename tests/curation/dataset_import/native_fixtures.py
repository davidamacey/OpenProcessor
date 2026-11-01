"""Offline Pascal VOC and Open Images dataset writers (no downloads)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from curation.dataset_import.harness import write_image


if TYPE_CHECKING:
    from pathlib import Path


VOC_XML = """<annotation>
  <folder>VOC2012</folder>
  <filename>{filename}</filename>
  {size}
  {objects}
</annotation>
"""


def voc_object(
    name: str, box: tuple[int, int, int, int], *, difficult: int = 0, truncated: int = 0
) -> str:
    x1, y1, x2, y2 = box
    return (
        f'<object><name>{name}</name><pose>Unspecified</pose><truncated>{truncated}</truncated>'
        f'<difficult>{difficult}</difficult><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>'
        f'<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>'
    )


def write_voc(
    root: Path,
    images: dict[str, list[str]],
    *,
    size: tuple[int, int] = (100, 100),
    splits: dict[str, list[str]] | None = None,
    xml_size: tuple[int, int] | str | None = 'same',
) -> None:
    """``images``: stem -> object snippets. Writes ``JPEGImages/<stem>.jpg``,
    ``Annotations/<stem>.xml`` and ``ImageSets/Main/<split>.txt``. ``xml_size``:
    ``'same'`` is the image size, ``None`` writes no ``<size>``."""
    (root / 'Annotations').mkdir(parents=True, exist_ok=True)
    for i, (stem, objects) in enumerate(images.items()):
        actual = (size[0] + i, size[1])  # distinct bytes per image: no dedup
        write_image(
            root / f'JPEGImages/{stem}.jpg',
            size=actual,
            color=['red', 'green', 'blue', 'white'][i % 4],
        )
        declared = actual if xml_size == 'same' else xml_size
        if declared is None:
            size_el = ''
        else:
            w, h = declared  # type: ignore[misc]
            size_el = f'<size><width>{w}</width><height>{h}</height><depth>3</depth></size>'
        (root / f'Annotations/{stem}.xml').write_text(
            VOC_XML.format(filename=f'{stem}.jpg', size=size_el, objects=''.join(objects))
        )
    if splits:
        (root / 'ImageSets/Main').mkdir(parents=True, exist_ok=True)
        for split, stems in splits.items():
            (root / f'ImageSets/Main/{split}.txt').write_text(''.join(f'{s}\n' for s in stems))
