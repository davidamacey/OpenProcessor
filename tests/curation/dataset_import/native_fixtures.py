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


OI_CLASSES = '/m/0bt9lr,Dog\n/m/01g317,Person\n/m/0k4j,Car\n'
OI_CLASSES_V7 = 'LabelName,DisplayName\n' + OI_CLASSES
OI_HEADER = (
    'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,'
    'IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n'
)
DOG = '/m/0bt9lr'
PERSON = '/m/01g317'
CAR = '/m/0k4j'


def oi_row(
    image_id: str,
    mid: str,
    xmin: float | str,
    xmax: float | str,
    ymin: float | str,
    ymax: float | str,
    *,
    group: int = 0,
    depiction: int = 0,
) -> str:
    return f'{image_id},xclick,{mid},1,{xmin},{xmax},{ymin},{ymax},0,0,{group},{depiction},0\n'


def _image_dir(root: Path, split: str, layout: str) -> Path:
    return {
        'images/<split>': root / 'images' / split,
        '<split>': root / split,
        'images': root / 'images',
        'root': root,
    }[layout]


def write_openimages(
    root: Path,
    rows_by_split: dict[str, list[str]],
    *,
    classes: str = OI_CLASSES,
    layout: str = 'images/<split>',
) -> None:
    """Writes ``class-descriptions-boxable.csv``, ``<split>-annotations-bbox.csv``
    (split in train/validation/test) and one JPEG per distinct ImageID under the
    chosen layout (``images/<split>``, ``<split>``, ``images`` or ``root``)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / 'class-descriptions-boxable.csv').write_text(classes)
    index = 0
    for split, rows in rows_by_split.items():
        (root / f'{split}-annotations-bbox.csv').write_text(OI_HEADER + ''.join(rows))
        for image_id in dict.fromkeys(r.split(',', 1)[0] for r in rows):
            write_image(
                _image_dir(root, split, layout) / f'{image_id}.jpg',
                size=(100 + index, 100),
                color=['red', 'green', 'blue', 'white'][index % 4],
            )
            index += 1
