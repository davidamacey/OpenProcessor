"""Load a frozen YOLO test split and expose it as COCO ground truth.

The bake-off scores every model against the same frozen ``test/`` split a
dataset export produced (``images/test`` + ``labels/test``). Every class in
the label files is loaded; which classes are *scored* is decided per call
(``coco_gt(scored_class_ids)``). Background frames have an empty ``.txt``
(or none) and still count -- a false positive on a background frame is a
false positive.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2

from .class_map import read_names


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path


class GtBox(NamedTuple):
    """One ground-truth box: eval class id + absolute-pixel xyxy."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(slots=True)
class GtImage:
    """One test frame: identity, pixel size, and its GT boxes (every class)."""

    image_id: int
    path: Path
    width: int
    height: int
    boxes: list[GtBox] = field(default_factory=list)


def category_id(eval_class_id: int) -> int:
    """COCO category id of an eval class (COCO ids start at 1)."""
    return eval_class_id + 1


class YoloTestSet:
    """A YOLO detection split loaded for COCO-style evaluation.

    Args:
        images_dir: Directory of test images.
        labels_dir: Directory of matching ``<stem>.txt`` YOLO labels.
        class_names: The dataset's label space (``data.yaml`` ``names``):
            a ``{class_id: name}`` mapping or a list indexed by class id.
            Classes seen in labels but missing here are named ``str(id)``.
        scored_class_ids: Classes the metrics score by default (``None`` =
            every class present in the split).
        stratum_map: Optional ``{image_stem: stratum}`` for per-cluster
            breakdowns; produced alongside the export.
    """

    _IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        *,
        class_names: Mapping[int, str] | Sequence[str],
        scored_class_ids: Iterable[int] | None = None,
        stratum_map: dict[str, str] | None = None,
    ) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        names = (
            dict(class_names.items())
            if hasattr(class_names, 'items')
            else dict(enumerate(class_names))
        )
        self.class_names: dict[int, str] = {int(k): str(v) for k, v in names.items()}
        self.stratum_map = stratum_map or {}
        self.images: list[GtImage] = []
        self._load()
        for cid in self.present_class_ids:
            self.class_names.setdefault(cid, str(cid))
        self.scored_class_ids: list[int] = sorted(
            self.present_class_ids if scored_class_ids is None else set(scored_class_ids)
        )

    @classmethod
    def from_dataset_root(
        cls,
        root: Path,
        *,
        split: str = 'test',
        scored_class_ids: Iterable[int] | None = None,
        stratum_map_path: Path | None = None,
    ) -> YoloTestSet:
        """Build from an export root containing ``images/<split>``, ``data.yaml`` etc."""
        stratum_map = None
        if stratum_map_path and stratum_map_path.is_file():
            stratum_map = json.loads(stratum_map_path.read_text(encoding='utf-8'))
        return cls(
            root / 'images' / split,
            root / 'labels' / split,
            class_names=read_names(root / 'data.yaml'),
            scored_class_ids=scored_class_ids,
            stratum_map=stratum_map,
        )

    def _iter_image_paths(self) -> list[Path]:
        out: list[Path] = []
        for ext in self._IMG_EXTS:
            out.extend(sorted(self.images_dir.glob(f'*{ext}')))
        return sorted(out)

    def _load(self) -> None:
        for image_id, img_path in enumerate(self._iter_image_paths(), start=1):
            dims = _image_size(img_path)
            if dims is None:
                continue
            w, h = dims
            boxes = self._read_label(self.labels_dir / f'{img_path.stem}.txt', w, h)
            self.images.append(GtImage(image_id, img_path, w, h, boxes))

    @staticmethod
    def _read_label(label_path: Path, w: int, h: int) -> list[GtBox]:
        if not label_path.is_file():
            return []
        boxes: list[GtBox] = []
        for line in label_path.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = parts
            cx_, cy_, bw_, bh_ = (float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h)
            boxes.append(
                GtBox(
                    int(float(cls_id)),
                    cx_ - bw_ / 2,
                    cy_ - bh_ / 2,
                    cx_ + bw_ / 2,
                    cy_ + bh_ / 2,
                )
            )
        return boxes

    @property
    def present_class_ids(self) -> set[int]:
        """Every class with at least one GT box in the split."""
        return {b.class_id for img in self.images for b in img.boxes}

    @property
    def n_gt_by_class(self) -> dict[int, int]:
        return dict(Counter(b.class_id for img in self.images for b in img.boxes))

    def stratum_for(self, img: GtImage) -> str:
        return self.stratum_map.get(img.path.stem, 'all')

    def coco_gt(
        self,
        scored_class_ids: Iterable[int] | None = None,
        *,
        images: Sequence[GtImage] | None = None,
    ) -> dict[str, Any]:
        """The pycocotools ground-truth dict: one category per scored class.

        ``category_id = eval_class_id + 1``. GT boxes of unscored classes are
        left out (they are not measured). ``images`` restricts the frames
        (per-stratum breakdowns).
        """
        scored = sorted(self.scored_class_ids if scored_class_ids is None else scored_class_ids)
        keep = set(scored)
        frames = self.images if images is None else images
        out_images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        ann_id = 1
        for img in frames:
            out_images.append(
                {
                    'id': img.image_id,
                    'file_name': img.path.name,
                    'width': img.width,
                    'height': img.height,
                }
            )
            for box in img.boxes:
                if box.class_id not in keep:
                    continue
                bw, bh = box.x2 - box.x1, box.y2 - box.y1
                annotations.append(
                    {
                        'id': ann_id,
                        'image_id': img.image_id,
                        'category_id': category_id(box.class_id),
                        'bbox': [box.x1, box.y1, bw, bh],
                        'area': bw * bh,
                        'iscrowd': 0,
                    }
                )
                ann_id += 1
        return {
            'images': out_images,
            'annotations': annotations,
            'categories': [
                {'id': category_id(c), 'name': self.class_names.get(c, str(c))} for c in scored
            ],
        }

    @property
    def n_positive_frames(self) -> int:
        return sum(1 for i in self.images if i.boxes)

    @property
    def n_background_frames(self) -> int:
        return sum(1 for i in self.images if not i.boxes)


def _image_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) without decoding the full image when possible."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h
