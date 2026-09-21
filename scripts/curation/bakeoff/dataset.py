"""Load a frozen YOLO test split and expose it as COCO ground truth.

The bake-off scores every model against the same frozen ``test/`` split a
dataset export produced (``images/test`` + ``labels/test``, single class).
Background frames have an empty ``.txt`` (no boxes) and still count — a
false positive on a background frame is a false positive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2


if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class GtImage:
    """One test frame: identity, pixel size, and abs-xyxy GT boxes."""

    image_id: int
    path: Path
    width: int
    height: int
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


class YoloTestSet:
    """A YOLO detection split loaded for COCO-style evaluation.

    Args:
        images_dir: Directory of test images.
        labels_dir: Directory of matching ``<stem>.txt`` YOLO labels.
        target_class_id: The class id under test (single-class export => 0).
        target_class_name: Display name for that class in the COCO GT
            ``categories`` block (cosmetic only, doesn't affect scoring).
        stratum_map: Optional ``{image_stem: stratum}`` for per-cluster
            breakdowns; produced alongside the export.
    """

    _IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        *,
        target_class_id: int = 0,
        target_class_name: str = 'object',
        stratum_map: dict[str, str] | None = None,
    ) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_class_id = target_class_id
        self.target_class_name = target_class_name
        self.stratum_map = stratum_map or {}
        self.images: list[GtImage] = []
        self._load()

    @classmethod
    def from_dataset_root(
        cls,
        root: Path,
        *,
        split: str = 'test',
        target_class_id: int = 0,
        target_class_name: str = 'object',
        stratum_map_path: Path | None = None,
    ) -> YoloTestSet:
        """Build from an export root containing ``images/<split>`` etc."""
        stratum_map = None
        if stratum_map_path and stratum_map_path.is_file():
            stratum_map = json.loads(stratum_map_path.read_text(encoding='utf-8'))
        return cls(
            root / 'images' / split,
            root / 'labels' / split,
            target_class_id=target_class_id,
            target_class_name=target_class_name,
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

    def _read_label(
        self, label_path: Path, w: int, h: int
    ) -> list[tuple[float, float, float, float]]:
        if not label_path.is_file():
            return []
        boxes: list[tuple[float, float, float, float]] = []
        for line in label_path.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = parts
            if int(float(cls_id)) != self.target_class_id:
                continue
            cx_, cy_, bw_, bh_ = (float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h)
            boxes.append((cx_ - bw_ / 2, cy_ - bh_ / 2, cx_ + bw_ / 2, cy_ + bh_ / 2))
        return boxes

    def stratum_for(self, img: GtImage) -> str:
        return self.stratum_map.get(img.path.stem, 'all')

    def coco_gt(self) -> dict[str, Any]:
        """Assemble the pycocotools-compatible ground-truth dict."""
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        ann_id = 1
        for img in self.images:
            images.append(
                {
                    'id': img.image_id,
                    'file_name': img.path.name,
                    'width': img.width,
                    'height': img.height,
                }
            )
            for x1, y1, x2, y2 in img.boxes:
                bw, bh = x2 - x1, y2 - y1
                annotations.append(
                    {
                        'id': ann_id,
                        'image_id': img.image_id,
                        'category_id': 1,
                        'bbox': [x1, y1, bw, bh],
                        'area': bw * bh,
                        'iscrowd': 0,
                    }
                )
                ann_id += 1
        return {
            'images': images,
            'annotations': annotations,
            'categories': [{'id': 1, 'name': self.target_class_name}],
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
