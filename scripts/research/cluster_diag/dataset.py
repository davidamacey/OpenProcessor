"""Load an exported project (vectors, metadata) with its labels into one aligned bundle."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.research.cluster_diag.diagnostics import l2_normalize
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@dataclass(frozen=True)
class Bundle:
    """Vectors of every exported crop plus the labelled subset the scores are computed on."""

    vectors: NDArray[np.float32]
    meta: list[dict[str, Any]]
    labelled: NDArray[np.int64]  # row indices of the labelled crops
    truth: NDArray[Any]  # class of each labelled crop
    stratum: NDArray[Any]  # stratum name of each labelled crop
    group: list[str]  # source image of each labelled crop
    aspect: NDArray[np.float64]  # box width / height in pixels, per labelled crop
    side: NDArray[np.float64]  # short box side in pixels, per labelled crop
    protected: NDArray[np.bool_]  # human- or import-validated, locked or holdout, per labelled crop

    def strata_masks(self) -> dict[str, NDArray[np.bool_]]:
        """Boolean masks over the labelled crops: label strata plus shape and size strata."""
        elongated = np.abs(np.log(self.aspect)) > math.log(2)
        masks = {str(name): self.stratum == name for name in sorted(set(self.stratum.tolist()))}
        masks['elongated>2:1'] = elongated
        masks['aspect<=2:1'] = ~elongated
        masks['side<32px'] = self.side < 32
        masks['side>=96px'] = self.side >= 96
        return masks


NONE_KEY = 'none'  # partition key of a crop the detector gave no class


def detector_keys(meta: list[dict[str, Any]]) -> NDArray[Any]:
    """Each crop's detector class as the product's name rule writes it (``NONE_KEY`` if absent)."""
    return np.array(
        [normalize_class_name(str(m.get('detector_class_name') or '')) or NONE_KEY for m in meta]
    )


def class_agreement(meta: list[dict[str, Any]], rows: Any, truth: Any) -> float:
    """Share of ``rows`` whose detector class names the same class as ``truth`` (same rule).

    A crop with no detector class never agrees.
    """
    det = detector_keys([meta[int(i)] for i in rows])
    wanted = np.array([normalize_class_name(str(t)) for t in truth])
    return float(np.mean((det == wanted) & (det != NONE_KEY)))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def image_name(row: dict[str, Any]) -> str:
    """File name the annotations are keyed by: the uploader's identifier when there is one."""
    return str(row.get('source_identifier') or row['image_path']).rsplit('/', 1)[-1]


def load_bundle(stem: Path, labels: Path, coco_gt: Path) -> Bundle:
    """``stem.npy`` and ``stem.meta.jsonl`` from ``fetch_vectors``; ``labels`` rows carry
    ``item_id`` (the crop id), ``label``, ``stratum`` and optionally ``group`` (the source image;
    the image id when absent); ``coco_gt`` is the public annotation
    file, used only for the pixel size of each source image.
    """
    vectors = l2_normalize(np.load(stem.with_suffix('.npy')))
    meta = read_jsonl(stem.with_suffix('.meta.jsonl'))
    label_of = {r['item_id']: r for r in read_jsonl(labels)}
    dims = {
        im['file_name']: (im['width'], im['height'])
        for im in json.loads(coco_gt.read_text())['images']
    }
    rows = [i for i, m in enumerate(meta) if m['crop_id'] in label_of]
    aspect, side = [], []
    for i in rows:
        w, h = dims[image_name(meta[i])]
        x1, y1, x2, y2 = meta[i]['bbox_norm']
        bw, bh = max(1e-6, (x2 - x1) * w), max(1e-6, (y2 - y1) * h)
        aspect.append(bw / bh)
        side.append(min(bw, bh))
    return Bundle(
        vectors=vectors,
        meta=meta,
        labelled=np.array(rows, dtype=np.int64),
        truth=np.array([label_of[meta[i]['crop_id']]['label'] for i in rows]),
        stratum=np.array([label_of[meta[i]['crop_id']]['stratum'] for i in rows]),
        group=[str(label_of[meta[i]['crop_id']].get('group', meta[i]['image_id'])) for i in rows],
        aspect=np.array(aspect),
        side=np.array(side),
        protected=np.array(
            [bool(label_of[meta[i]['crop_id']].get('protected', False)) for i in rows], dtype=bool
        ),
    )
