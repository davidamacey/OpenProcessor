"""Load an exported project (vectors, metadata) with its labels into one aligned bundle."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.research.cluster_diag.diagnostics import l2_normalize


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

    def strata_masks(self) -> dict[str, NDArray[np.bool_]]:
        """Boolean masks over the labelled crops: label strata plus shape and size strata."""
        elongated = np.abs(np.log(self.aspect)) > math.log(2)
        masks = {str(name): self.stratum == name for name in sorted(set(self.stratum.tolist()))}
        masks['elongated>2:1'] = elongated
        masks['aspect<=2:1'] = ~elongated
        masks['side<32px'] = self.side < 32
        masks['side>=96px'] = self.side >= 96
        return masks


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_bundle(stem: Path, labels: Path, coco_gt: Path) -> Bundle:
    """``stem.npy`` and ``stem.meta.jsonl`` from ``fetch_vectors``; ``labels`` rows carry
    ``item_id`` (the crop id), ``label`` and ``stratum``; ``coco_gt`` is the public annotation
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
        w, h = dims[meta[i]['image_path'].rsplit('/', 1)[-1]]
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
        group=[str(meta[i]['image_id']) for i in rows],
        aspect=np.array(aspect),
        side=np.array(side),
    )
