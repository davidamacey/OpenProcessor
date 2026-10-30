"""Labels and the stored production partition from a ``fetch_vectors.py`` export.

    build_labels.py --bundle STEM --coco-gt GT.json --eval-manifest PIN.json --out-dir DIR

The evaluation set's own builder reads crops over the HTTP API, which names an uploaded image
by its hash file name. The export carries the uploader's ``source_identifier``, so the same
matching rules (``cluster_eval_set.build``) run on it here. Writes ``labels.jsonl``,
``partition.jsonl`` (the ``cluster_id`` the pipeline stored) and ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.cluster_eval_set import MIN_IOU, build, load_coco
from scripts.research.cluster_diag.dataset import image_name, read_jsonl


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--coco-gt', type=Path, required=True)
    ap.add_argument('--eval-manifest', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--min-iou', type=float, default=MIN_IOU)
    args = ap.parse_args(argv)
    pin = json.loads(args.eval_manifest.read_text())
    crops = [
        {**m, 'image_path': image_name(m)}
        for m in read_jsonl(args.bundle.with_suffix('.meta.jsonl'))
    ]
    labels, partition, manifest = build(
        crops, load_coco(args.coco_gt), args.min_iou, set(pin['rare_classes'])
    )
    manifest |= {'eval_manifest_sha256': pin['manifest_sha256'], 'eval_seed': pin['seed']}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (('labels', labels), ('partition', partition)):
        (args.out_dir / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
