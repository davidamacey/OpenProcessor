"""What an uploaded ``.pt`` checkpoint is, read without running anything from it.

* :func:`is_original_yolov5` recognises a checkpoint written by the original ``ultralytics/yolov5``
  repository, which the Ultralytics library cannot load (YOLOv5u checkpoints are the supported
  form: https://docs.ultralytics.com/models/yolov5/).
"""

from __future__ import annotations

import pickletools  # nosec B403 - only disassembles a pickle stream, never loads it
import zipfile
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


YOLOV5_ORIGINAL_MESSAGE = (
    'Original YOLOv5 checkpoints are not supported by Ultralytics; retrain from YOLOv5u '
    '(yolov5su.pt), or export an ONNX with the official ultralytics/yolov5 export.py for the '
    'probe or the ingest secondary detector'
)
_YOLOV5_REPO_MODULES = ('models.yolo', 'models.common', 'models.experimental')


def is_original_yolov5(path: Path) -> bool:
    """True when the checkpoint pickles classes from the original yolov5 repo.

    The pickle stream is only disassembled (``pickletools``), never executed.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            member = next((n for n in archive.namelist() if n.endswith('data.pkl')), None)
            if member is None:
                return False
            data = archive.read(member)
    except (OSError, zipfile.BadZipFile):
        return False
    try:
        return any(
            isinstance(arg, str) and arg.startswith(_YOLOV5_REPO_MODULES)
            for _, arg, _ in pickletools.genops(data)
        )
    except (ValueError, EOFError, IndexError):  # a stream pickletools cannot read
        return False
