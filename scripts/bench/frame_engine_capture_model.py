"""Triton Python-backend model: the capture stand-in of the native engine's N0 comparison.

Copied by ``frame_engine_stand_ins.py`` to ``<model>/1/model.py`` of each ``capture_*`` model. It
answers like the null model it stands for (the same fixed tensor, so both engines cut the same
crops from every photo) and records every input row the model server hands it: the row's bytes to
``<CAPTURE_DIR>/<model>/<sha256>.bin`` (one file per distinct content) and a line
``<order>\\t<sha256>\\t<shape>`` to ``<CAPTURE_DIR>/<model>/index.txt``, in the order the rows
arrived. ``scripts/bench/gpu_prep_parity.py compare-inputs`` reads those two to decide whether two
engines wrote the same model inputs, byte for byte.

``FORCE_CPU_ONLY_INPUT_TENSORS`` is ``yes`` in the config, so the server copies the engine's CUDA
shared-memory inputs to host memory before ``execute`` runs
(https://github.com/triton-inference-server/python_backend/blob/r26.10/README.md, "Input Tensor
Device Placement"). The instance count is one, so ``order`` is the arrival order of the rows.

The model configuration's parameters (strings):

``CAPTURE_DIR``  where to write
``OUTPUT_NAME``  the output tensor's name
``OUTPUT_KIND``  ``detector`` (``OUTPUT_ROWS``: the JSON list of detection rows, padded with zero
                 rows to ``OUTPUT_ROW_COUNT``) or ``encoder`` (a unit vector of ``VECTOR_DIM``)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def output_row(kind: str, params: dict[str, str]) -> np.ndarray:
    """The one fixed answer per input row: ``[300, 6]`` detections or a ``[dim]`` unit vector."""
    if kind == 'detector':
        out = np.zeros((int(params['OUTPUT_ROW_COUNT']), 6), np.float32)
        rows = np.array(json.loads(params['OUTPUT_ROWS']), np.float32)
        out[: len(rows)] = rows
        return out
    dim = int(params['VECTOR_DIM'])
    return np.full(dim, 1.0 / np.sqrt(dim), np.float32)


class Recorder:
    """The capture of one model: content-addressed row files and the arrival-order index."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.index = root / 'index.txt'
        # a model reloaded under the same directory continues its order
        self.order = len(self.index.read_text().splitlines()) if self.index.exists() else 0

    def record(self, batch: np.ndarray) -> None:
        lines = []
        for row in batch:
            data = np.ascontiguousarray(row).tobytes()
            digest = hashlib.sha256(data).hexdigest()
            blob = self.root / f'{digest}.bin'
            if not blob.exists():
                blob.write_bytes(data)
            shape = 'x'.join(str(d) for d in row.shape)
            lines.append(f'{self.order}\t{digest}\t{shape}\n')
            self.order += 1
        with self.index.open('a') as f:
            f.writelines(lines)


class TritonPythonModel:
    """The server calls ``initialize`` once and ``execute`` per batch of requests."""

    def initialize(self, args: dict[str, str]) -> None:
        config = json.loads(args['model_config'])
        params = {k: v['string_value'] for k, v in config.get('parameters', {}).items()}
        self.output_name = params['OUTPUT_NAME']
        self.answer = output_row(params['OUTPUT_KIND'], params)
        self.recorder = Recorder(Path(params['CAPTURE_DIR']) / args['model_name'])

    def execute(self, requests: list[Any]) -> list[Any]:
        import triton_python_backend_utils as pb_utils

        responses = []
        for request in requests:
            batch = pb_utils.get_input_tensor_by_name(request, 'images').as_numpy()
            self.recorder.record(batch)
            out = np.broadcast_to(self.answer, (len(batch), *self.answer.shape)).copy()
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor(self.output_name, out)])
            )
        return responses
