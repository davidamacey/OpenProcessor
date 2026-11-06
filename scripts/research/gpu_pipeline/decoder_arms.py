"""Decoder adapters of gpu:PR-1: ``decode(list[bytes]) -> list[HWC uint8 CuPy array, EXIF applied]``.

Control D-a is the shipped nvJPEG ``GPU_HYBRID`` lane pool (``nvjpeg_lanes.py``). D-b is nvImageCodec
(``nvidia.nvimgcodec.Decoder``, backend ``HYBRID_CPU_GPU`` or ``GPU_ONLY``, option
``:fancy_upsampling=1``). D-c is DALI ``fn.decoders.image(device='mixed')`` with
``jpeg_fancy_upsampling=True`` and ``hybrid_huffman_threshold`` 0 or the default
(docs.nvidia.com/deeplearning/dali, "nvidia.dali.fn.decoders.image"; docs.nvidia.com/cuda/nvimagecodec).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import harness_common as common


# ruff: noqa: E402
common.use_shipped_prep_modules()

import cupy as cp
import numpy as np
from nvjpeg_lanes import NvJpegLanes
from prep_geometry import route
from shipped_pipeline import orient_gpu


class Decoder(Protocol):
    name: str

    def decode(self, datas: list[bytes]) -> list[Any]: ...


def routed(datas: list[bytes]) -> list[str]:
    """The shipped router's verdict per file: ``gpu`` or the reason it stays on the CPU path."""
    return [route(d, 0.0, 100.0)[0] for d in datas]


class ShippedLanes:
    """D-a. Threads check lanes out of the pool exactly as the shipped model's workers do."""

    def __init__(self, workers: int = 16) -> None:
        self.name = 'D-a nvjpeg-gpu-hybrid-lanes'
        self.lanes = NvJpegLanes(lanes=workers)
        self.pool = ThreadPoolExecutor(workers)

    def _one(self, data: bytes) -> Any:
        header = route(data, 0.0, 100.0)[1]
        with cp.cuda.Stream(non_blocking=True) as stream:
            frame, _ = self.lanes.decode(data)
            frame = orient_gpu(frame, header.orientation)
            stream.synchronize()
        return frame

    def decode(self, datas: list[bytes]) -> list[Any]:
        return list(self.pool.map(self._one, datas))


class NvImageCodec:
    """D-b."""

    def __init__(self, backend: str, batch: int, threads: int = 16) -> None:
        from nvidia import nvimgcodec

        self.name = f'D-b nvimgcodec-{backend}-b{batch}'
        self.batch = batch
        kind = {
            'hybrid': nvimgcodec.BackendKind.HYBRID_CPU_GPU,
            'gpu': nvimgcodec.BackendKind.GPU_ONLY,
        }[backend]
        self.params = nvimgcodec.DecodeParams(
            apply_exif_orientation=True, color_spec=nvimgcodec.ColorSpec.SRGB
        )
        self.dec = nvimgcodec.Decoder(
            device_id=0,
            max_num_cpu_threads=threads,
            backends=[nvimgcodec.Backend(kind)],
            options=':fancy_upsampling=1',
        )

    def decode(self, datas: list[bytes]) -> list[Any]:
        out: list[Any] = []
        for s in range(0, len(datas), self.batch):
            images = self.dec.decode(datas[s : s + self.batch], params=self.params)
            out.extend(None if im is None else cp.asarray(im) for im in images)
        cp.cuda.get_current_stream().synchronize()
        return out


class Dali:
    """D-c: one pipeline per batch size, fed with the encoded bytes."""

    def __init__(self, huffman_threshold: int | None, batch: int = 64, threads: int = 16) -> None:
        from nvidia.dali import fn, pipeline_def, types

        self.name = f'D-c dali-mixed-hht{"default" if huffman_threshold is None else huffman_threshold}-b{batch}'
        self.batch = batch
        extra = {} if huffman_threshold is None else {'hybrid_huffman_threshold': huffman_threshold}

        @pipeline_def(batch_size=batch, num_threads=threads, device_id=0, prefetch_queue_depth=1)
        def graph() -> Any:
            jpegs = fn.external_source(name='jpegs', dtype=types.UINT8)
            return fn.decoders.image(
                jpegs, device='mixed', output_type=types.RGB, jpeg_fancy_upsampling=True, **extra
            )

        self.pipe = graph()
        self.pipe.build()

    def decode(self, datas: list[bytes]) -> list[Any]:
        out: list[Any] = []
        for s in range(0, len(datas), self.batch):
            chunk = datas[s : s + self.batch]
            self.pipe.feed_input('jpegs', [np.frombuffer(d, np.uint8) for d in chunk])
            (res,) = self.pipe.run()
            out.extend(cp.asarray(res[i]) for i in range(len(chunk)))
        cp.cuda.get_current_stream().synchronize()
        return out


def build(spec: str) -> Decoder:
    """``control`` | ``nvimgcodec:<hybrid|gpu>:<batch>`` | ``dali:<default|0>:<batch>``."""
    parts = spec.split(':')
    if parts[0] == 'control':
        return ShippedLanes()
    if parts[0] == 'nvimgcodec':
        return NvImageCodec(parts[1], int(parts[2]))
    if parts[0] == 'dali':
        return Dali(None if parts[1] == 'default' else int(parts[1]), int(parts[2]))
    raise ValueError(f'unknown decoder spec {spec!r}')
