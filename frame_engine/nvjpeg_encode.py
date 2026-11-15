"""nvJPEG encoder lanes through ctypes (the same library and binding style as the decode lanes).

``encode(rgb, quality, ready)`` takes an HWC uint8 RGB array that is already on the GPU and returns
the baseline 4:2:0 JPEG bytes; only the bitstream crosses to the host. Each lane owns what nvJPEG
requires per thread (an encoder state, encode parameters, a CUDA stream), so encoding threads check
lanes out of a bounded pool. ``ready`` is an event recorded on the stream that wrote ``rgb``: the
lane's stream waits for it, and the caller keeps ``rgb`` alive until ``encode`` returns.
Optimized Huffman is off: one pass, within 0.5% of OpenCV's bytes at the same quality
(artifacts_local/speed_1113/engine_deriv/results.md). Library: nvJPEG encoder API
(docs.nvidia.com/cuda/nvjpeg, "JPEG Encoding").
"""

from __future__ import annotations

import ctypes
import queue
from typing import Any

import cupy as cp

from frame_engine.nvjpeg_lanes import NvJpegError, _check, _Image


_ENCODING_BASELINE_DCT = 0xC0
_CSS_420 = 2
_INPUT_RGBI = 5


class _EncLane:
    def __init__(self, lib: Any, handle: ctypes.c_void_p) -> None:
        self.stream = cp.cuda.Stream(non_blocking=True)
        s = ctypes.c_void_p(self.stream.ptr)
        self.state = ctypes.c_void_p()
        _check(
            lib.nvjpegEncoderStateCreate(handle, ctypes.byref(self.state), s), 'EncoderStateCreate'
        )
        self.params = ctypes.c_void_p()
        _check(
            lib.nvjpegEncoderParamsCreate(handle, ctypes.byref(self.params), s),
            'EncoderParamsCreate',
        )
        _check(
            lib.nvjpegEncoderParamsSetEncoding(self.params, _ENCODING_BASELINE_DCT, s),
            'SetEncoding',
        )
        _check(lib.nvjpegEncoderParamsSetSamplingFactors(self.params, _CSS_420, s), 'SetSampling')
        _check(lib.nvjpegEncoderParamsSetOptimizedHuffman(self.params, 0, s), 'SetOptimizedHuffman')


class NvJpegEncoders:
    def __init__(self, lanes: int, library: str = 'libnvjpeg.so.13') -> None:
        self.lib = ctypes.CDLL(library)
        self.handle = ctypes.c_void_p()
        _check(self.lib.nvjpegCreateEx(0, None, None, 0, ctypes.byref(self.handle)), 'CreateEx')
        self.pool: queue.Queue[_EncLane] = queue.Queue()
        for _ in range(lanes):
            self.pool.put(_EncLane(self.lib, self.handle))

    @staticmethod
    def mark() -> Any:
        """An event on the caller's current stream: everything launched so far is done before the
        encode reads the pixels."""
        event = cp.cuda.Event()
        event.record()
        return event

    def encode(self, rgb: Any, quality: int, ready: Any) -> bytes:
        lane = self.pool.get()
        try:
            return self._encode_on(lane, rgb, quality, ready)
        finally:
            self.pool.put(lane)

    def _encode_on(self, lane: _EncLane, rgb: Any, quality: int, ready: Any) -> bytes:
        lib = self.lib
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        s = ctypes.c_void_p(lane.stream.ptr)
        try:
            lane.stream.wait_event(ready)
            _check(lib.nvjpegEncoderParamsSetQuality(lane.params, quality, s), 'SetQuality')
            img = _Image()
            img.channel[0] = rgb.data.ptr
            img.pitch[0] = w * 3
            _check(
                lib.nvjpegEncodeImage(
                    self.handle, lane.state, lane.params, ctypes.byref(img), _INPUT_RGBI, w, h, s
                ),
                'EncodeImage',
            )
            n = ctypes.c_size_t()
            _check(
                lib.nvjpegEncodeRetrieveBitstream(
                    self.handle, lane.state, None, ctypes.byref(n), s
                ),
                'RetrieveBitstream(length)',
            )
            buf = (ctypes.c_ubyte * n.value)()
            _check(
                lib.nvjpegEncodeRetrieveBitstream(self.handle, lane.state, buf, ctypes.byref(n), s),
                'RetrieveBitstream',
            )
            lane.stream.synchronize()
        except BaseException:
            lane.stream.synchronize()  # the caller frees rgb next: nothing may still read it
            raise
        return bytes(buf[: n.value])


__all__ = ['NvJpegEncoders', 'NvJpegError']
