"""nvJPEG decode lanes through ctypes: the decoupled API with the GPU-Huffman backend.

nvImageCodec's ``GPU_ONLY`` backend decodes nothing on a card without a hardware JPEG engine
(the RTX A6000), and its hybrid backend runs the Huffman stage on the CPU (134 ms per 18 MP photo).
nvJPEG's own ``NVJPEG_BACKEND_GPU_HYBRID`` runs the Huffman stage on the GPU; reached through the
decoupled API (parse, ``DecodeJpegHost``, ``TransferToDevice``, ``DecodeJpegDevice``) it decodes
single images. Each lane owns what nvJPEG requires per thread: a decoder state, a pinned and a
device buffer, a jpeg stream, decode parameters and a CUDA stream; the handle and the decoders are
shared. Decoding threads check lanes out of a bounded pool (ctypes releases the GIL during the
native calls, so threads overlap). The output is an HWC uint8 CuPy array (``RGBI``) with
libjpeg-style chroma interpolation (``NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION``), so 4:2:0
photos decode within a grey level of Pillow.

nvJPEG ignores EXIF orientation: the caller re-orients. A stream nvJPEG refuses or fails on raises
:class:`NvJpegError`; the caller reports it and the API decodes that image on the CPU.
"""

from __future__ import annotations

import ctypes
import queue
from typing import Any

import cupy as cp


_BACKEND = {'hybrid': 1, 'gpu_hybrid': 2}
_OUTPUT_RGBI = 5
_FLAG_INTERPOLATE = 1 << 5


class _Image(ctypes.Structure):
    _fields_ = [('channel', ctypes.c_void_p * 4), ('pitch', ctypes.c_size_t * 4)]


class NvJpegError(RuntimeError):
    def __init__(self, call: str, status: int) -> None:
        super().__init__(f'{call} returned nvjpegStatus {status}')
        self.status = status


def _check(status: int, call: str) -> None:
    if status != 0:
        raise NvJpegError(call, status)


class _Lane:
    def __init__(
        self, lib: Any, handle: ctypes.c_void_p, decoders: dict[str, ctypes.c_void_p]
    ) -> None:
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.states: dict[str, ctypes.c_void_p] = {}
        for name, dec in decoders.items():
            st = ctypes.c_void_p()
            _check(
                lib.nvjpegDecoderStateCreate(handle, dec, ctypes.byref(st)), 'DecoderStateCreate'
            )
            pinned, device = ctypes.c_void_p(), ctypes.c_void_p()
            _check(
                lib.nvjpegBufferPinnedCreate(handle, None, ctypes.byref(pinned)),
                'BufferPinnedCreate',
            )
            _check(
                lib.nvjpegBufferDeviceCreate(handle, None, ctypes.byref(device)),
                'BufferDeviceCreate',
            )
            _check(lib.nvjpegStateAttachPinnedBuffer(st, pinned), 'AttachPinned')
            _check(lib.nvjpegStateAttachDeviceBuffer(st, device), 'AttachDevice')
            self.states[name] = st
        self.jstream = ctypes.c_void_p()
        _check(lib.nvjpegJpegStreamCreate(handle, ctypes.byref(self.jstream)), 'JpegStreamCreate')
        self.params = ctypes.c_void_p()
        _check(
            lib.nvjpegDecodeParamsCreate(handle, ctypes.byref(self.params)), 'DecodeParamsCreate'
        )
        _check(lib.nvjpegDecodeParamsSetOutputFormat(self.params, _OUTPUT_RGBI), 'SetOutputFormat')


class NvJpegLanes:
    """``decode(data) -> (frame, backend)``; ``frame`` is HWC uint8 on the current device."""

    def __init__(
        self,
        lanes: int,
        library: str = 'libnvjpeg.so.13',
        backends: tuple[str, ...] = ('gpu_hybrid', 'hybrid'),
    ) -> None:
        self.lib = ctypes.CDLL(library)
        self.handle = ctypes.c_void_p()
        _check(
            self.lib.nvjpegCreateEx(0, None, None, _FLAG_INTERPOLATE, ctypes.byref(self.handle)),
            'CreateEx',
        )
        self.backends = backends
        self.decoders: dict[str, ctypes.c_void_p] = {}
        for name in backends:
            d = ctypes.c_void_p()
            _check(
                self.lib.nvjpegDecoderCreate(self.handle, _BACKEND[name], ctypes.byref(d)),
                'DecoderCreate',
            )
            self.decoders[name] = d
        self.pool: queue.Queue[_Lane] = queue.Queue()
        for _ in range(lanes):
            self.pool.put(_Lane(self.lib, self.handle, self.decoders))

    def decode(self, data: bytes, alloc_stream: Any = None) -> tuple[Any, str]:
        """``alloc_stream`` is the stream whose memory-pool arena holds the frame; the default is
        the lane's own, so the frame cannot be reused by another worker."""
        lane = self.pool.get()
        try:
            return self._decode_on(lane, data, alloc_stream or lane.stream)
        finally:
            self.pool.put(lane)

    def _decode_on(self, lane: _Lane, data: bytes, alloc_stream: Any) -> tuple[Any, str]:
        lib = self.lib
        _check(
            lib.nvjpegJpegStreamParse(
                self.handle, ctypes.c_char_p(data), ctypes.c_size_t(len(data)), 0, 0, lane.jstream
            ),
            'JpegStreamParse',
        )
        w, h = ctypes.c_uint(), ctypes.c_uint()
        _check(
            lib.nvjpegJpegStreamGetFrameDimensions(lane.jstream, ctypes.byref(w), ctypes.byref(h)),
            'GetFrameDimensions',
        )
        for name in self.backends:
            dec = self.decoders[name]
            supported = ctypes.c_int(1)
            _check(
                lib.nvjpegDecoderJpegSupported(
                    dec, lane.jstream, lane.params, ctypes.byref(supported)
                ),
                'JpegSupported',
            )
            if supported.value != 0:
                continue
            state = lane.states[name]
            with alloc_stream:
                out = cp.empty((h.value, w.value, 3), dtype=cp.uint8)
            img = _Image()
            img.channel[0] = out.data.ptr
            img.pitch[0] = w.value * 3
            cs = ctypes.c_void_p(lane.stream.ptr)
            try:
                _check(
                    lib.nvjpegDecodeJpegHost(self.handle, dec, state, lane.params, lane.jstream),
                    'DecodeJpegHost',
                )
                _check(
                    lib.nvjpegDecodeJpegTransferToDevice(self.handle, dec, state, lane.jstream, cs),
                    'TransferToDevice',
                )
                _check(
                    lib.nvjpegDecodeJpegDevice(self.handle, dec, state, ctypes.byref(img), cs),
                    'DecodeJpegDevice',
                )
                lane.stream.synchronize()
            except BaseException:
                # the frame returns to a pool shared with other workers: nothing may still write it
                lane.stream.synchronize()
                raise
            return out, name
        raise NvJpegError('no nvJPEG backend supports this stream', -1)
