"""A lost CUDA context: recognise it, refuse work at once, and let the supervisor start a new process.

CUDA runtime API, ``cudaError_t``: after an illegal address (700), a launch timeout (702), a device
assert (710), a hardware stack error (714), an illegal instruction (715), a misaligned address
(716), an invalid address space (717), an invalid program counter (718) or a launch failure (719),
"the process is left in an inconsistent state and any further CUDA work will return the same
error. To continue using CUDA, the process must be terminated and relaunched." CuPy raises these
as ``cupy_backends.cuda.api.runtime.CUDARuntimeError`` with the code in ``status``. nvJPEG answers
its own status codes in such a context, so a decoder failure is checked against the context
before it is called the photo's fault.

The engine then answers every request UNAVAILABLE (retryable; the API tries again and the photo
is served by the next process), reports not ready, and the entry point exits non-zero so the
container's restart policy (``restart: always`` in docker-compose.yml) starts a fresh process.
"""

from __future__ import annotations


#: ``cudaError_t`` codes after which the context cannot be used again (CUDA runtime API).
STICKY = frozenset({700, 702, 710, 714, 715, 716, 717, 718, 719})


class ContextLostError(Exception):
    """The CUDA context of this process is unusable: answered UNAVAILABLE until it restarts."""


def is_sticky(exc: BaseException) -> bool:
    return getattr(exc, 'status', None) in STICKY
