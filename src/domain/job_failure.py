"""A job handler's way to end ``failed`` with a stable code the client can act on."""

from __future__ import annotations


class JobFailedError(RuntimeError):
    """Raised by a handler to fail its job. ``code`` is served as the job's ``error_code`` and
    the message as its ``error_message`` (any other exception is ``handler_error``)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
