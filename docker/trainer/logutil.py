"""Logger factory shared by every trainer-container module.

The trainer image installs ``structlog`` so its logs match the API's
structured style. The stdlib fallback exists so these modules stay
importable -- and unit-testable -- on a host without it.
"""

from __future__ import annotations

import logging
from typing import Any


class _StdlibKwargLogger:
    """Wrap a stdlib logger so it accepts structlog-style kwargs."""

    def __init__(self, inner: logging.Logger) -> None:
        self._inner = inner

    def _fmt(self, msg: str, **kwargs: Any) -> str:
        if not kwargs:
            return msg
        extras = ' '.join(f'{k}={v!r}' for k, v in kwargs.items())
        return f'{msg} {extras}'

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._inner.debug(self._fmt(msg, **kwargs))

    def info(self, msg: str, **kwargs: Any) -> None:
        self._inner.info(self._fmt(msg, **kwargs))

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._inner.warning(self._fmt(msg, **kwargs))

    def error(self, msg: str, **kwargs: Any) -> None:
        self._inner.error(self._fmt(msg, **kwargs))

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._inner.exception(self._fmt(msg, **kwargs))


try:
    import structlog

    def get_logger(name: str) -> Any:
        return structlog.get_logger(name)

except ImportError:

    def get_logger(name: str) -> Any:
        return _StdlibKwargLogger(logging.getLogger(name))


__all__ = ['get_logger']
