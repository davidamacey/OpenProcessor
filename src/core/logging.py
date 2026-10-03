"""
Structured logging configuration for the Visual AI API.

Provides:
- JSON-formatted logs for production
- Request correlation IDs (X-Request-ID)
- Structured context in all log messages
- Better error messages with full context
"""

import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

import structlog


# Request-scoped correlation ID. Set by the HTTP middleware in
# ``src.main`` (FastAPI) and by worker processes per task. Lives here
# (not in src.main) so service-layer modules can import it without
# pulling the FastAPI app at import time — background workers run
# outside the FastAPI process and would otherwise be unable to bind
# the ID.
request_id_ctx: ContextVar[str] = ContextVar('request_id', default='-')


def get_request_id() -> str:
    """Return the current request ID, or ``'-'`` when unbound."""
    return request_id_ctx.get()


def bind_request_id(request_id: str) -> None:
    """Bind ``request_id`` to both the ContextVar and structlog contextvars.

    Both bindings are needed: the ContextVar drives :func:`get_request_id`
    callers, and the structlog contextvars binding makes every subsequent
    ``logger.info(...)`` call on this asyncio task auto-attach
    ``request_id=<id>`` to its event dict (via the
    :func:`structlog.contextvars.merge_contextvars` processor wired into
    the chain below).
    """
    request_id_ctx.set(request_id)
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_id() -> None:
    """Reset both ContextVar + structlog contextvars to defaults."""
    request_id_ctx.set('-')
    structlog.contextvars.unbind_contextvars('request_id')


_OUTAGE_LOGGERS = ('opensearch', 'elastic_transport')


class OutageLogFilter(logging.Filter):
    """Turn the OpenSearch client's per-request connection failures (a full
    traceback each) into one backed-off warning line: while OpenSearch is down
    every poll of every worker would otherwise print one.

    The first failure is logged at once, then at most one more after 1 s, 2 s,
    4 s ... up to ``max_backoff_s``; after a quiet stretch it starts over.
    """

    def __init__(self, *, max_backoff_s: float = 60.0, clock: Any = time.monotonic) -> None:
        super().__init__()
        self._max = max_backoff_s
        self._clock = clock
        self._delay = 0.0
        self._next_at = 0.0

    @staticmethod
    def _is_outage(exc: BaseException | None) -> bool:
        from opensearchpy.exceptions import ConnectionError as OsConnectionError

        while exc is not None:
            if isinstance(exc, OsConnectionError | ConnectionError):
                return True
            exc = exc.__cause__ or exc.__context__
        return False

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith(_OUTAGE_LOGGERS) or not record.exc_info:
            return True
        if not self._is_outage(record.exc_info[1]):
            return True
        now = self._clock()
        if now < self._next_at:
            return False
        quiet = now - self._next_at > self._max * 2
        self._delay = 1.0 if quiet or not self._delay else min(self._delay * 2, self._max)
        self._next_at = now + self._delay
        record.msg = (
            f'opensearch unreachable ({type(record.exc_info[1]).__name__}); '
            f'retrying, repeats suppressed for {self._delay:.0f}s'
        )
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        return True


def _install_outage_filter() -> None:
    """On the client's own loggers, so it holds for every process that imports
    this module (the API and each worker), configured or not."""
    outage = OutageLogFilter()
    for name in _OUTAGE_LOGGERS:
        logging.getLogger(name).addFilter(outage)


_install_outage_filter()


def configure_logging(json_logs: bool = True, log_level: str = 'INFO') -> None:
    """
    Configure structured logging for the application.

    Args:
        json_logs: If True, output JSON logs. If False, use console format.
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Configure standard library logging
    logging.basicConfig(
        format='%(message)s',
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Shared pre-chain processors. These run for BOTH structlog calls and
    # foreign (stdlib logging) records before the final formatter renders.
    # IMPORTANT: do NOT include ProcessorFormatter.wrap_for_formatter here --
    # that processor wraps event_dict in a (args, kwargs, event_dict) tuple
    # and is only valid as the LAST step of a structlog-native chain. Putting
    # it in foreign_pre_chain triggers `'tuple' object does not support item
    # deletion` in ProcessorFormatter.format() when stdlib logs (uvicorn,
    # opensearch-py, etc.) flow through.
    shared_processors: list[Any] = [
        # ``merge_contextvars`` MUST be first so request_id (and any
        # other future contextvar) lands in the event dict before any
        # other processor inspects it.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Configure structlog: shared processors, then the wrap_for_formatter
    # bridge that hands control back to stdlib logging's formatter.
    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure formatter for stdlib logging
    if json_logs:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=shared_processors,
        )

    # Apply formatter to root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))
    # The client logs every request at INFO (the per-project config_revision
    # poll was ~64% of all API lines); failures are WARNING and still surface.
    for name in _OUTAGE_LOGGERS:
        logging.getLogger(name).setLevel(max(root_logger.level, logging.WARNING))


def get_logger(name: str) -> Any:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger with context binding support
    """
    return structlog.get_logger(name)
