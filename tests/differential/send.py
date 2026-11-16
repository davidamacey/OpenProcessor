"""Sending one case to one stack, and checking the answer against the committed schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from schemathesis.checks import CHECKS, load_all_checks
from schemathesis.core.failures import FailureGroup

from .compare import Observed


if TYPE_CHECKING:
    from .cases import DiffCase

SCHEMA_CHECK = 'response_schema_conformance'
EVENT_END = (b'\n\n', b'\r\n\r\n')


@dataclass(frozen=True)
class Sent:
    """What one stack answered: the compared parts, the raw response when whole, or the error."""

    observed: Observed | None
    response: httpx.Response | None
    error: str | None = None

    def asdict(self) -> dict[str, Any]:
        if self.observed is None:
            return {'error': self.error}
        return {
            'status': self.observed.status,
            'media_type': self.observed.media_type,
            'bytes': len(self.observed.body),
        }


def send(
    client: httpx.Client,
    case: DiffCase,
    *,
    event_stream: bool,
    stream_limit: int,
    stream_seconds: float,
) -> Sent:
    """Send ``case``; a transport failure is returned, never raised, so one stack cannot end the run."""
    try:
        if event_stream:
            request = client.build_request(
                case.method, case.path, timeout=stream_seconds, **case.request_kwargs()
            )
            return read_event_stream(client, request, stream_limit)
        request = client.build_request(case.method, case.path, **case.request_kwargs())
        response = client.send(request)
    except (httpx.HTTPError, UnicodeError, ValueError) as error:
        return Sent(None, None, f'{type(error).__name__}: {error}')
    return Sent(Observed.of(response.status_code, response.headers, response.content), response)


def read_event_stream(client: httpx.Client, request: httpx.Request, limit: int) -> Sent:
    """An event stream never ends: keep what arrives up to its first event, or ``limit`` bytes.

    The read timeout of the client ends a stream that sends nothing; what arrived by then is the body.
    """
    response = client.send(request, stream=True)
    received = b''
    try:
        for chunk in response.iter_raw():
            received += chunk
            if len(received) >= limit or any(end in received for end in EVENT_END):
                break
    except httpx.ReadTimeout:
        pass
    finally:
        response.close()
    return Sent(Observed.of(response.status_code, response.headers, received), None)


def schema_failures(case: DiffCase, response: httpx.Response) -> list[dict[str, Any]]:
    """Where the answer departs from the committed schema, by operation and schema path.

    Runs only Schemathesis's ``response_schema_conformance`` check, by passing it as the explicit
    ``checks`` list of ``Case.validate_response`` (the other built-in checks stay off).
    """
    load_all_checks()
    try:
        case.source.validate_response(response, checks=[CHECKS.get_one(SCHEMA_CHECK)])  # type: ignore[list-item]
    except FailureGroup as group:
        return [describe(failure) for failure in group.exceptions]
    return []


def describe(failure: Any) -> dict[str, Any]:
    return {
        'check': SCHEMA_CHECK,
        'kind': type(failure).__name__,
        'operation': failure.operation,
        'title': failure.title,
        'schema_path': [str(p) for p in getattr(failure, 'schema_path', [])],
        'instance_path': [str(p) for p in getattr(failure, 'instance_path', [])],
        'message': str(getattr(failure, 'validation_message', '') or failure.message).splitlines()[
            0
        ],
    }
