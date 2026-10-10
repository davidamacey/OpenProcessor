"""JSON responses for hand-built wire dicts.

FastAPI serializes straight to JSON bytes through Pydantic when a route has a
response model. The curation item pages (review, crops, search, ...) cannot:
they return the dict ``wire.serialize_item`` builds, documented through
``responses=`` and declared ``response_model=None`` so a stored value of an
unexpected type never turns a whole page into a 500. For those routes
FastAPI would run ``jsonable_encoder`` over every nested value before
``json.dumps`` (about 70 ms for a 100-item review page, against under 1 ms
here), so :class:`WireRoute` renders them directly with orjson.

The byte output is the one FastAPI's former orjson response class produced after
``jsonable_encoder``: orjson handles the JSON-native values (including
numpy arrays, non-string keys, datetimes, enums, UUIDs and dataclasses) and
hands every other type (``set``, ``Path``, ``Decimal``, Pydantic models, ...)
to ``jsonable_encoder``, so those still serialize.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any

import orjson
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute


if TYPE_CHECKING:
    from collections.abc import Callable

_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def dumps_wire(content: Any) -> bytes:
    """Serialize ``content`` to compact UTF-8 JSON bytes (NaN becomes ``null``)."""
    return orjson.dumps(content, default=jsonable_encoder, option=_OPTIONS)


class WireJSONResponse(JSONResponse):
    """``JSONResponse`` rendered by orjson instead of the stdlib encoder."""

    def render(self, content: Any) -> bytes:
        return dumps_wire(content)


def _respond_directly(endpoint: Callable[..., Any], status_code: int | None) -> Callable[..., Any]:
    """Wrap ``endpoint`` so a returned dict/list becomes a :class:`WireJSONResponse`.

    Anything else (a ``Response``, a model, ``None``) takes FastAPI's usual path.
    """

    def wrap(result: Any) -> Any:
        if not isinstance(result, dict | list):
            return result
        if status_code is None:
            return WireJSONResponse(result)
        return WireJSONResponse(result, status_code=status_code)

    if inspect.iscoroutinefunction(endpoint):

        @functools.wraps(endpoint)
        async def async_endpoint(*args: Any, **kwargs: Any) -> Any:
            return wrap(await endpoint(*args, **kwargs))

        return async_endpoint

    @functools.wraps(endpoint)
    def sync_endpoint(*args: Any, **kwargs: Any) -> Any:
        return wrap(endpoint(*args, **kwargs))

    return sync_endpoint


class WireRoute(APIRoute):
    """Route class for routers whose ``response_model=None`` routes return wire dicts.

    A route declared with an explicit ``response_model=None`` skips FastAPI's
    ``jsonable_encoder`` pass and answers with :class:`WireJSONResponse`;
    every other route is an ordinary ``APIRoute``. Handlers of such routes
    must not rely on an injected ``Response`` (its headers would be dropped)
    or on ``BackgroundTasks``.
    """

    def __init__(self, path: str, endpoint: Callable[..., Any], **kwargs: Any) -> None:
        if 'response_model' in kwargs and kwargs['response_model'] is None:
            endpoint = _respond_directly(endpoint, kwargs.get('status_code'))
        super().__init__(path, endpoint, **kwargs)
