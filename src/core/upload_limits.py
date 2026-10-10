"""Upload size limits.

``MAX_FILE_SIZE_MB`` caps each uploaded file, not the request: a multi-file
route (``/ingest/upload`` takes up to 128 images) is bounded by the files it
carries. The body is refused early from its ``Content-Length`` only when it is
larger than the biggest request any route accepts
(:func:`request_exceeds_ceiling`); the per-file check runs once the form is
parsed (:func:`enforce_file_cap`). Both answer ``413`` with the same body,
naming the served cap.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from src.config import get_curation_config
from src.config.settings import get_settings
from src.core.logging import get_request_id


# Boundaries, part headers and the small form fields around the files.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
# `/batch` routes take whole photo libraries.
BATCH_CEILING_BYTES = 10 * 1024 * 1024 * 1024


class UploadTooLargeError(Exception):
    """An uploaded file is larger than ``MAX_FILE_SIZE_MB``."""


def too_large_response() -> JSONResponse:
    """The ``413`` answer: the message and the cap the server is serving."""
    cap_mb = get_settings().max_file_size_mb
    req_id = get_request_id()
    return JSONResponse(
        status_code=413,
        content={
            'detail': f'File too large. Maximum: {cap_mb}MB',
            'max_file_size_mb': cap_mb,
            'request_id': req_id,
        },
        headers={'X-Request-ID': req_id},
    )


async def upload_too_large_handler(_request: Request, _exc: UploadTooLargeError) -> JSONResponse:
    return too_large_response()


def request_exceeds_ceiling(request: Request) -> bool:
    """Whether the declared body is larger than any upload this server accepts.

    A multipart body may carry many files, so it is bounded by the larger of
    one file and the byte total the curation upload route accepts; anything
    else (a JSON body) by one file's cap.
    """
    declared = request.headers.get('content-length')
    if not declared or not declared.isdigit():
        return False
    cap = get_settings().max_file_size_bytes
    if '/batch' in request.url.path:
        ceiling = BATCH_CEILING_BYTES
    elif request.headers.get('content-type', '').startswith('multipart/form-data'):
        ceiling = max(cap, get_curation_config().upload_max_bytes_per_request)
        ceiling += MULTIPART_OVERHEAD_BYTES
    else:
        ceiling = cap
    return int(declared) > ceiling


async def enforce_file_cap(request: Request) -> None:
    """App-level dependency: refuse any uploaded file over ``MAX_FILE_SIZE_MB``.

    Runs after the form is parsed (FastAPI reads the body before it resolves
    dependencies), so ``request.form()`` is the cached parse, not a second one.
    """
    if not request.headers.get('content-type', '').startswith('multipart/form-data'):
        return
    cap = get_settings().max_file_size_bytes
    for _name, value in (await request.form()).multi_items():
        if isinstance(value, UploadFile) and (value.size or 0) > cap:
            raise UploadTooLargeError
