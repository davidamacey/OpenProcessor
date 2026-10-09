"""Shared APIRouter for the /search endpoints.

Lives in its own module so each endpoint module can register routes on it
without importing the package ``__init__``.
"""

from fastapi import APIRouter
from fastapi.responses import ORJSONResponse


router = APIRouter(
    prefix='/search',
    tags=['Visual Search'],
    default_response_class=ORJSONResponse,
)
