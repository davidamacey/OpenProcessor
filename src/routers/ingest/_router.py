"""Shared APIRouter for the /ingest endpoints.

Lives in its own module so each endpoint module can register routes on it
without importing the package ``__init__``.
"""

from fastapi import APIRouter
from fastapi.responses import ORJSONResponse


router = APIRouter(
    prefix='/ingest',
    tags=['Data Ingestion'],
    default_response_class=ORJSONResponse,
)
