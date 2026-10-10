"""Shared APIRouter for the /ingest endpoints.

Lives in its own module so each endpoint module can register routes on it
without importing the package ``__init__``.
"""

from fastapi import APIRouter


router = APIRouter(
    prefix='/ingest',
    tags=['Data Ingestion'],
)
