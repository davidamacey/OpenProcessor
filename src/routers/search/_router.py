"""Shared APIRouter for the /search endpoints.

Lives in its own module so each endpoint module can register routes on it
without importing the package ``__init__``.
"""

from fastapi import APIRouter


router = APIRouter(
    prefix='/search',
    tags=['Visual Search'],
)
