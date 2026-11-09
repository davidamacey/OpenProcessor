"""The ingest-run, ingest-source and dispatcher tests share the routes-over-PostgreSQL app."""

from curation.ingest_db.app_fixtures import app, client, served_root, store


__all__ = ['app', 'client', 'served_root', 'store']
