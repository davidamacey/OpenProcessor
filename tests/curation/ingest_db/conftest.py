"""The ingest-run, ingest-source and dispatcher tests share the routes-over-PostgreSQL app."""

from fixtures.job_fixtures import job_env, kinds, producer, wakes

from curation.ingest_db.app_fixtures import app, client, served_root, store


__all__ = ['app', 'client', 'job_env', 'kinds', 'producer', 'served_root', 'store', 'wakes']
