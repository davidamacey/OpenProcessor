"""Registers the three-project ``leak_env`` fixture for the isolation tests in this package."""

from fixtures.job_fixtures import kinds, producer  # noqa: F401  # fixture registration: a DELETE queues its job on a fake producer

from curation.cross_project_leak._env import leak_env  # noqa: F401  # fixture registration
