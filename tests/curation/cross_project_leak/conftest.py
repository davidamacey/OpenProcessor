"""Registers the three-project ``leak_env`` fixture for the isolation tests in this package."""

from fixtures.job_fixtures import FakeProducer, kinds, producer, wakes  # noqa: F401  # fixture registration

from curation.cross_project_leak._env import leak_env  # noqa: F401  # fixture registration
