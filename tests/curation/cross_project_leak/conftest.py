"""Registers the three-project ``leak_env`` fixture for the isolation tests in this package."""

from curation.cross_project_leak._env import leak_env  # noqa: F401  # fixture registration
