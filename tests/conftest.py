"""Pytest configuration for the test suite.

Some files under ``tests/`` are standalone smoke-test *scripts* meant to be
run against a live deployment (``python tests/test_full_system.py``), not
pytest suites. They define helper functions named ``test_*(name, ...)`` that
pytest would otherwise miscollect as test cases — failing with
``fixture 'name' not found``. Exclude them from collection here; run them
directly per the project README / CLAUDE.md.
"""

from __future__ import annotations


collect_ignore = [
    'test_full_system.py',
    'test_scrfd_pipeline.py',
    'test_validate_models.py',
    'validate_visual_results.py',
]
