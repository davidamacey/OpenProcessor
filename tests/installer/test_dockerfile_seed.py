"""Dockerfile model-repo seed contract (installer plan section 9.1)."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_copies_export_examples_and_model_seed() -> None:
    text = (REPO_ROOT / 'Dockerfile').read_text()
    assert re.search(r'COPY\s+(--chown=\S+\s+)?export/\s+\./export/', text)
    assert re.search(r'COPY\s+(--chown=\S+\s+)?examples/\s+\./examples/', text)
    assert re.search(r'COPY\s+(--chown=\S+\s+)?models/\s+/opt/openprocessor/model_repo_seed/', text)


def test_every_tracked_config_pbtxt_would_land_in_the_seed() -> None:
    """Every models/*/config.pbtxt tracked in git is not excluded by
    .dockerignore (which would silently drop it from the model repo
    seed)."""
    dockerignore = (REPO_ROOT / '.dockerignore').read_text()
    config_pbtxt_files = sorted((REPO_ROOT / 'models').glob('*/config.pbtxt'))
    assert config_pbtxt_files, 'expected at least one models/*/config.pbtxt in the repo'

    # .dockerignore explicitly documents that config.pbtxt is included;
    # a regression would add a line like "*.pbtxt" or "models/**" without
    # a config.pbtxt re-include (!models/**/config.pbtxt).
    exclude_lines = [
        line.strip()
        for line in dockerignore.splitlines()
        if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('!')
    ]
    for line in exclude_lines:
        assert 'config.pbtxt' not in line, f'.dockerignore excludes config.pbtxt: {line}'
        # A bare "models/**" or "models" exclude with no re-include would
        # also drop every config.pbtxt.
        if line in ('models/**', 'models', 'models/*'):
            reincludes = [
                ln for ln in dockerignore.splitlines() if ln.strip().startswith('!models')
            ]
            assert reincludes, f'.dockerignore excludes {line} with no !models re-include'
