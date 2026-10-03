"""The shipped docs and env template must not steer users to a vehicle-only detector."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'README.md',
    'INSTALLATION.md',
    'env.template',
    'docs/CURATION.md',
    'docs-site/docs/getting-started/quick-start.mdx',
    'docs-site/docs/guides/use-your-own-domain.mdx',
    'docs-site/docs/configuration/basic.mdx',
    'docs-site/docs/configuration/advanced.mdx',
]
# An assignment of the class-id filter to a value, active or commented out,
# e.g. ``OP_INGEST_PRIMARY_CLASS_IDS=2,3,5,7``.
_ASSIGNED = re.compile(r'OP_INGEST_PRIMARY_CLASS_IDS\s*=\s*[0-9]')


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text()


def test_no_doc_recommends_a_vehicle_class_filter() -> None:
    offenders = [rel for rel in FILES if _ASSIGNED.search(_read(rel))]
    assert offenders == [], f'these files still show a class-id filter value: {offenders}'


def test_no_doc_lists_the_vehicle_id_set() -> None:
    offenders = [rel for rel in FILES if '2,3,5,7' in _read(rel)]
    assert offenders == [], f'these files still recommend the vehicle ids: {offenders}'


def test_env_template_quick_block_leaves_filter_unset() -> None:
    text = _read('env.template')
    active = [line for line in text.splitlines() if line.startswith('OP_INGEST_PRIMARY_CLASS_IDS')]
    assert active == []


def test_quick_start_says_full_vocabulary_is_kept() -> None:
    text = _read('docs-site/docs/getting-started/quick-start.mdx')
    assert 'all 80 COCO classes' in text
    assert 'Switching detectors' in text
