"""Every reader of the ingest detector resolves it through
``project_ingest_profiles``; only the resolver (and the deployment-wide unload
guard, which protects the env models of every project) read the env profile."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pytest


ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r'\bingest_(primary|secondary)_profile\(')
ALLOWED = {
    'src/config/ingest_profiles.py',
    'src/services/curation/ingest_detector.py',
    # Global routes call it with no project bound: it guards the env models.
    'src/services/model_unload_guard.py',
}
RESOLVER_CONSUMERS = [
    'src/services/curation/ingest_service_factory.py',
    'src/services/curation/class_sources.py',
    'src/services/curation/model_choices.py',
    'src/services/curation/region_vocabulary.py',
    'src/services/curation/empty_vlm_answer_repair.py',
    'src/services/training/promoted_models.py',
    'src/services/config_store/open_vocab_inputs.py',
    'src/routers/curation/ingest.py',
    'src/routers/curation/class_seed.py',
    # dataset import runs as a queued job; the job resolves the profile, the route no longer does
    'src/services/curation/dataset_import/job.py',
    'src/routers/curation/ingest_policy.py',
]


def test_nothing_outside_the_resolver_reads_the_env_profile() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / 'src').rglob('*.py'))
        if str(path.relative_to(ROOT)) not in ALLOWED and PATTERN.search(path.read_text())
    ]
    assert offenders == []


def test_the_known_consumers_still_go_through_the_resolver() -> None:
    # An empty scan above would also pass if the consumers moved; pin them.
    missing = [
        rel
        for rel in RESOLVER_CONSUMERS
        if not re.search(
            r'project_ingest_profiles|refreshed_ingest_profiles', (ROOT / rel).read_text()
        )
    ]
    assert missing == []


def test_the_secondary_classifier_source_is_listed_without_an_env_secondary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A project can run a secondary through its ingest profile while the
    deployment configures none; its ``secondary_model`` rows must still count as
    classifier output."""
    from src.domain.ingest_class_sources import classifier_class_sources

    monkeypatch.delenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', raising=False)
    assert 'secondary_model' in classifier_class_sources()
