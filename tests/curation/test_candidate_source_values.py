"""S3: stored ``region_source`` / ``candidate_source`` provenance values.

The reference deployment's vendor names (``sam3``, ``lpr``) must never be
the *stored* candidate-source vocabulary — see
``docs/design/naming_sweep_plan.md`` S3. The worker's four write sites and
``candidate_detector()``'s lookup table must agree on the generic names.
"""

from __future__ import annotations

from scripts.curation.worker.region_text_stage import candidate_detector
from scripts.curation.worker.state import (
    CANDIDATE_DETECTOR,
    CANDIDATE_DETECTOR_EXISTING,
    CANDIDATE_SEGMENTER,
    CANDIDATE_SEGMENTER_TEXT_HINT,
    _ItemTask,
)


def test_candidate_source_constants_are_generic() -> None:
    assert CANDIDATE_SEGMENTER == 'segmenter'
    assert CANDIDATE_SEGMENTER_TEXT_HINT == 'segmenter_text_hint'
    assert CANDIDATE_DETECTOR == 'detector'
    assert CANDIDATE_DETECTOR_EXISTING == 'detector_existing'
    # The old vendor-named values must never leak back in.
    assert {
        CANDIDATE_SEGMENTER,
        CANDIDATE_SEGMENTER_TEXT_HINT,
        CANDIDATE_DETECTOR,
        CANDIDATE_DETECTOR_EXISTING,
    }.isdisjoint({'sam3', 'sam3_text_hint', 'lpr', 'lpr_existing'})


def test_candidate_detector_lookup_uses_generic_keys(monkeypatch) -> None:
    from src.services.detection.reference_profiles import REFERENCE_LICENSE_PLATE_PROFILE

    task = _ItemTask.__new__(_ItemTask)  # bypass required-field construction

    for source, expected in (
        (
            CANDIDATE_SEGMENTER,
            (
                REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name,
                REFERENCE_LICENSE_PLATE_PROFILE.segmenter_version,
            ),
        ),
        (
            CANDIDATE_SEGMENTER_TEXT_HINT,
            (
                REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name,
                REFERENCE_LICENSE_PLATE_PROFILE.segmenter_version,
            ),
        ),
        (
            CANDIDATE_DETECTOR,
            (
                REFERENCE_LICENSE_PLATE_PROFILE.detector_model,
                REFERENCE_LICENSE_PLATE_PROFILE.detector_version,
            ),
        ),
        (
            CANDIDATE_DETECTOR_EXISTING,
            (
                REFERENCE_LICENSE_PLATE_PROFILE.detector_model,
                REFERENCE_LICENSE_PLATE_PROFILE.detector_version,
            ),
        ),
    ):
        task.candidate_source = source
        assert candidate_detector(task, REFERENCE_LICENSE_PLATE_PROFILE) == expected

    task.candidate_source = 'sam3'  # a retired value must not resolve specially anymore
    assert candidate_detector(task, REFERENCE_LICENSE_PLATE_PROFILE) == ('sam3', '1')
