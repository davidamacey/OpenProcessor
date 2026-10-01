"""W10 fix-pass regression: unexclusion_update's legacy
``excluded_prior_class_validated``-absent branch must ask "was this a
human write" (``is_human_marker``), not "is this locked"
(``is_locked_class``) -- the two are not the same claim.

``is_locked_class`` is broader than "a human validated this": it is
also true for ``test_holdout`` items (regardless of who set the class)
and for validated imports. Using it here would incorrectly restore an
un-excluded ``test_holdout`` item with a machine-set class as
``class_validated=True``, contradicting the legacy branch's actual
intent (Opus review 2026-09-28, lock-rule call-site m1/exclusion.py:93).
"""

from __future__ import annotations

from src.services.curation.exclusion import EXCLUDED_CLUSTER_ID, unexclusion_update


def _excluded_doc(**extra):
    return {
        'class_excluded': True,
        'class_id': 7,
        'cluster_id': EXCLUDED_CLUSTER_ID,
        **extra,
    }


def test_legacy_holdout_with_machine_class_is_not_restored_as_validated():
    """No PRIOR_VALIDATED recorded (legacy doc), test_holdout set, class
    from a machine writer -- must NOT come back validated."""
    doc = _excluded_doc(class_source='vlm', test_holdout=True)
    update = unexclusion_update(doc, now='2026-09-28T00:00:00+00:00')
    assert update['class_validated'] is False


def test_legacy_human_class_is_restored_as_validated():
    """No PRIOR_VALIDATED recorded, but the class IS human-set -- the
    legacy branch's actual intended case."""
    doc = _excluded_doc(class_source='human')
    update = unexclusion_update(doc, now='2026-09-28T00:00:00+00:00')
    assert update['class_validated'] is True


def test_legacy_validated_import_is_not_restored_as_validated():
    """A validated import is locked (is_locked_class == True) but was
    never a human write -- must not be restored as validated either."""
    doc = _excluded_doc(class_source='external_label', class_validated=True)
    update = unexclusion_update(doc, now='2026-09-28T00:00:00+00:00')
    assert update['class_validated'] is False


def test_recorded_prior_validated_still_wins_over_the_legacy_branch():
    doc = _excluded_doc(class_source='vlm', excluded_prior_class_validated=True, test_holdout=False)
    update = unexclusion_update(doc, now='2026-09-28T00:00:00+00:00')
    assert update['class_validated'] is True
