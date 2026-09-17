"""Tests for the class/region history helpers."""

from __future__ import annotations

import pytest

from src.services.curation.history import (
    MAX_HISTORY_ENTRIES,
    MAX_PLATE_CHAIN_ENTRIES,
    append_plate_chain_entry,
    record_class_history,
)


class TestRecordClassHistory:
    def test_no_class_id_returns_existing_unchanged(self):
        # First labeling of a brand-new crop: nothing to preserve.
        src: dict[str, object] = {}
        result = record_class_history(src, writer='ingest')
        assert result == []

    def test_no_class_id_preserves_existing_history(self):
        # Edge case: somehow history exists but class_id was cleared.
        # Don't lose history; just don't append.
        src = {
            'class_id': None,
            'class_id_history': [{'class_id': 5, 'writer': 'vlm'}],
        }
        result = record_class_history(src, writer='ingest')
        assert result == [{'class_id': 5, 'writer': 'vlm'}]

    def test_appends_entry_with_current_state(self):
        src = {
            'class_id': 47,
            'class_name': 'pickup_truck',
            'class_source': 'v6_model',
            'label_source': '',
            'confidence': 0.91,
        }
        result = record_class_history(src, writer='ingest', now='2026-05-15T00:00:00+00:00')
        assert len(result) == 1
        entry = result[0]
        assert entry['class_id'] == 47
        assert entry['class_name'] == 'pickup_truck'
        assert entry['class_source'] == 'v6_model'
        assert entry['confidence'] == 0.91
        assert entry['writer'] == 'ingest'
        assert entry['at'] == '2026-05-15T00:00:00+00:00'

    def test_appends_to_existing_history(self):
        src = {
            'class_id': 47,
            'class_name': 'pickup_truck',
            'class_source': 'vlm',
            'class_id_history': [
                {'class_id': 47, 'class_source': 'v6_model', 'writer': 'ingest'},
            ],
        }
        result = record_class_history(src, writer='vlm_pipeline')
        assert len(result) == 2
        assert result[0]['class_source'] == 'v6_model'
        assert result[1]['class_source'] == 'vlm'
        assert result[1]['writer'] == 'vlm_pipeline'

    def test_caps_at_max_entries(self):
        # Pre-populate at the cap, then append one more.
        history = [
            {'class_id': i, 'class_source': 'vlm', 'writer': f'w{i}'}
            for i in range(MAX_HISTORY_ENTRIES)
        ]
        src = {
            'class_id': 99,
            'class_source': 'human',
            'class_id_history': history,
        }
        result = record_class_history(src, writer='human')
        assert len(result) == MAX_HISTORY_ENTRIES
        # Newest entry is at the tail.
        assert result[-1]['class_id'] == 99
        # Oldest dropped (since no seed_backfill stub).
        assert result[0]['class_id'] == 1

    def test_caps_preserves_seed_backfill_origin(self):
        history = [{'class_id': -1, 'writer': 'seed_backfill'}]
        history += [
            {'class_id': i, 'class_source': 'vlm', 'writer': f'w{i}'}
            for i in range(MAX_HISTORY_ENTRIES)
        ]
        src = {
            'class_id': 99,
            'class_source': 'human',
            'class_id_history': history,
        }
        result = record_class_history(src, writer='human')
        assert len(result) == MAX_HISTORY_ENTRIES
        # Seed stays at index 0.
        assert result[0]['writer'] == 'seed_backfill'
        # Newest is at the tail.
        assert result[-1]['class_id'] == 99

    # =========================================================================
    # Dedupe — skip the append when class_id AND class_source are both
    # unchanged from the last recorded entry.
    # =========================================================================

    def test_no_append_when_class_unchanged(self):
        # Call twice with the same class_id/class_source. Before the fix,
        # the second call always appends, producing length 2.
        src = {'class_id': 47, 'class_source': 'v6_model', 'class_id_history': []}
        history_after_first = record_class_history(src, writer='ingest')
        assert len(history_after_first) == 1

        src_second_call = dict(src)
        src_second_call['class_id_history'] = history_after_first
        history_after_second = record_class_history(src_second_call, writer='ingest')
        assert len(history_after_second) == 1
        assert history_after_second == history_after_first

    def test_append_when_source_changes_but_class_does_not(self):
        # The inverse of test_no_append_when_class_unchanged: class_id is
        # the same but class_source changed — must still append.
        src = {'class_id': 47, 'class_source': 'v6_model', 'class_id_history': []}
        history_after_first = record_class_history(src, writer='ingest')
        assert len(history_after_first) == 1

        src_second_call = dict(src)
        src_second_call['class_source'] = 'vlm'
        src_second_call['class_id_history'] = history_after_first
        history_after_second = record_class_history(src_second_call, writer='vlm_pipeline')
        assert len(history_after_second) == 2
        assert history_after_second[0]['class_source'] == 'v6_model'
        assert history_after_second[1]['class_source'] == 'vlm'

    def test_append_when_class_changes_but_source_does_not(self):
        # Same class_source, different class_id — must still append.
        src = {'class_id': 47, 'class_source': 'human', 'class_id_history': []}
        history_after_first = record_class_history(src, writer='human:label_crop')
        src_second_call = dict(src)
        src_second_call['class_id'] = 12
        src_second_call['class_id_history'] = history_after_first
        history_after_second = record_class_history(src_second_call, writer='human:label_crop')
        assert len(history_after_second) == 2
        assert history_after_second[0]['class_id'] == 47
        assert history_after_second[1]['class_id'] == 12

    # =========================================================================
    # No cap once class_validated=true.
    # =========================================================================

    def test_validated_crop_history_is_not_truncated_at_32(self):
        history = [
            {'class_id': i, 'class_source': 'vlm', 'writer': f'w{i}'}
            for i in range(MAX_HISTORY_ENTRIES)
        ]
        src = {
            'class_id': 99,
            'class_source': 'human',
            'class_validated': True,
            'class_id_history': history,
        }
        result = record_class_history(src, writer='human:label_crop')
        # Uncapped: MAX_HISTORY_ENTRIES existing entries + 1 new one.
        assert len(result) == MAX_HISTORY_ENTRIES + 1
        assert result[0]['class_id'] == 0
        assert result[-1]['class_id'] == 99

    def test_unvalidated_crop_history_still_capped(self):
        # Non-regression: the cap still applies when class_validated is
        # falsy/absent — only the validated cohort is exempt.
        history = [
            {'class_id': i, 'class_source': 'vlm', 'writer': f'w{i}'}
            for i in range(MAX_HISTORY_ENTRIES)
        ]
        src = {
            'class_id': 99,
            'class_source': 'vlm',
            'class_validated': False,
            'class_id_history': history,
        }
        result = record_class_history(src, writer='vlm_pipeline')
        assert len(result) == MAX_HISTORY_ENTRIES


class TestAppendPlateChainEntry:
    def test_appends_to_empty(self):
        result = append_plate_chain_entry(
            None,
            detector='primary_detector',
            detector_version='1',
            outcome='hit',
        )
        assert len(result) == 1
        assert result[0].startswith('primary_detector:1:hit@')

    def test_appends_to_existing(self):
        chain = ['primary_detector:1:miss@2026-05-15T00:00:00+00:00']
        result = append_plate_chain_entry(
            chain,
            detector='secondary_detector',
            detector_version='2',
            outcome='hit',
        )
        assert len(result) == 2
        assert result[-1].startswith('secondary_detector:2:hit@')

    def test_caps_at_max(self):
        chain = [
            f'primary_detector:{i}:hit@2026-05-15T00:00:00+00:00'
            for i in range(MAX_PLATE_CHAIN_ENTRIES)
        ]
        result = append_plate_chain_entry(
            chain,
            detector='secondary_detector',
            detector_version='1',
            outcome='hit',
        )
        assert len(result) == MAX_PLATE_CHAIN_ENTRIES
        # Newest at tail.
        assert result[-1].startswith('secondary_detector:1:hit@')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
