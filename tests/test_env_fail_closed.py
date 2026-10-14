"""A malformed or out-of-range environment value refuses the read and names the variable.

Before the typed env module most of these fell back to the default (or read a
mistyped boolean as false), so a typo in ``.env`` ran on a setting nobody chose.
"""

import pytest

from src.config import CurationConfig, DetectionProfile
from src.config.env import EnvError
from src.services.curation import strategy_flags
from src.services.curation.dataset_import import limits


@pytest.mark.parametrize(
    ('var', 'value', 'read'),
    [
        ('OP_DATASET_IMPORT_CHUNK', 'abc', limits.import_chunk_size),
        ('OP_DATASET_IMPORT_CHUNK', '0', limits.import_chunk_size),
        ('OP_OPEN_VOCAB_SWEEP_S', '-1', limits.open_vocab_sweep_interval_s),
        ('OP_SCORES_ENABLED', 'ture', strategy_flags._scores_enabled),
    ],
)
def test_runtime_reader_refuses_a_bad_value(monkeypatch, var, value, read):
    monkeypatch.setenv(var, value)
    with pytest.raises(EnvError, match=var):
        read()


def test_good_and_blank_values_still_read(monkeypatch):
    monkeypatch.setenv('OP_DATASET_IMPORT_CHUNK', '128')
    assert limits.import_chunk_size() == 128
    monkeypatch.setenv('OP_DATASET_IMPORT_CHUNK', '')
    assert limits.import_chunk_size() == 64
    monkeypatch.setenv('OP_SCORES_ENABLED', 'Yes')
    assert strategy_flags._scores_enabled() is True


def test_curation_config_refuses_a_malformed_number(monkeypatch):
    monkeypatch.setenv('OP_EXPORT_KEEP_LAST', 'many')
    with pytest.raises(EnvError, match='OP_EXPORT_KEEP_LAST'):
        CurationConfig.from_env()


@pytest.mark.parametrize(
    ('field', 'value'),
    [('CONFIDENCE_FLOOR', 'high'), ('BATCH_LIMIT', '1.5'), ('LETTERBOX_FILL', '1,x,3')],
)
def test_detection_profile_refuses_a_malformed_override(monkeypatch, field, value):
    var = f'OP_TEST_PROFILE_{field}'
    monkeypatch.setenv(var, value)
    with pytest.raises(EnvError, match=var):
        DetectionProfile.from_env('OP_TEST_PROFILE_', name='p')
