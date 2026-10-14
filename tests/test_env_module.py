"""``src/config/env.py``: unset or blank gives the default; malformed or out of range raises naming the variable."""

from pathlib import Path

import pytest

from src.config.env import (
    EnvError,
    env_bool,
    env_float,
    env_int,
    env_is_set,
    env_names_with_prefix,
    env_optional_path,
    env_optional_str,
    env_path,
    env_str,
)


VAR = 'OP_TEST_ENV_MODULE_VAR'


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)


@pytest.mark.parametrize('value', [None, '', '   '])
def test_unset_or_blank_gives_the_default(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(VAR, value)
    assert env_str(VAR, 'd') == 'd'
    assert env_optional_str(VAR) is None
    assert env_int(VAR, 7) == 7
    assert env_float(VAR, 1.5) == 1.5
    assert env_bool(VAR, True) is True
    assert env_path(VAR, '/x') == Path('/x')
    assert env_optional_path(VAR) is None


def test_values_are_stripped_and_typed(monkeypatch):
    monkeypatch.setenv(VAR, '  12 ')
    assert env_str(VAR) == '12'
    assert env_int(VAR, 0) == 12
    assert env_float(VAR, 0.0) == 12.0
    assert env_path(VAR, '/x') == Path('12')


@pytest.mark.parametrize('value', ['1', 'true', 'YES', ' On '])
def test_bool_true_spellings(monkeypatch, value):
    monkeypatch.setenv(VAR, value)
    assert env_bool(VAR) is True


@pytest.mark.parametrize('value', ['0', 'false', 'No', 'off'])
def test_bool_false_spellings(monkeypatch, value):
    monkeypatch.setenv(VAR, value)
    assert env_bool(VAR, True) is False


@pytest.mark.parametrize(
    'reader',
    [
        lambda: env_int(VAR, 1),
        lambda: env_float(VAR, 1.0),
        lambda: env_bool(VAR),
    ],
)
def test_malformed_raises_naming_the_variable(monkeypatch, reader):
    monkeypatch.setenv(VAR, 'banana')
    with pytest.raises(EnvError, match=VAR) as exc:
        reader()
    assert exc.value.name == VAR
    assert 'banana' in str(exc.value)


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_float_rejects_non_finite(monkeypatch, value):
    monkeypatch.setenv(VAR, value)
    with pytest.raises(EnvError, match=VAR):
        env_float(VAR, 1.0)


def test_int_rejects_a_fraction(monkeypatch):
    monkeypatch.setenv(VAR, '1.5')
    with pytest.raises(EnvError, match=VAR):
        env_int(VAR, 1)


def test_range_is_enforced_on_both_ends(monkeypatch):
    monkeypatch.setenv(VAR, '0')
    with pytest.raises(EnvError, match=VAR):
        env_int(VAR, 5, minimum=1)
    monkeypatch.setenv(VAR, '11')
    with pytest.raises(EnvError, match=VAR):
        env_int(VAR, 5, maximum=10)
    monkeypatch.setenv(VAR, '-0.5')
    with pytest.raises(EnvError, match=VAR):
        env_float(VAR, 1.0, minimum=0.0)
    monkeypatch.setenv(VAR, '10')
    assert env_int(VAR, 5, minimum=1, maximum=10) == 10


def test_default_is_not_range_checked(monkeypatch):
    assert env_int(VAR, 0, minimum=1) == 0


def test_is_set_counts_blank_and_prefix_scan(monkeypatch):
    assert not env_is_set(VAR)
    monkeypatch.setenv(VAR, '')
    assert env_is_set(VAR)
    assert VAR in env_names_with_prefix('OP_TEST_ENV_MODULE_')
