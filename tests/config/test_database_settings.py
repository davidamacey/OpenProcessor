"""DatabaseSettings: every bad input refuses with the variable's name, valid input loads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from pathlib import Path

_VARS = (
    'OP_DB_HOST',
    'OP_DB_PORT',
    'OP_DB_NAME',
    'OP_DB_APP_USER',
    'OP_DB_APP_PASSWORD_FILE',
    'OP_DB_OWNER_USER',
    'OP_DB_OWNER_PASSWORD_FILE',
    'OP_DB_SUPER_PASSWORD_FILE',
    'OP_API_REPLICAS',
    'OP_DB_DIRECT_HOST',
    'OP_DB_SSLMODE',
    'OP_DB_POOLER',
    'OP_DB_API_CONNECTIONS_TOTAL',
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)


def _secret(tmp_path: Path, name: str, content: str = 's3cret\n') -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


def _load():
    from src.config.database import DatabaseSettings

    return DatabaseSettings.from_env()


def test_defaults_load_without_any_variable() -> None:
    settings = _load()
    assert settings.host == 'postgres'
    assert settings.port == 5432
    assert settings.direct_host == 'postgres'
    assert settings.app_user == 'op_app'
    assert settings.owner_user == 'op_owner'
    assert settings.pooler == 'none'
    assert settings.api_replicas == 1
    assert settings.api_connections_total == 128


def test_valid_values_and_password_files_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('OP_DB_HOST', 'pgbouncer')
    monkeypatch.setenv('OP_DB_DIRECT_HOST', 'db.internal')
    monkeypatch.setenv('OP_DB_PORT', '6432')
    monkeypatch.setenv('OP_DB_POOLER', 'pgbouncer')
    monkeypatch.setenv('OP_DB_SSLMODE', 'require')
    monkeypatch.setenv('OP_DB_APP_PASSWORD_FILE', str(_secret(tmp_path, 'app', 'pw-app\n')))
    monkeypatch.setenv('OP_DB_OWNER_PASSWORD_FILE', str(_secret(tmp_path, 'own', 'pw-own')))
    monkeypatch.setenv('OP_DB_SUPER_PASSWORD_FILE', str(_secret(tmp_path, 'sup', 'pw-sup')))
    settings = _load()
    assert (settings.host, settings.direct_host, settings.port) == (
        'pgbouncer',
        'db.internal',
        6432,
    )
    assert settings.pooler == 'pgbouncer'
    assert settings.password('app') == 'pw-app'
    assert settings.password('owner') == 'pw-own'
    assert settings.password('super') == 'pw-sup'


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('OP_DB_HOST', 'postgresql://postgres:5432'),
        ('OP_DB_HOST', 'user@postgres'),
        ('OP_DB_HOST', 'postgres/db'),
        ('OP_DB_DIRECT_HOST', 'postgres://db'),
        ('OP_DB_PORT', 'abc'),
        ('OP_DB_PORT', '0'),
        ('OP_DB_PORT', '70000'),
        ('OP_DB_POOLER', 'pgcat'),
        ('OP_DB_SSLMODE', 'sometimes'),
        ('OP_API_REPLICAS', '0'),
        ('OP_API_REPLICAS', 'two'),
        ('OP_DB_API_CONNECTIONS_TOTAL', '0'),
        ('OP_DB_NAME', 'a b'),
        ('OP_DB_APP_USER', 'op app'),
    ],
)
def test_a_bad_value_refuses_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        _load()


@pytest.mark.parametrize(
    ('role', 'variable'),
    [
        ('app', 'OP_DB_APP_PASSWORD_FILE'),
        ('owner', 'OP_DB_OWNER_PASSWORD_FILE'),
        ('super', 'OP_DB_SUPER_PASSWORD_FILE'),
    ],
)
def test_a_missing_password_file_refuses_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str, variable: str
) -> None:
    monkeypatch.setenv(variable, str(tmp_path / 'absent'))
    with pytest.raises(ValueError, match=variable):
        _load().password(role)


def test_an_empty_password_file_refuses_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('OP_DB_APP_PASSWORD_FILE', str(_secret(tmp_path, 'app', '  \n')))
    with pytest.raises(ValueError, match='OP_DB_APP_PASSWORD_FILE'):
        _load().password('app')


def test_the_error_never_contains_the_password(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('OP_DB_APP_PASSWORD_FILE', str(tmp_path / 'absent'))
    with pytest.raises(ValueError, match='OP_DB_APP_PASSWORD_FILE') as caught:
        _load().password('app')
    assert 's3cret' not in str(caught.value)


def test_an_unknown_role_is_refused() -> None:
    with pytest.raises(ValueError, match='role'):
        _load().password('admin')
