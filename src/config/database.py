"""Connection settings of the control-plane database (PostgreSQL).

Everything is read through ``src.config.env``. A malformed value refuses startup naming
its variable. Passwords are never environment values: each role has a secret file, read
only when that role connects, so a process that does not hold a role's file (the API
holds only the application one) is not asked for it. A missing or empty file raises
:class:`DatabaseConfigError` naming the variable that points at it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.config.env import env_int, env_str


POOLERS = ('none', 'pgbouncer')
SSLMODES = ('disable', 'allow', 'prefer', 'require', 'verify-ca', 'verify-full')
ROLES = ('app', 'owner', 'super')

# The image's POSTGRES_USER, used only by the migration runner to create the other two roles.
SUPER_USER = 'op_super'

_IDENTIFIER = re.compile(r'^[a-z_][a-z0-9_]{0,62}$')
_HOST = re.compile(r'^[A-Za-z0-9._-]+$')

_PASSWORD_FILE_VARIABLE = {
    'app': 'OP_DB_APP_PASSWORD_FILE',
    'owner': 'OP_DB_OWNER_PASSWORD_FILE',
    'super': 'OP_DB_SUPER_PASSWORD_FILE',
}


class DatabaseConfigError(ValueError):
    """A database setting or its secret file cannot be used."""

    def __init__(self, name: str, why: str):
        super().__init__(f'{name}: {why}')
        self.name = name


def _host(name: str, value: str) -> str:
    if not _HOST.fullmatch(value):
        raise DatabaseConfigError(
            name, f'must be a bare host name or address, not a URL or path (got {value!r})'
        )
    return value


def _identifier(name: str, value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise DatabaseConfigError(name, f'not a valid PostgreSQL identifier (got {value!r})')
    return value


def _choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
    value = value.lower()
    if value not in allowed:
        raise DatabaseConfigError(name, f'unknown value {value!r}; use one of {list(allowed)}')
    return value


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    direct_host: str
    port: int
    name: str
    app_user: str
    owner_user: str
    app_password_file: Path
    owner_password_file: Path
    super_password_file: Path
    sslmode: str
    pooler: str
    api_replicas: int
    api_connections_total: int

    @classmethod
    def from_env(cls) -> DatabaseSettings:
        host = _host('OP_DB_HOST', env_str('OP_DB_HOST', 'postgres'))
        return cls(
            host=host,
            # Session advisory locks and DDL never go through a transaction pooler.
            direct_host=_host('OP_DB_DIRECT_HOST', env_str('OP_DB_DIRECT_HOST', host)),
            port=env_int('OP_DB_PORT', 5432, minimum=1, maximum=65535),
            name=_identifier('OP_DB_NAME', env_str('OP_DB_NAME', 'openprocessor')),
            app_user=_identifier('OP_DB_APP_USER', env_str('OP_DB_APP_USER', 'op_app')),
            owner_user=_identifier('OP_DB_OWNER_USER', env_str('OP_DB_OWNER_USER', 'op_owner')),
            app_password_file=Path(
                env_str('OP_DB_APP_PASSWORD_FILE', '/run/secrets/op_pg/app_password')
            ),
            owner_password_file=Path(
                env_str('OP_DB_OWNER_PASSWORD_FILE', '/run/secrets/op_pg/owner_password')
            ),
            super_password_file=Path(
                env_str('OP_DB_SUPER_PASSWORD_FILE', '/run/secrets/op_pg/super_password')
            ),
            sslmode=_choice('OP_DB_SSLMODE', env_str('OP_DB_SSLMODE', 'prefer'), SSLMODES),
            pooler=_choice('OP_DB_POOLER', env_str('OP_DB_POOLER', 'none'), POOLERS),
            api_replicas=env_int('OP_API_REPLICAS', 1, minimum=1),
            api_connections_total=env_int('OP_DB_API_CONNECTIONS_TOTAL', 128, minimum=1),
        )

    def password(self, role: str) -> str:
        """The stored password of ``role`` (``app``, ``owner`` or ``super``)."""
        if role not in ROLES:
            raise DatabaseConfigError('role', f'unknown database role {role!r}; use one of {ROLES}')
        variable = _PASSWORD_FILE_VARIABLE[role]
        path: Path = getattr(self, f'{role}_password_file')
        try:
            value = path.read_text(encoding='utf-8').strip()
        except OSError as exc:
            raise DatabaseConfigError(
                variable, f'the password file {path} cannot be read ({type(exc).__name__})'
            ) from None
        if not value:
            raise DatabaseConfigError(variable, f'the password file {path} is empty')
        return value

    def user(self, role: str) -> str:
        if role == 'super':
            return SUPER_USER
        if role == 'owner':
            return self.owner_user
        if role == 'app':
            return self.app_user
        raise DatabaseConfigError('role', f'unknown database role {role!r}; use one of {ROLES}')
