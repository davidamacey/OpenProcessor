"""Typed environment reads: the one place ``src`` touches ``os.environ``.

An unset or blank variable yields the caller's default. A value that is set
but malformed or out of range raises :class:`EnvError` naming the variable, so
a typo in ``.env`` refuses startup instead of silently running on a default.
"""

from __future__ import annotations

import math
import os
from pathlib import Path


TRUE_VALUES = frozenset({'1', 'true', 'yes', 'on'})
FALSE_VALUES = frozenset({'0', 'false', 'no', 'off'})


class EnvError(ValueError):
    """A set environment variable holds a value the reader cannot use."""

    def __init__(self, name: str, value: str, why: str):
        super().__init__(f'environment variable {name}={value!r}: {why}')
        self.name = name


def env_raw(name: str) -> str | None:
    """The value exactly as set (``None`` only when unset), for readers where a
    blank value is itself an override, such as an emptied profile field."""
    return os.environ.get(name)


def env_optional_str(name: str) -> str | None:
    """The stripped value, or ``None`` when unset or blank."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def env_str(name: str, default: str = '') -> str:
    """The stripped value, or ``default`` when unset or blank."""
    value = env_optional_str(name)
    return default if value is None else value


def env_path(name: str, default: Path | str) -> Path:
    return Path(env_str(name, str(default)))


def env_optional_path(name: str) -> Path | None:
    value = env_optional_str(name)
    return None if value is None else Path(value)


def env_int(
    name: str, default: int, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    raw = env_optional_str(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise EnvError(name, raw, 'not an integer') from None
    _check_range(name, raw, value, minimum, maximum)
    return value


def env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = env_optional_str(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise EnvError(name, raw, 'not a number') from None
    if not math.isfinite(value):
        raise EnvError(name, raw, 'not a finite number')
    _check_range(name, raw, value, minimum, maximum)
    return value


def env_bool(name: str, default: bool = False) -> bool:
    raw = env_optional_str(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    raise EnvError(name, raw, f'not a boolean (use one of {sorted(TRUE_VALUES | FALSE_VALUES)})')


def env_is_set(name: str) -> bool:
    """Whether the variable exists at all, even blank."""
    return name in os.environ


def env_names_with_prefix(prefix: str) -> list[str]:
    return sorted(k for k in os.environ if k.startswith(prefix))


def _check_range(
    name: str, raw: str, value: float, minimum: float | None, maximum: float | None
) -> None:
    if minimum is not None and value < minimum:
        raise EnvError(name, raw, f'below the minimum {minimum}')
    if maximum is not None and value > maximum:
        raise EnvError(name, raw, f'above the maximum {maximum}')
