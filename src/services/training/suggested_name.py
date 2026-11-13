"""The readable Triton model name a run is promoted under unless the person picks another."""

from __future__ import annotations

import datetime as dt
import re


TRITON_NAME_MAX = 64
_UNSAFE = re.compile(r'[^A-Za-z0-9_-]+')


def suggested_triton_name(project_slug: str, when: dt.datetime) -> str:
    """``<project>_<YYYYMMDD>_<HHMM>`` (UTC): who it is for and which run, within the 64
    characters of ``[A-Za-z0-9_-]`` the promote route accepts."""
    stamp = when.astimezone(dt.UTC).strftime('%Y%m%d_%H%M')
    slug = _UNSAFE.sub('_', project_slug).strip('_') or 'model'
    return f'{slug[: TRITON_NAME_MAX - len(stamp) - 1]}_{stamp}'
