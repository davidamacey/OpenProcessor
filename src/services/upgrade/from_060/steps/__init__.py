"""The load as a registry of step modules (research seams, lane 2E-8).

Every module in this package with an integer ``ORDER`` and an ``async def run(ctx)`` is a step
of the load: discovered with :func:`pkgutil.iter_modules`
(https://docs.python.org/3/library/pkgutil.html#pkgutil.iter_modules), run in ``ORDER`` inside
the load's transaction, after the control rows are written. A step may also define
``async def verify(ctx) -> list[str]`` returning the problems it finds after the load. Later
lanes add their step (clustering, near duplicates) as one new file here and edit nothing else.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from types import ModuleType

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.upgrade.from_060.admin_client import ControlReader, ControlWriter
    from src.services.upgrade.from_060.layout import Layout
    from src.services.upgrade.from_060.load import Parsed
    from src.services.upgrade.from_060.model import LoadReport


@dataclass
class StepContext:
    """What a step may use. ``reader`` and ``writer`` reach OpenSearch (the live items and
    images indexes stay); ``config_ids`` maps ``(project slug or None, kind, name)`` to the id the
    load wrote."""

    session: AsyncSession
    parsed: Parsed
    project_ids: dict[str, uuid.UUID]
    config_ids: dict[tuple[str | None, str, str], uuid.UUID]
    reader: ControlReader
    writer: ControlWriter
    layout: Layout
    report: LoadReport
    now: dt.datetime
    notes: dict[str, Any] = field(default_factory=dict)


def discover() -> list[ModuleType]:
    """The step modules in ``ORDER`` (ties by module name)."""
    found: list[ModuleType] = []
    for info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f'{__name__}.{info.name}')
        if isinstance(getattr(module, 'ORDER', None), int) and hasattr(module, 'run'):
            found.append(module)
    return sorted(found, key=lambda m: (m.ORDER, m.__name__))


def step_name(module: ModuleType) -> str:
    return module.__name__.rsplit('.', 1)[-1]


async def run_all(ctx: StepContext) -> list[str]:
    """Run every step in order; returns the names run."""
    ran: list[str] = []
    for module in discover():
        await module.run(ctx)
        ran.append(step_name(module))
    return ran


async def verify_all(ctx: StepContext) -> list[str]:
    """The problems every step's ``verify`` reports, each prefixed with the step name."""
    problems: list[str] = []
    for module in discover():
        check = getattr(module, 'verify', None)
        if check is not None:
            problems.extend(f'{step_name(module)}: {p}' for p in await check(ctx))
    return problems
