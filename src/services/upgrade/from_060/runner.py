"""The four steps of the 0.6.0 upgrade in order, with the gate and the ``upgrade_runs`` record.

``export_bundle`` -> ``load`` (and the step registry) -> ``verify`` -> ``delete_sources``. Each
step is recorded in ``upgrade_runs`` when it completes, so a run that stopped resumes at the
first step not completed; a verify that failed leaves every source in place (trap T-25) and the
gate closed (``upgrade_state`` stays ``in_progress``), and the operator either fixes the cause
and runs again (verify repeats) or runs with ``reset`` (the load repeats from the same bundle).

``load`` is the one sanctioned large transaction (review fix R-25): the precondition is that the
0.6.0 API and workers are stopped and nothing else writes.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, text

from src.db.models import UpgradeRun
from src.db.session import deployment_tx
from src.services.upgrade import gate
from src.services.upgrade.from_060 import direct_rows, steps
from src.services.upgrade.from_060.bundle import BundleError, check_bundle
from src.services.upgrade.from_060.delete_sources import delete_sources
from src.services.upgrade.from_060.export import export_bundle
from src.services.upgrade.from_060.load import Parsed, run_load
from src.services.upgrade.from_060.snapshot import read_snapshot
from src.services.upgrade.from_060.verify import VerifyReport, verify_load


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.clients.storage.base import BlobStore
    from src.services.upgrade.from_060.admin_client import ControlReader, ControlWriter
    from src.services.upgrade.from_060.layout import Layout

IDLE_LIMIT = "'30min'"
BUNDLE_DIR_FORMAT = 'upgrade-0.6.0-%Y%m%dT%H%M%SZ'
REPORT_NAME = 'UPGRADE_REPORT.json'


class VerifyFailedError(RuntimeError):
    """Verify found a mismatch; nothing was deleted. ``report_path`` has the details."""

    def __init__(self, report_path: Path, problems: list[str]):
        super().__init__(f'verify failed ({len(problems)} problems); see {report_path}')
        self.report_path = report_path
        self.problems = problems


class AlreadyCompletedError(RuntimeError):
    """The upgrade finished earlier; there is nothing to do."""


class NothingToUpgradeError(RuntimeError):
    """No upgrade was started and OpenSearch holds no 0.6.0 control data (a fresh 0.7.0 install)."""


@dataclass
class UpgradeResult:
    bundle: Path
    steps_run: list[str] = field(default_factory=list)
    load_counts: dict[str, int] = field(default_factory=dict)
    verify: VerifyReport | None = None
    report_path: Path | None = None


@dataclass
class UpgradeRunner:
    sessions: async_sessionmaker[AsyncSession]
    reader: ControlReader
    writer: ControlWriter
    layout: Layout
    store: BlobStore
    install_dir: Path
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC)

    async def run(self) -> UpgradeResult:
        await self._begin()
        bundle = await self._export()
        result = UpgradeResult(bundle=bundle)
        await self._load(result)
        await self._verify(result)
        await self._delete(result)
        return result

    async def _begin(self) -> None:
        """Close the gate (``upgrade_state`` in progress) before anything else happens."""
        async with self.sessions() as session, deployment_tx(session, '0.6.0 upgrade: gate'):
            if await gate.step_done(session, 'delete_sources'):
                raise AlreadyCompletedError('the 0.6.0 upgrade already completed')
            if await gate.read_state(session) is None and not await self.reader.index_exists(
                self.layout.projects_index
            ):
                raise NothingToUpgradeError(
                    f'no {self.layout.projects_index} index: nothing to upgrade'
                )
            await gate.mark_in_progress(session)

    # ---------------------------------------------------------------- 1 export_bundle

    async def _export(self) -> Path:
        async with self.sessions() as session, deployment_tx(session, '0.6.0 upgrade: export'):
            done = (
                await session.execute(
                    select(UpgradeRun).where(
                        UpgradeRun.from_version == gate.FROM_VERSION,
                        UpgradeRun.step == 'export_bundle',
                        UpgradeRun.status == 'completed',
                    )
                )
            ).scalar_one_or_none()
        if done is not None:
            from pathlib import Path

            bundle = Path((done.report or {})['bundle'])
            check_bundle(bundle)
            return bundle
        stamp = self.now()
        directory = self.install_dir / stamp.strftime(BUNDLE_DIR_FORMAT)
        manifest = await export_bundle(self.reader, self.layout, self.store, directory, now=stamp)
        async with self.sessions() as session, deployment_tx(session, '0.6.0 upgrade: export'):
            await gate.record_step(
                session,
                'export_bundle',
                status='completed',
                fingerprint=manifest.sha256,
                report={'bundle': str(directory), 'counts': manifest.counts},
            )
        return directory

    async def _bundle_fingerprint(self, bundle: Path) -> str:
        return check_bundle(bundle).sha256

    # ---------------------------------------------------------------- 2 load

    async def _load(self, result: UpgradeResult) -> None:
        async with self.sessions() as session, deployment_tx(session, '0.6.0 upgrade: gate'):
            if await gate.step_done(session, 'load'):
                return
        fingerprint = await self._bundle_fingerprint(result.bundle)
        async with self.sessions() as session, deployment_tx(session, '0.6.0 upgrade: load'):
            await session.execute(
                text(f'SET LOCAL idle_in_transaction_session_timeout = {IDLE_LIMIT}')
            )
            await session.execute(text(f'SET LOCAL statement_timeout = {IDLE_LIMIT}'))
            loaded = await run_load(session, result.bundle, self.store, now=self.now())
            ctx = steps.StepContext(
                session=session,
                parsed=loaded.parsed,
                project_ids=loaded.project_ids,
                config_ids=loaded.config_ids,
                reader=self.reader,
                writer=self.writer,
                layout=self.layout,
                report=loaded.report,
                now=self.now(),
            )
            result.steps_run = await steps.run_all(ctx)
            await gate.record_step(
                session,
                'load',
                status='completed',
                fingerprint=fingerprint,
                report={
                    'counts': dict(sorted(loaded.report.counts.items())),
                    'steps': result.steps_run,
                    'notes': ctx.notes,
                },
            )
        result.load_counts = dict(loaded.report.counts)

    # ---------------------------------------------------------------- 3 verify

    async def _verify(self, result: UpgradeResult) -> None:
        async with self.sessions() as session, deployment_tx(session, '0.6.0 upgrade: gate'):
            if await gate.step_done(session, 'verify'):
                result.report_path = result.bundle / REPORT_NAME
                return
        fingerprint = await self._bundle_fingerprint(result.bundle)
        async with self.sessions() as session, deployment_tx(session, '0.6.0 upgrade: verify'):
            parsed = Parsed(read_snapshot(result.bundle), now=self.now())
            project_ids, config_ids = await direct_rows.id_maps(session)
            ctx = steps.StepContext(
                session=session,
                parsed=parsed,
                project_ids=project_ids,
                config_ids=config_ids,
                reader=self.reader,
                writer=self.writer,
                layout=self.layout,
                report=_empty_report(),
                now=self.now(),
            )
            report = await verify_load(session, self.reader, parsed, result.bundle, ctx)
            await gate.record_step(
                session,
                'verify',
                status='completed' if report.ok else 'failed',
                fingerprint=fingerprint,
                report=report.to_json(),
            )
        result.verify = report
        result.report_path = result.bundle / REPORT_NAME
        result.report_path.write_text(
            json.dumps(report.to_json(), indent=2, sort_keys=True), encoding='utf-8'
        )
        if not report.ok:
            raise VerifyFailedError(result.report_path, report.problems)

    # ---------------------------------------------------------------- 4 delete_sources

    async def _delete(self, result: UpgradeResult) -> None:
        async with self.sessions() as session, deployment_tx(session, '0.6.0 upgrade: gate'):
            if await gate.step_done(session, 'delete_sources'):
                return
        fingerprint = await self._bundle_fingerprint(result.bundle)
        deleted = await delete_sources(result.bundle, self.writer, self.layout)
        async with (
            self.sessions() as session,
            deployment_tx(session, '0.6.0 upgrade: delete_sources'),
        ):
            await gate.record_step(
                session,
                'delete_sources',
                status='completed',
                fingerprint=fingerprint,
                report=deleted.to_json(),
            )
            await gate.mark_completed(session)


def _empty_report() -> Any:
    from src.services.upgrade.from_060.model import LoadReport

    return LoadReport()


__all__ = [
    'AlreadyCompletedError',
    'BundleError',
    'NothingToUpgradeError',
    'UpgradeResult',
    'UpgradeRunner',
    'VerifyFailedError',
]
