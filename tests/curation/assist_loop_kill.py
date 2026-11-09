"""Crash simulation for the auto-label loop tests: a switch that kills the loop at a chosen
record write, after which nothing more reaches the tables or the item store."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from curation.query_fakes import QueryFakeOpenSearch
    from src.services.curation.assist_loop.record import LoopRecord


class Killed(BaseException):
    """The process died: nothing after this point reaches the tables or the item store."""


class KillSwitch:
    """Kills the loop at the ``occurrence``-th call of the record method ``event`` (``when`` it
    is made: just ``before`` or just ``after`` its rows are committed). From then on every
    write to the tables and the item store raises :class:`Killed`, like a dead process."""

    WRITES = (
        'admit',
        'select',
        'record_answers',
        'forget_failed',
        'decide',
        'gate_pass',
        'begin_propagate',
        'finish_propagate',
        'split',
        'record_audit_chunk',
        'set_audit',
        'save_counts',
        'pin_roots',
    )

    def __init__(self, event: str, occurrence: int, when: str = 'after') -> None:
        self.event = event
        self.occurrence = occurrence
        self.when = when
        self.seen = 0
        self.tripped = False

    def _check(self) -> None:
        if self.tripped:
            raise Killed

    def _count(self, kind: str, when: str) -> None:
        if kind == self.event and self.when == when:
            self.seen += 1
            if self.seen == self.occurrence:
                self.tripped = True
                raise Killed

    def arm_record(self, record: LoopRecord) -> None:
        for name in self.WRITES:
            real = getattr(record, name)

            def wrapped(*a: Any, _real: Any = real, _name: str = name, **kw: Any) -> Any:
                async def run() -> Any:
                    self._check()
                    self._count(_name, 'before')
                    out = await _real(*a, **kw)
                    self._count(_name, 'after')
                    return out

                return run()

            setattr(record, name, wrapped)

    def arm_os(self, os_fake: QueryFakeOpenSearch) -> None:
        real_bulk = os_fake.bulk
        real_update = os_fake.update

        async def bulk(**kw: Any) -> dict[str, Any]:
            self._check()
            return await real_bulk(**kw)

        async def update(**kw: Any) -> dict[str, Any]:
            self._check()
            return await real_update(**kw)

        os_fake.bulk = bulk  # type: ignore[method-assign]
        os_fake.update = update  # type: ignore[method-assign]
        self._restore = (os_fake, real_bulk, real_update)

    def disarm(self) -> None:
        """The process restarts: the item store accepts writes again."""
        os_fake, real_bulk, real_update = self._restore
        os_fake.bulk = real_bulk  # type: ignore[method-assign]
        os_fake.update = real_update  # type: ignore[method-assign]
