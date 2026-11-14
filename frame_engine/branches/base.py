"""What every branch module builds on: the photo it reads, the photo's model calls, the interface.

The contract and a worked example are in the package docstring (:mod:`frame_engine.branches`).
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from frame_engine.transport import ModelCallError


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import numpy as np

    from frame_engine.prep_kernels import PrepKernels
    from frame_engine.stage_metrics import StageClock
    from frame_engine.transport import BodyTransport, CudaShmTransport, ModelInput, Pending, Staged
    from src.domain import frame_engine_wire as wire


class DependencyFailedError(Exception):
    """A branch cannot run: an input it needs from round 1 is missing (that branch failed)."""


@dataclass
class Photo:
    """One decoded photo as the branches see it, and the answers they leave for each other."""

    plan: wire.EnginePlan
    #: The resident full-resolution frame, HWC uint8 on the GPU; every model input is cut from it.
    frame: Any
    size: tuple[int, int]
    calls: PhotoCalls
    clock: StageClock
    kernels: PrepKernels
    #: Output tensor name to array (``frame_engine.messages.EngineResult.outputs``).
    out: dict[str, Any]
    # -- round-1 answers later rounds read (set in ``settle``) --------------------------------
    #: Object rectangles in frame pixels (``x1, y1, x2, y2``), from the detector or the caller's
    #: stored boxes; ``None`` when none are available (the detector failed or did not run).
    rects: list[tuple[int, int, int, int]] | None = None
    #: The detector's class id of each rectangle; ``None`` for stored boxes.
    classes: np.ndarray | None = None


class Branch:
    """One branch module. Every hook is optional; the orchestrator calls them in this order for
    every branch of the plan: ``round1`` (inside one commit), ``settle``, ``round2``, ``finish``,
    then ``close`` always, even after a failure. A hook that raises
    :class:`~frame_engine.transport.ModelCallError` or :class:`DependencyFailedError` fails this
    branch only: its calls in flight are cancelled and its later hooks are skipped."""

    #: Its ``branch_status`` entry; ``None`` for an output with no entry (the crop JPEGs).
    status: str | None = None
    #: The ``branch_status`` names whose round-1 answers it reads: when one of them fails or does
    #: not run, this branch ends the same way without running (``face_embed``: ``('faces',)``).
    needs: tuple[str, ...] = ()

    def round1(self, photo: Photo) -> None:
        """Stage and submit the calls that need only the frame."""

    def settle(self, photo: Photo) -> None:
        """Collect the round-1 answers other branches read, and leave them on ``photo``."""

    def round2(self, photo: Photo) -> None:
        """Stage and submit the calls cut from round-1 answers; start host work."""

    def finish(self, photo: Photo) -> None:
        """Collect the rest and write this branch's outputs into ``photo.out``."""

    def close(self, photo: Photo) -> None:
        """Give up host work still queued (after a failure; a no-op after ``finish``)."""


@dataclass(eq=False)
class Call:
    """One model call of a photo, from its staged slice to its answer."""

    owner: Branch
    stage: str  # the stage clock entry its commit, submit and wait count in
    staged: Staged
    model: str
    tensor: ModelInput
    outputs: tuple[str, ...]
    pending: Pending | None = None
    answer: dict[str, np.ndarray] | None = None
    error: ModelCallError | None = None


@dataclass(eq=False)
class PhotoCalls:
    """The model calls of one photo: its slices of the region, its calls in flight, its window.

    ``window`` bounds the region bytes one photo holds while it has calls in flight: staging a
    slice that would pass it first collects the photo's oldest call (its answer is kept for
    :meth:`result`). A call is always allowed when none is in flight, so the bound is
    ``max(window, one call)`` and a photo never waits on itself.
    """

    transport: BodyTransport | CudaShmTransport
    clock: StageClock
    deadline: float
    window: int
    sync: Callable[[], None]
    held: list[Staged] = field(default_factory=list)
    _open: deque[Call] = field(default_factory=deque)
    _round: list[Call] | None = None

    @property
    def held_bytes(self) -> int:
        return sum(s.nbytes for s in self.held)

    def stage(self, nbytes: int, stage: str) -> Staged:
        """A slice for one call's input; the waits for room count in ``stage``."""
        with self.clock.stage(stage):
            while self._open and self.held_bytes + nbytes > self.window:
                self._settle(self._open[0])
            staged = self.transport.stage(nbytes)
        self.held.append(staged)
        return staged

    @contextmanager
    def together(self) -> Iterator[None]:
        """Calls submitted inside go out together at the end, after one wait for the stream that
        covers every kernel that wrote them (round 1). The wait and the sends count in the stage
        of the first call."""
        self._round = []
        try:
            yield
            calls = self._round
        finally:
            self._round = None
        if calls:
            with self.clock.stage(calls[0].stage):
                self.transport.commit([c.staged for c in calls], self.sync)
                for c in calls:
                    self._send(c)

    def submit(
        self,
        owner: Branch,
        stage: str,
        staged: Staged,
        model: str,
        tensor: ModelInput,
        outputs: Sequence[str],
    ) -> Call:
        """Send one call reading ``staged`` (written by the kernels just launched) and return
        without waiting for its answer."""
        call = Call(owner, stage, staged, model, tensor, tuple(outputs))
        if self._round is not None:
            self._round.append(call)
            return call
        with self.clock.stage(stage):
            self.transport.commit([staged], self.sync)
            self._send(call)
        return call

    def _send(self, call: Call) -> None:
        try:
            call.pending = self.transport.submit(
                call.model, call.staged, call.tensor, call.outputs, deadline=self.deadline
            )
        except ModelCallError as exc:  # refused when sent: it never reads the slice
            call.error = exc
            self._release(call.staged)
        else:
            self._open.append(call)

    def result(self, call: Call) -> dict[str, np.ndarray]:
        """The call's outputs (waited for in its stage), or its :class:`ModelCallError`."""
        if call in self._open:
            with self.clock.stage(call.stage):
                self._settle(call)
        if call.error is not None:
            raise call.error
        if call.answer is None:
            raise RuntimeError(f'{call.model}: result of a call never sent')
        return call.answer

    def _settle(self, call: Call) -> None:
        self._open.remove(call)
        if call.pending is None:
            raise RuntimeError(f'{call.model}: settle of a call never sent')
        try:
            call.answer = self.transport.collect(call.pending)
        except ModelCallError as exc:
            call.error = exc
        finally:  # answered, failed or cancelled at its deadline: the server is done with it
            self._release(call.staged)

    def cancel(self, owner: Branch) -> None:
        """Give up ``owner``'s calls in flight and give their slices back (it failed)."""
        for call in [c for c in self._open if c.owner is owner]:
            self._open.remove(call)
            if call.pending is not None:
                self.transport.cancel(call.pending)
            self._release(call.staged)

    def _release(self, staged: Staged) -> None:
        self.held.remove(staged)
        self.transport.release(staged, self.sync)

    def close(self) -> None:
        """Cancel every call still in flight, then give every slice back (each after the stream
        was waited for, the transport's rule)."""
        while self._open:
            call = self._open.popleft()
            if call.pending is not None:
                self.transport.cancel(call.pending)
        for staged in list(self.held):
            self._release(staged)
