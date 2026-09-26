"""V-1 (fresh-start E2E findings 2026-09-25, coordinator visual review):
serving the region-drain stall cause.

Persisted to a state.json under ``OP_REGION_DRAIN_STATE_DIR`` (same
shared-``/jobs``-volume convention as ``region_drain.py``'s streak
state), so the "unavailable since" timestamp is stable across
``yolo-api --workers`` processes and repeated polls, rather than
resetting to "just now" every time.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from src.config import DetectionProfile
from src.services.curation import region_dependency_health as rdh


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    d = tmp_path / 'region_drain'
    monkeypatch.setenv('OP_REGION_DRAIN_STATE_DIR', str(d))
    yield d
    rdh._reset_for_tests()


def _profile(**overrides: object) -> DetectionProfile:
    defaults: dict[str, object] = {'name': 'test', 'detector_model': 'det_v1'}
    defaults.update(overrides)
    return DetectionProfile(**defaults)  # type: ignore[arg-type]


async def _index(*entries: tuple[str, str]) -> list[dict[str, str]]:
    return [{'name': name, 'state': state} for name, state in entries]


class TestNoActiveProfile:
    @pytest.mark.asyncio
    async def test_empty_detector_model_means_nothing_to_check(self) -> None:
        """No active region profile (the off/neutral case) must report
        no dependencies at all -- not a false 'unavailable'."""

        async def _boom() -> list[dict[str, str]]:
            msg = 'must not be called when there is no active profile'
            raise AssertionError(msg)

        result = await rdh.check_region_dependencies(_boom, _profile(detector_model=''))
        assert result == []


class TestReadyDependencies:
    @pytest.mark.asyncio
    async def test_detector_and_segmenter_both_ready(self) -> None:
        """The segmenter is checked via its own HTTP health endpoint, not
        the Triton repository index -- SAM 3 (or whatever's configured)
        never appears in that index, so it must not need to for this to
        report ready (V-1 follow-up)."""

        async def _idx() -> list[dict[str, str]]:
            return await _index(('det_v1', 'READY'))

        async def _segmenter_ok() -> tuple[bool, str]:
            return True, 'http://segmenter:8000/health loaded=true'

        results = await rdh.check_region_dependencies(
            _idx, _profile(segmenter_name='sam3'), check_segmenter_health=_segmenter_ok
        )

        assert {r.model for r in results} == {'det_v1', 'sam3'}
        assert all(r.ready for r in results)
        assert all(r.unavailable_since is None for r in results)


class TestUnavailableDependencies:
    @pytest.mark.asyncio
    async def test_model_missing_from_repository_index(self) -> None:
        async def _idx() -> list[dict[str, str]]:
            return []

        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        results = await rdh.check_region_dependencies(_idx, _profile(segmenter_name=''), now=t0)

        [det] = results
        assert det.role == 'detector'
        assert det.ready is False
        assert det.unavailable_since == t0.isoformat()
        assert 'never loaded' in det.detail

    @pytest.mark.asyncio
    async def test_model_present_but_not_ready(self) -> None:
        async def _idx() -> list[dict[str, str]]:
            return await _index(('det_v1', 'UNAVAILABLE'))

        results = await rdh.check_region_dependencies(_idx, _profile(segmenter_name=''))
        [det] = results
        assert det.ready is False
        assert 'UNAVAILABLE' in det.detail

    @pytest.mark.asyncio
    async def test_unavailable_since_is_stable_across_polls(self) -> None:
        """The timestamp must be the FIRST time it was seen unavailable,
        not the current poll's time on every call."""

        async def _idx() -> list[dict[str, str]]:
            return []

        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        first = await rdh.check_region_dependencies(_idx, _profile(), now=t0)
        second = await rdh.check_region_dependencies(
            _idx, _profile(), now=t0 + timedelta(minutes=5)
        )

        assert first[0].unavailable_since == t0.isoformat()
        assert second[0].unavailable_since == t0.isoformat()  # unchanged

    @pytest.mark.asyncio
    async def test_recovering_clears_the_persisted_since_timestamp(self) -> None:
        t0 = datetime(2026, 1, 1, tzinfo=UTC)

        async def _down() -> list[dict[str, str]]:
            return []

        async def _up() -> list[dict[str, str]]:
            return await _index(('det_v1', 'READY'))

        await rdh.check_region_dependencies(_down, _profile(), now=t0)
        recovered = await rdh.check_region_dependencies(
            _up, _profile(), now=t0 + timedelta(minutes=1)
        )
        assert recovered[0].ready is True
        assert recovered[0].unavailable_since is None

        # And a later outage starts a fresh "since", not the old one.
        later_down = await rdh.check_region_dependencies(
            _down, _profile(), now=t0 + timedelta(hours=1)
        )
        assert later_down[0].unavailable_since == (t0 + timedelta(hours=1)).isoformat()

    @pytest.mark.asyncio
    async def test_triton_itself_unreachable_reports_detector_down(self) -> None:
        """The segmenter dependency is independent of Triton -- a Triton
        outage must only affect the detector's status, not fabricate a
        segmenter failure via the wrong subsystem."""

        async def _boom() -> list[dict[str, str]]:
            msg = 'connection refused'
            raise ConnectionError(msg)

        async def _segmenter_ok() -> tuple[bool, str]:
            return True, 'loaded=true'

        results = await rdh.check_region_dependencies(
            _boom, _profile(segmenter_name='sam3'), check_segmenter_health=_segmenter_ok
        )
        by_role = {r.role: r for r in results}
        assert by_role['detector'].ready is False
        assert 'repository index unavailable' in by_role['detector'].detail
        assert by_role['segmenter'].ready is True

    @pytest.mark.asyncio
    async def test_segmenter_down_reports_unavailable_independent_of_triton(self) -> None:
        """Segmenter health check failing must not need the Triton
        repository index to say so (V-1 follow-up: this used to look the
        segmenter up in Triton's index, which it never appears in)."""

        async def _idx() -> list[dict[str, str]]:
            return await _index(('det_v1', 'READY'))

        async def _segmenter_down() -> tuple[bool, str]:
            return False, 'http://segmenter:8000/health unreachable: connection refused'

        results = await rdh.check_region_dependencies(
            _idx, _profile(segmenter_name='sam3'), check_segmenter_health=_segmenter_down
        )
        by_role = {r.role: r for r in results}
        assert by_role['detector'].ready is True
        assert by_role['segmenter'].ready is False
        assert by_role['segmenter'].unavailable_since is not None
        assert 'unreachable' in by_role['segmenter'].detail

    @pytest.mark.asyncio
    async def test_segmenter_healthy_clears_stall_reason(self) -> None:
        async def _idx() -> list[dict[str, str]]:
            return await _index(('det_v1', 'READY'))

        async def _segmenter_ok() -> tuple[bool, str]:
            return True, 'loaded=true'

        results = await rdh.check_region_dependencies(
            _idx, _profile(segmenter_name='sam3'), check_segmenter_health=_segmenter_ok
        )
        assert rdh.stall_reason(results, pending_detection=42) is None

    @pytest.mark.asyncio
    async def test_segmenter_down_names_it_in_stall_reason(self) -> None:
        async def _idx() -> list[dict[str, str]]:
            return await _index(('det_v1', 'READY'))

        async def _segmenter_down() -> tuple[bool, str]:
            return False, 'not loaded'

        results = await rdh.check_region_dependencies(
            _idx, _profile(segmenter_name='sam3'), check_segmenter_health=_segmenter_down
        )
        reason = rdh.stall_reason(results, pending_detection=7)
        assert reason is not None
        assert 'segmenter (sam3)' in reason

    @pytest.mark.asyncio
    async def test_triton_detector_missing_is_named_in_stall_reason(self) -> None:
        async def _idx() -> list[dict[str, str]]:
            return []

        results = await rdh.check_region_dependencies(_idx, _profile(segmenter_name=''))
        reason = rdh.stall_reason(results, pending_detection=5)
        assert reason is not None
        assert 'detector (det_v1)' in reason

    @pytest.mark.asyncio
    async def test_default_segmenter_health_checker_reports_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No OP_SEGMENTER_URL/OP_SEGMENTER_URLS configured must be a clear,
        specific unavailable detail, not an opaque connection error."""
        monkeypatch.delenv('OP_SEGMENTER_URL', raising=False)
        monkeypatch.delenv('OP_SEGMENTER_URLS', raising=False)

        async def _idx() -> list[dict[str, str]]:
            return await _index(('det_v1', 'READY'))

        results = await rdh.check_region_dependencies(_idx, _profile(segmenter_name='sam3'))
        by_role = {r.role: r for r in results}
        assert by_role['segmenter'].ready is False
        assert 'not configured' in by_role['segmenter'].detail


class TestStallReason:
    def test_none_when_nothing_pending(self) -> None:
        deps = [rdh.RegionDependencyStatus('detector', 'det_v1', False, '2026-01-01', 'down')]
        assert rdh.stall_reason(deps, pending_detection=0) is None

    def test_none_when_everything_is_ready(self) -> None:
        deps = [rdh.RegionDependencyStatus('detector', 'det_v1', True, None, 'READY')]
        assert rdh.stall_reason(deps, pending_detection=50) is None

    def test_names_the_down_dependency_and_since_when_pending(self) -> None:
        deps = [
            rdh.RegionDependencyStatus(
                'segmenter', 'sam3', False, '2026-09-25T14:02:11+00:00', 'not loaded'
            )
        ]
        reason = rdh.stall_reason(deps, pending_detection=3516)
        assert reason is not None
        assert '3516' in reason
        assert 'segmenter (sam3)' in reason
        assert '2026-09-25T14:02:11+00:00' in reason


class TestSegmenterOnlyMode:
    """A profile's detector_model may not ship (the public plate example names
    one nothing provides). The cascade then falls through to the segmenter,
    so a missing detector with a healthy segmenter is not a stall."""

    def test_missing_detector_with_ready_segmenter_is_not_a_stall(self) -> None:
        deps = [
            rdh.RegionDependencyStatus('detector', 'det_v1', False, '2026-09-25', 'missing'),
            rdh.RegionDependencyStatus('segmenter', 'sam3', True, None, 'loaded=true'),
        ]
        assert rdh.stall_reason(deps, pending_detection=3508) is None

    def test_both_down_names_both(self) -> None:
        deps = [
            rdh.RegionDependencyStatus('detector', 'det_v1', False, '2026-09-25', 'missing'),
            rdh.RegionDependencyStatus('segmenter', 'sam3', False, '2026-09-25', 'not loaded'),
        ]
        reason = rdh.stall_reason(deps, pending_detection=10)
        assert reason is not None
        assert 'detector (det_v1)' in reason
        assert 'segmenter (sam3)' in reason


class TestStatePersistenceIsBestEffort:
    @pytest.mark.asyncio
    async def test_an_unwritable_state_dir_still_returns_a_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A state-persistence failure (dir not writable, volume not
        mounted) must never turn into an exception that 500s the whole
        /stats/dataset or /ingest/region_drain response."""
        readonly_parent = tmp_path / 'readonly'
        readonly_parent.mkdir()
        readonly_parent.chmod(0o500)
        monkeypatch.setenv('OP_REGION_DRAIN_STATE_DIR', str(readonly_parent / 'region_drain'))

        async def _idx() -> list[dict[str, str]]:
            return []

        try:
            results = await rdh.check_region_dependencies(_idx, _profile(segmenter_name=''))
        finally:
            readonly_parent.chmod(0o700)  # allow tmp_path cleanup
        [det] = results
        assert det.ready is False
        assert det.unavailable_since is not None
