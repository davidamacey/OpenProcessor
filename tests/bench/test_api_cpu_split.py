"""The API port is served by the Rust front (service `api`) and the Python API runs behind it
(service `api-py`). A bench that reports "API CPU" must measure the Python API container, where
the pixel work and the stage series are, and report the front next to it, each under its name."""

from __future__ import annotations

import pytest

from scripts.bench import suite_probes as sp


def _fake_cgroup(monkeypatch: pytest.MonkeyPatch, values: dict[str, list[float]]) -> list[str]:
    asked: list[str] = []

    def fake(container: str) -> float:
        asked.append(container)
        return values[container].pop(0)

    monkeypatch.setattr(sp, 'cgroup_cpu_s', fake)
    monkeypatch.setattr(sp.time, 'sleep', lambda _s: None)
    return asked


def test_the_idle_probe_measures_the_python_api_and_the_front_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {f'p-{c}': [0.0, 10.0] for c in sp.BACKGROUND}
    values['p-api-py'] = [0.0, 5.0]
    asked = _fake_cgroup(monkeypatch, values)
    cores = sp.background_cores('p', seconds=10.0)
    assert 'p-api-py' in asked
    assert 'p-api' in asked
    assert cores['api-py'] == pytest.approx(0.5)
    assert cores['api'] == pytest.approx(1.0)


def test_the_stack_is_quiet_when_the_python_api_is_quiet_whatever_the_front_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cores = dict.fromkeys(sp.BACKGROUND, 0.0)
    cores['api'] = 5.0
    monkeypatch.setattr(sp, 'background_cores', lambda _p, seconds=10.0: cores)
    assert sp.wait_idle('p', limit_s=0.0)['quiet'] is True
    cores['api-py'] = 5.0
    assert sp.wait_idle('p', limit_s=0.0)['quiet'] is False


def test_both_api_containers_are_reported_per_image_under_their_own_names() -> None:
    fields = sp.api_cpu_fields({'api-py': 30.0, 'api': 3.0, 'triton': 1.0}, images=10)
    assert fields == {'api_cpu_s_per_image': 3.0, 'front_cpu_s_per_image': 0.3}


def test_no_images_does_not_divide_by_zero() -> None:
    fields = sp.api_cpu_fields({'api-py': 2.0, 'api': 1.0}, images=0)
    assert fields == {'api_cpu_s_per_image': 2.0, 'front_cpu_s_per_image': 1.0}
