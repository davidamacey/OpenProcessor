"""The opt-in ingest load profile overlay stays the base compose file minus the unused models."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from compose._helpers import COMPOSE_PATH, REPO_ROOT, _load_yaml


INGEST_PROFILE_MODELS = ('yolov11_small_trt_end2end', 'pe_image_encoder', 'pe_text_encoder')


OVERLAY = REPO_ROOT / 'docker-compose.ingest-focus.yml'


def _loads(command: list[str]) -> list[str]:
    return [c.split('=', 1)[1] for c in command if c.startswith('--load-model=')]


def _command(doc: dict[str, Any]) -> list[str]:
    return list(doc['services']['triton-server']['command'])


def test_the_overlay_loads_only_what_ingest_calls() -> None:
    assert _loads(_command(_load_yaml(OVERLAY))) == list(INGEST_PROFILE_MODELS)


def test_the_overlay_command_is_the_base_command_without_the_other_models() -> None:
    base = _command(_load_yaml(COMPOSE_PATH))
    wanted = [
        c
        for c in base
        if not c.startswith('--load-model=') or c.split('=', 1)[1] in INGEST_PROFILE_MODELS
    ]
    assert _command(_load_yaml(OVERLAY)) == wanted


def test_every_model_the_profile_preloads_is_in_the_default_load_list() -> None:
    base = _loads(_command(_load_yaml(COMPOSE_PATH)))
    assert set(INGEST_PROFILE_MODELS) <= set(base)


@pytest.mark.parametrize(
    'service',
    ['api', 'curation-detection-worker', 'curation-auto-label-worker', 'curation-evaluator'],
)
def test_the_callers_of_triton_are_told_to_load_on_demand(service: str) -> None:
    env = _load_yaml(OVERLAY)['services'][service]['environment']
    assert 'OP_LOAD_PROFILE=ingest' in env


def test_every_triton_caller_of_the_base_file_is_covered() -> None:
    """A service that reaches Triton and is missing from the overlay would fail on a model the
    profile does not preload."""
    base = yaml.safe_load(COMPOSE_PATH.read_text())['services']
    calls_triton = {
        name
        for name, svc in base.items()
        if any('TRITON_URL' in str(e) for e in svc.get('environment', []))
    }
    covered = set(_load_yaml(OVERLAY)['services']) - {'triton-server'}
    assert calls_triton - covered == set(), 'add these services to docker-compose.ingest-focus.yml'
