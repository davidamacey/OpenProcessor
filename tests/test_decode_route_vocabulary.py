"""The decode-route vocabulary is closed and covers every status the frame engine can answer."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.domain.decode_route import (
    CPU_REASONS,
    DECODE_ROUTES,
    PREP_STATUSES,
    check_route,
    cpu_reason_route,
    prep_status_route,
)


ENGINE_DIR = Path(__file__).resolve().parents[1] / 'frame_engine'


def _route_reasons() -> tuple[set[str], set[str]]:
    """What ``prep_geometry.route`` answers, by kind: the reasons of ``Route('cpu', reason, ...)``
    (and what ``_structural_reason`` returns) and the statuses of ``Route('refused', ...)``."""
    tree = ast.parse((ENGINE_DIR / 'prep_geometry.py').read_text())
    cpu: set[str] = set()
    refused: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'Route'
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Constant)
        ):
            kind = node.args[0].value
            if kind == 'cpu':
                cpu.add(str(node.args[1].value))
            elif kind == 'refused':
                refused.add(str(node.args[1].value))
        if isinstance(node, ast.FunctionDef) and node.name == '_structural_reason':
            cpu.update(
                str(r.value.value)
                for r in ast.walk(node)
                if isinstance(r, ast.Return) and isinstance(r.value, ast.Constant)
            )
            cpu.update(
                str(c.value)
                for r in ast.walk(node)
                if isinstance(r, ast.Return) and isinstance(r.value, ast.IfExp)
                for c in ast.walk(r.value)
                if isinstance(c, ast.Constant)
            )
    return cpu - {''}, refused


def _engine_statuses() -> set[str]:
    """The literals ``engine.py`` answers a photo with: the status it passes to ``_failed`` and
    what ``_source``, ``_decode`` and ``_decode_host`` return as a status string."""
    tree = ast.parse((ENGINE_DIR / 'engine.py').read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == '_failed'
            and isinstance(node.args[1], ast.Constant)
        ):
            found.add(str(node.args[1].value))
        if isinstance(node, ast.FunctionDef) and node.name in (
            '_source',
            '_decode',
            '_decode_host',
        ):
            found.update(
                str(r.value.value)
                for r in ast.walk(node)
                if isinstance(r, ast.Return) and isinstance(r.value, ast.Constant)
            )
    return found


def test_every_status_the_engine_answers_has_a_route() -> None:
    engine = _engine_statuses()
    assert {'over_budget', 'gpu_memory', 'decoder', 'huge'} <= engine, engine
    _, refused = _route_reasons()
    answered = refused | engine
    assert answered, 'the scan found no statuses'
    assert answered <= set(PREP_STATUSES), answered - set(PREP_STATUSES)
    for status in answered:
        assert prep_status_route(status) in DECODE_ROUTES


def test_every_reason_the_engine_decodes_on_the_host_has_a_route() -> None:
    cpu, _ = _route_reasons()
    assert cpu == set(CPU_REASONS), (cpu, CPU_REASONS)
    for reason in cpu:
        assert cpu_reason_route(reason) == f'cpu_{reason}'
        assert cpu_reason_route(reason) in DECODE_ROUTES


def test_a_reason_the_engine_decodes_is_not_also_a_status_it_refuses_with() -> None:
    assert not set(CPU_REASONS) & set(PREP_STATUSES)


def test_the_names_are_unique_and_cpu_routes_are_prefixed() -> None:
    assert len(set(DECODE_ROUTES)) == len(DECODE_ROUTES)
    assert all(r == 'gpu' or r.startswith('cpu_') for r in DECODE_ROUTES)


@pytest.mark.parametrize('status', ['telepathy', 'ok', ''])
def test_a_status_outside_the_vocabulary_is_rejected(status: str) -> None:
    with pytest.raises(ValueError, match='unknown decode route'):
        prep_status_route(status)
    with pytest.raises(ValueError, match='unknown decode route'):
        check_route(f'cpu_{status}')


@pytest.mark.parametrize('reason', ['telepathy', 'ok', 'huge', 'decoder', ''])
def test_a_cpu_reason_outside_the_vocabulary_is_rejected(reason: str) -> None:
    with pytest.raises(ValueError, match='unknown decode route reason'):
        cpu_reason_route(reason)


def test_the_call_failure_word_is_in_the_vocabulary() -> None:
    assert prep_status_route('call_failed') == 'cpu_call_failed'
