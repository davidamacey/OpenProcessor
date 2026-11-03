"""Worker recycling is staggered: a service that recycles after N requests also sets a jitter."""

from __future__ import annotations

from compose._helpers import _services


def _args(service: dict[str, object]) -> list[str]:
    command = service.get('command') or []
    return command.split() if isinstance(command, str) else [str(a) for a in command]


def _flag(args: list[str], name: str) -> str | None:
    for arg in args:
        if arg.startswith(f'{name}='):
            return arg.split('=', 1)[1]
    return None


def test_api_recycle_sets_jitter() -> None:
    args = _args(_services()['api'])
    assert _flag(args, '--limit-max-requests') is not None, 'api no longer recycles workers'
    jitter = _flag(args, '--limit-max-requests-jitter')
    assert jitter is not None, 'workers recycle in lockstep: --limit-max-requests-jitter missing'
    assert 0 < int(jitter) < int(_flag(args, '--limit-max-requests') or 0)


def test_every_recycling_service_sets_jitter() -> None:
    for name, svc in _services().items():
        args = _args(svc)
        if _flag(args, '--limit-max-requests') is not None:
            assert _flag(args, '--limit-max-requests-jitter') is not None, name


def test_control_flag_parser_tells_present_from_absent() -> None:
    assert _flag(['--limit-max-requests=10', '--x'], '--limit-max-requests') == '10'
    assert _flag(['--limit-max-requests=10'], '--limit-max-requests-jitter') is None
    assert (
        _flag(
            ['--limit-max-requests=10', '--limit-max-requests-jitter=1'],
            '--limit-max-requests-jitter',
        )
        == '1'
    )
