"""triton_unload_if_ready: a ready model is unloaded and awaited before any load."""

from __future__ import annotations

from typing import TYPE_CHECKING

from installer_harness import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path

LIB = REPO_ROOT / 'scripts/lib/triton_load.sh'


def _run(bash, tmp_path: Path, ready_codes: list[str], timeout: int = 3):
    state = tmp_path / 'ready'
    state.write_text('\n'.join(ready_codes) + '\n')
    log = tmp_path / 'calls'
    script = f"""
sleep() {{ :; }}
curl() {{
    local url="${{@: -1}}"
    case "$url" in
        */ready) echo "ready" >> {log}
            local c; c="$(head -n1 {state})"
            if [[ "$(wc -l < {state})" -gt 1 ]]; then sed -i 1d {state}; fi
            printf '%s' "$c" ;;
        */unload) echo unload >> {log} ;;
    esac
}}
source {LIB}
triton_unload_if_ready http://t m {timeout}
echo "rc=$?"
"""
    result = bash(script)
    calls = log.read_text().split() if log.exists() else []
    return result.stdout.strip().splitlines()[-1], calls


def test_ready_model_is_unloaded_and_awaited(bash, tmp_path: Path) -> None:
    rc, calls = _run(bash, tmp_path, ['200', '200', '400'])
    assert rc == 'rc=0'
    assert calls == ['ready', 'unload', 'ready', 'ready']


def test_not_loaded_model_sends_no_unload(bash, tmp_path: Path) -> None:
    rc, calls = _run(bash, tmp_path, ['400'])
    assert rc == 'rc=0'
    assert 'unload' not in calls


def test_a_model_that_never_unloads_fails(bash, tmp_path: Path) -> None:
    rc, _ = _run(bash, tmp_path, ['200'])
    assert rc == 'rc=1'


def test_an_unknown_state_fails_without_unloading(bash, tmp_path: Path) -> None:
    rc, calls = _run(bash, tmp_path, ['503'])
    assert rc == 'rc=2'
    assert 'unload' not in calls
