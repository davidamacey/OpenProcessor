"""Unit tests for scripts/lib/model_setup.sh's retry classifier and skip
logic (installer plan section 9.2, last paragraph)."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


LIB = 'scripts/lib/model_setup.sh'


def _source(extra: str) -> str:
    return f'set -euo pipefail; source {LIB}; {extra}'


def test_classify_failure_transient(tmp_path: Path, bash) -> None:
    log = tmp_path / 'step.log'
    log.write_text('some noise\nCUDA initialization failure: error 100\nmore noise\n')
    result = bash(_source(f'classify_failure "{log}"'))
    assert result.returncode == 0
    assert result.stdout.strip() == 'transient'


def test_classify_failure_permanent_module_not_found(tmp_path: Path, bash) -> None:
    log = tmp_path / 'step.log'
    log.write_text("Traceback ...\nModuleNotFoundError: No module named 'core'\n")
    result = bash(_source(f'classify_failure "{log}"'))
    assert result.stdout.strip() == 'permanent:stale_image'


def test_classify_failure_permanent_gated(tmp_path: Path, bash) -> None:
    log = tmp_path / 'step.log'
    log.write_text('huggingface_hub.utils.GatedRepoError: access denied\n')
    result = bash(_source(f'classify_failure "{log}"'))
    assert result.stdout.strip() == 'permanent:gated'


def test_classify_failure_unknown(tmp_path: Path, bash) -> None:
    log = tmp_path / 'step.log'
    log.write_text('something totally unrelated blew up\n')
    result = bash(_source(f'classify_failure "{log}"'))
    assert result.stdout.strip() == 'unknown'


def test_permanent_hint_matches_class(bash) -> None:
    result = bash(_source('permanent_hint "permanent:stale_image"'))
    assert 'repair --images' in result.stdout


def test_retry_step_retries_on_transient_then_succeeds(tmp_path: Path, bash) -> None:
    # A fake step that fails once with a transient signature, then succeeds.
    marker = tmp_path / 'attempt_count'
    fake_step = tmp_path / 'fake_step.sh'
    fake_step.write_text(
        '#!/bin/bash\n'
        f'count_file="{marker}"\n'
        'n=$(cat "$count_file" 2>/dev/null || echo 0)\n'
        'n=$((n + 1))\n'
        'echo "$n" > "$count_file"\n'
        'if [[ "$n" -lt 2 ]]; then\n'
        '  echo "CUDA initialization failure: error 100" >&2\n'
        '  exit 1\n'
        'fi\n'
        'echo ok\n'
    )
    fake_step.chmod(0o755)

    script = _source(
        f'MODEL_SETUP_LOGDIR="{tmp_path}/logs" retry_step fake_transient 3 -- bash "{fake_step}"'
    )
    # Backoff is 15s/45s; patch sleep away for a fast test.
    script = f'sleep() {{ :; }}; {script}'
    result = bash(script)
    assert result.returncode == 0, result.stderr
    assert marker.read_text().strip() == '2'


def test_retry_step_no_retry_on_permanent(tmp_path: Path, bash) -> None:
    marker = tmp_path / 'attempt_count'
    fake_step = tmp_path / 'fake_step.sh'
    fake_step.write_text(
        '#!/bin/bash\n'
        f'count_file="{marker}"\n'
        'n=$(cat "$count_file" 2>/dev/null || echo 0)\n'
        'n=$((n + 1))\n'
        'echo "$n" > "$count_file"\n'
        'echo "ModuleNotFoundError: No module named \'core\'" >&2\n'
        'exit 1\n'
    )
    fake_step.chmod(0o755)

    script = _source(
        f'MODEL_SETUP_LOGDIR="{tmp_path}/logs" retry_step fake_permanent 3 -- bash "{fake_step}"'
    )
    script = f'sleep() {{ :; }}; {script}'
    result = bash(script)
    assert result.returncode != 0
    assert marker.read_text().strip() == '1'


def test_retry_step_unknown_gets_exactly_one_retry(tmp_path: Path, bash) -> None:
    marker = tmp_path / 'attempt_count'
    fake_step = tmp_path / 'fake_step.sh'
    fake_step.write_text(
        '#!/bin/bash\n'
        f'count_file="{marker}"\n'
        'n=$(cat "$count_file" 2>/dev/null || echo 0)\n'
        'n=$((n + 1))\n'
        'echo "$n" > "$count_file"\n'
        'echo "totally unrelated failure" >&2\n'
        'exit 1\n'
    )
    fake_step.chmod(0o755)

    script = _source(
        f'MODEL_SETUP_LOGDIR="{tmp_path}/logs" retry_step fake_unknown 5 -- bash "{fake_step}"'
    )
    script = f'sleep() {{ :; }}; {script}'
    result = bash(script)
    assert result.returncode != 0
    # One retry: two total attempts, then the group fails.
    assert marker.read_text().strip() == '2'


def test_group_should_skip_true_when_outputs_and_state_match(tmp_path: Path, bash) -> None:
    out1 = tmp_path / 'a.plan'
    out1.write_text('x')
    state = tmp_path / 'state.json'
    state.write_text('{"triton_image_digest": "sha256:abc"}')

    script = _source(
        f'group_should_skip "{out1}" "" "{state}" "sha256:abc" && echo SKIP || echo RUN'
    )
    result = bash(script)
    assert result.stdout.strip() == 'SKIP'


def test_group_should_skip_false_when_digest_differs(tmp_path: Path, bash) -> None:
    out1 = tmp_path / 'a.plan'
    out1.write_text('x')
    state = tmp_path / 'state.json'
    state.write_text('{"triton_image_digest": "sha256:old"}')

    script = _source(
        f'group_should_skip "{out1}" "" "{state}" "sha256:new" && echo SKIP || echo RUN'
    )
    result = bash(script)
    assert result.stdout.strip() == 'RUN'


def test_group_should_skip_false_when_output_missing(tmp_path: Path, bash) -> None:
    state = tmp_path / 'state.json'
    state.write_text('{"triton_image_digest": "sha256:abc"}')

    script = _source(
        f'group_should_skip "{tmp_path}/missing.plan" "" "{state}" "sha256:abc" && echo SKIP || echo RUN'
    )
    result = bash(script)
    assert result.stdout.strip() == 'RUN'


def test_group_should_skip_false_when_no_state_file(tmp_path: Path, bash) -> None:
    out1 = tmp_path / 'a.plan'
    out1.write_text('x')

    script = _source(
        f'group_should_skip "{out1}" "" "{tmp_path}/missing_state.json" "sha256:abc" && echo SKIP || echo RUN'
    )
    result = bash(script)
    assert result.stdout.strip() == 'RUN'
