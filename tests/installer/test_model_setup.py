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


def test_classify_failure_ignores_a_bare_403_count(tmp_path: Path, bash) -> None:
    log = tmp_path / 'step.log'
    log.write_text('downloaded 403 files in 12s\n')
    assert bash(_source(f'classify_failure "{log}"')).stdout.strip() == 'unknown'


def test_classify_failure_http_auth_errors_are_gated(tmp_path: Path, bash) -> None:
    for text in (
        'urllib.error.HTTPError: HTTP Error 403: Forbidden',
        'requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url',
        'huggingface_hub: status code 403',
    ):
        log = tmp_path / 'step.log'
        log.write_text(text + '\n')
        assert bash(_source(f'classify_failure "{log}"')).stdout.strip() == 'permanent:gated', text


def test_retry_step_leaves_the_callers_errexit_alone(tmp_path: Path, bash) -> None:
    off = bash(
        f'source {LIB}; MODEL_SETUP_LOGDIR="{tmp_path}" retry_step ok 1 -- true; '
        '[[ $- == *e* ]] && echo ON || echo OFF'
    )
    assert off.stdout.strip().splitlines()[-1] == 'OFF'
    on = bash(
        f'set -e; source {LIB}; MODEL_SETUP_LOGDIR="{tmp_path}" retry_step ok 1 -- true; '
        '[[ $- == *e* ]] && echo ON || echo OFF'
    )
    assert on.stdout.strip().splitlines()[-1] == 'ON'


def test_step_logs_are_private_and_redacted(tmp_path: Path, bash) -> None:
    secret = 'hf_ABCDEFGHIJKLMNOP'  # gitleaks:allow
    result = bash(
        f'source {LIB}; OP_DIR="{tmp_path}" retry_step leaky 1 -- '
        f'bash -c \'echo "token {secret}"; echo "Authorization: Bearer abc"; echo "X_API_KEY=zzz"\''
    )
    assert result.returncode == 0, result.stderr
    log = tmp_path / '.install' / 'logs' / 'leaky.log'
    text = log.read_text()
    assert secret not in text
    assert 'Bearer abc' not in text
    assert 'zzz' not in text
    assert (log.stat().st_mode & 0o777) == 0o600
    assert (log.parent.stat().st_mode & 0o777) == 0o700


def test_step_logs_never_default_to_the_working_directory(tmp_path: Path, bash) -> None:
    result = bash(
        f'cd "{tmp_path}"; source {LIB}; unset OP_DIR MODEL_SETUP_LOGDIR; retry_step x 1 -- true'
    )
    assert result.returncode != 0
    assert not (tmp_path / 'x.log').exists()


def test_groups_follow_the_tiers(bash) -> None:
    core = bash(_source('model_setup_groups_for_tiers "core"')).stdout.split()
    assert core == ['preflight', 'base', 'yolo', 'mobileclip', 'faces', 'ocr']
    cur = bash(_source('model_setup_groups_for_tiers "core curation"')).stdout.split()
    assert cur[-1] == 'pe'


def test_a_failed_group_does_not_stop_later_groups(tmp_path: Path, bash) -> None:
    script = (
        f'source {LIB}; OP_DIR="{tmp_path}"; mkdir -p "{tmp_path}/.install"; sleep() {{ :; }}; '
        'triton_load_and_wait() { return 0; }; '
        'dc() { if [[ "$*" == *export_mobileclip_image* ]]; then echo "ModuleNotFoundError: x"; return 1; fi; return 0; }; '
        'model_setup_run_groups "core"; echo "rc=$?"'
    )
    result = bash(script)
    assert 'failed_groups=mobileclip' in result.stdout
    assert 'rc=1' in result.stdout
    statuses = dict(
        ln.split('\t')[:2] for ln in (tmp_path / '.install' / 'groups.tsv').read_text().splitlines()
    )
    assert statuses['mobileclip'] == 'failed'
    assert statuses['faces'] == statuses['ocr'] == 'ok'


def test_classify_failure_trt_cuda_init_error_2_is_oom(tmp_path: Path, bash) -> None:
    """TensorRT's message when the card is full (#111): not a transient race."""
    log = tmp_path / 'step.log'
    log.write_text(
        '[TRT] [E] createInferBuilder: Error Code 6: API Usage Error '
        '(CUDA initialization failure with error: 2. Please check your CUDA installation)\n'
    )
    assert bash(_source(f'classify_failure "{log}"')).stdout.strip() == 'permanent:oom'


def test_classify_failure_cuda_init_other_errors_stay_transient(tmp_path: Path, bash) -> None:
    for text in (
        'CUDA initialization failure with error: 100\n',
        'CUDA initialization failure with error: 205\n',
    ):
        log = tmp_path / 'step.log'
        log.write_text(text)
        assert bash(_source(f'classify_failure "{log}"')).stdout.strip() == 'transient', text


def test_retry_step_does_not_retry_an_export_oom(tmp_path: Path, bash) -> None:
    marker = tmp_path / 'attempt_count'
    fake_step = tmp_path / 'fake_step.sh'
    fake_step.write_text(
        '#!/bin/bash\n'
        f'count_file="{marker}"\n'
        'n=$(cat "$count_file" 2>/dev/null || echo 0)\n'
        'echo $((n + 1)) > "$count_file"\n'
        'echo "[TRT] [E] createInferBuilder: CUDA initialization failure with error: 2" >&2\n'
        'exit 1\n'
    )
    fake_step.chmod(0o755)
    script = _source(
        f'MODEL_SETUP_LOGDIR="{tmp_path}/logs" retry_step yolo_export 3 -- bash "{fake_step}"'
    )
    result = bash(f'sleep() {{ :; }}; {script}')
    assert result.returncode != 0
    assert marker.read_text().strip() == '1'
    assert 'permanent failure (oom)' in result.stdout + result.stderr
    assert '--gpu-plan' in result.stdout + result.stderr
