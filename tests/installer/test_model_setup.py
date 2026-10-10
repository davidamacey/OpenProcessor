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


# ---- PE precision fallback must be loud, recorded and never a clean success ----

FP16_TRT_ERROR = (
    '[E] Error[4]: ITensor::getDimensions: Error Code 4: API Usage Error '
    '(/visual/Einsum: IEinsumLayer must have all inputs of same type.)'
)


def _pe_group_script(tmp_path: Path, *, fp16_trtexec_fails: bool) -> str:
    """Runs the real `pe` group with dc() standing in for docker compose:
    trtexec fails on the FP16-baked ONNX (first call) when asked to, and
    succeeds once the FP32 ONNX has been copied into place."""
    fail = 'true' if fp16_trtexec_fails else 'false'
    return (
        f'source {LIB}; OP_DIR="{tmp_path}"; mkdir -p "{tmp_path}/.install"; sleep() {{ :; }}; '
        'triton_load_and_wait() { return 0; }; '
        'fp32_staged=0; '
        'dc() { '
        '  if [[ "$*" == *"cp /app/pytorch_models/pe_image_encoder.onnx"* ]]; then touch "$OP_DIR/fp32_staged"; fi; '
        f'  if [[ "$*" == *trtexec* && {fail} == true && ! -e "$OP_DIR/fp32_staged" ]]; then '
        f"    echo '{FP16_TRT_ERROR}'; return 1; "
        '  fi; return 0; }; '
        'model_setup_run_group pe; echo "rc=$?"'
    )


def test_pe_fp32_fallback_warns_loudly_and_is_recorded_as_degraded(tmp_path: Path, bash) -> None:
    result = bash(_pe_group_script(tmp_path, fp16_trtexec_fails=True))
    out = result.stdout + result.stderr
    assert 'rc=0' in out  # the FP32 engine is kept: better than none
    assert '[WARN]' in out
    assert 'pe_image_encoder' in out
    assert 'FP32' in out
    assert 'IEinsumLayer must have all inputs of same type' in out  # the reason
    assert 'throughput' in out  # the consequence
    assert 'models install --only pe' in out
    statuses = dict(
        ln.split('\t')[:2] for ln in (tmp_path / '.install' / 'groups.tsv').read_text().splitlines()
    )
    assert statuses['pe'] == 'degraded'
    model, precision, reason = (
        (tmp_path / '.install' / 'precision.tsv').read_text().rstrip().split('\t')
    )
    assert (model, precision) == ('pe_image_encoder', 'fp32_fallback')
    assert 'IEinsumLayer' in reason


def test_pe_fp16_build_is_a_clean_ok_with_no_warning(tmp_path: Path, bash) -> None:
    result = bash(_pe_group_script(tmp_path, fp16_trtexec_fails=False))
    out = result.stdout + result.stderr
    assert 'rc=0' in out
    assert '[WARN]' not in out
    statuses = dict(
        ln.split('\t')[:2] for ln in (tmp_path / '.install' / 'groups.tsv').read_text().splitlines()
    )
    assert statuses['pe'] == 'ok'
    assert (tmp_path / '.install' / 'precision.tsv').read_text().split('\t')[:2] == [
        'pe_image_encoder',
        'fp16',
    ]


def test_a_degraded_group_is_not_skipped_as_up_to_date(tmp_path: Path, bash) -> None:
    models = tmp_path / 'models'
    (models / 'pe_image_encoder' / '1').mkdir(parents=True)
    (models / 'pe_image_encoder' / '1' / 'model.plan').write_text('x')
    (models / 'pe_text_encoder' / '1').mkdir(parents=True)
    (models / 'pe_text_encoder' / '1' / 'model.onnx').write_text('x')
    (tmp_path / '.install').mkdir()
    (tmp_path / '.install' / 'state.json').write_text('{"triton_image_digest": "sha256:abc"}')
    base = (
        f'source {LIB}; OP_DIR="{tmp_path}"; MODEL_SETUP_TRITON_DIGEST=sha256:abc; '
        'triton_model_ready() { return 0; }; triton_load_and_wait() { return 0; }; '
        'dc() { echo STEP-RAN; return 0; }; '
    )
    (tmp_path / '.install' / 'groups.tsv').write_text('pe\tok\t5\n')
    assert 'up to date, skipped' in bash(base + 'model_setup_run_group pe').stdout
    (tmp_path / '.install' / 'groups.tsv').write_text('pe\tdegraded\t5\n')
    rerun = bash(base + 'model_setup_run_group pe').stdout
    assert 'skipped' not in rerun
    assert 'STEP-RAN' not in rerun  # step output goes to the step log...
    assert 'step pe_weights: ok' in rerun  # ...but the steps did run again


def test_precision_report_lists_only_non_fp16_models(tmp_path: Path, bash) -> None:
    (tmp_path / '.install').mkdir()
    (tmp_path / '.install' / 'precision.tsv').write_text(
        'yolo\tfp16\t\npe_image_encoder\tfp32_fallback\tIEinsumLayer mismatch\n'
    )
    out = bash(f'source {LIB}; OP_DIR="{tmp_path}" model_setup_precision_report').stdout
    assert out.strip() == 'pe_image_encoder\tfp32_fallback\tIEinsumLayer mismatch'
