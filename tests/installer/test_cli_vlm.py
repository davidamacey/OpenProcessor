"""``openprocessor vlm status|use|apply|probe`` (any-domain W9.7), driven
through the installer shim harness: docker, curl, nvidia-smi are test
doubles, so nothing here starts a container or touches a GPU."""

from __future__ import annotations

import stat
from typing import TYPE_CHECKING

import pytest
import test_cli_project
from installer_harness import GPU_48, PROJECT
from test_cli_project import HOSTILE, cli, make_hostile


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

# The fixture that runs the real installer into a temp dir (defined once, in
# the project-safety tests) is reused as is.
inst = test_cli_project.inst

QWEN = 'qwen3-vl-4b'
QWEN_REPO = 'Qwen/Qwen3-VL-4B-Instruct'
GPU_12 = '0, NVIDIA GeForce RTX 3080 Ti, 12288, 0, 8.6\n'
# one fast poll so a failed wait ends the test in a couple of seconds
WAIT = {'OP_VLM_WAIT_S': '2', 'OP_VLM_POLL_S': '1'}


def env_value(inst: Path, key: str) -> str | None:
    found = None
    for line in (inst / '.env').read_text().splitlines():
        if line.startswith(f'{key}='):
            found = line.split('=', 1)[1]
    return found


def set_env(inst: Path, **values: str) -> None:
    text = (inst / '.env').read_text()
    for key, value in values.items():
        lines = [ln for ln in text.splitlines() if not ln.startswith(f'{key}=')]
        lines.append(f'{key}={value}')
        text = '\n'.join(lines) + '\n'
    (inst / '.env').write_text(text)


@pytest.fixture
def stack(shimmed: Shimmed, inst: Path) -> Path:
    """An install with an in-compose vlm, on one 48 GB card, mutations on."""
    shimmed.flag('allow_mutations')
    shimmed.gpus(GPU_48)
    set_env(inst, OP_LOCAL_VLM_ENDPOINT='env', VLM_GPU_ID='0')
    return inst


def use(shimmed: Shimmed, inst: Path, *args: str, root: str = QWEN_REPO, **env: str):
    return cli(shimmed, inst, 'vlm', 'use', QWEN, '--yes', *args, SHIM_VLM_ROOT=root, **WAIT, **env)


def compose_calls(shimmed: Shimmed) -> list[str]:
    """Each state-changing compose call, as what follows the last ``-f FILE``;
    the shim writes one ``EXEC <action>`` line per state-volume action."""
    calls = []
    for line in shimmed.mutating_docker_calls():
        tokens = line.split()
        if tokens[:3] == ['docker', 'compose', 'EXEC']:
            calls.append(f'exec {tokens[3]}')
            continue
        if tokens[:2] != ['docker', 'compose'] or '-f' not in tokens:
            continue  # a continuation line of a multi-line exec script
        last_file = max(i for i, t in enumerate(tokens) if t == '-f')
        call = ' '.join(tokens[last_file + 2 :])
        if not call.startswith('exec '):  # its EXEC line stands for it
            calls.append(call)
    return calls


def http_calls(shimmed: Shimmed) -> list[str]:
    return [ln for ln in shimmed.log_lines('curl') if '/curation/' in ln]


# ---- the happy path ----------------------------------------------------------


def test_use_rewrites_env_recreates_the_vlm_probes_and_unpauses(
    shimmed: Shimmed, stack: Path
) -> None:
    result = use(shimmed, stack)
    assert result.returncode == 0, result.stdout + result.stderr

    assert env_value(stack, 'VLM_CATALOG_ID') == QWEN
    assert env_value(stack, 'VLM_MODEL') == QWEN_REPO
    assert (env_value(stack, 'VLM_IMAGE') or '').startswith('vllm/vllm-openai')
    assert '@sha256:' in (env_value(stack, 'VLM_IMAGE') or '')
    assert env_value(stack, 'VLM_REASONING_PARSER') == ''  # set but empty: no parser
    assert env_value(stack, 'VLM_CHAT_TEMPLATE') == ''
    assert env_value(stack, 'VLM_MAX_MODEL_LEN') == '16384'
    assert env_value(stack, 'VLM_LIMIT_MM_IMAGES') == '8'
    assert env_value(stack, 'OP_VLM_MAX_IMAGES_PER_CALL') == '8'
    # clamp((vram_gb - 3) / card_gb, 0.2, 0.9): 17 GB on a 48 GB card
    assert env_value(stack, 'VLM_GPU_MEMORY_UTILIZATION') == '0.29'
    assert stat.S_IMODE((stack / '.env').stat().st_mode) == 0o600
    assert sorted(p.name for p in stack.glob('.env*')) == ['.env']  # no backup left behind

    calls = compose_calls(shimmed)
    assert [c for c in calls if 'up -d' in c] == ['up -d vlm']
    assert calls == ['exec pause-create', 'up -d vlm', 'exec pause-remove']
    urls = http_calls(shimmed)
    probe = [i for i, u in enumerate(urls) if '/vlm/endpoints/env/probe' in u]
    clear = [i for i, u in enumerate(urls) if '/vlm/local/select' in u]
    assert probe
    assert clear
    assert probe[0] < clear[0]  # the request is dropped only after the new identity was recorded
    assert not (shimmed.state / 'pause_sentinel').exists()
    assert 'warning: Multi-box regions' in result.stdout  # the probe's pairing warnings


def test_the_first_use_migrates_to_the_stable_alias_and_recreates_the_running_services(
    shimmed: Shimmed, stack: Path
) -> None:
    (shimmed.state / 'running_services').write_text(
        'yolo-api\ncuration-detection-worker\nsegmenter\n'
    )
    result = use(shimmed, stack)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'this one time the API and workers are recreated too' in result.stdout + result.stderr
    assert env_value(stack, 'VLM_SERVED_MODEL_NAME') == 'local-vlm'
    assert env_value(stack, 'OP_VLM_MODEL') == 'local-vlm'
    ups = [c for c in compose_calls(shimmed) if 'up -d' in c]
    assert ups == ['up -d vlm yolo-api curation-detection-worker']  # never the segmenter


def test_once_on_the_alias_only_the_vlm_container_is_recreated(
    shimmed: Shimmed, stack: Path
) -> None:
    set_env(stack, VLM_SERVED_MODEL_NAME='local-vlm', OP_VLM_MODEL='local-vlm')
    (shimmed.state / 'running_services').write_text('yolo-api\ncurie\n')
    assert use(shimmed, stack).returncode == 0
    assert [c for c in compose_calls(shimmed) if 'up -d' in c] == ['up -d vlm']


def test_a_pause_someone_else_made_is_left_in_place(shimmed: Shimmed, stack: Path) -> None:
    (shimmed.state / 'pause_sentinel').write_text('')
    assert use(shimmed, stack).returncode == 0
    assert (shimmed.state / 'pause_sentinel').exists()
    assert 'exec pause-remove' not in compose_calls(shimmed)


# ---- refusals before anything changes -----------------------------------------


def _untouched(shimmed: Shimmed, stack: Path, before: str) -> None:
    assert (stack / '.env').read_text() == before
    assert [c for c in compose_calls(shimmed) if 'up -d' in c] == []
    assert not (shimmed.state / 'pause_sentinel').exists()


def test_an_unknown_id_is_refused(shimmed: Shimmed, stack: Path) -> None:
    before = (stack / '.env').read_text()
    result = cli(shimmed, stack, 'vlm', 'use', 'not-a-model', '--yes')
    assert result.returncode == 1
    assert 'not in the catalog' in result.stdout + result.stderr
    _untouched(shimmed, stack, before)


def test_a_model_that_does_not_fit_is_refused_unless_forced(shimmed: Shimmed, stack: Path) -> None:
    shimmed.gpus(GPU_12)
    before = (stack / '.env').read_text()
    refused = use(shimmed, stack)
    assert refused.returncode == 1
    assert 'GPU 0 has 12.0 GB' in refused.stdout + refused.stderr  # the card, not just free memory
    _untouched(shimmed, stack, before)
    forced = use(shimmed, stack, '--force')
    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert env_value(stack, 'VLM_CATALOG_ID') == QWEN


def test_a_card_with_too_little_free_memory_is_refused(shimmed: Shimmed, stack: Path) -> None:
    shimmed.gpus('0, NVIDIA RTX A6000, 49140, 40000, 8.6\n')
    before = (stack / '.env').read_text()
    result = use(shimmed, stack)
    assert result.returncode == 1
    assert 'free' in result.stdout + result.stderr
    _untouched(shimmed, stack, before)


def test_a_running_training_job_blocks_the_switch(shimmed: Shimmed, stack: Path) -> None:
    (shimmed.state / 'training_lock').write_text('')
    before = (stack / '.env').read_text()
    result = use(shimmed, stack)
    assert result.returncode == 1
    assert 'training' in result.stdout + result.stderr
    _untouched(shimmed, stack, before)


def test_a_stack_without_the_in_compose_vlm_is_refused(shimmed: Shimmed, stack: Path) -> None:
    set_env(stack, OP_LOCAL_VLM_ENDPOINT='')
    result = use(shimmed, stack)
    assert result.returncode == 1
    assert 'OP_LOCAL_VLM_ENDPOINT' in result.stdout + result.stderr


def test_an_unreachable_api_container_aborts_before_touching_env(
    shimmed: Shimmed, stack: Path
) -> None:
    (shimmed.state / 'allow_mutations').unlink()  # exec is refused, like a stopped stack
    before = (stack / '.env').read_text()
    result = use(shimmed, stack)
    assert result.returncode == 1
    assert 'cannot reach the yolo-api container' in result.stdout + result.stderr
    assert (stack / '.env').read_text() == before


def test_a_gated_model_needs_a_working_token_and_it_never_reaches_an_argv(
    shimmed: Shimmed, stack: Path
) -> None:
    catalog = stack / 'examples' / 'vlm' / 'catalog.tsv'
    rows = catalog.read_text().splitlines()
    header = rows[0].split('\t')
    gated = header.index('gated')
    edited = []
    for row in rows:
        cols = row.split('\t')
        if cols[0] == QWEN:
            cols[gated] = 'true'
        edited.append('\t'.join(cols))
    catalog.write_text('\n'.join(edited) + '\n')

    before = (stack / '.env').read_text()
    no_token = use(shimmed, stack)
    assert no_token.returncode == 1
    assert 'HF_TOKEN' in no_token.stdout + no_token.stderr
    _untouched(shimmed, stack, before)

    set_env(stack, HF_TOKEN='hf_secret_token_value')
    before = (stack / '.env').read_text()
    (shimmed.state / 'hf_code').write_text('403')
    denied = use(shimmed, stack)
    assert denied.returncode == 1
    assert '403' in denied.stdout + denied.stderr
    _untouched(shimmed, stack, before)

    (shimmed.state / 'hf_code').write_text('200')
    assert use(shimmed, stack).returncode == 0
    log = shimmed.log.read_text()
    assert 'hf_secret_token_value' not in log
    assert '-H @' in log  # through a header file


# ---- failure after the change: everything is put back ----------------------------


def _restored(shimmed: Shimmed, stack: Path, before: str) -> None:
    assert (stack / '.env').read_text() == before
    assert stat.S_IMODE((stack / '.env').stat().st_mode) == 0o600
    assert sorted(p.name for p in stack.glob('.env*')) == ['.env']
    ups = [c for c in compose_calls(shimmed) if 'up -d' in c]
    assert ups == ['up -d vlm', 'up -d vlm']  # the new values, then the old ones
    assert not (shimmed.state / 'pause_sentinel').exists()


def test_a_model_that_never_comes_up_restores_env_and_unpauses(
    shimmed: Shimmed, stack: Path
) -> None:
    before = (stack / '.env').read_text()
    result = use(shimmed, stack, root='some/other-model')  # the server keeps serving another root
    assert result.returncode == 1
    assert 'did not serve' in result.stdout + result.stderr
    _restored(shimmed, stack, before)
    assert not [u for u in http_calls(shimmed) if '/vlm/endpoints/env/probe' in u]


def test_a_failed_probe_restores_env_and_leaves_the_request_in_place(
    shimmed: Shimmed, stack: Path
) -> None:
    (shimmed.state / 'probe_fail').write_text('')
    before = (stack / '.env').read_text()
    result = use(shimmed, stack)
    assert result.returncode == 1
    assert 'probing' in result.stdout + result.stderr
    _restored(shimmed, stack, before)
    assert not [u for u in http_calls(shimmed) if '/vlm/local/select' in u]


def test_a_pause_that_was_not_ours_survives_a_failed_switch(shimmed: Shimmed, stack: Path) -> None:
    (shimmed.state / 'pause_sentinel').write_text('')
    result = use(shimmed, stack, root='some/other-model')
    assert result.returncode == 1
    assert (shimmed.state / 'pause_sentinel').exists()


# ---- apply, status, probe ------------------------------------------------------------


def test_apply_uses_the_model_the_api_recorded(shimmed: Shimmed, stack: Path) -> None:
    (shimmed.state / 'vlm_local.json').write_text(
        '{"configured":true,"desired":{"catalog_id":"qwen3-vl-4b","requested_at":"t"}}'
    )
    result = cli(shimmed, stack, 'vlm', 'apply', '--yes', SHIM_VLM_ROOT=QWEN_REPO, **WAIT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert env_value(stack, 'VLM_CATALOG_ID') == QWEN


def test_apply_with_nothing_requested_changes_nothing(shimmed: Shimmed, stack: Path) -> None:
    (shimmed.state / 'vlm_local.json').write_text('{"configured":true,"desired":null}')
    before = (stack / '.env').read_text()
    result = cli(shimmed, stack, 'vlm', 'apply')
    assert result.returncode == 0
    assert 'nothing to apply' in result.stdout + result.stderr
    _untouched(shimmed, stack, before)


def test_status_reports_what_env_and_the_api_say(shimmed: Shimmed, stack: Path) -> None:
    set_env(stack, VLM_CATALOG_ID='gemma-4-e4b', VLM_MODEL='google/gemma-4-E4B-it')
    (shimmed.state / 'vlm_local.json').write_text(
        '{"served":{"root":"google/gemma-4-E4B-it"},"desired":{"catalog_id":"qwen3-vl-4b"},'
        '"restart_required":true}'
    )
    result = cli(shimmed, stack, 'vlm', 'status')
    assert result.returncode == 0
    out = result.stdout + result.stderr
    assert 'gemma-4-e4b' in out
    assert 'google/gemma-4-E4B-it' in out
    assert 'qwen3-vl-4b' in out
    assert 'openprocessor vlm apply' in out
    assert shimmed.mutating_docker_calls() == []


def test_probe_posts_to_the_local_endpoint_and_prints_pairing_warnings(
    shimmed: Shimmed, stack: Path
) -> None:
    result = cli(shimmed, stack, 'vlm', 'probe')
    assert result.returncode == 0, result.stdout + result.stderr
    assert any('/curation/vlm/endpoints/env/probe' in u for u in http_calls(shimmed))
    assert 'warning: Multi-box regions' in result.stdout


def test_a_failed_probe_command_exits_nonzero(shimmed: Shimmed, stack: Path) -> None:
    (shimmed.state / 'probe_fail').write_text('')
    assert cli(shimmed, stack, 'vlm', 'probe').returncode == 1


# ---- the same ownership rules as every other state-changing command ------------------


@pytest.mark.parametrize('case', HOSTILE)
def test_use_can_never_touch_a_project_it_did_not_create(
    shimmed: Shimmed, stack: Path, case: str
) -> None:
    env = make_hostile(shimmed, stack, case)
    result = cli(shimmed, stack, 'vlm', 'use', QWEN, '--yes', **env)
    assert result.returncode != 0, case
    assert shimmed.mutating_docker_calls() == [], f'{case}: {shimmed.mutating_docker_calls()}'
    assert PROJECT  # the fixture project name is the only one ever used


def test_the_help_lists_every_subcommand(shimmed: Shimmed, stack: Path) -> None:
    result = cli(shimmed, stack, 'help')
    text = result.stdout + result.stderr
    assert 'vlm list|status|use <id> [--force] [--yes]|apply|probe|key set <slug>' in text
