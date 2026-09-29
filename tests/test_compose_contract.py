"""The ``vlm`` compose service after W9.7: one compose file serves every
catalog entry, and its defaults are the command this stack was tested with.

Two independent checks: (1) a hermetic evaluation of the service's own shell
command (compose interpolation is reimplemented here, ``exec vllm`` is
replaced by ``printf``), which needs no Docker; (2) when a Docker CLI is
present, ``docker compose config`` renders the same file and the result must
agree.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / 'docker-compose.yml'
SERVER = 'python3 -m vllm.entrypoints.openai.api_server'

#: The argv the vlm service ran before W9 (docker-compose.yml at b3163f8c).
PRE_W9_ARGV = [
    '--model=google/gemma-4-E4B-it',
    '--served-model-name=gemma-4-e4b',
    '--dtype=bfloat16',
    '--max-model-len=8192',
    '--gpu-memory-utilization=0.4',
    '--limit-mm-per-prompt={"image":8,"audio":0}',
    '--chat-template=/vllm-workspace/examples/tool_chat_template_gemma4.jinja',
    '--reasoning-parser=gemma4',
    '--enable-prefix-caching',
]

_VAR = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?-)([^}]*))?\}')


def _interpolate(text: str, env: dict[str, str]) -> str:
    """Docker Compose's ``${VAR}``, ``${VAR:-d}`` (unset OR empty -> d) and
    ``${VAR-d}`` (only unset -> d)."""

    def sub(match: re.Match[str]) -> str:
        name, op, default = match.group(1), match.group(2), match.group(3)
        value = env.get(name)
        if op == ':-':
            return value if value else (default or '')
        if op == '-':
            return value if value is not None else (default or '')
        return value or ''

    return _VAR.sub(sub, text)


def _vlm_service() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE.read_text())['services']['vlm']


def _container_env(overrides: dict[str, str]) -> dict[str, str]:
    """The environment the vlm container sees, from its ``environment:``
    list with ``overrides`` playing the role of the shell / ``.env``."""
    env: dict[str, str] = {}
    for item in _vlm_service()['environment']:
        key, _, value = item.partition('=')
        env[key] = _interpolate(value, overrides)
    return env


def _evaluate(overrides: dict[str, str] | None = None) -> list[str]:
    service = _vlm_service()
    assert service['entrypoint'] == ['/bin/sh', '-c']
    (script,) = service['command']
    assert SERVER in script
    # compose turns `$$` into a literal `$`; the container's sh then expands it
    script = script.replace('$$', '$')
    script = script.replace(f'exec {SERVER}', "printf '%s\\n'")
    result = subprocess.run(
        ['/bin/sh', '-c', script],
        check=True,
        capture_output=True,
        text=True,
        env={'PATH': '/usr/bin:/bin', **_container_env(overrides or {})},
    )
    return result.stdout.splitlines()


def test_the_defaults_are_exactly_the_pre_w9_command() -> None:
    assert _evaluate() == PRE_W9_ARGV


def test_the_service_execs_the_server_so_it_receives_signals() -> None:
    (script,) = _vlm_service()['command']
    assert script.lstrip().startswith(f'exec {SERVER}')


@pytest.mark.parametrize(
    ('override', 'flag'),
    [
        ({'VLM_REASONING_PARSER': ''}, '--reasoning-parser'),
        ({'VLM_CHAT_TEMPLATE': ''}, '--chat-template'),
    ],
)
def test_a_set_but_empty_value_drops_that_flag(override: dict[str, str], flag: str) -> None:
    argv = _evaluate(override)
    assert not [a for a in argv if a.startswith(flag)]
    assert len(argv) == len(PRE_W9_ARGV) - 1


def test_an_unset_value_keeps_the_gemma_defaults_but_an_empty_one_is_not_a_default() -> None:
    """``${VAR-default}`` (not ``:-``): the CLI writes an empty value for a
    catalog model that needs no parser, and that must not fall back."""
    assert '--reasoning-parser=gemma4' in _evaluate()
    assert '--reasoning-parser=gemma4' not in _evaluate({'VLM_REASONING_PARSER': ''})


def test_a_catalog_models_own_values_replace_the_defaults() -> None:
    argv = _evaluate(
        {
            'VLM_MODEL': 'Qwen/Qwen3-VL-4B-Instruct',
            'VLM_SERVED_MODEL_NAME': 'local-vlm',
            'VLM_MAX_MODEL_LEN': '16384',
            'VLM_GPU_MEMORY_UTILIZATION': '0.55',
            'VLM_LIMIT_MM_IMAGES': '4',
            'VLM_REASONING_PARSER': '',
            'VLM_CHAT_TEMPLATE': '',
        }
    )
    assert argv == [
        '--model=Qwen/Qwen3-VL-4B-Instruct',
        '--served-model-name=local-vlm',
        '--dtype=bfloat16',
        '--max-model-len=16384',
        '--gpu-memory-utilization=0.55',
        '--limit-mm-per-prompt={"image":4,"audio":0}',
        '--enable-prefix-caching',
    ]


def test_extra_args_are_appended_and_split_on_whitespace() -> None:
    argv = _evaluate({'VLM_EXTRA_ARGS': '--quantization fp8 --enforce-eager'})
    assert argv[-3:] == ['--quantization', 'fp8', '--enforce-eager']
    assert argv[: len(PRE_W9_ARGV)] == PRE_W9_ARGV


def test_a_value_with_spaces_or_shell_syntax_stays_one_argument() -> None:
    argv = _evaluate({'VLM_MODEL': 'org/model with space; echo pwned'})
    assert argv[0] == '--model=org/model with space; echo pwned'
    assert 'pwned' not in ''.join(argv[1:])


def test_the_catalog_default_matches_the_compose_default() -> None:
    from src.services.labeling.vlm_catalog import catalog_entry

    default = catalog_entry('gemma-4-e4b')
    assert default is not None
    env = _container_env({})
    assert env['VLM_MODEL'] == default.hf_repo
    assert env['VLM_MAX_MODEL_LEN'] == str(default.max_model_len)
    assert env['VLM_LIMIT_MM_IMAGES'] == str(default.max_images)
    assert env['VLM_REASONING_PARSER'] == default.reasoning_parser
    assert env['VLM_CHAT_TEMPLATE'] == default.chat_template


def test_the_api_and_workers_get_the_local_endpoint_and_the_read_only_secrets_mount() -> None:
    services = yaml.safe_load(COMPOSE.read_text())['services']
    for name in ('yolo-api', 'curation-detection-worker'):
        env = ' '.join(services[name]['environment'])
        assert 'OP_LOCAL_VLM_ENDPOINT=${OP_LOCAL_VLM_ENDPOINT:-}' in env, name
        assert 'OP_LOCAL_VLM_GPU_TOTAL_MIB=${VLM_GPU_TOTAL_MIB:-}' in env, name
    for name in ('yolo-api', 'curation-detection-worker', 'curation-auto-label-worker'):
        assert './secrets/vlm:/run/secrets/op_vlm:ro' in services[name]['volumes'], name


@pytest.mark.skipif(shutil.which('docker') is None, reason='no docker CLI')
def test_docker_compose_renders_the_same_command() -> None:
    result = subprocess.run(
        ['docker', 'compose', '-f', str(COMPOSE), '--profile', 'vlm', 'config', '--format', 'json'],
        check=False,
        capture_output=True,
        text=True,
        env={'PATH': '/usr/bin:/bin:/usr/local/bin', 'HOME': str(ROOT)},
    )
    if result.returncode != 0:
        pytest.skip(f'docker compose could not render the file here: {result.stderr[:200]}')
    rendered = json.loads(result.stdout)['services']['vlm']
    assert rendered['entrypoint'] == ['/bin/sh', '-c']
    (script,) = rendered['command']
    # `docker compose config` keeps `$$` as written; the source file is the
    # authority, so the rendered script must equal it after interpolation
    (source,) = _vlm_service()['command']
    assert script.replace('$$', '$') == source.replace('$$', '$') or script == source
    assert rendered['environment']['VLM_REASONING_PARSER'] == 'gemma4'
