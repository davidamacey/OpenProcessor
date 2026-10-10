"""The provenance block of a parity capture: hardware, git sha, image digests, engine record.

Reuses the baseline suite's environment probe (GPU, CPU, RAM, Triton and OpenSearch versions,
dataset pin, foreign GPU load) and adds what a precision comparison needs: the image digests of
the compose project, whether the working tree was clean, the Triton model configuration of every
model the capture touches, and the operator's statement of the engine under test together with the
hash of the plan file when one is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.bench.parity_store import sha256_file
from scripts.bench.suite_probes import phase_env, run_cmd
from src.config.settings import TritonModelConfig


if TYPE_CHECKING:
    import httpx

MODELS = {
    'detector': TritonModelConfig.YOLO_MODEL,
    'pe_image': TritonModelConfig.PE_IMAGE_MODEL,
    'clip_image': TritonModelConfig.CLIP_IMAGE_MODEL,
    'face_detect': TritonModelConfig.FACE_DETECT_MODEL,
    'face_embed': TritonModelConfig.ARCFACE_MODEL,
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_inspect(text: str) -> dict[str, dict[str, str]]:
    """``docker inspect --format '{{.Name}} {{.Config.Image}} {{.Image}}'`` -> name -> image, id."""
    out: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3:
            out[parts[0].lstrip('/')] = {'image': parts[1], 'image_id': parts[2]}
    return out


def image_digests(compose_project: str) -> dict[str, dict[str, str]]:
    """Image reference and image id of every container of one compose project."""
    names = run_cmd(
        [
            'docker',
            'ps',
            '--filter',
            f'label=com.docker.compose.project={compose_project}',
            '--format',
            '{{.Names}}',
        ]
    ).split()
    if not names:
        return {}
    return parse_inspect(
        run_cmd(['docker', 'inspect', '--format', '{{.Name}} {{.Config.Image}} {{.Image}}', *names])
    )


def git_state() -> dict[str, Any]:
    status = run_cmd(['git', '-C', str(REPO_ROOT), 'status', '--porcelain'])
    return {
        'git_sha': run_cmd(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD']).strip(),
        'git_branch': run_cmd(
            ['git', '-C', str(REPO_ROOT), 'rev-parse', '--abbrev-ref', 'HEAD']
        ).strip(),
        'git_dirty': bool(status.strip()),
    }


def triton_models(client: httpx.Client, triton_url: str) -> dict[str, Any]:
    """Platform, batch limit and tensor types of each model the capture touches."""
    out: dict[str, Any] = {}
    for role, model in MODELS.items():
        resp = client.get(f'{triton_url}/v2/models/{model}/config')
        if resp.status_code != 200:
            out[role] = {'name': model, 'available': False}
            continue
        cfg = resp.json()
        out[role] = {
            'name': model,
            'available': True,
            'platform': cfg.get('platform') or cfg.get('backend'),
            'max_batch_size': cfg.get('max_batch_size'),
            'input': [(t['name'], t['data_type']) for t in cfg.get('input', [])],
            'output': [(t['name'], t['data_type']) for t in cfg.get('output', [])],
        }
    return out


def engine_record(label: str, plan: Path | None, extra: list[str]) -> dict[str, Any]:
    """The operator's statement of the engine under test plus the plan file's size and hash."""
    record: dict[str, Any] = {'label': label}
    for pair in extra:
        key, sep, value = pair.partition('=')
        if not sep:
            raise ValueError(f'--engine-record expects KEY=VALUE, got {pair!r}')
        record[key] = value
    if plan is not None:
        record['plan_file'] = plan.name
        record['plan_bytes'] = plan.stat().st_size
        record['plan_sha256'] = sha256_file(plan)
    return record


def environment_record(
    client: httpx.Client,
    *,
    gpu: str,
    compose_project: str,
    api_url: str,
    triton_url: str,
    opensearch_url: str,
    manifest: Path,
    idle_samples: int = 3,
) -> dict[str, Any]:
    probe_args = argparse.Namespace(
        gpu=gpu,
        project=compose_project,
        manifest=manifest,
        idle_samples=idle_samples,
        api_url=api_url,
        triton_url=triton_url,
        opensearch_url=opensearch_url,
    )
    env = phase_env(probe_args, client)
    return {
        **env,
        **git_state(),
        'compose_project': compose_project,
        'image_digests': image_digests(compose_project),
        'triton_models': triton_models(client, triton_url),
    }
