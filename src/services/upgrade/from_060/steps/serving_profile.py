"""Step ``serving_profile``: the deployment's default serving profile (research GPU S-5).

Generated from what the install runs today, not from a default: the profile the installer
applied (``GPU_PROFILE``, a file of ``config_templates/profiles``), and the instance count of
every model in the Triton repository (``config.pbtxt``). Written as revision 1 of the
deployment-wide document ``default`` of kind ``serving_profile`` and activated through
``deployment_settings('active_serving_profile')``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from src.config.env import env_str
from src.db.models import ConfigRevision, DeploymentSetting
from src.services.upgrade.from_060 import control_rows
from src.services.upgrade.from_060.model import ConfigDocSrc, ConfigRevisionSrc


if TYPE_CHECKING:
    from src.services.upgrade.from_060.steps import StepContext

ORDER = 10
DOC_NAME = 'default'
SERVER_NAME = 'triton-server'
PROFILES_DIR = Path(__file__).resolve().parents[5] / 'config_templates' / 'profiles'
_COUNT = re.compile(r'\bcount\s*:\s*(\d+)')
_INSTANCE_GROUP = re.compile(r'instance_group\s*\[(.*?)\n\]', re.DOTALL)


def instance_counts(model_repo: Path) -> dict[str, int]:
    """``{model: instances}`` from each ``<model>/config.pbtxt``; a model with no instance group
    (an ensemble) is listed with 0, since the server loads it but starts no instance of it."""
    out: dict[str, int] = {}
    for config in sorted(model_repo.glob('*/config.pbtxt')):
        text = config.read_text(encoding='utf-8')
        group = _INSTANCE_GROUP.search(text)
        out[config.parent.name] = (
            sum(int(n) for n in _COUNT.findall(group.group(1))) if group else 0
        )
    return out


def build_body(profile: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    wanted = profile.get('models', 'all')
    load_list = sorted(counts) if wanted == 'all' else [m for m in wanted if m in counts]
    return {
        'profile': profile['name'],
        'servers': [
            {
                'name': SERVER_NAME,
                'grpc_url': env_str('TRITON_URL', f'{SERVER_NAME}:8001'),
                'metrics_url': f'{SERVER_NAME}:8002',
                'gpu_uuids': [],
                'numa_node': None,
            }
        ],
        'load_list': {SERVER_NAME: load_list},
        'instances': {m: counts[m] for m in load_list},
        'image_prep': {
            'instances': profile.get('gpu_image_prep_instances', 1),
            'device_budget_mib': profile.get('python_backend_budget_mib'),
        },
        'priorities': {'interactive': 'high', 'bulk': 'low'},
        'ingest_gpu_decode': bool(profile.get('ingest_gpu_decode', False)),
        'max_batch_size': profile.get('max_batch_size'),
    }


def load_profile(name: str) -> dict[str, Any]:
    path = PROFILES_DIR / f'{name}.json'
    profile: dict[str, Any] = json.loads(path.read_text(encoding='utf-8'))
    return profile


def generated_body(ctx: StepContext) -> dict[str, Any]:
    return build_body(
        load_profile(env_str('GPU_PROFILE', 'standard')), instance_counts(ctx.layout.model_repo)
    )


async def run(ctx: StepContext) -> None:
    doc = ConfigDocSrc(
        project_slug=None,
        kind='serving_profile',
        name=DOC_NAME,
        description='generated from the 0.6.0 install by the upgrade',
        revisions=(ConfigRevisionSrc(1, generated_body(ctx), ctx.now, None),),
        created_at=ctx.now,
        updated_at=ctx.now,
    )
    config_id = await control_rows.load_config(ctx.session, None, doc)
    ctx.config_ids[(None, 'serving_profile', DOC_NAME)] = config_id
    ctx.session.add(
        DeploymentSetting(
            key='active_serving_profile',
            value={'config_id': str(config_id), 'revision': 1},
            updated_by='upgrade_from_060',
        )
    )
    await ctx.session.flush()
    ctx.report.add('serving_profile')


async def verify(ctx: StepContext) -> list[str]:
    """The stored body equals what today's install values generate."""
    config_id = ctx.config_ids.get((None, 'serving_profile', DOC_NAME))
    if config_id is None:
        return ['no serving_profile document was written']
    stored = (
        await ctx.session.execute(
            select(ConfigRevision.body).where(
                ConfigRevision.config_id == config_id, ConfigRevision.revision == 1
            )
        )
    ).scalar_one()
    expected = generated_body(ctx)
    return (
        []
        if stored == json.loads(json.dumps(expected))
        else ['the stored body differs from the install values']
    )
