"""The CI ``db`` job: same Postgres as compose, ``-m db``, code changes only; and no offline
pytest command of the workflow runs the ``db`` tier.

GitHub Actions service containers: https://docs.github.com/en/actions/using-containerized-services/creating-postgresql-service-containers
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.safe_load((ROOT / '.github' / 'workflows' / 'ci.yml').read_text())
COMPOSE = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
OFFLINE_EXPRESSION = "-m 'not live and not families and not db'"


def _run_text(job: dict) -> str:
    return '\n'.join(step.get('run', '') for step in job['steps'])


def test_db_job_uses_the_compose_postgres_image() -> None:
    service = WORKFLOW['jobs']['db']['services']['postgres']
    assert service['image'] == COMPOSE['services']['postgres']['image']


def test_db_job_runs_the_db_tier_against_the_service() -> None:
    job = WORKFLOW['jobs']['db']
    assert re.search(r'pytest .*-m db\b', _run_text(job))
    assert job['services']['postgres']['env']['POSTGRES_USER'] == 'op_super'
    url = job['env']['OP_TEST_DATABASE_URL']
    assert url.startswith('postgresql://op_super:')
    assert '@localhost:5432/' in url
    assert '5432:5432' in job['services']['postgres']['ports']
    assert '--health-cmd' in job['services']['postgres']['options']


def test_db_job_is_gated_by_the_code_path_filter() -> None:
    job = WORKFLOW['jobs']['db']
    assert job['needs'] == 'changes'
    assert "needs.changes.outputs.code == 'true'" in job['if']


def test_every_offline_pytest_command_excludes_db() -> None:
    commands = []
    for name, job in WORKFLOW['jobs'].items():
        if name == 'db':
            continue
        for step in job.get('steps', []):
            text = ' '.join(step.get('run', '').split())
            commands += re.findall(r'pytest [^|;]*?-m \'[^\']*\'', text)
    selecting = [c for c in commands if '-m families' not in c]
    assert selecting, 'no offline pytest command found'
    for command in selecting:
        assert "-m 'not live and not families and not db'" in command, command
