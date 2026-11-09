"""The model-export handler on fakes: a stub build script, a fake plan reader and Triton load.

Imported by name into ``tests/models/conftest.py``."""

from __future__ import annotations

import itertools
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fixtures.job_rows import deployment_session, new_job, sql

from src.db.engine import get_sync_engine
from src.db.repos import engine_builds, jobs as repo


# The export scripts and their handler import each other by bare name.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'export'))

_NAMES = itertools.count(1)
STUB_BUILD = textwrap.dedent(
    """
    import pathlib, sys
    root, name = pathlib.Path(sys.argv[1]), sys.argv[2]
    print("ONNX End2End Export", flush=True)
    print("Building TensorRT engine", flush=True)
    (root / name / "1").mkdir(parents=True, exist_ok=True)
    (root / name / "1" / "model.plan").write_bytes(b"plan-bytes")
    print("Engine saved", flush=True)
    """
)


def plan_facts(input_dtype: str = 'uint8', shape: tuple[int, ...] = (-1, 3, 640, 640)) -> Any:
    from engine_facts import PlanFacts, PlanTensor

    return PlanFacts(
        inputs=(PlanTensor('images', input_dtype, shape),),
        outputs=(
            PlanTensor('num_dets', 'int32', (-1, 1)),
            PlanTensor('det_boxes', 'fp32', (-1, 300, 4)),
            PlanTensor('det_scores', 'fp16', (-1, 300)),
            PlanTensor('det_classes', 'int32', (-1, 300)),
        ),
        trt_version='11.1.0.106',
        gpu_arch='sm_86',
    )


@dataclass
class ExportEnv:
    models_dir: Path
    loads: list[tuple[str, bool]]
    plan: Any
    measured: int | None = 900

    def start_job(self, *, name: str | None = None, auto_load: bool = True) -> repo.JobRow:
        engine = get_sync_engine()
        name = name or f'car{next(_NAMES)}'
        params = {
            'pt_file': str(self.models_dir / f'{name}.pt'),
            'triton_name': name,
            'model_name': name,
            'max_batch': 16,
            'formats': ['onnx_end2end', 'trt_end2end'],
            'normalize_boxes': True,
            'auto_load': auto_load,
            'head_kind': 'nms',
            'num_classes': 2,
            'class_names': ['car', 'bus'],
        }
        (self.models_dir / f'{name}.pt').write_bytes(b'weights')
        with deployment_session(engine) as session:
            engine_builds.add_model_version(
                session,
                name=name,
                labels=['car', 'bus'],
                family='yolo',
                artifact_storage_key=f'pytorch_models/{name}.pt',
                created_by='test',
            )
            return repo.insert_queued(
                session,
                new_job(
                    None,
                    'model_export',
                    queue='gpu_export',
                    scope=name,
                    exclusive_group='model_export',
                    params=params,
                ),
            ).job

    def engine_of(self, job: repo.JobRow) -> str:
        return f'{job.params["triton_name"]}_trt_end2end'

    def statuses(self, pg: Any, job: repo.JobRow) -> list[str]:
        rows = sql(
            pg,
            'SELECT status FROM model_engines WHERE triton_model_name = %s ORDER BY status',
            self.engine_of(job),
        )
        return [status for (status,) in rows]

    def run(self, job: repo.JobRow) -> None:
        from export_tasks import export_model

        from src.config.job_kinds import KINDS
        from src.tasks.runtime.job_task import run_job

        run_job(KINDS['model_export'], export_model, {}, str(job.id))

    def job_row(self, pg: Any, job: repo.JobRow) -> tuple[Any, ...]:
        [row] = sql(
            pg, 'SELECT status, error_code, error_message, result FROM jobs WHERE id = %s', job.id
        )
        return row


@pytest.fixture
def export_env(
    pg: Any, job_env: Any, wakes: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> ExportEnv:
    import engine_facts
    import export_tasks

    models_dir = tmp_path / 'models'
    models_dir.mkdir()
    stub = tmp_path / 'stub_build.py'
    stub.write_text(STUB_BUILD)
    env = ExportEnv(models_dir=models_dir, loads=[], plan=plan_facts())
    monkeypatch.setattr(export_tasks, 'MODELS_DIR', models_dir)
    monkeypatch.setattr(export_tasks, 'APP_DIR', tmp_path)
    monkeypatch.setattr(
        export_tasks,
        'build_command',
        lambda job: [
            sys.executable,
            str(stub),
            str(models_dir),
            f'{job["triton_name"]}_trt_end2end',
        ],
    )
    monkeypatch.setattr(engine_facts, 'read_plan', lambda _path: env.plan)

    async def load(name: str, *, keep: bool) -> tuple[bool, int | None]:
        env.loads.append((name, keep))
        return True, env.measured

    monkeypatch.setattr(export_tasks, '_load_and_measure', load)
    return env
