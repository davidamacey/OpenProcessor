"""2E-8: the generated serving profile equals today's values (one server, load lists, instance counts)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from src.services.upgrade.from_060.steps import serving_profile


if TYPE_CHECKING:
    from pathlib import Path

    from tests.upgrade._v060 import V060Install


PBTXT = 'name: "{name}"\ninstance_group [\n  {{\n    count: {count}\n    kind: KIND_GPU\n  }}\n]\n'


def _model(repo: Path, name: str, count: int | None) -> None:
    folder = repo / name
    folder.mkdir(parents=True, exist_ok=True)
    body = (
        'name: "x"\nplatform: "ensemble"\n'
        if count is None
        else PBTXT.format(name=name, count=count)
    )
    (folder / 'config.pbtxt').write_text(body)


def test_instance_counts_are_read_from_each_config(tmp_path: Path) -> None:
    _model(tmp_path, 'yolov11_small_trt_end2end', 2)
    _model(tmp_path, 'ocr_pipeline', None)
    (tmp_path / 'no_config').mkdir()
    assert serving_profile.instance_counts(tmp_path) == {
        'yolov11_small_trt_end2end': 2,
        'ocr_pipeline': 0,
    }


def test_a_profile_that_lists_models_loads_only_those() -> None:
    profile = serving_profile.load_profile('minimal')
    counts = {'yolov11_small_trt_end2end': 1, 'scrfd_10g_bnkps': 1, 'ocr_pipeline': 0}
    body = serving_profile.build_body(profile, counts)
    assert body['load_list'] == {'triton-server': ['yolov11_small_trt_end2end', 'scrfd_10g_bnkps']}
    assert body['instances'] == {'yolov11_small_trt_end2end': 1, 'scrfd_10g_bnkps': 1}
    assert body['profile'] == 'minimal'


def test_a_profile_with_all_loads_every_model() -> None:
    body = serving_profile.build_body(serving_profile.load_profile('standard'), {'a': 2, 'b': 1})
    assert body['load_list'] == {'triton-server': ['a', 'b']}
    assert len(body['servers']) == 1
    assert body['image_prep']['instances'] == 1
    assert body['max_batch_size'] == 32


@pytest.mark.db
@pytest.mark.asyncio
async def test_the_upgrade_stores_revision_1_and_activates_it(
    make_runner, clean_db, v060: V060Install, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('GPU_PROFILE', 'standard')
    _model(v060.layout.model_repo, 'yolov11_small_trt_end2end', 2)
    _model(v060.layout.model_repo, 'arcface_w600k_r50', 1)
    await make_runner().run()
    with clean_db.engine('super').connect() as conn:
        doc = conn.execute(
            text(
                "SELECT id, project_id, current_revision FROM config_documents WHERE kind = 'serving_profile'"
            )
        ).one()
        body = conn.execute(
            text('SELECT body FROM config_revisions WHERE config_id = :id AND revision = 1'),
            {'id': doc.id},
        ).scalar_one()
        active = conn.execute(
            text("SELECT value FROM deployment_settings WHERE key = 'active_serving_profile'")
        ).scalar_one()
    assert (doc.project_id, doc.current_revision) == (None, 1)
    assert body['instances'] == {'arcface_w600k_r50': 1, 'yolov11_small_trt_end2end': 2}
    assert body['load_list'] == {
        'triton-server': ['arcface_w600k_r50', 'yolov11_small_trt_end2end']
    }
    assert active == {'config_id': str(doc.id), 'revision': 1}
    assert json.loads(json.dumps(body)) == body
