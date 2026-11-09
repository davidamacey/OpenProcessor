"""2E-6: ``db verify`` and the upgrade's verify report class drift (report only) and a clean fixture reports nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.db.session import deployment_tx
from src.services.upgrade.class_reconcile import reconcile_classes
from src.services.upgrade.db_verify import verify_database
from src.services.upgrade.from_060.runner import UpgradeResult


if TYPE_CHECKING:
    from tests.upgrade._v060 import V060Install


pytestmark = pytest.mark.db


@pytest.mark.asyncio
async def test_the_upgrade_verify_reports_the_drift_it_finds_but_does_not_fail(make_runner) -> None:
    result = await make_runner().run()
    drift = result.verify.class_drift
    assert drift['default']['items'] == 5
    assert drift['default']['unknown_class_id'] == 1  # class 99
    assert drift['default']['name_disagrees'] == 1  # class 0 named 'truck' on the item
    kinds = {s['kind'] for s in drift['default']['samples']}
    assert kinds == {'unknown_class_id', 'name_disagrees'}
    assert drift['demo'] == {'items': 5, 'unknown_class_id': 0, 'name_disagrees': 0, 'samples': []}


@pytest.mark.asyncio
async def test_db_verify_reports_drift_and_missing_storage_after_the_upgrade(
    make_runner, owner_sessions, v060: V060Install
) -> None:
    await make_runner().run()
    async with owner_sessions() as session, deployment_tx(session, 'test db verify'):
        report = await verify_database(session, v060.reader, v060.store)
    assert not report.clean
    assert report.missing_indexes == []
    assert report.class_drift['default']['unknown_class_id'] == 1
    # the promoted model's version directory is not in the fixture; the run log's prefix is
    assert [m['kind'] for m in report.missing_storage] == ['model']


@pytest.mark.asyncio
async def test_a_clean_install_reports_nothing(
    make_runner, owner_sessions, v060: V060Install
) -> None:
    for doc_id in ('i3', 'i4'):  # the two drifted items of the fixture
        del v060.os.docs('op_prj_default__items')[doc_id]
    v060.store.write_atomic('models/demo__car_detector/1/model.plan', b'engine')
    await make_runner().run()
    async with owner_sessions() as session, deployment_tx(session, 'test db verify'):
        report = await verify_database(session, v060.reader, v060.store)
    assert report.clean, report.to_json()


@pytest.mark.asyncio
async def test_a_project_whose_index_is_missing_is_reported(
    make_runner, owner_sessions, v060: V060Install
) -> None:
    runner = make_runner()
    result = UpgradeResult(bundle=await runner._export())
    await runner._load(result)
    del v060.os.indexes['op_prj_demo__images']
    async with owner_sessions() as session, deployment_tx(session, 'test db verify'):
        report = await verify_database(session, v060.reader, v060.store)
    assert report.missing_indexes == [
        {'project': 'demo', 'role': 'images', 'index': 'op_prj_demo__images'}
    ]


@pytest.mark.asyncio
async def test_reconcile_compares_names_by_the_one_equality_rule(v060: V060Install) -> None:
    drift = await reconcile_classes(
        v060.reader, 'op_prj_default__items', {0: 'car', 2: 'traffic_light', 5: 'bus'}
    )
    # 'Traffic Light' on an item equals the registry's 'traffic_light'; 'truck' and class 99 do not
    assert (drift.unknown_class_id, drift.name_disagrees) == (1, 1)


@pytest.mark.asyncio
async def test_an_uploaded_model_in_the_pytorch_models_mount_is_found(
    make_runner, owner_sessions, tmp_path, v060: V060Install
) -> None:
    from sqlalchemy import text

    from src.clients.storage.roots import build_rooted_store
    from src.config.storage import StorageRootPaths

    await make_runner().run()
    mount = tmp_path / 'pytorch_models'
    mount.mkdir()
    (mount / 'car.pt').write_bytes(b'w')
    store = build_rooted_store(
        StorageRootPaths(
            source=tmp_path / 's',
            projects=tmp_path / 'p',
            models=tmp_path / 'm',
            cache=tmp_path / 'c',
            upgrade=tmp_path / 'u',
            bakeoff_results=tmp_path / 'b',
            pytorch_models=mount,
        )
    )
    async with owner_sessions() as session, deployment_tx(session, 'test seed model'):
        project = (await session.execute(text('SELECT id FROM projects LIMIT 1'))).scalar_one()
        await session.execute(
            text(
                'INSERT INTO models (project_id, name, version, origin, family, task, labels, '
                "artifact_storage_key, created_by) VALUES (:p, 'up', 1, 'uploaded', 'yolo', "
                "'detect', ARRAY['car'], 'models/pytorch_models/car.pt', 'test')"
            ),
            {'p': project},
        )
    async with owner_sessions() as session, deployment_tx(session, 'test db verify'):
        report = await verify_database(session, v060.reader, store)
    assert not [m for m in report.missing_storage if m['key'] == 'models/pytorch_models/car.pt']
