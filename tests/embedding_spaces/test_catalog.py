"""The catalog file, its sync into Postgres, and the fail-closed availability of a model."""

from __future__ import annotations

import copy
import uuid
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from src.db.repos import embedding_spaces as repo
from src.services.embedding_spaces.catalog import (
    CatalogError,
    availability,
    load_catalog,
    sync_catalog,
)
from tests.embedding_spaces._seed import shipped_engines


def _version(model_key: str = 'pe_core_l14_336') -> repo.ModelVersionRow:
    e = next(e for e in load_catalog() if e.model_key == model_key)
    return repo.ModelVersionRow(
        id=uuid.uuid4(),
        model_key=e.model_key,
        version=e.version,
        modalities=e.modalities,
        native_dim=e.native_dim,
        matryoshka_dims=e.matryoshka_dims,
        distance=e.distance,
        input_spec=e.input_spec,
        input_spec_sha256=e.input_spec_sha256,
        weights_ref=e.weights_ref,
        weights_sha256=e.weights_sha256,
        precision=e.precision,
        triton_image_model=e.triton_image_model,
        triton_text_model=e.triton_text_model,
        engine_parity_min_cos=None,
        licence=e.licence,
        status=e.status,
    )


def test_the_shipped_catalog_lists_both_models() -> None:
    keys = {(e.model_key, e.version, e.native_dim) for e in load_catalog()}
    assert keys == {('pe_core_l14_336', 1, 1024), ('mobileclip2_s2', 1, 512)}


def test_a_row_whose_triton_model_is_missing_is_unavailable() -> None:
    engines = shipped_engines()
    del engines['pe_text_encoder']
    state = availability(_version(), engines)
    assert not state.available
    assert 'pe_text_encoder' in (state.reason or '')
    assert not availability(_version(), {}).available


def test_an_input_spec_that_differs_from_the_engine_contract_is_refused() -> None:
    engines = copy.deepcopy(shipped_engines())
    engines['pe_image_encoder']['size'] = [224, 224]
    state = availability(_version(), engines)
    assert not state.available
    assert 'differs' in (state.reason or '')


def test_the_two_shipped_models_are_available() -> None:
    engines = shipped_engines()
    assert availability(_version('pe_core_l14_336'), engines).available
    assert availability(_version('mobileclip2_s2'), engines).available


def test_a_deprecated_version_cannot_back_a_space() -> None:
    v = _version()
    deprecated = repo.ModelVersionRow(**{**v.__dict__, 'status': 'deprecated'})
    assert not availability(deprecated, shipped_engines()).available


def test_a_malformed_catalog_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / 'catalog.tsv'
    bad.write_text('model_key\tversion\nx\t1\n')
    with pytest.raises(CatalogError, match='missing column'):
        load_catalog(bad)


@pytest.mark.db
def test_sync_inserts_once_and_refuses_a_drifted_row(pg, tmp_path: Path) -> None:
    with Session(pg.engine('app')) as session:
        first = sync_catalog(session)
    assert set(first.inserted) == {'pe_core_l14_336_v1', 'mobileclip2_s2_v1'}
    with Session(pg.engine('app')) as session:
        assert sync_catalog(session).inserted == ()
    drifted = tmp_path / 'catalog.tsv'
    lines = (
        Path(__file__).resolve().parents[2].joinpath('examples/embedding/catalog.tsv').read_text()
    )
    drifted.write_text(lines.replace('"size":[336,336]', '"size":[448,448]'))
    with Session(pg.engine('app')) as session, pytest.raises(CatalogError, match='new version'):
        sync_catalog(session, drifted)


@pytest.mark.db
def test_a_changed_status_in_the_file_moves_the_stored_status(pg, tmp_path: Path) -> None:
    with Session(pg.engine('app')) as session:
        sync_catalog(session)
    text = (
        Path(__file__).resolve().parents[2].joinpath('examples/embedding/catalog.tsv').read_text()
    )
    changed = tmp_path / 'catalog.tsv'
    changed.write_text(text.replace('Apache-2.0\tavailable', 'Apache-2.0\tdeprecated'))
    with Session(pg.engine('app')) as session:
        assert sync_catalog(session, changed).status_changed == ('pe_core_l14_336_v1',)
        row = repo.get_model_version(session, 'pe_core_l14_336', 1)
        session.rollback()
    assert row is not None
    assert row.status == 'deprecated'
