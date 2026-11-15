"""The catalog file, its sync into Postgres, and the fail-closed availability of a model."""

from __future__ import annotations

import copy
import uuid
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from src.config.crop_encoder import CROP_ENCODERS
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


def test_the_shipped_catalog_lists_every_model() -> None:
    keys = {(e.model_key, e.version, e.native_dim) for e in load_catalog()}
    assert keys == {
        ('pe_core_l14_336', 1, 1024),
        ('pe_core_l14_crop224', 1, 1024),
        ('pe_core_l14_crop336', 1, 1024),
        ('mobileclip2_s2', 1, 512),
    }


@pytest.mark.parametrize('size', sorted(CROP_ENCODERS))
def test_the_default_spaces_are_catalog_versions_and_crops_use_the_crop_encoder(
    size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new project's frame space is the frame encoder's and its item and box spaces the crop
    encoder's at the deployment's size: the crop version names the crop model, that size and the
    letterbox rule, and its image contract is the one the engine of that size declares."""
    from src.clients import pe_encoder
    from src.clients.pe_encoder import PE_CROP_MODEL, PE_IMAGE_MODEL
    from src.domain.image_geometry import CROP_FILL
    from src.services.projects.store import FRAME_ENCODER, crop_encoder
    from src.services.training import builtin_engines as builtin

    monkeypatch.setattr(pe_encoder, 'PE_CROP_SIZE', size)
    by_key = {(e.model_key, e.version): e for e in load_catalog()}
    frame, crop = by_key[FRAME_ENCODER], by_key[crop_encoder()]
    assert crop.model_key == f'pe_core_l14_crop{size}'
    assert frame.triton_image_model == PE_IMAGE_MODEL
    assert crop.triton_image_model == PE_CROP_MODEL
    assert crop.triton_text_model == frame.triton_text_model  # one text tower for both spaces
    assert (crop.weights_ref, crop.weights_sha256) == (frame.weights_ref, frame.weights_sha256)
    image = crop.input_spec['image']
    assert image['size'] == [size, size]
    assert (image['resize'], image['fill']) == ('letterbox', CROP_FILL)
    assert crop.input_spec_sha256 != frame.input_spec_sha256  # a space of its own
    other = by_key[CROP_ENCODERS[336 if size == 224 else 224]]
    assert crop.input_spec_sha256 != other.input_spec_sha256  # and not the other size's
    shipped = {e.name: e.input_contract for e in builtin.SHIPPED}
    contract = shipped[PE_CROP_MODEL].model_dump(mode='json', exclude_none=True)
    assert {**contract, 'size': [size, size]} == image  # the engine of this size declares this
    engines = {**shipped_engines(), PE_CROP_MODEL: {**contract, 'size': [size, size]}}
    assert availability(_version(crop.model_key), engines).available
    assert not availability(_version(other.model_key), engines).available


def test_every_supported_size_has_a_catalog_row_of_that_size() -> None:
    by_key = {(e.model_key, e.version): e for e in load_catalog()}
    for size, key in CROP_ENCODERS.items():
        assert by_key[key].input_spec['image']['size'] == [size, size]


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
    assert set(first.inserted) == {
        'pe_core_l14_336_v1',
        'pe_core_l14_crop224_v1',
        'mobileclip2_s2_v1',
    }
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
