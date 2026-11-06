"""Every shipped Triton engine has a ready registry row with a contract that matches its config.

The rows are seeded by `db-migrate` from `src/services/training/builtin_engines.py`; the shipped
`config.pbtxt` is the source of dtype, dimensions, batch size and outputs, so a drift between the
two fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fixtures.job_rows import deployment_session, sql

from src.db.repos import embedding_spaces as spaces_repo
from src.schemas.models import (
    EncoderInputSpec,
    EngineInputContract,
    EngineOutputContract,
    SizeRange,
)
from src.services.embedding_spaces.catalog import availability, load_catalog
from src.services.training import builtin_engines as builtin


MODELS = Path(__file__).resolve().parents[2] / 'models'
DTYPES = {'UINT8': 'uint8', 'FP32': 'fp32', 'FP16': 'fp16', 'INT32': 'int32', 'INT64': 'int64'}


def shipped_engines() -> list[str]:
    """Models whose config.pbtxt names a platform (TensorRT or ONNX); the python-backend
    models run no engine of their own."""
    return sorted(
        d.name
        for d in MODELS.iterdir()
        if (d / 'config.pbtxt').is_file()
        and re.search(r'^platform:', (d / 'config.pbtxt').read_text(), re.MULTILINE)
    )


def tensors(name: str) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    """(dtype, dims) of every input and output of a config.pbtxt, comments removed."""
    text = re.sub(r'#.*', '', (MODELS / name / 'config.pbtxt').read_text())
    _, _, rest = text.partition('input')
    inputs_text, _, outputs_text = rest.partition('output')
    pattern = re.compile(r'data_type:\s*TYPE_(\w+)\s*dims:\s*\[([^\]]*)\]')

    def parse(chunk: str) -> list[tuple[str, list[int]]]:
        return [
            (DTYPES[m.group(1)], [int(d) for d in m.group(2).split(',') if d.strip()])
            for m in pattern.finditer(chunk)
        ]

    return parse(inputs_text), parse(
        outputs_text.split('dynamic_batching')[0].split('instance_group')[0]
    )


def test_the_seed_covers_exactly_the_shipped_engines() -> None:
    assert sorted(s.name for s in builtin.SHIPPED) == shipped_engines()


@pytest.mark.parametrize('shipped', builtin.SHIPPED, ids=lambda s: s.name)
def test_a_contract_agrees_with_the_shipped_config(shipped: builtin.Shipped) -> None:
    inputs, outputs = tensors(shipped.name)
    [(dtype, dims)] = inputs
    contract = shipped.input_contract
    if isinstance(contract, EngineInputContract):
        assert contract.dtype == dtype
        assert len(dims) == 3  # C, H, W without the batch dimension
        assert dims[0] == 3
        for declared, wanted in zip(dims[1:], contract.size, strict=True):
            if isinstance(wanted, SizeRange):
                assert declared == -1
            else:
                assert declared == wanted
    else:
        assert (dtype, dims[0] if dims else None) == (
            contract.dtype,
            contract.length if contract.length is not None else -1,
        )
    assert [
        (d, s)
        for d, s in zip(shipped.output_contract.dtypes, shipped.output_contract.shapes, strict=True)
    ] == outputs
    assert builtin.read_pbtxt(MODELS, shipped.name).max_batch >= 1


@pytest.mark.db
def test_every_shipped_engine_has_a_ready_row_with_a_valid_contract(pg) -> None:
    with deployment_session(pg.engine('app')) as session:
        inserted = builtin.seed_builtin_engines(session)
    assert sorted(inserted) == shipped_engines()

    rows = sql(
        pg,
        'SELECT e.triton_model_name, e.status, e.input_contract, e.output_contract, e.max_batch, '
        'm.origin, m.project_id FROM model_engines e JOIN models m ON m.id = e.model_id '
        "WHERE m.origin = 'builtin' ORDER BY 1",
    )
    assert [r[0] for r in rows] == shipped_engines()
    for _name, status, input_contract, output_contract, max_batch, origin, project in rows:
        assert (status, origin, project) == ('ready', 'builtin', None)
        assert max_batch >= 1
        EngineOutputContract.model_validate(output_contract)
        if input_contract.get('kind') == 'tokens':
            builtin.TokenInputContract.model_validate(input_contract)
        else:
            EngineInputContract.model_validate(input_contract)


@pytest.mark.db
def test_seeding_twice_inserts_nothing_and_keeps_a_measured_figure(pg) -> None:
    with deployment_session(pg.engine('app')) as session:
        builtin.seed_builtin_engines(session)
    sql(
        pg,
        "UPDATE model_engines SET device_mib_measured = 4321 WHERE triton_model_name = 'pe_image_encoder'",
    )
    with deployment_session(pg.engine('app')) as session:
        assert builtin.seed_builtin_engines(session) == []
    assert sql(
        pg,
        "SELECT device_mib_measured FROM model_engines WHERE triton_model_name = 'pe_image_encoder'",
    ) == [(4321,)]


@pytest.mark.db
def test_the_catalogued_encoders_are_available_with_the_seeded_engines(pg) -> None:
    with deployment_session(pg.engine('app')) as session:
        builtin.seed_builtin_engines(session)
    with deployment_session(pg.engine('app')) as session:
        ready = spaces_repo.ready_engine_contracts(session)
    for entry in load_catalog():
        wanted = EncoderInputSpec.model_validate(entry.input_spec).image
        assert wanted == EngineInputContract.model_validate(ready[entry.triton_image_model])
        assert availability(entry, ready).available is True, entry.model_key
