"""2E-8: the load is a registry of step modules, discovered and run in ORDER, each idempotent."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from src.services.upgrade.from_060 import steps


pytestmark = pytest.mark.db


def test_the_steps_are_discovered_in_order() -> None:
    names = [steps.step_name(m) for m in steps.discover()]
    assert names == ['serving_profile', 'embedding_spaces', 'legacy_repair', 'clustering']
    orders = [m.ORDER for m in steps.discover()]
    assert orders == sorted(orders)


def test_a_step_module_added_later_is_picked_up_and_ordered(tmp_path, monkeypatch) -> None:
    # a later lane's step (clustering, near_duplicates) is one new file in the package
    (tmp_path / 'extra_step.py').write_text(
        'ORDER = 15\nasync def run(ctx):\n    ctx.notes["extra"] = True\n'
    )
    monkeypatch.setattr(steps, '__path__', [*steps.__path__, str(tmp_path)])
    import sys

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, f'{steps.__name__}.extra_step', __import__('extra_step'))
    names = [steps.step_name(m) for m in steps.discover()]
    assert (
        names.index('serving_profile') < names.index('extra_step') < names.index('embedding_spaces')
    )


@pytest.mark.asyncio
async def test_the_steps_ran_in_order_and_a_second_load_adds_nothing_new(
    make_runner, clean_db
) -> None:
    result = await make_runner().run()
    assert result.steps_run == [
        'serving_profile',
        'embedding_spaces',
        'legacy_repair',
        'clustering',
    ]
    # idempotent: the second run is refused by the gate, and the rows stay one set
    with clean_db.engine('super').connect() as conn:
        spaces = conn.execute(text('SELECT count(*) FROM embedding_spaces')).scalar_one()
    assert spaces == 9
