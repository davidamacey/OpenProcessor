"""``GET /review/tabs`` serves ``default_tab``: the first queue in served order that has work."""

from __future__ import annotations

from src.services.curation.review_tabs import default_tab


def _tabs(**counts: int) -> list[dict[str, object]]:
    return [{'id': name, 'count': n} for name, n in counts.items()]


def test_the_default_is_the_first_queue_in_served_order_with_work() -> None:
    tabs = _tabs(to_label=0, check_machine=7, disagreements=3)
    assert default_tab(tabs) == 'check_machine'


def test_the_first_queue_wins_when_several_have_work() -> None:
    assert default_tab(_tabs(to_label=2, check_machine=7)) == 'to_label'


def test_no_queue_with_work_has_no_default() -> None:
    assert default_tab(_tabs(to_label=0, check_machine=0)) is None
    assert default_tab([]) is None
