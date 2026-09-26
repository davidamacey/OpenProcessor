"""F8 D1 (OpenProcessor d817605): an item outside the probe's classes
(`probe_in_scope: false`, `probe_disagreement: null`) reads "no opinion"
and never offers "Accept model's class"; a served disagreement does.

OpenProcessor main 9e217f0 adds `probe_actionable` — Accept requires
`probe_actionable === true`, not just a served disagreement. A
disagreement the server didn't mark actionable (below its own confidence
threshold) renders as a muted "model unsure: <predicted class>" instead."""

from __future__ import annotations

from fixtures.wire import make_item

CLASSES = [
    {"id": 1, "name": "widget_a", "group": "widgets", "hotkey_letter": None,
     "count": 10, "validated_count": 5, "cluster_size": 12, "deprecated": False},
    {"id": 2, "name": "widget_b", "group": "widgets", "hotkey_letter": None,
     "count": 10, "validated_count": 5, "cluster_size": 12, "deprecated": False},
]


def _item(crop_id: str, in_scope, disagreement, actionable=None) -> dict:
    item = make_item(
        crop_id=crop_id,
        image_id=f"img-{crop_id}",
        class_id=1,
        class_name="widget_a",
        thumbnail_url=f"/curation/crops/{crop_id}/thumbnail",
    )
    item.update(
        reason="model disagreement",
        probe_pred_class="widget_b",
        probe_pred_class_id=2,
        probe_in_scope=in_scope,
        probe_disagreement=disagreement,
        probe_actionable=actionable,
    )
    return item


def _open(stub, page, app_url, item) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", {"strategies": [], "flags": {}})
    stub.on(
        "GET",
        r"/review/",
        lambda _r, _m: (200, {"items": [item], "total": 1, "page": 1, "page_size": 30}),
    )
    page.goto(f"{app_url}/review?tab=model_disagreements")
    page.get_by_role("button", name="Confirm", exact=True).wait_for(timeout=15000)


def test_out_of_scope_item_shows_no_opinion_and_no_accept(stub, page, app_url):
    _open(stub, page, app_url, _item("c-out", False, None, None))
    assert "no opinion (outside the probe's classes)" in page.get_by_test_id(
        "probe-no-opinion"
    ).inner_text()
    assert page.get_by_role("button", name="Accept model's class").count() == 0


def test_null_disagreement_never_offers_accept(stub, page, app_url):
    _open(stub, page, app_url, _item("c-null", None, None, None))
    assert page.get_by_role("button", name="Accept model's class").count() == 0


def test_actionable_disagreement_offers_accept(stub, page, app_url):
    _open(stub, page, app_url, _item("c-dis", True, True, True))
    assert page.get_by_role("button", name="Accept model's class").count() == 1


def test_disagreement_not_actionable_shows_unsure_and_no_accept(stub, page, app_url):
    _open(stub, page, app_url, _item("c-unsure", True, True, False))
    assert "model unsure: widget_b" in page.get_by_test_id("probe-unsure").inner_text()
    assert page.get_by_role("button", name="Accept model's class").count() == 0
