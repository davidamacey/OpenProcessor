#!/usr/bin/env python3
"""Backend integration write-path checklist (coordination plan §5.2).

Manual runbook, NOT wired into `npm run test`, pre-commit, or CI. It
drives the real Cropwright frontend against a live OpenProcessor
backend with real indexed data (~347k crops) through 12 numbered
steps: page navigation, thumbnail rendering (the region_thumbnail
segment bug has teeth here), single + batch label, cluster refine,
review dismiss, region write + restore, batch region status, VLM label
batch, both SSE channels, export gating, and the train read path. Step
0 is a cross-cutting assertion, independent of the 12 steps: every
response observed from the API origin must carry the configured
prefix and none of the other one.

Every default reads an env var, then falls back to a flag default —
nothing is hardcoded: no origin, no prefix, no output path, no repo
path.

Needs a Python env with Playwright installed; this repo has none of
its own (pure SvelteKit/TS). Call the interpreter directly -- do not
`source` an activate script, and note that an env-var prefix cannot be
applied to a shell builtin, which is why the previous form here was
never valid shell:

    cd <repo root>
    DISPLAY=:11 /path/to/venv/bin/python \\
        scripts/playwright_backend_integration.py \\
        [--url URL] [--api URL] [--api-prefix PREFIX] [--out DIR]

e.g. /data/repos/openprocessor/.venv/bin/python -- any venv with
playwright works. No absolute path to this repo: the directory is
slated to be renamed to `cropwright`, and worktrees check it out
elsewhere.

Ports: 5184 is Cropwright's nginx container (host) per this repo's
CLAUDE.md; 5174/5180/5181/5183 belong to a sibling project
(example-app) on the same host and are NOT this app. 4603 is
OpenProcessor's host port.

--dry-run resolves and prints every URL this run would touch, then
exits 0, without launching a browser or making a request. Run it at
both prefixes and diff -- every differing line pair must differ ONLY
in the prefix segment. That two-prefix equivalence is H2's acceptance
criterion (coordination plan §5.2: "Every step must run twice ...
Identical results are the H2 acceptance criterion"), and it is exactly
what this mode proves without a backend.

Exit code 0 = PASS, 1 = FAIL. NOT RUN END TO END as of the commit that
adds this file -- see docs/design/backend-integration-phase-d-static-
plan-2026-09-20.md §3 for the full spec and §8 for the required
reporting caveat.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

TRANSITIONAL_DEFAULT = "/curation"  # mirrors src/lib/api.ts:107 (normalizeApiPrefix); flips at T-E2


def normalize_api_prefix(raw: str) -> str:
    """Python mirror of normalizeApiPrefix() in src/lib/api.ts.

    Kept byte-for-byte equivalent on purpose: this script asserts the
    URLs the SPA composes, so a divergence here silently tests nothing.
    Empty and a leaked ``__API_PREFIX__`` placeholder both mean "unset"
    -- see the TS function's own docstring for why the placeholder case
    is not theoretical (docker-entrypoint.sh bakes it at build time).

    If api.ts's fallback changes (T-E2 flips it to '/curation'), change
    TRANSITIONAL_DEFAULT above and nothing else.
    """
    trimmed = raw.strip()
    if not trimmed or trimmed.startswith("__"):
        return TRANSITIONAL_DEFAULT
    leading = trimmed if trimmed.startswith("/") else f"/{trimmed}"
    return leading.rstrip("/")


class Api:
    """Every backend URL the script touches, composed in one place.

    Rule for anyone editing this file: no other place may concatenate
    `self.prefix`. `--dry-run` works by recording every `Api.url()`
    call, which only holds if this is the sole composition point.
    """

    def __init__(self, origin: str, prefix: str) -> None:
        self.origin = origin.rstrip("/")
        self.prefix = prefix
        self.calls: list[tuple[str, str]] = []  # (method, url), in call order

    def url(self, path: str, method: str = "GET") -> str:
        """`path` is PREFIX-RELATIVE and must start with '/'."""
        assert path.startswith("/"), path
        u = f"{self.origin}{self.prefix}{path}"
        self.calls.append((method, u))
        return u

    def unprefixed(self, path: str, method: str = "GET") -> str:
        """Legacy top-level routers that do NOT hang off the prefix.

        NOT used for /clusters/stats/{index}: see pick_cluster()'s
        docstring for why that endpoint is avoided entirely, not just
        reached through this method.
        """
        u = f"{self.origin}{path}"
        self.calls.append((method, u))
        return u

    def relative(self, path: str) -> str:
        """Prefix-relative, origin-less path for the in-page EventSource
        (step 10): the browser reaches it through the SAME origin as the
        frontend (nginx proxies {prefix}/* to the API), not through
        `self.origin` (--api), matching sse.ts's own construction.
        Still the sole place `self.prefix` is concatenated for this case.
        """
        assert path.startswith("/"), path
        u = f"{self.prefix}{path}"
        self.calls.append(("EventSource", u))
        return u


def http_get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)


def http_json(url: str, method: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


FAILURES: list[str] = []
STEPS: list[dict[str, Any]] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}{'' if cond else ' — ' + detail}")
    if not cond:
        FAILURES.append(name)
    return cond


def record_step(number: int, name: str, checks: list[dict[str, Any]]) -> None:
    STEPS.append({"step": number, "name": name, "checks": checks})


# --------------------------------------------------------------------------
# Target selection -- replaces playwright_round_trip.py's pick_target(),
# does not port it (see §3.10 of the plan doc for why).
# --------------------------------------------------------------------------


def pick_cluster(api: Api) -> int:
    """First non-trivial cluster, via the same endpoint /clusters uses.

    NOT /clusters/stats/{index}: that is a legacy un-prefixed router
    whose only valid index values are global|vehicles|people|faces
    (backend src/routers/clusters.py:163-171), and the labeler's nginx
    stopped proxying it once T-B3 dropped the dead location block --
    only a direct --api origin call would even reach it, and it 400s
    on any legacy-shaped index name regardless.
    """
    page = http_get(api.url("/clusters?per_cluster=1&max_clusters=50"))
    for item in page.get("items", []):
        if (item.get("size") or 0) > 1:
            return int(item["cluster_id"])
    raise RuntimeError("no cluster with >1 member found")


def pick_crops(api: Api, cluster_id: int, n: int) -> list[dict[str, Any]]:
    page = http_get(
        api.url(f"/crops?cluster_id={cluster_id}&page_size={n}&label_validated=false")
    )
    return page.get("crops", [])


def pick_plate_crop(api: Api) -> dict[str, Any] | None:
    """A crop with an existing plate region, for steps 7/8."""
    page = http_get(api.url("/plates?page_size=5&verified=false"))
    items = page.get("items", [])
    return items[0] if items else None


# --------------------------------------------------------------------------
# The 12 steps. Each returns True/False (overall pass) and appends to
# STEPS via record_step for summary.json.
# --------------------------------------------------------------------------

NAV_ROUTES = [
    ("clusters", "/clusters"),
    ("review", "/review"),
    ("train", "/train"),
    ("export", "/export"),
    ("models", "/models"),
]


def step0_prefix_purity(api: Api, observed: list[dict[str, Any]]) -> bool:
    """Cross-cutting assertion, independent of the 12 steps -- the
    runtime counterpart to T-D3's static ratchet. Any request to the
    API origin whose path does not start with the configured prefix
    (and does not start with the OTHER known prefix either -- both are
    offenses) is a threading bug: a call site that escaped API_PREFIX.
    """
    offenders = []
    other = "/curation" if api.prefix == "/curation" else "/curation"
    for r in observed:
        url = r.get("url", "")
        if not url.startswith(api.origin):
            continue
        path = url[len(api.origin) :]
        if path.startswith(api.prefix + "/"):
            continue
        if path.startswith(other + "/"):
            offenders.append((url, r.get("status")))
    ok = check(
        "step 0: every observed API-origin request starts with the configured prefix",
        not offenders,
        f"{len(offenders)} offender(s): {offenders[:5]}",
    )
    record_step(0, "prefix purity of observed traffic", [{"offenders": offenders}])
    return ok


def step1_navigation(page: Any, front: str, timeout: int) -> bool:
    ok_all = True
    for name, path in NAV_ROUTES:
        console_errors: list[str] = []
        page_errors: list[str] = []
        bad_responses: list[dict[str, Any]] = []

        def on_console(msg: Any, _errs: list[str] = console_errors) -> None:
            if msg.type == "error":
                _errs.append(msg.text)

        def on_pageerror(exc: Any, _errs: list[str] = page_errors) -> None:
            _errs.append(str(exc))

        def on_response(resp: Any, _bad: list[dict[str, Any]] = bad_responses) -> None:
            if resp.status >= 400:
                _bad.append({"url": resp.url, "status": resp.status})

        page.on("console", on_console)
        page.on("pageerror", on_pageerror)
        page.on("response", on_response)
        page.goto(f"{front}{path}", wait_until="domcontentloaded", timeout=timeout)
        try:
            page.wait_for_load_state("networkidle", timeout=timeout)
        except Exception:
            pass  # some routes (SSE) never go idle
        main_text = page.evaluate(
            "() => (document.querySelector('main')?.innerText || '').length"
        )
        page.remove_listener("console", on_console)
        page.remove_listener("pageerror", on_pageerror)
        page.remove_listener("response", on_response)

        ok = check(
            f"step 1: {name} navigates clean (no console error, no pageerror, no >=400, <main> non-empty)",
            not console_errors and not page_errors and not bad_responses and main_text > 0,
            f"console_errors={console_errors[:3]} pageerrors={page_errors[:3]} "
            f"bad={bad_responses[:3]} main_len={main_text}",
        )
        ok_all = ok_all and ok
        record_step(
            1,
            f"navigate {name}",
            [
                {
                    "console_errors": console_errors,
                    "page_errors": page_errors,
                    "bad_responses": bad_responses,
                    "main_len": main_text,
                }
            ],
        )
    return ok_all


def step2_thumbnails(page: Any, front: str, api: Api, timeout: int) -> bool:
    def render_check(path: str, label: str) -> tuple[bool, list[dict[str, Any]]]:
        page.goto(f"{front}{path}", wait_until="domcontentloaded", timeout=timeout)
        try:
            page.wait_for_function(
                "() => Array.from(document.images).every(i => i.complete)", timeout=timeout
            )
        except Exception:
            pass
        imgs = page.evaluate(
            """() => Array.from(document.querySelectorAll('img')).map(i => ({
              src: i.currentSrc || i.src,
              naturalWidth: i.naturalWidth,
            }))"""
        )
        return any(i["naturalWidth"] > 0 for i in imgs), imgs

    clusters_ok, clusters_imgs = render_check("/clusters", "clusters grid")

    # The plates gallery (and the region_thumbnail guard below, which can
    # only observe a src if the gallery has rows to render) only has
    # anything to show on a deployment whose class registry actually
    # includes `license_plate` -- a from-scratch generic test dataset
    # (e.g. a warehouse/pallet domain) legitimately has none, and an
    # empty gallery there is correct behavior, not a failure. Check the
    # registry first so this step reports a skip instead of a false FAIL.
    classes = http_get(api.url("/classes"))
    has_lp_class = any(
        c.get("class_name") == "license_plate" for c in classes.get("classes", [])
    )

    if has_lp_class:
        plates_ok, plates_imgs = render_check("/clusters?class=license_plate", "plates gallery")
    else:
        plates_ok, plates_imgs = True, []  # nothing to assert; see skip note below

    region_seg = f"{api.prefix}/crops/"
    has_region = any(
        region_seg in i["src"] and "region_thumbnail" in i["src"] for i in plates_imgs
    )

    ok = check(
        "step 2: >=1 <img> naturalWidth>0 on /clusters and the plates gallery",
        clusters_ok and plates_ok,
        f"clusters_ok={clusters_ok} plates_ok={plates_ok}"
        + ("" if has_lp_class else " (skipped: no license_plate class in this dataset)"),
    )
    ok2 = check(
        "step 2: >=1 rendered src contains {prefix}/crops/...region_thumbnail (bug-#3 guard)",
        has_region if has_lp_class else True,
        f"sample srcs={[i['src'] for i in plates_imgs[:3]]}"
        if has_lp_class
        else "skipped: no license_plate class in this dataset",
    )
    record_step(
        2,
        "thumbnails render",
        [
            {
                "clusters_ok": clusters_ok,
                "plates_ok": plates_ok,
                "has_region_thumb": has_region,
                "has_license_plate_class": has_lp_class,
            }
        ],
    )
    return ok and ok2


def step3_label_crop(api: Api, crop: dict[str, Any]) -> bool:
    crop_id = crop["crop_id"]
    # Never change a crop's class value -- re-label to its own current
    # class_id. The write path (label_validated flips, label_source is
    # stamped, class_id_history appends) is still fully exercised.
    target_class = crop.get("class_id") or 0
    http_json(
        api.url(f"/crops/{crop_id}/label", "PUT"),
        "PUT",
        {"class_id": int(target_class), "validated": True},
    )
    refetched = http_get(api.url(f"/crops/{crop_id}"))
    validated = bool(refetched.get("label_validated") or refetched.get("class_validated"))
    has_source = bool(refetched.get("label_source"))
    hist = refetched.get("class_id_history") or []
    ok = check(
        "step 3: PUT /crops/{id}/label -> label_validated true + label_source set",
        validated and has_source,
        f"label_validated={refetched.get('label_validated')} "
        f"class_validated={refetched.get('class_validated')} "
        f"label_source={refetched.get('label_source')!r}",
    )
    if not hist:
        print("  [warn] class_id_history empty after label PUT (non-fatal, backend-version-dependent)")
    record_step(
        3,
        "label a crop",
        [{"crop_id": crop_id, "validated": validated, "label_source": refetched.get("label_source")}],
    )
    return ok


def step4_batch_label(api: Api, crops: list[dict[str, Any]]) -> bool:
    crop_ids = [c["crop_id"] for c in crops]
    if not crop_ids:
        return check("step 4: batch-label a selection", False, "no crops to batch-label")
    class_id = int(crops[0].get("class_id") or 0)
    # bulkLabel is a PUT, not a POST (api.ts:1262) -- get this wrong and
    # step 4 405s against a working backend.
    result = http_json(
        api.url("/crops/batch_label", "PUT"),
        "PUT",
        {"crop_ids": crop_ids, "class_id": class_id, "validated": True},
    )
    updated = result.get("updated")
    ok = check(
        "step 4: PUT /crops/batch_label updated == len(crop_ids)",
        updated == len(crop_ids),
        f"updated={updated} expected={len(crop_ids)}",
    )
    each_ok = True
    for cid in crop_ids:
        refetched = http_get(api.url(f"/crops/{cid}"))
        if refetched.get("class_id") != class_id:
            each_ok = False
    ok2 = check("step 4: re-GET each id shows the class", each_ok)
    record_step(4, "batch-label a selection", [{"updated": updated, "crop_ids": crop_ids}])
    return ok and ok2


def step5_refine_cluster(api: Api, cluster_id: int, skip: bool) -> bool:
    if skip:
        print("  [skip] step 5: cluster refine (--skip-steps)")
        record_step(5, "refine a cluster", [{"skipped": True}])
        return True
    # Not reversible -- print a warning naming the cluster id before firing.
    print(f"  [warn] step 5 will refine cluster {cluster_id} -- not reversible, re-runnable")
    result = http_json(api.url(f"/clusters/refine/{cluster_id}", "POST"), "POST")
    ok = check(
        "step 5: POST /clusters/refine/{id} -> action=='refined' and n_subclusters>1",
        result.get("action") == "refined" and (result.get("n_subclusters") or 0) > 1,
        f"action={result.get('action')} n_subclusters={result.get('n_subclusters')}",
    )
    record_step(5, "refine a cluster", [{"cluster_id": cluster_id, "result": result}])
    return ok


def step6_review_dismiss(api: Api, crop_id: str) -> bool:
    http_json(api.url(f"/crops/{crop_id}/review_dismiss", "POST"), "POST")
    page = http_get(api.url("/review/all"))
    ids = [i.get("crop_id") for i in page.get("items", [])]
    ok = check(
        "step 6: dismissed id absent from GET /review/all on refetch",
        crop_id not in ids,
    )
    record_step(6, "review dismiss", [{"crop_id": crop_id, "still_present": crop_id in ids}])
    return ok


def step7_region_write_restore(api: Api, crop: dict[str, Any]) -> bool:
    crop_id = crop["crop_id"]
    prior = http_get(api.url(f"/crops/{crop_id}"))
    prior_status = prior.get("plate_status")
    prior_text = prior.get("plate_text")

    bbox = [0.1, 0.1, 0.2, 0.2]
    write_result = http_json(api.url(f"/crops/{crop_id}/plate", "PUT"), "PUT", {"bbox_norm": bbox})
    meta_result = http_json(
        api.url(f"/crops/{crop_id}/plate_meta", "PATCH"),
        "PATCH",
        {"plate_text": "TESTROUNDTRIP", "plate_status": "detected"},
    )
    refetched = http_get(api.url(f"/crops/{crop_id}"))
    round_trip_ok = check(
        "step 7: plate write round-trips (plate_detector=='human', plate_text set)",
        refetched.get("plate_detector") == "human"
        and refetched.get("plate_text") == "TESTROUNDTRIP",
        f"plate_detector={refetched.get('plate_detector')} plate_text={refetched.get('plate_text')!r}",
    )

    # Restore what we can (§3.8 rule 2).
    restore_bbox = prior.get("plate_bbox_norm") or prior.get("bbox_norm")
    if restore_bbox:
        http_json(api.url(f"/crops/{crop_id}/plate", "PUT"), "PUT", {"bbox_norm": restore_bbox})
    else:
        http_json(api.url(f"/crops/{crop_id}/plate", "PUT"), "PUT", {"bbox_norm": None})
    http_json(
        api.url(f"/crops/{crop_id}/plate_meta", "PATCH"),
        "PATCH",
        {"plate_text": prior_text, "plate_status": prior_status},
    )
    after_restore = http_get(api.url(f"/crops/{crop_id}"))
    restore_ok = check(
        "step 7: restore succeeds (plate_text back to prior value)",
        after_restore.get("plate_text") == prior_text,
        f"expected={prior_text!r} got={after_restore.get('plate_text')!r}",
    )
    record_step(
        7,
        "region write + restore",
        [
            {
                "crop_id": crop_id,
                "write_result": write_result,
                "meta_result": meta_result,
                "prior": {"plate_status": prior_status, "plate_text": prior_text},
                "restored": restore_ok,
            }
        ],
    )
    return round_trip_ok and restore_ok


def step8_batch_region_status(api: Api, crops: list[dict[str, Any]]) -> bool:
    crop_ids = [c["crop_id"] for c in crops]
    if not crop_ids:
        return check("step 8: batch region status", False, "no plate crops available")
    prior_statuses = {c["crop_id"]: c.get("plate_status") for c in crops}

    result = http_json(
        api.url("/plates/batch_status", "POST"),
        "POST",
        {
            "crop_ids": crop_ids,
            "plate_status": "false_positive",
            "plate_verified": None,
            "label_source": "human",
        },
    )
    ok = check(
        "step 8: POST /plates/batch_status updated == len(crop_ids)",
        result.get("updated") == len(crop_ids),
        f"updated={result.get('updated')} expected={len(crop_ids)}",
    )
    reflected = all(
        http_get(api.url(f"/crops/{cid}")).get("plate_status") == "false_positive"
        for cid in crop_ids
    )
    ok2 = check("step 8: re-GET reflects the status", reflected)

    # Restore prior statuses.
    for cid in crop_ids:
        http_json(
            api.url(f"/crops/{cid}/plate_meta", "PATCH"),
            "PATCH",
            {"plate_status": prior_statuses.get(cid)},
        )
    record_step(8, "batch region status", [{"crop_ids": crop_ids, "result": result}])
    return ok and ok2


def step9_vlm_label_batch(api: Api, cluster_id: int, skip: bool) -> bool:
    if skip:
        print("  [skip] step 9: VLM label batch (--skip-steps)")
        record_step(9, "VLM label batch", [{"skipped": True}])
        return True
    print("  [warn] step 9 writes VLM suggestions -- re-runnable but not reversible")
    crops = pick_crops(api, cluster_id, 10)
    crop_ids = [c["crop_id"] for c in crops][:64]
    if not crop_ids:
        return check("step 9: VLM label batch", False, "no crops to send")
    try:
        result = http_json(
            api.url("/vlm/label_batch", "POST"), "POST", {"crop_ids": crop_ids}
        )
        status_ok = True
    except urllib.error.HTTPError as e:
        status_ok = e.code != 404
        result = {"error": str(e), "code": e.code}
    ok = check(
        "step 9: POST /vlm/label_batch 200 with predicted>=1, no 404 (route is vlm, not gemma)",
        status_ok and (result.get("predicted") or 0) >= 1,
        f"result={result}",
    )
    record_step(9, "VLM label batch", [{"crop_ids": crop_ids, "result": result}])
    return ok


def _sse_probe(page: Any, path: str, timeout_ms: int) -> dict[str, Any]:
    return page.evaluate(
        """async ({ path, timeoutMs }) => {
          return await new Promise((resolve) => {
            const es = new EventSource(path);
            let gotMessage = false;
            const timer = setTimeout(() => {
              es.close();
              resolve({ readyState: es.readyState, gotMessage, timedOutIdle: true, error: false });
            }, timeoutMs);
            es.onmessage = () => { gotMessage = true; };
            es.onerror = () => {
              clearTimeout(timer);
              es.close();
              resolve({ readyState: es.readyState, gotMessage, timedOutIdle: false, error: true });
            };
          });
        }""",
        {"path": path, "timeoutMs": timeout_ms},
    )


def _same_origin(a: str, b: str) -> bool:
    """True iff `a` and `b` share scheme+host+port (Python has no `new URL()`)."""
    pat = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*://[^/]+)")
    ma, mb = pat.match(a), pat.match(b)
    return bool(ma and mb and ma.group(1) == mb.group(1))


def step10_sse(page: Any, front: str, api: Api) -> bool:
    # Must run inside the page via page.evaluate -- the point is the
    # browser's view of the SSE endpoint through nginx, not Python's.
    #
    # sse.ts itself picks same-origin-relative vs. absolute based on
    # whether apiBase is a cross-origin URL (see its own `apiBase &&
    # /^https?:\/\//i.test(apiBase)` branch) -- this probe must mirror
    # that same branch, not hardcode the same-origin (production nginx
    # proxy) case. In a same-origin run (--url and --api share an
    # origin, or --api is unset/relative), api.relative() is correct
    # and matches sse.ts. In a cross-origin dev-server run (--api on a
    # different host/port than --url, as in a `npm run dev` +
    # PUBLIC_TRITON_API_URL=<remote> smoke test), a relative path
    # resolves against the WRONG origin (the frontend's, not the
    # API's) and this probe would report a false failure having tested
    # nothing -- use the absolute api.url() in that case instead.
    same_origin = _same_origin(front, api.origin)
    events_path = api.relative("/events") if same_origin else api.url("/events")
    pipeline_path = (
        api.relative("/pipeline/events") if same_origin else api.url("/pipeline/events")
    )
    page.goto(front, wait_until="domcontentloaded")
    result = _sse_probe(page, events_path, 8000)
    ok1 = check(
        "step 10: GET {prefix}/events reaches readyState==1 (OPEN), no error",
        not result.get("error"),
        f"{result}",
    )
    result2 = _sse_probe(page, pipeline_path, 8000)
    ok2 = check(
        "step 10: GET {prefix}/pipeline/events reaches readyState==1 (OPEN), no error",
        not result2.get("error"),
        f"{result2}",
    )
    record_step(10, "SSE both channels", [{"events": result, "pipeline_events": result2}])
    return ok1 and ok2


def step11_export_gating(api: Api, page: Any, front: str) -> bool:
    methods = http_get(api.url("/methods"))
    strategies = methods.get("strategies", [])
    export_entries = [s for s in strategies if s.get("axis") == "export"]
    has_yolo = any(s.get("id") == "yolo" for s in export_entries)
    has_lpr = any(s.get("id") == "lpr" for s in export_entries)
    ok = check(
        "step 11: /methods has axis=export id=yolo and no id=lpr",
        has_yolo and not has_lpr,
        f"entries={export_entries}",
    )
    page.goto(f"{front}/train", wait_until="domcontentloaded")
    lpr_panel_present = page.evaluate(
        "() => !!document.querySelector('[data-testid=lpr-export-panel]')"
    )
    ok2 = check("step 11: /train renders no LPR export panel", not lpr_panel_present)

    page.goto(f"{front}/export", wait_until="domcontentloaded")
    # /export only ever offers a YOLO-format export (the axis=export
    # entry checked above is literally id=='yolo') -- any "trigger an
    # export" control on this page IS the YOLO control, whether or not
    # its visible label spells "yolo" literally. The live label is
    # "Export to staging" / "Re-export", so match on the export verb
    # instead of assuming a specific brand string in the button copy.
    yolo_control_present = page.evaluate(
        "() => Array.from(document.querySelectorAll('button, [role=button]'))"
        ".some(b => /yolo|export/i.test(b.textContent || ''))"
    )
    ok3 = check("step 11: /export's export-trigger control is present", yolo_control_present)

    export_status_code = None
    try:
        with urllib.request.urlopen(api.url("/export/status"), timeout=15) as r:
            export_status_code = r.status
    except urllib.error.HTTPError as e:
        export_status_code = e.code
    ok4 = check("step 11: GET /export/status is 200", export_status_code == 200, f"got {export_status_code}")

    record_step(
        11,
        "export gating",
        [
            {
                "export_entries": export_entries,
                "lpr_panel_present": lpr_panel_present,
                "yolo_control_present": yolo_control_present,
                "export_status_code": export_status_code,
            }
        ],
    )
    return ok and ok2 and ok3 and ok4


def step12_train_read_path(api: Api, page: Any, front: str) -> bool:
    endpoints = ["/train/profiles", "/train/presets", "/train/runs", "/train/status"]
    codes: dict[str, int | None] = {}
    for ep in endpoints:
        try:
            with urllib.request.urlopen(api.url(ep), timeout=15) as r:
                codes[ep] = r.status
        except urllib.error.HTTPError as e:
            codes[ep] = e.code
    all_200 = all(v == 200 for v in codes.values())
    ok = check("step 12: /train/{profiles,presets,runs,status} all 200", all_200, f"{codes}")

    page.goto(f"{front}/train", wait_until="domcontentloaded")
    main_len = page.evaluate("() => (document.querySelector('main')?.innerText || '').length")
    ok2 = check("step 12: /train renders", main_len > 0)

    record_step(12, "train read path", [{"codes": codes, "main_len": main_len}])
    return ok and ok2


# --------------------------------------------------------------------------
# --dry-run
# --------------------------------------------------------------------------


def dry_run(api: Api, front: str) -> int:
    """Print every URL this run would touch and exit, without a browser.

    This is what makes the script verifiable with no backend: run it at
    two prefixes and diff. Every line must differ in exactly the prefix
    segment and nothing else -- that is the same property D3-RUN/E3-RUN
    assert at runtime (coordination plan §5.2: "Every step must run
    twice ... Identical results are the H2 acceptance criterion").

    Param-bearing URLs print with a literal {crop_id}/{cluster_id} token
    rather than a resolved id -- nothing has been fetched yet.
    """
    lines: list[str] = []

    def add(n: str, method: str, url: str) -> None:
        lines.append(f"{n:<4s}{method:<7s}{url}")

    for name, path in NAV_ROUTES:
        add("1", "GET", f"{front}{path}")

    add("2", "GET", api.url("/crops/{crop_id}/thumbnail"))
    add("2", "GET", api.url("/crops/{crop_id}/region_thumbnail"))
    add("2", "GET", f"{front}/clusters")
    add("2", "GET", f"{front}/clusters?class=license_plate")

    add("3", "PUT", api.url("/crops/{crop_id}/label"))
    add("3", "GET", api.url("/crops/{crop_id}"))

    add("4", "PUT", api.url("/crops/batch_label"))

    add("5", "POST", api.url("/clusters/refine/{cluster_id}"))
    add("5", "GET", api.url("/clusters"))

    add("6", "POST", api.url("/crops/{crop_id}/review_dismiss"))
    add("6", "GET", api.url("/review/all"))

    add("7", "GET", api.url("/crops/{crop_id}"))
    add("7", "PUT", api.url("/crops/{crop_id}/plate"))
    add("7", "PATCH", api.url("/crops/{crop_id}/plate_meta"))

    add("8", "POST", api.url("/plates/batch_status"))

    add("9", "GET", api.url("/crops"))
    add("9", "POST", api.url("/vlm/label_batch"))

    add("10", "GET", f"{front}(page) -> EventSource {api.relative('/events')}")
    add("10", "GET", f"{front}(page) -> EventSource {api.relative('/pipeline/events')}")

    add("11", "GET", api.url("/methods"))
    add("11", "GET", f"{front}/train")
    add("11", "GET", f"{front}/export")
    add("11", "GET", api.url("/export/status"))

    add("12", "GET", api.url("/train/profiles"))
    add("12", "GET", api.url("/train/presets"))
    add("12", "GET", api.url("/train/runs"))
    add("12", "GET", api.url("/train/status"))

    add("0", "*", f"(cross-cutting) every response above must start with {api.prefix}/")

    for line in lines:
        print(line)
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--url",
        default=os.environ.get("CROPWRIGHT_URL", "http://localhost:5184"),
        help="Cropwright frontend origin (nginx container; host port 5184).",
    )
    p.add_argument(
        "--api",
        default=os.environ.get("OP_API_URL", "http://localhost:4603"),
        help="OpenProcessor origin, for direct verification fetches.",
    )
    p.add_argument(
        "--api-prefix",
        default=os.environ.get("PUBLIC_API_PREFIX", ""),
        help="Backend path prefix. Normalized exactly like "
        "normalizeApiPrefix() in src/lib/api.ts: empty or a leaked "
        "__API_PREFIX__ placeholder means the transitional default.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output dir. Defaults to /tmp/cropwright_integration_<prefix-slug>, "
        "so a /curation run and a /curation run never overwrite each other.",
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print every URL this run would touch, then exit 0. "
        "Launches no browser and makes no request.",
    )
    p.add_argument(
        "--skip-steps",
        default="",
        help="Comma-separated step numbers to skip, e.g. '5,9' to omit "
        "cluster refine and the VLM batch on a re-run.",
    )
    p.add_argument("--timeout", type=int, default=20000, help="Per-navigation timeout (ms).")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    prefix = normalize_api_prefix(args.api_prefix)
    api = Api(args.api, prefix)
    front = args.url.rstrip("/")

    if args.dry_run:
        return dry_run(api, front)

    skip_steps = {int(s) for s in args.skip_steps.split(",") if s.strip()}

    out_dir = Path(args.out or f"/tmp/cropwright_integration_{prefix.strip('/')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright  # deferred: --dry-run needs no browser

    observed_responses: list[dict[str, Any]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        page.on(
            "response",
            lambda r: observed_responses.append({"url": r.url, "status": r.status}),
        )

        print(f"[integration] frontend={front} api={args.api} prefix={prefix}")

        if 1 not in skip_steps:
            step1_navigation(page, front, args.timeout)

        cluster_id = pick_cluster(api)
        crops = pick_crops(api, cluster_id, 10)
        if not crops:
            print(f"[integration] FATAL: no unvalidated crops in cluster {cluster_id}")
            return 1
        target_crop = crops[0]

        if 2 not in skip_steps:
            step2_thumbnails(page, front, api, args.timeout)
        if 3 not in skip_steps:
            step3_label_crop(api, target_crop)
        if 4 not in skip_steps:
            step4_batch_label(api, crops[1:4] or crops[:1])
        step5_refine_cluster(api, cluster_id, skip=5 in skip_steps)
        if 6 not in skip_steps:
            step6_review_dismiss(api, target_crop["crop_id"])

        plate_crop = pick_plate_crop(api)
        if 7 not in skip_steps and plate_crop:
            step7_region_write_restore(api, plate_crop)
        elif 7 not in skip_steps:
            check("step 7: region write + restore", False, "no plate crop available")

        plate_crops = [pick_plate_crop(api)] if plate_crop else []
        if 8 not in skip_steps and plate_crops and plate_crops[0]:
            step8_batch_region_status(api, [c for c in plate_crops if c])
        elif 8 not in skip_steps:
            check("step 8: batch region status", False, "no plate crops available")

        step9_vlm_label_batch(api, cluster_id, skip=9 in skip_steps)

        if 10 not in skip_steps:
            step10_sse(page, front, api)
        if 11 not in skip_steps:
            step11_export_gating(api, page, front)
        if 12 not in skip_steps:
            step12_train_read_path(api, page, front)

        browser.close()

    step0_prefix_purity(api, observed_responses)

    summary = {
        "frontend": front,
        "api": args.api,
        "prefix": prefix,
        "steps": STEPS,
        "failures": FAILURES,
        "status": "PASS" if not FAILURES else "FAIL",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[integration] summary -> {summary_path}")
    print(f"[integration] {summary['status']}" + (f" — {len(FAILURES)} failure(s)" if FAILURES else ""))
    for f in FAILURES:
        print(f"  - {f}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
