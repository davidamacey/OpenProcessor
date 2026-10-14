"""Read-only request guard for the docs screenshot capture: GET/HEAD pass,
every other request is aborted except the explicit allow-list below."""

from __future__ import annotations

import re
from urllib.parse import urlsplit


# The ONLY non-GET/HEAD calls the capture lets through. Each one is
# side-effect-free (a dry run, a validation, a preview or a test that writes
# nothing); anything else is aborted. (method, path regex, why, optional
# predicate over (query string, request body text)).
READ_ONLY_CALLS: list[tuple[str, re.Pattern[str], str, object]] = [
    ("POST", re.compile(r"/train/preflight$"), "preflight report", None),
    (
        "DELETE",
        re.compile(r"/projects/[^/]+$"),
        "project delete DRY RUN (dry_run=true, no confirm) shown in the delete dialog",
        lambda q, body: "dry_run=true" in q and "confirm=" not in q,
    ),
    ("POST", re.compile(r"/projects/combine/preview$"), "combine preview report", None),
    (
        "POST",
        re.compile(r"/projects/[^/]+/reprocess$"),
        "multi-item Reprocess 'Check what would run' (body dry_run: true)",
        lambda q, body: '"dry_run":true' in body.replace(" ", ""),
    ),
    ("POST", re.compile(r"/prompt_packs/validate$"), "prompt-pack validation report", None),
    (
        "POST",
        re.compile(r"/prompt_packs/test$"),
        "prompt-pack 'Test on a crop' (runs the VLM on stored crops, 'Nothing is written')",
        None,
    ),
    ("POST", re.compile(r"/region_profiles/validate$"), "region-profile validation report", None),
    (
        "POST",
        re.compile(r"/region_profiles/test$"),
        "region-profile 'Test on a crop' (runs the detector/segmenter on stored crops, writes nothing)",
        None,
    ),
    (
        "POST",
        re.compile(r"/datasets/preview$"),
        "import wizard preview (dry run: 'writes nothing', dataset problems come back as issues)",
        None,
    ),
    ("POST", re.compile(r"/keymap/validate$"), "keymap validation report", None),
    ("POST", re.compile(r"/ingest/policy/preview$"), "ingest policy cost preview (a report over stored detections)", None),
    (
        "POST",
        re.compile(r"/region_profiles/validate_segmenter_prompt$"),
        "'Check segmenter prompt' (text-only validation report)",
        None,
    ),
    ("POST", re.compile(r"/open_vocab/validate$"), "open-vocabulary set validation report", None),
    (
        "POST",
        re.compile(r"/open_vocab/test$"),
        "open-vocabulary 'Test on an image' (runs SAM 3 on one stored image, writes nothing)",
        None,
    ),
    ("POST", re.compile(r"/vlm/endpoints/validate$"), "VLM endpoint validation report", None),
]


def _allowed(method: str, url: str, body: str) -> bool:
    if method in ("GET", "HEAD"):
        return True
    parts = urlsplit(url)
    for m, pattern, _why, pred in READ_ONLY_CALLS:
        if method == m and pattern.search(parts.path) and (pred is None or pred(parts.query, body)):
            return True
    return False


def _read_only(route) -> None:
    req = route.request
    if _allowed(req.method, req.url, req.post_data or ""):
        route.continue_()
    else:
        print(f"  blocked {req.method} {req.url}")
        route.abort()
