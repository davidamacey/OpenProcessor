"""Behavioural tests for ``scripts/codegen/check_no_literal_region_fields.py``
(plan Wave 5 W5.d, T-5). Before this file the guard had zero tests
beyond hook-registration/path-existence checks in
``tests/curation/test_precommit_paths.py`` — none of its accumulated
skip rules (8 commits' worth) were ever exercised.

The guard's ``main()`` resolves every scanned path against the real
repo root via ``Path(__file__).resolve().parents[2]`` (never
overridable), so a path under an arbitrary ``tmp_path`` can never match
a ``PORTED_PATHS`` prefix — subprocessing the CLI against synthetic
fixture files can only prove the *skip* path (untracked/non-ported
files are ignored), not the *catch* path. The catch-path skip rules
(Pydantic attr, wire-key, fields_set) and the genuine-violation case
are exercised directly against ``_scan_file``, which is pure content
scanning with no path-identity dependency — the more precise test for
regex behavior anyway. ``main()``'s CLI wiring (argv, exit codes,
stderr) is covered separately via subprocess against real,
already-committed files whose exemption status is known.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import scripts.codegen.check_no_literal_region_fields as guard


REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_SCRIPT = REPO_ROOT / 'scripts' / 'codegen' / 'check_no_literal_region_fields.py'


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / 'fixture.py'
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# _scan_file — each accumulated skip rule
# ---------------------------------------------------------------------------


def test_scan_file_ignores_a_plain_comment_mentioning_an_unrelated_literal(
    tmp_path: Path,
) -> None:
    # The literal regex only matches a quoted `plate_...`-prefixed token;
    # prose describing the *value* (not the field name) never matches.
    content = "# Human reviewed and said plate_status='no_region_visible'\n"
    path = _write(tmp_path, content)
    assert guard._scan_file(path) == []


def test_scan_file_ignores_pydantic_attribute_declaration(tmp_path: Path) -> None:
    content = 'plate_status: str | None = None\n'
    path = _write(tmp_path, content)
    assert guard._scan_file(path) == []


def test_scan_file_ignores_wire_response_key_routed_through_region_fields(
    tmp_path: Path,
) -> None:
    content = "        'plate_status': src.get(F.status),\n"
    path = _write(tmp_path, content)
    assert guard._scan_file(path) == []


def test_scan_file_ignores_wire_response_key_routed_through_payload_attribute(
    tmp_path: Path,
) -> None:
    content = "        'plate_status': payload.plate_status,\n"
    path = _write(tmp_path, content)
    assert guard._scan_file(path) == []


def test_scan_file_ignores_wire_response_key_routed_through_dict_bracket_field(
    tmp_path: Path,
) -> None:
    content = "        'plate_bbox_norm': doc[F.bbox_norm],\n"
    path = _write(tmp_path, content)
    assert guard._scan_file(path) == []


def test_scan_file_ignores_url_literal_after_the_key(tmp_path: Path) -> None:
    content = (
        "        'plate_thumbnail_url': f'{config.api_prefix}/crops/{crop_id}/region_thumbnail',\n"
    )
    path = _write(tmp_path, content)
    assert guard._scan_file(path) == []


def test_scan_file_ignores_model_fields_set_membership_check(tmp_path: Path) -> None:
    content = "    if 'plate_text' in fields_set:\n"
    path = _write(tmp_path, content)
    assert guard._scan_file(path) == []


def test_scan_file_catches_a_genuine_bare_status_literal_outside_any_exemption(
    tmp_path: Path,
) -> None:
    """The case that MUST be caught: a bare `'plate_status'` dict-key
    literal whose *value* is a hardcoded string, not routed through
    RegionFields/payload/a URL — exactly what a copy-paste from the
    reference tree without genericization would leave behind."""
    content = "        'plate_status': 'detected',\n"
    path = _write(tmp_path, content)
    violations = guard._scan_file(path)
    assert len(violations) == 1
    lineno, line = violations[0]
    assert lineno == 1
    assert "'plate_status'" in line


def test_scan_file_catches_a_bare_literal_used_as_a_plain_value(tmp_path: Path) -> None:
    content = "region_status = 'plate_status'\n"
    path = _write(tmp_path, content)
    violations = guard._scan_file(path)
    assert len(violations) == 1


def test_scan_file_reports_every_violating_line_with_its_line_number(tmp_path: Path) -> None:
    content = "x = 'plate_status'\ny = 'not_a_match'\nz = 'plate_bbox_norm'\n"
    path = _write(tmp_path, content)
    violations = guard._scan_file(path)
    assert [v[0] for v in violations] == [1, 3]


def test_scan_file_missing_file_returns_no_violations_rather_than_raising(
    tmp_path: Path,
) -> None:
    assert guard._scan_file(tmp_path / 'does-not-exist.py') == []


# ---------------------------------------------------------------------------
# _is_ported / _is_exempt — path-gating logic
# ---------------------------------------------------------------------------


def test_is_ported_matches_an_exact_file_in_ported_paths() -> None:
    assert guard._is_ported('src/routers/curation/regions.py') is True


def test_is_ported_matches_a_file_under_a_directory_prefix() -> None:
    # 'scripts/curation/worker/' is a directory-prefix entry.
    assert guard._is_ported('scripts/curation/worker/runner.py') is True


def test_is_ported_false_for_an_unrelated_path() -> None:
    assert guard._is_ported('src/routers/detect.py') is False


def test_is_ported_does_not_match_a_lookalike_sibling_file() -> None:
    # 'src/routers/curation/regions.py' is listed; a differently-named
    # sibling file must not match by prefix accident.
    assert guard._is_ported('src/routers/curation/regions_extra.py') is False


def test_is_exempt_true_for_the_two_hardcoded_exemptions() -> None:
    assert guard._is_exempt('src/config/region_fields.py') is True
    assert guard._is_exempt('tests/curation/test_region_fields.py') is True


def test_is_exempt_true_for_anything_under_docs() -> None:
    assert guard._is_exempt('docs/design/curation_design_rationale.md') is True


def test_is_exempt_false_for_an_ordinary_ported_file() -> None:
    assert guard._is_exempt('src/routers/curation/regions.py') is False


# ---------------------------------------------------------------------------
# main() CLI wiring — exit codes, against real, already-committed files
# whose exemption status is known (never mutates the working tree).
# ---------------------------------------------------------------------------


def _run_guard(*rel_paths: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GUARD_SCRIPT), *rel_paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_exits_zero_for_the_fully_exempt_region_fields_module() -> None:
    result = _run_guard('src/config/region_fields.py')
    assert result.returncode == 0, result.stderr


def test_cli_exits_zero_for_a_file_never_listed_in_ported_paths() -> None:
    # Not in PORTED_PATHS at all -> skipped regardless of content.
    result = _run_guard('src/routers/detect.py')
    assert result.returncode == 0, result.stderr


def test_cli_exits_zero_and_scans_nothing_for_no_arguments() -> None:
    result = _run_guard()
    assert result.returncode == 0
