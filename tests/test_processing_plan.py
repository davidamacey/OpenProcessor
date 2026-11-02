"""Processing-plan resolution (#67 P1): pure rules, no I/O.

One case per reason code, the default plan per route, the faces-off switch, class-name
normalization of ``if_detected``, and properties over every subset of the vocabulary.
"""

from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from src.domain.processing_plan import (
    ALLOWED,
    BRANCHES,
    CONDITIONAL,
    DEFAULT,
    REQUIRES,
    PlanError,
    PlanRequest,
    RouteKey,
    resolve,
)
from src.schemas.processing_plan import PlanReport, ProcessingPlan


ROUTES: tuple[RouteKey, ...] = ('analyze', 'analyze_batch', 'ingest_core', 'curation_ingest')


def _plan(route: RouteKey, *branches: str, when: dict[str, list[str]] | None = None):
    return resolve(route, PlanRequest(branches, when or {}), faces_enabled=True)


def _reason(route: RouteKey, *branches: str, when=None, faces_enabled: bool = True) -> PlanError:
    with pytest.raises(PlanError) as err:
        resolve(route, PlanRequest(branches, when or {}), faces_enabled=faces_enabled)
    return err.value


class TestReasonCodes:
    def test_unknown_branch(self) -> None:
        err = _reason('analyze', 'detect', 'segment')
        assert (err.reason, err.branch) == ('unknown_branch', 'segment')

    def test_empty_plan(self) -> None:
        assert _reason('analyze').reason == 'empty_plan'

    def test_duplicate_branch(self) -> None:
        err = _reason('analyze', 'detect', 'faces', 'detect')
        assert (err.reason, err.branch) == ('duplicate_branch', 'detect')

    @pytest.mark.parametrize(
        ('route', 'branch'),
        [
            ('analyze', 'near_duplicates'),
            ('analyze', 'region'),
            ('analyze_batch', 'region'),
            ('ingest_core', 'region'),
            ('curation_ingest', 'faces'),
            ('curation_ingest', 'ocr'),
            ('curation_ingest', 'near_duplicates'),
        ],
    )
    def test_branch_not_supported_names_the_branch(self, route: RouteKey, branch: str) -> None:
        # Ask for the branch together with whatever it needs so only the route is at fault.
        needs = {
            'face_embed': ('faces',),
            'near_duplicates': ('embed_frame',),
            'region': ('detect',),
        }
        err = _reason(route, branch, *needs.get(branch, ()))
        assert (err.reason, err.branch) == ('branch_not_supported', branch)

    @pytest.mark.parametrize(
        ('branches', 'missing'),
        [
            (('embed_crops',), 'detect'),
            (('face_embed',), 'faces'),
            (('near_duplicates',), 'embed_frame'),
            (('region',), 'detect'),
        ],
    )
    def test_missing_dependency_is_refused_not_added(
        self, branches: tuple[str, ...], missing: str
    ) -> None:
        route: RouteKey = 'curation_ingest' if branches == ('region',) else 'ingest_core'
        err = _reason(route, *branches)
        assert (err.reason, err.branch) == ('missing_dependency', missing)

    def test_condition_on_an_unconditional_branch(self) -> None:
        err = _reason('analyze', 'detect', 'embed_frame', when={'embed_frame': ['car']})
        assert (err.reason, err.branch) == ('condition_on_unconditional', 'embed_frame')

    def test_condition_on_a_branch_not_in_the_plan(self) -> None:
        err = _reason('analyze', 'detect', when={'ocr': ['car']})
        assert err.reason == 'condition_on_unconditional'

    def test_condition_needs_detect(self) -> None:
        err = _reason('analyze', 'ocr', when={'ocr': ['car']})
        assert (err.reason, err.branch) == ('missing_dependency', 'detect')

    def test_condition_with_no_usable_class_name(self) -> None:
        err = _reason('analyze', 'detect', 'ocr', when={'ocr': ['', '  ']})
        assert (err.reason, err.branch) == ('empty_condition', 'ocr')

    def test_core_ingest_detect_without_crop_vectors(self) -> None:
        err = _reason('ingest_core', 'detect', 'embed_frame')
        assert (err.reason, err.branch) == ('paired_branches', 'embed_crops')

    def test_curation_without_a_plan_is_refused(self) -> None:
        with pytest.raises(PlanError) as err:
            resolve('curation_ingest', None, faces_enabled=True)
        assert err.value.reason == 'plan_required'


class TestFacesSwitch:
    @pytest.mark.parametrize('branches', [('faces',), ('faces', 'face_embed')])
    def test_explicit_faces_are_refused_when_off(self, branches: tuple[str, ...]) -> None:
        err = _reason('analyze', *branches, faces_enabled=False)
        assert (err.reason, err.branch) == ('faces_disabled', 'faces')

    def test_face_embed_alone_names_face_embed_first_only_if_faces_is_absent(self) -> None:
        err = _reason('analyze', 'face_embed', faces_enabled=False)
        assert (err.reason, err.branch) == ('faces_disabled', 'face_embed')

    @pytest.mark.parametrize('route', ['analyze', 'analyze_batch', 'ingest_core'])
    def test_default_plan_omits_the_face_branches_when_off(self, route: RouteKey) -> None:
        plan = resolve(route, None, faces_enabled=False)
        assert 'faces' not in plan.branches
        assert 'face_embed' not in plan.branches
        assert plan.branches == tuple(b for b in DEFAULT[route] if b not in ('faces', 'face_embed'))

    def test_non_face_plan_resolves_when_off(self) -> None:
        plan = resolve('analyze', PlanRequest(['detect', 'ocr']), faces_enabled=False)
        assert plan.branches == ('detect', 'ocr')


class TestDefaults:
    @pytest.mark.parametrize('route', ['analyze', 'analyze_batch', 'ingest_core'])
    def test_default_plan_is_todays_chain(self, route: RouteKey) -> None:
        plan = resolve(route, None, faces_enabled=True)
        assert plan.branches == DEFAULT[route]

    def test_the_documented_chains(self) -> None:
        assert resolve('analyze', None, faces_enabled=True).key == 'detect+faces+ocr'
        assert resolve('analyze_batch', None, faces_enabled=True).key == 'detect+faces'
        assert resolve('ingest_core', None, faces_enabled=True).branches == (
            'detect',
            'embed_frame',
            'embed_crops',
            'faces',
            'face_embed',
            'ocr',
            'near_duplicates',
        )

    @pytest.mark.parametrize('route', ['analyze', 'analyze_batch', 'ingest_core'])
    def test_every_default_is_itself_a_valid_request(self, route: RouteKey) -> None:
        plan = resolve(route, PlanRequest(DEFAULT[route]), faces_enabled=True)
        assert plan.branches == DEFAULT[route]


class TestResolution:
    def test_canonical_order_and_key_ignore_request_order(self) -> None:
        a = _plan('analyze', 'ocr', 'faces', 'detect')
        b = _plan('analyze', 'detect', 'faces', 'ocr')
        assert a.branches == b.branches == ('detect', 'faces', 'ocr')
        assert a.key == b.key == 'detect+faces+ocr'

    def test_runs(self) -> None:
        plan = _plan('analyze', 'faces', 'face_embed')
        assert plan.runs('faces')
        assert not plan.runs('ocr')

    def test_if_detected_uses_the_shared_class_name_normalization(self) -> None:
        plan = _plan(
            'analyze', 'detect', 'ocr', when={'ocr': ['Traffic Light', 'traffic-light', 'CAR']}
        )
        assert plan.if_detected == {'ocr': ('car', 'traffic_light')}

    def test_unconditional_branches_carry_no_condition(self) -> None:
        plan = _plan('analyze', 'detect', 'faces', 'ocr', when={'ocr': ['car']})
        assert 'faces' not in plan.if_detected

    def test_a_plan_can_narrow_to_one_branch(self) -> None:
        assert _plan('analyze', 'faces', 'face_embed').key == 'faces+face_embed'


class TestProperties:
    """Over every subset of the vocabulary: PlanError, or a plan closed under REQUIRES."""

    @pytest.mark.parametrize('route', ROUTES)
    def test_every_subset(self, route: RouteKey) -> None:
        accepted = 0
        for size in range(1, len(BRANCHES) + 1):
            for subset in combinations(BRANCHES, size):
                try:
                    plan = resolve(route, PlanRequest(subset), faces_enabled=True)
                except PlanError:
                    continue
                accepted += 1
                assert set(plan.branches) == set(subset)
                assert set(plan.branches) <= ALLOWED[route]
                for branch in plan.branches:
                    assert set(REQUIRES.get(branch, ())) <= set(plan.branches)
                shuffled = resolve(route, PlanRequest(tuple(reversed(subset))), faces_enabled=True)
                assert shuffled.key == plan.key
        assert accepted > 0

    def test_conditional_branches_are_a_subset_of_the_vocabulary(self) -> None:
        assert set(BRANCHES) >= CONDITIONAL

    def test_every_route_has_an_allowlist_inside_the_vocabulary(self) -> None:
        assert set(ALLOWED) == set(ROUTES)
        for allowed in ALLOWED.values():
            assert allowed <= set(BRANCHES)


class TestWireModels:
    def test_plan_round_trips_into_a_request(self) -> None:
        plan = ProcessingPlan.model_validate_json(
            '{"branches": ["detect", "ocr"], "when": {"ocr": {"if_detected": ["car"]}}}'
        )
        resolved = resolve('analyze', plan.to_request(), faces_enabled=True)
        assert resolved.if_detected == {'ocr': ('car',)}

    def test_an_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProcessingPlan.model_validate_json('{"branches": ["detect"], "extra": 1}')

    def test_an_unknown_condition_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProcessingPlan.model_validate_json(
                '{"branches": ["detect"], "when": {"ocr": {"if_missing": ["car"]}}}'
            )

    def test_an_unknown_branch_reaches_resolve_with_its_reason_code(self) -> None:
        plan = ProcessingPlan.model_validate_json('{"branches": ["detect", "teleport"]}')
        with pytest.raises(PlanError) as err:
            resolve('analyze', plan.to_request(), faces_enabled=True)
        assert err.value.reason == 'unknown_branch'

    def test_report_carries_the_disabled_outcome(self) -> None:
        report = PlanReport.model_validate(
            {
                'branches': ['detect', 'faces'],
                'outcomes': {
                    'detect': {'status': 'ok', 'ms': 3.2},
                    'faces': {'status': 'disabled', 'ms': 0.0},
                },
            }
        )
        assert report.outcomes['faces'].status == 'disabled'
