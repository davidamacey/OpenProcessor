"""Processing plans: which branches of the image pipeline a request runs (#67).

A plan is a short list of named branches from a closed vocabulary, one condition kind
(``if_detected``, by class name) and a per-route allowlist. :func:`resolve` either returns a plan
that runs exactly as written or raises :class:`PlanError`: nothing is added, dropped or guessed.
Pure rules, no I/O and no pydantic; the wire models live in ``src/schemas/processing_plan.py``.

The class-name check against a detector's vocabulary (``unknown_class``), the curation policy
intersection (``branch_not_in_policy``) and the curation default derived from the ingest policy
need state outside this module and arrive with the routes that own that state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


Branch = Literal[
    'detect',
    'embed_frame',
    'embed_crops',
    'faces',
    'face_embed',
    'ocr',
    'near_duplicates',
    'region',
]
RouteKey = Literal['analyze', 'analyze_batch', 'ingest_core', 'curation_ingest']
PlanErrorReason = Literal[
    'unknown_branch',
    'empty_plan',
    'duplicate_branch',
    'branch_not_supported',
    'missing_dependency',
    'condition_on_unconditional',
    'empty_condition',
    'paired_branches',
    'faces_disabled',
    'plan_required',
]

# Canonical order, also the execution order.
BRANCHES: tuple[Branch, ...] = get_args(Branch)

REQUIRES: Mapping[Branch, tuple[Branch, ...]] = {
    'embed_crops': ('detect',),
    'face_embed': ('faces',),
    'near_duplicates': ('embed_frame',),
    'region': ('detect',),
}

# Branches that may carry an ``if_detected`` condition. The condition itself needs ``detect``.
CONDITIONAL: frozenset[Branch] = frozenset({'faces', 'ocr'})

FACE_BRANCHES: frozenset[Branch] = frozenset({'faces', 'face_embed'})

ALLOWED: Mapping[RouteKey, frozenset[Branch]] = {
    'analyze': frozenset({'detect', 'embed_frame', 'embed_crops', 'faces', 'face_embed', 'ocr'}),
    'analyze_batch': frozenset(
        {'detect', 'embed_frame', 'embed_crops', 'faces', 'face_embed', 'ocr'}
    ),
    'ingest_core': frozenset(
        {
            'detect',
            'embed_frame',
            'embed_crops',
            'faces',
            'face_embed',
            'ocr',
            'near_duplicates',
        }
    ),
    # No curation store exists for faces, text or duplicate groups: asking for them is a 422.
    'curation_ingest': frozenset({'detect', 'embed_frame', 'embed_crops', 'region'}),
}

# What a request without a plan returns today (design section 3.4). Curation has no entry: its
# default is the project's ingest policy, so a curation request without a plan is refused here
# until the route supplies one.
DEFAULT: Mapping[RouteKey, tuple[Branch, ...]] = {
    'analyze': ('detect', 'faces', 'ocr'),
    'analyze_batch': ('detect', 'faces'),
    'ingest_core': (
        'detect',
        'embed_frame',
        'embed_crops',
        'faces',
        'face_embed',
        'ocr',
        'near_duplicates',
    ),
}


class PlanError(ValueError):
    """A plan the server cannot run exactly as written (HTTP 422)."""

    def __init__(self, reason: PlanErrorReason, branch: str | None = None) -> None:
        self.reason = reason
        self.branch = branch
        super().__init__(f'{reason}: {branch}' if branch else reason)


@dataclass(frozen=True)
class PlanRequest:
    """A requested plan before resolution: branch names as sent, and ``when`` as sent."""

    branches: Sequence[str]
    when: Mapping[str, Sequence[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedPlan:
    route: RouteKey
    branches: tuple[Branch, ...]
    # Normalized class names per conditional branch; a branch absent here is unconditional.
    if_detected: Mapping[Branch, tuple[str, ...]] = field(default_factory=dict)

    def runs(self, branch: Branch) -> bool:
        return branch in self.branches

    @property
    def key(self) -> str:
        """Canonical label, e.g. ``detect+faces`` (order-insensitive to the request)."""
        return '+'.join(self.branches)


def _default_branches(route: RouteKey, *, faces_enabled: bool) -> tuple[Branch, ...]:
    if route not in DEFAULT:
        raise PlanError('plan_required', route)
    return tuple(b for b in DEFAULT[route] if faces_enabled or b not in FACE_BRANCHES)


def _check_branches(
    route: RouteKey, requested: PlanRequest, *, faces_enabled: bool
) -> tuple[Branch, ...]:
    names = list(requested.branches)
    if not names:
        raise PlanError('empty_plan')
    for name in names:
        if name not in BRANCHES:
            raise PlanError('unknown_branch', str(name))
    if len(set(names)) != len(names):
        dup = next(n for n in names if names.count(n) > 1)
        raise PlanError('duplicate_branch', dup)
    chosen: set[Branch] = {b for b in BRANCHES if b in names}
    for branch in BRANCHES:
        if branch in chosen and branch not in ALLOWED[route]:
            raise PlanError('branch_not_supported', branch)
    for branch in BRANCHES:
        if branch in chosen and not faces_enabled and branch in FACE_BRANCHES:
            raise PlanError('faces_disabled', branch)
    for branch in BRANCHES:
        if branch in chosen:
            for needed in REQUIRES.get(branch, ()):
                if needed not in chosen:
                    raise PlanError('missing_dependency', needed)
    # Core ingest stores per-category vector indexes: a detection with no vector has nowhere to go.
    if route == 'ingest_core' and 'detect' in chosen and 'embed_crops' not in chosen:
        raise PlanError('paired_branches', 'embed_crops')
    return tuple(b for b in BRANCHES if b in chosen)


def _check_conditions(
    branches: tuple[Branch, ...], when: Mapping[str, Sequence[str]]
) -> dict[Branch, tuple[str, ...]]:
    out: dict[Branch, tuple[str, ...]] = {}
    for name, classes in when.items():
        if name not in CONDITIONAL or name not in branches:
            raise PlanError('condition_on_unconditional', name)
        if 'detect' not in branches:
            raise PlanError('missing_dependency', 'detect')
        normalized = tuple(sorted({n for n in map(normalize_class_name, classes) if n}))
        if not normalized:
            raise PlanError('empty_condition', name)
        out[name] = normalized
    return out


def resolve(route: RouteKey, requested: PlanRequest | None, *, faces_enabled: bool) -> ResolvedPlan:
    """The plan ``route`` runs for ``requested`` (``None``: the route's default plan).

    ``faces_enabled`` is the deployment switch (``OP_FACES_ENABLED``): the default plan then
    leaves the face branches out, and an explicit request for one is refused with
    ``faces_disabled``.

    Raises:
        PlanError: the plan cannot run exactly as written.
    """
    if requested is None:
        return ResolvedPlan(route, _default_branches(route, faces_enabled=faces_enabled))
    branches = _check_branches(route, requested, faces_enabled=faces_enabled)
    return ResolvedPlan(route, branches, _check_conditions(branches, requested.when))
