"""Guard against CFG-4 (see docs/design/curation_design_rationale.md and
the OSS completion plan §0.6): ``env.template`` used to document *zero*
curation vars while the code read over a hundred ``OP_*`` env vars, and
there is no mechanism to keep the two in sync as new ones get added.

Two directions:

1. Every ``OP_*`` literal passed to ``os.environ.get(...)``/``os.getenv(...)``
   anywhere under :data:`_SCANNED_ROOTS`, PLUS every var implied by the
   three ``from_env``-driven config dataclasses (``CurationConfig``,
   ``RegionFields``, ``DetectionProfile``), must appear somewhere in
   ``env.template`` (commented out is fine -- this repo's convention is to
   ship every optional var pre-commented with its default).
2. Every ``OP_*``-shaped token in ``env.template`` must correspond to a
   real var read somewhere in the code (catches documenting a var that
   was renamed/removed and never updated).

``RegionFields``/``DetectionProfile`` are documented via a 2-3-example +
"see the dataclass" pattern rather than enumerating all ~30 fields each
(matching the pre-existing ``OP_REGION_FIELD_*`` convention) -- both
prefixes are allowlisted for direction 1 with that reason.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

from scripts.curation.bakeoff.profile import BakeoffProfile
from src.config.detection_profile import DetectionProfile
from src.config.region_fields import RegionFields


REPO_ROOT = Path(__file__).resolve().parents[1]

_DIRECT_ENV_RE = re.compile(r"os\.(?:environ\.get|getenv)\(\s*['\"](OP_[A-Z0-9_]+)['\"]")

# First-party Python this repo ships. ``docker/`` holds the side-car container
# sources (the trainer watcher, the test-harness fakes) -- they are not
# importable from ``src/`` but they read ``OP_*`` vars a deployment has to set,
# so leaving them unscanned would let the trainer's whole env surface drift out
# of env.template unnoticed.
_SCANNED_ROOTS = ('src', 'scripts', 'docker')

# Prefixes documented via a "2-3 examples + see the dataclass" convention
# rather than one env.template line per field (RegionFields has ~30
# fields, DetectionProfile ~30) -- direction 1 (code -> env.template) is
# allowlisted for these; direction 2 (env.template -> code) still checks
# every literal env.template actually spells out.
_PATTERN_DOCUMENTED_PREFIXES = (
    'OP_REGION_FIELD_',
    'OP_INGEST_PRIMARY_',
    'OP_INGEST_SECONDARY_',
    'OP_REGION_DETECTION_',
    'OP_BAKEOFF_PROFILE_',
)

# Vars read by code but intentionally not surfaced in env.template: none
# yet. Keep this real -- if something lands here, document why.
_CODE_SIDE_ALLOWLIST: dict[str, str] = {}

# Tokens that look like OP_* but are prose/pattern placeholders in
# env.template, not real env vars a user would set verbatim.
_TEMPLATE_SIDE_ALLOWLIST = {
    'OP_',  # pattern-prefix mentions, e.g. "Pattern: OP_REGION_FIELD_<ATTR>"
    'OP_REGION_FIELD_',
    'OP_DETECTION_',  # the retired prefix, named in the migration note
    'OP_INGEST_PRIMARY_',
    'OP_INGEST_SECONDARY_',
    'OP_REGION_DETECTION_',
    'OP_BAKEOFF_PROFILE_',
}


def _direct_env_vars_read_by_code() -> set[str]:
    found: set[str] = set()
    for root in _SCANNED_ROOTS:
        for path in (REPO_ROOT / root).rglob('*.py'):
            text = path.read_text(encoding='utf-8', errors='ignore')
            found.update(_DIRECT_ENV_RE.findall(text))
    return found


def _from_env_derived_vars() -> set[str]:
    derived = set()
    for f in fields(RegionFields):
        derived.add(f'OP_REGION_FIELD_{f.name.upper()}')
    for f in fields(DetectionProfile):
        for prefix in ('OP_INGEST_PRIMARY_', 'OP_INGEST_SECONDARY_', 'OP_REGION_DETECTION_'):
            derived.add(f'{prefix}{f.name.upper()}')
    for f in fields(BakeoffProfile):
        derived.add(f'OP_BAKEOFF_PROFILE_{f.name.upper()}')
    # CurationConfig.from_env uses manual per-field keys (not always the
    # field name uppercased, e.g. class_registry_path -> OP_REGISTRY_PATH)
    # -- regex-scan the classmethod's own source for the literal keys it
    # passes to its _str/_path/_int/_float/_bool helpers, rather than guessing.
    curation_src = (REPO_ROOT / 'src/config/curation.py').read_text()
    for m in re.finditer(r"_(?:str|path|int|float|bool)\(\s*'([A-Z0-9_]+)'", curation_src):
        derived.add(f'OP_{m.group(1)}')
    return derived


def _env_template_tokens() -> set[str]:
    text = (REPO_ROOT / 'env.template').read_text()
    return set(re.findall(r'OP_[A-Z0-9_]*', text))


def test_every_op_env_var_read_by_code_is_documented_in_env_template() -> None:
    code_vars = _direct_env_vars_read_by_code() | _from_env_derived_vars()
    template_tokens = _env_template_tokens()

    missing = sorted(
        v
        for v in code_vars
        if v not in template_tokens
        and v not in _CODE_SIDE_ALLOWLIST
        and not v.startswith(_PATTERN_DOCUMENTED_PREFIXES)
    )
    assert not missing, (
        f'OP_* env vars read by {"/, ".join(_SCANNED_ROOTS)}/ but not documented '
        f'anywhere in env.template: {missing}'
    )


def test_every_op_env_var_in_env_template_is_read_by_code() -> None:
    code_vars = _direct_env_vars_read_by_code() | _from_env_derived_vars()
    template_tokens = _env_template_tokens()

    orphaned = sorted(
        t
        for t in template_tokens
        if t not in code_vars
        and t not in _TEMPLATE_SIDE_ALLOWLIST
        and not t.startswith(_PATTERN_DOCUMENTED_PREFIXES)
    )
    assert not orphaned, (
        f'env.template documents OP_* var(s) that nothing in {"/, ".join(_SCANNED_ROOTS)}/ '
        f'actually reads (stale/renamed?): {orphaned}'
    )


def test_bakeoff_profile_env_examples_name_real_fields() -> None:
    """``OP_BAKEOFF_PROFILE_<FIELD>`` examples in env.template must be real fields.

    The prefix is pattern-documented, so the orphan check above skips it; this
    keeps its examples honest when a BakeoffProfile field is removed.
    """
    fields_ = {f'OP_BAKEOFF_PROFILE_{f.name.upper()}' for f in fields(BakeoffProfile)}
    examples = {
        t
        for t in _env_template_tokens()
        if t.startswith('OP_BAKEOFF_PROFILE_') and t != 'OP_BAKEOFF_PROFILE_'
    }
    assert examples, 'env.template should show at least one OP_BAKEOFF_PROFILE_<FIELD> example'
    assert examples <= fields_, sorted(examples - fields_)
    assert 'OP_BAKEOFF_PROFILE_CLASS_FILTER' in examples
