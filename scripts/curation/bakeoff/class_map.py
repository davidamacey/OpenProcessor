"""Model class id -> eval class id mapping for one (model, dataset) pair.

The one owner of the class-mapping algorithm (plan D5). Stdlib only: the
API imports it at enqueue time (when the run manifest or an explicit map
already says how the model's classes relate to the eval export), and the
harness imports it inside the evaluator (when only the loaded model knows
its class names).

Id spaces:

* **eval** -- the eval dataset's dense ids ``0..nc-1`` (its label files,
  ``data.yaml`` ``names``).
* **model** -- the model's own class ids ``0..k-1``.
* **registry** -- class-registry ids; an export's ``class_registry.json``
  ``export_id_map`` translates ``{registry_id: export_id}``.

Resolution (first that applies wins): an explicit name map, a run's
``class_remap`` (model -> registry -> eval), a full-export run (model =
training export dense id -> registry -> eval), and names (both sides
normalized by :func:`normalize_name`). :func:`finalize` turns the resulting
``{model_id: eval_id}`` dict into a :class:`ClassMapping` that also lists
what is left over on each side, so nothing is dropped silently.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


METHODS = ('explicit', 'run_class_remap', 'registry_ids', 'names', 'single_class_fallback')

_SEP_RE = re.compile(r'[\s\-]+')


def normalize_name(name: str) -> str:
    """``'  Mini Cooper '`` -> ``'mini_cooper'`` (casefold, strip, spaces/dashes -> ``_``)."""
    return _SEP_RE.sub('_', str(name).strip().casefold())


def _names_from_value(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        return {i: str(n) for i, n in enumerate(value)}
    if isinstance(value, dict):
        return {int(k): str(v) for k, v in value.items()}
    return {}


def _unquote(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def _parse_inline(text: str) -> dict[int, str]:
    for loader in (json.loads, ast.literal_eval):
        try:
            return _names_from_value(loader(text))
        except (ValueError, SyntaxError, TypeError):
            continue
    if text.startswith('[') and text.endswith(']'):  # unquoted flow list: [a, b]
        return {i: _unquote(p) for i, p in enumerate(text[1:-1].split(',')) if p.strip()}
    return {}


def read_names(data_yaml_path: str | Path) -> dict[int, str]:
    """``{class_id: name}`` from a YOLO ``data.yaml`` without a YAML dependency.

    Handles the forms the exporters and converters write: an inline list or
    dict after ``names:`` (``names: ["a", "b"]``) and a block of
    ``  <id>: <name>`` or ``  - <name>`` lines below ``names:``. Missing
    file or no ``names`` key -> ``{}``.
    """
    path = Path(data_yaml_path)
    if not path.is_file():
        return {}
    lines = path.read_text(encoding='utf-8').splitlines()
    for i, line in enumerate(lines):
        if not line.startswith('names:'):
            continue
        rest = line.split(':', 1)[1].strip()
        if rest:
            return _parse_inline(rest)
        out: dict[int, str] = {}
        for item in lines[i + 1 :]:
            if not item.strip() or item.lstrip().startswith('#'):
                continue
            if not item[0].isspace():
                break
            body = item.strip()
            if body.startswith('- '):
                out[len(out)] = _unquote(body[2:])
            elif ':' in body:
                key, val = body.split(':', 1)
                try:
                    out[int(key.strip())] = _unquote(val)
                except ValueError:
                    continue
        return out
    return {}


def read_export_id_map(root: str | Path) -> dict[int, int]:
    """``{registry_id: export_id}`` from ``<root>/class_registry.json`` (``{}`` if absent)."""
    path = Path(root) / 'class_registry.json'
    try:
        raw = json.loads(path.read_text(encoding='utf-8')).get('export_id_map') or {}
        return {int(k): int(v) for k, v in raw.items()}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def invert_export_id_map(export_id_map: Mapping[int, int]) -> dict[int, int]:
    """``{registry_id: export_id}`` -> ``{export_id: registry_id}``.

    For a run trained on a full export, the model's class ids ARE its
    training export's dense ids, so this is the model -> registry map.
    """
    return {int(e): int(r) for r, e in export_id_map.items()}


def resolve_explicit(
    explicit: Mapping[str, str], eval_names: Mapping[int, str]
) -> tuple[dict[int, int], list[str]]:
    """``{"<model_class_id>": "<eval class name>"}`` -> ``{model_id: eval_id}``.

    Names (not ids) because a baseline's map is dataset-independent. An
    unknown name leaves that model class unmapped, with a warning.
    """
    by_name = {normalize_name(n): i for i, n in eval_names.items()}
    out: dict[int, int] = {}
    warnings: list[str] = []
    for k, name in explicit.items():
        eid = by_name.get(normalize_name(name))
        if eid is None:
            warnings.append(f'class_map name {name!r} (model class {k}) is not an eval class')
            continue
        out[int(k)] = eid
    return out, warnings


def resolve_by_registry(
    model_to_registry: Mapping[int, int],
    eval_export_id_map: Mapping[int, int],
    *,
    model_names: Mapping[int, str] | None = None,
    eval_names: Mapping[int, str] | None = None,
) -> tuple[dict[int, int], list[str]]:
    """model -> registry id -> eval export id.

    A registry id the eval export does not carry leaves that model class
    unmapped. When both sides have names they are cross-checked: a mismatch
    (a registry rename between exports) keeps the id mapping and warns.
    """
    out: dict[int, int] = {}
    warnings: list[str] = []
    for mid, rid in sorted(model_to_registry.items()):
        eid = eval_export_id_map.get(int(rid))
        if eid is None:
            continue
        out[int(mid)] = int(eid)
        m_name = (model_names or {}).get(int(mid))
        e_name = (eval_names or {}).get(int(eid))
        if (
            m_name is not None
            and e_name is not None
            and normalize_name(m_name) != normalize_name(e_name)
        ):
            warnings.append(
                f'model class {mid} {m_name!r} maps to eval class {eid} {e_name!r} by registry id'
            )
    return out, warnings


def resolve_by_names(
    model_names: Mapping[int, str], eval_names: Mapping[int, str]
) -> dict[int, int]:
    """Map model classes to eval classes whose normalized names are equal."""
    by_name = {normalize_name(n): i for i, n in eval_names.items()}
    out: dict[int, int] = {}
    for mid, name in sorted(model_names.items()):
        eid = by_name.get(normalize_name(name))
        if eid is not None:
            out[int(mid)] = eid
    return out


@dataclass
class ClassMapping:
    """A resolved (model, dataset) class mapping (plan section 3.3 shape).

    ``model_to_eval`` is ``None`` only for an unresolved mapping the
    evaluator still has to name-match (the API's placeholder).
    """

    method: str
    model_to_eval: dict[int, int] | None
    unmapped_model_classes: list[dict[str, Any]] = field(default_factory=list)
    not_covered_eval_classes: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'method': self.method,
            'model_to_eval': (
                None
                if self.model_to_eval is None
                else {str(k): v for k, v in sorted(self.model_to_eval.items())}
            ),
            'unmapped_model_classes': list(self.unmapped_model_classes),
            'not_covered_eval_classes': list(self.not_covered_eval_classes),
            'warnings': list(self.warnings),
        }


def finalize(
    method: str,
    model_to_eval: Mapping[int, int],
    *,
    model_names: Mapping[int, str] | None,
    eval_names: Mapping[int, str],
    scored_class_ids: Iterable[int],
    warnings: Iterable[str] = (),
) -> ClassMapping:
    """Build the :class:`ClassMapping`: leftovers on both sides + merge warnings.

    ``unmapped_model_classes`` lists model classes (from ``model_names``)
    with no eval counterpart; ``not_covered_eval_classes`` lists scored eval
    classes no model class maps to. Several model classes mapping to one
    eval class is allowed (their predictions merge) but warned about.
    """
    if method not in METHODS:
        raise ValueError(f'unknown class-mapping method {method!r}')
    m2e = {int(k): int(v) for k, v in model_to_eval.items()}
    notes = list(warnings)
    by_eval: dict[int, list[int]] = {}
    for mid, eid in sorted(m2e.items()):
        by_eval.setdefault(eid, []).append(mid)
    for eid, mids in sorted(by_eval.items()):
        if len(mids) > 1:
            notes.append(
                f'model classes {", ".join(map(str, mids))} all map to eval class {eid}; '
                'their predictions are merged'
            )
    unmapped = [
        {'model_class_id': mid, 'name': name}
        for mid, name in sorted((model_names or {}).items())
        if mid not in m2e
    ]
    covered = set(m2e.values())
    not_covered = [
        {'eval_class_id': eid, 'name': eval_names.get(eid, str(eid))}
        for eid in sorted(set(scored_class_ids))
        if eid not in covered
    ]
    return ClassMapping(method, m2e, unmapped, not_covered, notes)


def resolve_for_loaded_model(
    model_names: Mapping[int, str] | None,
    eval_names: Mapping[int, str],
    scored_class_ids: Iterable[int],
) -> ClassMapping:
    """Harness-side resolution when the job carries no map (plan rules 4-5).

    Name-match the loaded model's own class names. A model without names
    (e.g. a Triton engine) is single-class only: its class 0 maps to the
    dataset's one scored class when exactly one is scored; otherwise
    nothing maps and the row reports zero coverage.
    """
    scored = sorted(set(scored_class_ids))
    if model_names:
        return finalize(
            'names',
            resolve_by_names(model_names, eval_names),
            model_names=model_names,
            eval_names=eval_names,
            scored_class_ids=scored,
        )
    if len(scored) == 1:
        return finalize(
            'single_class_fallback',
            {0: scored[0]},
            model_names=None,
            eval_names=eval_names,
            scored_class_ids=scored,
        )
    return finalize(
        'names',
        {},
        model_names=None,
        eval_names=eval_names,
        scored_class_ids=scored,
        warnings=[
            'model exposes no class names and the eval split scores '
            f'{len(scored)} classes; give it an explicit class_map'
        ],
    )


def from_given(
    given: Mapping[str, Any],
    *,
    model_names: Mapping[int, str] | None,
    eval_names: Mapping[int, str],
    scored_class_ids: Iterable[int],
) -> ClassMapping:
    """A map resolved before the evaluator ran (job spec ``class_map_by_dataset``).

    Accepts the plain ``{"<model_id>": <eval_id>}`` form (recorded as
    method ``explicit``) or a full :meth:`ClassMapping.to_dict` dict, whose
    ``method`` and ``warnings`` are kept.
    """
    if 'model_to_eval' in given:
        method = str(given.get('method') or 'explicit')
        raw = given.get('model_to_eval') or {}
        warnings = [str(w) for w in given.get('warnings') or []]
    else:
        method, raw, warnings = 'explicit', given, []
    return finalize(
        method,
        {int(k): int(v) for k, v in raw.items()},
        model_names=model_names,
        eval_names=eval_names,
        scored_class_ids=scored_class_ids,
        warnings=warnings,
    )


__all__ = [
    'METHODS',
    'ClassMapping',
    'finalize',
    'from_given',
    'invert_export_id_map',
    'normalize_name',
    'read_export_id_map',
    'read_names',
    'resolve_by_names',
    'resolve_by_registry',
    'resolve_explicit',
    'resolve_for_loaded_model',
]
