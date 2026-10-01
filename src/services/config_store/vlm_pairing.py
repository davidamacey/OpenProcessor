"""Pack <-> region profile <-> VLM endpoint compatibility (W9.5).

:func:`vlm_pairing_issues` runs whenever one side of the triple changes:
a pack activation (against the active VLM and profile), a profile
activation, a VLM activation, a per-run ``?vlm=`` / ``?prompt_pack=``
resolution and the draft tests. Image *counts* are safe by construction (the
labeler chunks every batch to ``max_images_per_call`` and the numbered
overlay puts all N boxes on one image per item), so these checks are about
context capacity and verified behaviour.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING, Any, Literal

from src.services.config_store.vlm_validation import issue
from src.services.labeling.vlm_catalog import catalog_entry


if TYPE_CHECKING:
    from src.config.detection_profile import DetectionProfile
    from src.routers.curation._config_common_models import ValidationIssue
    from src.services.labeling.vlm_endpoints import VlmEndpoint
    from src.services.labeling.vlm_prompts import PromptPack

PairingMode = Literal['validate', 'activate', 'run']

#: Rendered prompt characters per token (a deliberate over-estimate of
#: tokens: the check is a warning-grade estimate, never a measurement).
CHARS_PER_TOKEN = 3.5

#: ``(call, system field, user field, max_tokens the labeler sends, which
#: image cap applies)``. ``max_tokens`` mirrors the constants in
#: ``vlm_labeler.py``; ``tests/curation/test_vlm_pairing.py`` pins them.
_CALLS: tuple[tuple[str, str, str, int, str], ...] = (
    ('class_batch', 'class_system', 'class_user_template', 512, 'max'),
    ('open_class_batch', 'open_class_system', 'open_class_user_template', 768, 'open'),
    ('region_verify_batch', 'region_batch_system', 'region_batch_user', 2048, 'max'),
    ('combined_batch', 'combined_batch_system', 'combined_batch_rules', 6144, 'max'),
    ('region_visible_batch', 'region_visible_system', 'region_visible_user', 1024, 'max'),
)

_VLM_TEXT_READERS = frozenset({'vlm', 'vlm_then_ocr', 'both'})


class _Blank(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ''


def _render(template: str, values: dict[str, str]) -> str:
    """``template`` with its placeholders filled; a malformed template (the
    pack validator's job to reject) falls back to its raw text."""
    try:
        return string.Formatter().vformat(template, (), _Blank(values))
    except (ValueError, IndexError, KeyError):
        return template


def _prompt_chars(pack: PromptPack, system: str, user: str, class_names: list[str]) -> int:
    names_csv = ', '.join(class_names)
    block = '\n'.join(
        f'- {n}: {pack.class_descriptions.get(n, "")}'.rstrip(': ') for n in class_names
    )
    values = {'class_names_csv': names_csv, 'class_block': block, 'region_block': ''}
    return len(getattr(pack, system)) + len(_render(getattr(pack, user), values))


def context_breakdown(
    endpoint: VlmEndpoint, pack: PromptPack, class_names: list[str]
) -> list[dict[str, Any]] | None:
    """Per-call token estimates, or ``None`` when the probe has not yet
    learned ``image_tokens`` and ``max_model_len`` (then ``vlm_not_probed``
    is the signal, not a guess)."""
    probe = endpoint.last_probe
    if probe is None or probe.image_tokens is None or probe.max_model_len is None:
        return None
    rows: list[dict[str, Any]] = []
    for call, system, user, max_tokens, cap_kind in _CALLS:
        images = (
            endpoint.body.effective_open_images
            if cap_kind == 'open'
            else endpoint.body.max_images_per_call
        )
        prompt_tokens = int(_prompt_chars(pack, system, user, class_names) / CHARS_PER_TOKEN)
        fixed = prompt_tokens + max_tokens
        estimate = fixed + images * probe.image_tokens
        room = probe.max_model_len - fixed
        fit = room // probe.image_tokens if probe.image_tokens > 0 else images
        rows.append(
            {
                'call': call,
                'estimate': estimate,
                'max_model_len': probe.max_model_len,
                'images_per_call': images,
                'breakdown': {
                    'prompt_tokens': prompt_tokens,
                    'image_tokens_each': probe.image_tokens,
                    'max_tokens': max_tokens,
                },
                'suggest_max_images': int(fit) if fit >= 1 else None,
            }
        )
    return rows


def vlm_pairing_issues(
    endpoint: VlmEndpoint,
    pack: PromptPack | None,
    profile: DetectionProfile | None,
    *,
    mode: PairingMode,
    class_names: list[str] | None = None,
) -> list[ValidationIssue]:
    """Every pairing problem between ``endpoint`` and ``pack``/``profile``.

    ``mode`` sets the severities: ``validate`` (warnings), ``activate``
    (the context estimate is an error ``force`` may bypass) and ``run`` (a
    per-run selection carries no ``force``, so the estimate stays a
    warning and only the certain failure blocks). ``vlm_max_images_exceeds_server``
    is an error in every mode and never bypassable: every multi-image call
    would fail.
    """
    out: list[ValidationIssue] = []
    probe = endpoint.last_probe
    body = endpoint.body

    if probe is not None and any(
        i.get('code') == 'vlm_max_images_exceeds_server' for i in probe.issues
    ):
        out.append(
            issue(
                'vlm_max_images_exceeds_server',
                'error',
                'The endpoint rejects calls with its own image cap; lower max_images_per_call.',
                field='max_images_per_call',
                detail={'max_images_per_call': body.max_images_per_call},
            )
        )

    if pack is not None:
        out.extend(
            issue(
                'vlm_context_too_small',
                'error' if mode == 'activate' else 'warning',
                f'The {row["call"]} call needs about {row["estimate"]} tokens; the '
                f'model serves {row["max_model_len"]}.',
                detail={
                    'call': row['call'],
                    'estimate': row['estimate'],
                    'max_model_len': row['max_model_len'],
                    'breakdown': row['breakdown'],
                    'suggest_max_images': row['suggest_max_images'],
                },
            )
            for row in context_breakdown(endpoint, pack, class_names or []) or []
            if row['estimate'] > row['max_model_len']
        )

    entry = catalog_entry(body.catalog_id)
    if profile is not None:
        if profile.max_regions_per_item > 1 and not (entry and entry.multi_box_verified):
            out.append(
                issue(
                    'vlm_multi_box_unverified',
                    'warning',
                    'Multi-box regions have not been verified on this model.',
                    detail={
                        'max_regions_per_item': profile.max_regions_per_item,
                        'catalog_id': body.catalog_id,
                    },
                )
            )
        if profile.text_reader in _VLM_TEXT_READERS and not (entry and entry.text_reading_verified):
            out.append(
                issue(
                    'vlm_reads_text_unverified',
                    'warning',
                    'Reading region text through the VLM has not been verified on this model.',
                    detail={'text_reader': profile.text_reader, 'catalog_id': body.catalog_id},
                )
            )

    if not endpoint.json_mode_on and probe is not None and probe.reasoning_channel:
        out.append(
            issue(
                'vlm_json_mode_off',
                'warning',
                'JSON mode is off and the model answers on a reasoning channel; empty replies '
                'become more likely.',
            )
        )
    if (
        body.open_images_per_call is not None
        and body.open_images_per_call > body.max_images_per_call
    ):
        out.append(
            issue(
                'vlm_open_images_clamped',
                'warning',
                'open_images_per_call is larger than max_images_per_call and is clamped to it.',
                field='open_images_per_call',
                detail={
                    'open_images_per_call': body.open_images_per_call,
                    'max_images_per_call': body.max_images_per_call,
                },
            )
        )
    return out


__all__ = ['CHARS_PER_TOKEN', 'PairingMode', 'context_breakdown', 'vlm_pairing_issues']
