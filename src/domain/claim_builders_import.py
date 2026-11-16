"""Dataset import and combine as claims (15a section 3.5).

A validated import label is a ``class_set`` whose actor is the import: it ranks below every live
human claim (B2), and where a person decides the import reports ``locked_conflict`` from the
fold's conflicts. An unvalidated (``label_trust: suggestion``) label is a machine opinion, not a
claim, and has no builder here. Combine is the same with a combine actor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain.claim_builders import ItemView, human_draft
from src.domain.claim_kinds import ActorKind, ClaimKind, SubjectKind
from src.domain.claim_payloads import (
    BoxCreated,
    ClassProfile,
    ClassSet,
    CombineConflict,
    ItemDrawn,
    Origin,
)
from src.domain.claims import ClaimDraft, Subject


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.domain.claim_payloads import Box4


def _import_class_set(
    view: ItemView,
    *,
    class_name: str,
    class_id_hint: int | None,
    via: str,
    detector_version: str,
    actor_kind: ActorKind,
    actor: str,
    op_id: str,
) -> ClaimDraft:
    payload = ClassSet(
        class_name=class_name,
        class_id_hint=class_id_hint,
        label_source='import',
        via=via,
        labeler='import',
        profile=ClassProfile(detector='import', detector_version=detector_version),
    )
    return human_draft(
        ClaimKind.CLASS_SET, view, payload, op_id=op_id, actor_kind=actor_kind, actor=actor
    )


def import_label(
    view: ItemView, *, import_id: str, class_name: str, class_id_hint: int | None, op_id: str
) -> list[ClaimDraft]:
    """A validated label of a dataset import on an item it matched."""
    return [
        _import_class_set(
            view,
            class_name=class_name,
            class_id_hint=class_id_hint,
            via='import',
            detector_version=import_id,
            actor_kind=ActorKind.IMPORT,
            actor=f'import:{import_id}',
            op_id=op_id,
        )
    ]


def import_item(
    view: ItemView,
    *,
    import_id: str,
    class_name: str,
    class_id_hint: int | None,
    standalone_region: bool = False,
    origin: Origin | None = None,
    op_id: str,
) -> list[ClaimDraft]:
    """An item the import creates because no detection matched its box: the item and its label."""
    actor = f'import:{import_id}'
    payload = ItemDrawn(bbox=view.bbox, standalone_region=standalone_region, origin=origin)
    drawn = human_draft(
        ClaimKind.ITEM_DRAWN, view, payload, op_id=op_id, actor_kind=ActorKind.IMPORT, actor=actor
    )
    return [
        drawn,
        *import_label(
            view,
            import_id=import_id,
            class_name=class_name,
            class_id_hint=class_id_hint,
            op_id=op_id,
        ),
    ]


def import_box(
    view: ItemView,
    *,
    import_id: str,
    box_id: str,
    bbox: Box4,
    state: str,
    region_seq: int,
    op_id: str,
) -> list[ClaimDraft]:
    """A validated box an import brings (not a ``proposed`` one: that is a machine result)."""
    subject = Subject(
        SubjectKind.BOX,
        photo_key=view.photo_key,
        crop_id=view.crop_id,
        box_id=box_id,
        bbox=bbox,
        frame=view.frame,
    )
    payload = BoxCreated(item_bbox=view.bbox, bbox=bbox, state=state, region_seq=region_seq)
    return [
        ClaimDraft(
            ActorKind.IMPORT,
            f'import:{import_id}',
            op_id,
            ClaimKind.BOX_CREATED,
            subject,
            payload,
        )
    ]


def combine_label(
    view: ItemView, *, job_id: str, class_name: str, class_id_hint: int | None, op_id: str
) -> list[ClaimDraft]:
    """A class a project combine copies onto an item (ranked like an import's)."""
    return [
        _import_class_set(
            view,
            class_name=class_name,
            class_id_hint=class_id_hint,
            via='combine',
            detector_version=job_id,
            actor_kind=ActorKind.COMBINE,
            actor=f'combine:{job_id}',
            op_id=op_id,
        )
    ]


def combine_conflict(
    view: ItemView,
    *,
    job_id: str,
    origins: Sequence[str],
    merged_origins: Sequence[str],
    op_id: str,
) -> list[ClaimDraft]:
    """The combine found two origins of one item disagreeing and left it for a person."""
    payload = CombineConflict(origins=tuple(origins), merged_origins=tuple(merged_origins))
    return [
        human_draft(
            ClaimKind.COMBINE_CONFLICT,
            view,
            payload,
            op_id=op_id,
            actor_kind=ActorKind.COMBINE,
            actor=f'combine:{job_id}',
        )
    ]
