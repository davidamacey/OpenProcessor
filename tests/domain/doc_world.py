"""Today's world: one item document, changed by today's pure mergers and lock predicates.

Every human act calls the function the route calls; every machine write lands only where
today's guard lets it (``class_write_locked``, ``is_locked_item``, ``human_owned``, the
adoption and acceptance mergers). Nothing here knows the claim model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.occ_locks import is_locked_item
from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.domain.region_boxes import RegionBox, RegionBoxWriteError, is_human_owned, read_boxes
from src.services.curation.class_label import ItemLabel, class_label_update
from src.services.curation.class_sources import unmatched_class_clear
from src.services.curation.class_undo import class_undo_update
from src.services.curation.class_write_guard import class_write_locked
from src.services.curation.dataset_import.item_labels import human_owned
from src.services.curation.exclusion import exclusion_update, unexclusion_update
from src.services.curation.item_geometry import (
    INSIDE_TOLERANCE,
    RegionsOutsideItemError,
    geometry_edit_merger,
)
from src.services.curation.label_ops.accept_merge import accept_merger
from src.services.curation.label_ops.adopt_merge import adoption_merger
from src.services.curation.label_undo import (
    NoProbeDisagreementError,
    NoVlmSuggestionError,
    discard_merger,
    probe_dismiss_merger,
    vlm_dismiss_merger,
)
from src.services.curation.region_box_edits import apply_put_boxes
from src.services.curation.region_box_pass import box_pass_update
from src.services.curation.region_writes import human_box_write, human_status_box_write
from src.services.curation.reprocess_locks import region_set_locked
from src.services.curation.wire import serialize_item
from tests.domain.fold_support import BOX, CROP, PRIMARY


if TYPE_CHECKING:
    from tests.domain.world_acts import Act


F = get_region_fields()
NOW = '2026-01-01T00:00:00+00:00'
IDS = {'car': 1, 'truck': 2, 'bus': 3, 'van': 4}
VLM_SCORE = 0.92  # VLM_TEXT_CONFIDENCE['high'], the category the vlm writes below carry


class RefusedError(Exception):
    """The route refuses the act (422); the fold world must refuse it too."""


def _label(
    class_name: str, *, source: str = 'human', op_id: str | None = None, **kw: Any
) -> ItemLabel:
    if source == 'human':
        return ItemLabel.human(
            class_id=IDS[class_name],
            class_name=class_name,
            label_source='human',
            writer='human:label_crop',
            op_id=op_id,
        )
    return ItemLabel.imported(
        import_id='i1', class_id=IDS[class_name], class_name=class_name, now=NOW, **kw
    )


class DocWorld:
    def __init__(self, classifier_class: str | None) -> None:
        self.doc: dict[str, Any] = {
            'crop_id': CROP,
            'image_id': 'img',
            'bbox_norm': list(BOX),
            'crop_area_norm': 0.16,
            'class_id': None,
            'class_name': None,
            'class_source': f'{PRIMARY}_proposal',
            'class_validated': False,
            'detector_class_name': 'car',
            'detector_confidence': 0.9,
            'confidence': 0.9,
            'probe_pred_class': 'x',
            'probe_disagreement': True,
            'probe_model_version': 'v1',
            F.box_seq: 0,
        }
        if classifier_class is not None:
            self.doc.update(
                class_id=IDS[classifier_class],
                class_name=classifier_class,
                class_source=f'{PRIMARY}_model',
            )
        self.op = 0
        self.dropped_machine_sources: list[str] = []
        self.landed_machine_sources: list[str] = []
        self.refused_imports = 0
        self.last_bbox_refused_by_lock = False
        self.bbox_refused_count = 0
        self.redetect_log: list[tuple[bool, bool]] = []
        self.last_machine_boxes: list[RegionBox] = []

    def _op(self) -> str:
        self.op += 1
        return f'op{self.op}'

    def _merge(self, update: dict[str, Any]) -> None:
        self.doc.update(update)

    # ----- the view the wire serializer reads -------------------------------------------
    def wire(self) -> dict[str, Any]:
        return serialize_item(self.doc, CROP, state_fields={}, api_prefix='/p', label='Region')

    def boxes(self) -> list[RegionBox]:
        return read_boxes(self.doc)

    # ----- machine class writers: today's guard decides whether the write lands ----------
    def _machine_class(self, source: str, guard_open: bool, fields: dict[str, Any]) -> None:
        if not guard_open:
            self.dropped_machine_sources.append(source)
            return
        self.landed_machine_sources.append(source)
        self._merge({'class_validated': False, **fields})

    def _class_fields(self, cls: str, source: str) -> dict[str, Any]:
        return {'class_id': IDS[cls], 'class_name': cls, 'class_source': source}

    def apply(self, act: Act) -> None:
        getattr(self, f'act_{act[0]}')(*act[1:])

    # ----- human acts --------------------------------------------------------------------
    def act_label(self, cls: str) -> None:
        self._merge(class_label_update(self.doc, _label(cls, op_id=self._op())))

    def act_accept(self) -> None:
        merge = accept_merger(
            classes={c: (i, c) for c, i in IDS.items()},
            min_confidence=0.5,
            sources={'detector', 'vlm'},
            op_id=self._op(),
        )
        self._merge(merge(self.doc))

    def act_unlabel(self) -> None:
        self._merge(class_undo_update(self.doc))

    def act_discard(self, clear: bool, dismiss: bool) -> None:
        if not (clear or dismiss):
            return
        self._merge(discard_merger(clear_class=clear, dismiss_from_review=dismiss)(self.doc))

    def act_exclude(self, reason: str) -> None:
        self._merge(exclusion_update(self.doc, reason=reason, now=NOW))

    def act_restore(self) -> None:
        self._merge(unexclusion_update(self.doc, now=NOW))

    def act_review_dismiss(self) -> None:
        self._merge({'review_dismissed_at': NOW, 'review_dismissed_by': 'human'})

    def act_review_undismiss(self) -> None:
        self._merge({'review_dismissed_at': None, 'review_dismissed_by': None})

    def act_vlm_dismiss(self) -> None:
        try:
            self._merge(vlm_dismiss_merger(self.doc))
        except NoVlmSuggestionError as exc:
            raise RefusedError from exc

    def act_probe_dismiss(self) -> None:
        try:
            self._merge(probe_dismiss_merger(self.doc))
        except NoProbeDisagreementError as exc:
            raise RefusedError from exc

    def act_freeze(self) -> None:
        self._merge({'test_holdout': True})

    def act_import_label(self, cls: str, trust: str) -> None:
        if human_owned(self.doc):
            self.refused_imports += 1  # locked_conflict: nothing is written
            return
        label = _label(cls, source='import', trust=trust)
        fields = class_label_update(self.doc, label)
        fields['import_ids'] = [*self.doc.get('import_ids', []), 'i1']
        self._merge(fields)

    # ----- region acts -------------------------------------------------------------------
    def _box_write(self, boxes: list[RegionBox]) -> None:
        self._merge(
            human_box_write(
                self.doc,
                boxes,
                label_source='human',
                now=NOW,
                human_name='human',
                human_version='1',
            )
        )

    def _shown_ids(self) -> list[str]:
        return [b.box_id for b in self.boxes()]

    def _put(self, elements: list[dict[str, Any]]) -> None:
        try:
            boxes = apply_put_boxes(self.doc, elements, frame='source', now=NOW)
        except RegionBoxWriteError as exc:
            raise RefusedError from exc
        self._box_write(boxes)

    def _keep(self) -> list[dict[str, Any]]:
        return [{'box_id': b} for b in self._shown_ids()]

    def act_box_new(self, bbox: tuple[float, ...]) -> None:
        self._put([*self._keep(), {'box_id': None, 'bbox_norm': list(bbox)}])

    def _element(self, idx: int, **patch: Any) -> list[dict[str, Any]]:
        els = self._keep()
        if idx >= len(els):
            raise RefusedError
        els[idx] = {**els[idx], **patch}
        return els

    def act_box_state(self, idx: int, state: str) -> None:
        self._put(self._element(idx, state=state))

    def act_box_text(self, idx: int, text: str) -> None:
        self._put(self._element(idx, text=text))

    def act_box_move(self, idx: int, bbox: tuple[float, ...]) -> None:
        self._put(self._element(idx, bbox_norm=list(bbox)))

    def act_box_delete(self, idx: int) -> None:
        els = self._keep()
        if idx >= len(els):
            raise RefusedError
        del els[idx]
        self._put(els)

    def act_box_none(self) -> None:
        self._put([])

    def act_box_batch_state(self, idxs: list[int], state: str) -> None:
        ids = self._shown_ids()
        if any(i >= len(ids) for i in idxs):
            raise RefusedError
        from src.services.curation.region_box_edits import with_state

        self._box_write(
            [with_state(b, state) if i in idxs else b for i, b in enumerate(self.boxes())]
        )

    def act_region_status(self, status: str) -> None:
        try:
            update = human_status_box_write(status, self.doc, label_source='human')
        except RegionBoxWriteError as exc:
            raise RefusedError from exc
        self._merge({'updated_at': NOW, F.label_source: 'human', F.validated: True, **update})

    def act_geometry(self, bbox: tuple[float, ...]) -> None:
        slack = INSIDE_TOLERANCE
        outside = [
            b.box_id
            for b in self.boxes()
            if not (
                b.bbox_norm[0] >= bbox[0] - slack
                and b.bbox_norm[1] >= bbox[1] - slack
                and b.bbox_norm[2] <= bbox[2] + slack
                and b.bbox_norm[3] <= bbox[3] + slack
            )
        ]
        merger = geometry_edit_merger(
            bbox,
            expected_revision=int(self.doc.get('geometry_revision') or 0),
            drop_region_box_ids=outside,
            resets={},
        )
        try:
            self._merge(merger(self.doc))
        except RegionsOutsideItemError as exc:  # pragma: no cover - every outside id is listed
            raise RefusedError from exc

    # ----- machine acts ------------------------------------------------------------------
    def act_redetect(self, shift: float) -> None:
        whole_set = bool(self.doc.get(F.validated)) and self.doc.get(F.verifier) not in (
            'human',
            'import',
        )
        self.redetect_log.append((not is_locked_item(self.doc, F), whole_set))
        self.last_bbox_refused_by_lock = False
        if is_locked_item(self.doc, F):
            self.last_bbox_refused_by_lock = True
            self.bbox_refused_count += 1
            return
        x1, y1, x2, y2 = BOX
        box = [x1 + shift, y1 + shift, x2 + shift, y2 + shift]
        self._merge({'bbox_norm': box, 'detector_bbox_norm': box, 'crop_area_norm': 0.16})

    def act_vlm(self, outcome: str, cls: str) -> None:
        open_ = not class_write_locked(self.doc)
        if outcome == 'empty':
            return  # the attempt is recorded, never a class (M17)
        if outcome == 'matched':
            fields = {**self._class_fields(cls, 'vlm'), 'vlm_confidence': 'high'}
        elif outcome == 'unmatched':
            fields = {
                'class_source': 'vlm_unmatched',
                **(unmatched_class_clear(self.doc) if open_ else {}),
            }
        else:
            fields = {
                'class_source': 'vlm_new_class_pending',
                'vlm_proposed_class': cls,
                'needs_new_class': True,
            }
        self._machine_class('vlm', open_, fields)

    def act_propagate(self, cls: str) -> None:
        self._machine_class(
            'assist_propagated',
            not class_write_locked(self.doc),
            self._class_fields(cls, 'assist_propagated'),
        )

    def act_open_vocab_target(self, cls: str) -> None:
        self._machine_class(
            'open_vocab_target',
            not is_locked_item(self.doc, F),
            self._class_fields(cls, 'open_vocab_target'),
        )

    def act_classifier(self, cls: str, low: bool) -> None:
        source = f'{PRIMARY}_low_conf' if low else f'{PRIMARY}_model'
        self._machine_class(
            'classifier', not is_locked_item(self.doc, F), self._class_fields(cls, source)
        )

    def act_adopt(self) -> None:
        update = adoption_merger({'car': (1, 'car')}, self._op())(self.doc)
        if update:
            self.landed_machine_sources.append('adopted_detector')
            self._merge(update)
        else:
            self.dropped_machine_sources.append('adopted_detector')

    def act_region_write(self, boxes: list[tuple[float, ...]], state: str) -> None:
        # Today's gate for a region re-run is the validated flag (reprocess_locks.region_set_locked):
        # a set a person closed has no machine output to regenerate; boxes a person owns are
        # kept by the merge (box_pass_update, merge_machine_boxes).
        if region_set_locked(self.doc, F):
            self.dropped_machine_sources.append('region')
            return
        pending = [
            RegionBox(
                box_id=f'\x00pending:{i}',
                bbox_norm=b,  # type: ignore[arg-type]
                state=state,
                score=0.8,
                detector='seg',
                detector_version='1',
                source='segmenter',
            )
            for i, b in enumerate(boxes)
        ]
        result = box_pass_update(
            self.doc,
            pending,
            reverify=False,
            merge_machine_boxes=True,
            baseline=None,
            status=None,
            empty_status=RegionStatus.NO_REGION_BOX,
        )
        kept = [b for b in read_boxes(self.doc) if is_human_owned(b)]
        self.last_machine_boxes = result.finalized[len(kept) :]
        self.landed_machine_sources.append('region')
        self._merge(result.update)
