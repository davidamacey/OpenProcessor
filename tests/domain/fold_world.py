"""The redesigned world: machine results as rows, a person's acts as appended claims, and the
state of the item as ``fold`` of the two. The same act tuples as :mod:`tests.domain.doc_world`."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from src.config.region_state import RegionStatus
from src.domain import (
    claim_builders as cb,
    claim_builders_import as cbi,
    claim_builders_region as cbr,
)
from src.domain.fold import fold_item
from src.domain.fold_class import freeze_machine_class
from src.domain.item_geometry import ItemBoxError
from src.domain.region_boxes import RegionBox, RegionBoxWriteError, derive_status
from src.domain.region_edit_rules import BoxEdit
from src.domain.results import ClassOpinion, ItemInputs, ItemRow, MachineBox, RegionResult
from tests.domain.doc_world import IDS, VLM_SCORE, RefusedError
from tests.domain.fold_support import (
    BOX,
    CROP,
    FRAME,
    PHOTO,
    PRIMARY,
    REGISTRY,
    SETTINGS,
    Log,
    detection,
    opinion,
    t,
)


if TYPE_CHECKING:
    from src.domain.claims import ClaimDraft
    from src.domain.fold_state import ItemFold
    from tests.domain.world_acts import Act


DETECTOR_SOURCES = frozenset({'detector', f'{PRIMARY}_model'})
VLM_SOURCES = frozenset({'vlm', 'vlm_reclassified'})


class FoldWorld:
    def __init__(self, classifier_class: str | None) -> None:
        self.log = Log(clock=0)
        self.detections = [detection(at=0, class_name='car')]
        self.opinions: dict[str, ClassOpinion] = {}
        self.boxes: list[MachineBox] = []
        self.region: RegionResult | None = None
        self.item = ItemRow(crop_id=CROP, photo_key=PHOTO, created_at=t(0))
        self.ops: dict[str, str] = {}
        self.op = 0
        if classifier_class is not None:
            self._opinion('classifier', classifier_class, outcome='model', confidence=0.9)

    # ----- plumbing ------------------------------------------------------------------
    def tick(self) -> float:
        self.log.clock += 1
        return self.log.clock

    def inputs(self) -> ItemInputs:
        return ItemInputs(
            item=self.item,
            frame=FRAME,
            detections=tuple(self.detections),
            opinions=tuple(self.opinions.values()),
            region=self.region,
            boxes=tuple(self.boxes),
            claims=tuple(self.log.claims),
        )

    def fold(self) -> ItemFold:
        result = fold_item(self.inputs(), REGISTRY, SETTINGS)
        assert result is not None
        return result

    def view(self) -> cb.ItemView:
        det = self.detections[-1]
        return cb.ItemView(
            crop_id=CROP,
            photo_key=PHOTO,
            frame=FRAME,
            bbox=self.fold().state.bbox,
            machine_bbox=det.bbox,
        )

    def _op(self, kind: str) -> str:
        self.op += 1
        op_id = f'op{self.op}'
        self.ops[op_id] = kind
        return op_id

    def _add(self, drafts: list[ClaimDraft]) -> None:
        self.log.add(drafts)

    def _opinion(self, source: str, cls: str | None, **kw: Any) -> None:
        new = opinion(source, cls, at=self.tick(), **kw)
        old = self.opinions.get(source)
        # An answer without a class never replaces a row that has one (M17).
        if (
            old is not None
            and old.class_name is not None
            and new.class_name is None
            and new.outcome == 'empty'
        ):
            return
        self.opinions[source] = new

    def shown(self) -> list[RegionBox]:
        return [
            RegionBox(
                box_id=b.box_id,
                bbox_norm=b.bbox,
                state=b.state,
                rejection_reason=b.rejection_reason,
                text=b.text,
                text_source=b.text_source,
                source=b.source,
            )
            for b in self.fold().boxes
        ]

    def apply(self, act: Act) -> None:
        getattr(self, f'act_{act[0]}')(*act[1:])

    # ----- human acts ----------------------------------------------------------------
    def act_label(self, cls: str) -> None:
        op = self._op('label')
        self._add(cb.label(self.view(), class_name=cls, class_id_hint=IDS[cls], op_id=op))

    def act_accept(self) -> None:
        state = self.fold().state
        if (
            not cb.accept_machine_eligible(
                state,
                min_confidence=0.5,
                detector_sources=DETECTOR_SOURCES,
                vlm_sources=VLM_SOURCES,
            )
            or state.class_name is None
        ):
            return
        op = self._op('accept_machine')
        self._add(
            cb.accept_machine(
                self.view(), class_name=state.class_name, class_id_hint=state.class_id, op_id=op
            )
        )

    def act_unlabel(self) -> None:
        op = self._op('unlabel')
        history = []
        from src.domain.claim_layers import split_claims

        live = split_claims(self.log.claims).live
        for op_id, kind in self.ops.items():
            ids = tuple(c.claim_id for c in live if c.op_id == op_id)
            if ids:
                history.append(cb.LiveOperation(op_id, kind, ids))
        self._add(cb.unlabel(self.view(), history, op_id=op))

    def act_discard(self, clear: bool, dismiss: bool) -> None:
        if not (clear or dismiss):
            return
        op = self._op('discard')
        self._add(
            cb.discard(self.view(), clear_class_too=clear, dismiss_from_review=dismiss, op_id=op)
        )

    def act_exclude(self, reason: str) -> None:
        self._add(cb.exclude(self.view(), reason=reason, op_id=self._op('exclude')))

    def act_restore(self) -> None:
        self._add(cb.restore(self.view(), op_id=self._op('restore')))

    def act_review_dismiss(self) -> None:
        self._add(cb.review_dismiss(self.view()))

    def act_review_undismiss(self) -> None:
        self._add(cb.review_undismiss(self.view()))

    def act_vlm_dismiss(self) -> None:
        state = self.fold().state
        if state.vlm_suggestion_dismissed or (
            state.vlm_suggestion_class_id is None and state.vlm_suggestion_name is None
        ):
            raise RefusedError
        self._add(
            cb.vlm_dismiss(
                self.view(),
                class_name=state.vlm_suggestion_name,
                class_id_hint=state.vlm_suggestion_class_id,
            )
        )

    def act_probe_dismiss(self) -> None:
        self._add(cb.probe_dismiss(self.view(), class_name='x', probe_analyser_id=1))

    def act_freeze(self) -> None:
        machine_class = freeze_machine_class(self.inputs(), REGISTRY, SETTINGS)
        self._add(
            cb.holdout_freeze(
                self.view(), freeze_id='f1', machine_class=machine_class, op_id=self._op('freeze')
            )
        )

    def act_import_label(self, cls: str, trust: str) -> None:
        if trust == 'suggestion':
            self._opinion('import_suggestion', cls, import_id='i1')
            return
        self._add(
            cbi.import_label(
                self.view(),
                import_id='i1',
                class_name=cls,
                class_id_hint=IDS[cls],
                op_id=self._op('import'),
            )
        )

    # ----- region acts ---------------------------------------------------------------
    def _write_region(self, call: Any) -> None:
        try:
            write = call()
        except RegionBoxWriteError as exc:
            raise RefusedError from exc
        self._add(write.drafts)
        self.item = dataclasses.replace(self.item, box_seq=write.box_seq)

    def _put(self, edits: list[BoxEdit]) -> None:
        self._write_region(
            lambda: cbr.put_regions(
                self.view(),
                self.shown(),
                edits,
                box_seq=self.item.box_seq,
                region_seq=self.fold().state.region_seen_seq,
                op_id=self._op('region_edit'),
            )
        )

    def _keep(self) -> list[BoxEdit]:
        return [BoxEdit(b.box_id) for b in self.shown()]

    def _element(self, idx: int, **patch: Any) -> list[BoxEdit]:
        edits = self._keep()
        if idx >= len(edits):
            raise RefusedError
        edits[idx] = dataclasses.replace(edits[idx], **patch)
        return edits

    def act_box_new(self, bbox: tuple[float, ...]) -> None:
        self._put([*self._keep(), BoxEdit(bbox=bbox)])  # type: ignore[arg-type]

    def act_box_state(self, idx: int, state: str) -> None:
        self._put(self._element(idx, state=state))

    def act_box_text(self, idx: int, text: str) -> None:
        self._put(self._element(idx, text=text))

    def act_box_move(self, idx: int, bbox: tuple[float, ...]) -> None:
        self._put(self._element(idx, bbox=bbox))

    def act_box_delete(self, idx: int) -> None:
        edits = self._keep()
        if idx >= len(edits):
            raise RefusedError
        del edits[idx]
        self._put(edits)

    def act_box_none(self) -> None:
        self._put([])

    def act_box_batch_state(self, idxs: list[int], state: str) -> None:
        shown = self.shown()
        if any(i >= len(shown) for i in idxs):
            raise RefusedError
        ids = [shown[i].box_id for i in idxs]
        self._write_region(
            lambda: cbr.batch_box_state(
                self.view(),
                shown,
                ids,
                state,
                box_seq=self.item.box_seq,
                region_seq=self.fold().state.region_seen_seq,
                op_id=self._op('region_edit'),
            )
        )

    def act_region_status(self, status: str) -> None:
        self._write_region(
            lambda: cbr.set_region_status(
                self.view(),
                self.shown(),
                status,
                box_seq=self.item.box_seq,
                region_seq=self.fold().state.region_seen_seq,
                op_id=self._op('region_edit'),
            )
        )

    def act_geometry(self, bbox: tuple[float, ...]) -> None:
        slack = 1e-4
        outside = [
            b
            for b in self.shown()
            if not (
                b.bbox_norm[0] >= bbox[0] - slack
                and b.bbox_norm[1] >= bbox[1] - slack
                and b.bbox_norm[2] <= bbox[2] + slack
                and b.bbox_norm[3] <= bbox[3] + slack
            )
        ]
        try:
            drafts = cbr.set_geometry(
                self.view(),
                bbox,
                drop=outside,
                region_seq=self.fold().state.region_seen_seq,
                op_id=self._op('geometry'),
            )
        except ItemBoxError as exc:
            raise RefusedError from exc
        self._add(drafts)

    # ----- machine acts --------------------------------------------------------------
    def act_redetect(self, shift: float) -> None:
        x1, y1, x2, y2 = BOX
        box = (x1 + shift, y1 + shift, x2 + shift, y2 + shift)
        self.detections = [detection(bbox=box, at=self.tick(), class_name='car')]

    def act_vlm(self, outcome: str, cls: str) -> None:
        if outcome == 'matched':
            self._opinion(
                'vlm', cls, outcome='matched', confidence=VLM_SCORE, confidence_cat='high'
            )
        elif outcome == 'unmatched':
            self._opinion('vlm', None, outcome='unmatched', raw_class=cls)
        elif outcome == 'new_class_pending':
            self._opinion('vlm', None, outcome='new_class_pending', proposed_class=cls)
        else:
            self._opinion('vlm', None, outcome='empty', empty_reason='no_answer')

    def act_propagate(self, cls: str) -> None:
        self._opinion('assist_propagated', cls, op_id='prop')

    def act_open_vocab_target(self, cls: str) -> None:
        self._opinion('open_vocab_target', cls)

    def act_classifier(self, cls: str, low: bool) -> None:
        self._opinion('classifier', cls, outcome='low_conf' if low else 'model', confidence=0.8)

    def act_adopt(self) -> None:
        self._opinion('adopted_detector', 'car', op_id='adopt', confidence=0.9)

    def act_region_write(self, boxes: list[tuple[float, ...]], state: str) -> None:
        raise NotImplementedError  # replaced by the harness (needs the document's minted ids)

    def region_write_with(self, machine_boxes: list[RegionBox], *, bump_seq: bool = True) -> None:
        """A region-stage write: the machine boxes are replaced (human boxes are claims)."""
        seq = self.item.region_seq + 1
        new = [
            MachineBox(
                box_id=b.box_id,
                bbox=b.bbox_norm,
                state=b.state,
                region_seq=seq,
                score=b.score,
                detector=b.detector,
                detector_version=b.detector_version,
                source=b.source,
            )
            for b in machine_boxes
        ]
        top = max((int(b.box_id[1:]) for b in machine_boxes), default=0) if bump_seq else 0
        self.item = dataclasses.replace(
            self.item, region_seq=seq, box_seq=max(self.item.box_seq, top)
        )
        self.boxes = [*self.boxes, *new]  # generations are kept (fold_boxes picks the one seen)
        stubs = [RegionBox(box_id=b.box_id, bbox_norm=b.bbox, state=b.state) for b in new]
        status = derive_status(stubs, empty_status=RegionStatus.NO_REGION_BOX).value
        self.region = RegionResult(status=status)
