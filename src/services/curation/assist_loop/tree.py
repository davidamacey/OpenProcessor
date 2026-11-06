"""The loop's tree as the tables describe it: what a slice rebuilds from at its start.

A node is a row of ``assist_loop_nodes``; the answers of a node are the proposals of the items
selected for it (``assist_loop_selections`` names the node, ``assist_loop_proposals`` the
answer); a selected item with no proposal is *pending*: asked, answer not on record. Nothing is
replayed from a log. :func:`build_tree` is the only reader of the rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.services.curation.assist.proposal import ClassAnswer


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from src.db.repos.assist_loop_record import NodeRow, ProposalRow, SelectionRow

VLM = 'vlm'
"""Budget pool of the primary VLM's requests."""
AUDIT = 'audit'
"""Budget pool of the second-opinion audit's requests."""
ASK_PURPOSES = frozenset({'node_draw', 'node_topup', 'redraw'})
"""Selection purposes of the primary VLM's asks (the audit's own are ``audit``)."""
DECISION_KEYS = (
    'sampled',
    'reason',
    'dissent',
    'need',
    'vetoed',
    'drift',
)


@dataclass
class NodeRecord:
    """What the tables say about one node."""

    path: str
    parent: str | None
    level: int
    n: int
    """Eligible members at snapshot."""
    member_count: int
    digest: str
    snapshot_key: str
    step: str = 'open'
    answers: dict[str, ClassAnswer] = field(default_factory=dict)
    """Usable VLM answers by crop id (asked here, or reused from the parent)."""
    failed: set[str] = field(default_factory=set)
    """Asked, no usable answer."""
    pending: list[str] = field(default_factory=list)
    """Selected, answer not on record: in flight, or lost in a crash."""
    decision: dict[str, Any] | None = None
    children: list[str] | None = None
    child_n: dict[str, int] = field(default_factory=dict)
    crops: int = 0
    """VLM crops selected for this node."""
    claim: int | None = None
    """Crops the scheduler admitted the node against; ``None`` until it is admitted."""
    propagate_begun: bool = False
    propagated: dict[str, Any] | None = None
    label_op_id: str | None = None


@dataclass
class LoopTree:
    nodes: dict[str, NodeRecord] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    """The audit state kept on the iteration row (see :mod:`~.audit`)."""
    usage_missing: int = 0
    """Settled calls over all slices whose response had no usage object."""

    def node(self, path: str) -> NodeRecord:
        try:
            return self.nodes[path]
        except KeyError:
            msg = f'no node {path} in the loop'
            raise ValueError(msg) from None


def decision_columns(fields: Mapping[str, Any]) -> dict[str, Any]:
    """The node columns that store a gate result. ``distribution`` carries the votes, the
    new-class votes and the few fields no column exists for."""
    detail = {k: fields[k] for k in DECISION_KEYS if fields.get(k) is not None}
    return {
        'class_name': fields.get('class_name'),
        'answered': fields.get('answered'),
        'agree': fields.get('agree'),
        'bound': fields.get('bound'),
        'distribution': {
            'votes': [list(v) for v in fields.get('distribution') or []],
            'new_class_proposals': [list(v) for v in fields.get('new_class_proposals') or []],
            'detail': detail,
        },
    }


def decision_of(kind: str, row: NodeRow) -> dict[str, Any]:
    """The decision dict a node row stands for (the inverse of :func:`decision_columns`)."""
    stored = row.distribution or {}
    return {
        'kind': kind,
        'class_name': row.class_name,
        'answered': row.answered,
        'agree': row.agree,
        'bound': row.bound,
        'distribution': [tuple(v) for v in stored.get('votes') or []],
        'new_class_proposals': [tuple(v) for v in stored.get('new_class_proposals') or []],
        **(stored.get('detail') or {}),
    }


def answer_of(row: ProposalRow) -> ClassAnswer:
    return ClassAnswer(
        class_name=row.class_name, proposed=row.new_class_name or '', crop_id=row.item_id
    )


def build_tree(
    nodes: Iterable[NodeRow],
    selections: Iterable[SelectionRow],
    proposals: Iterable[ProposalRow],
    *,
    counts: Mapping[str, Any] | None = None,
) -> LoopTree:
    stored = counts or {}
    tree = LoopTree(
        audit=dict(stored.get('audit') or {}), usage_missing=int(stored.get('usage_missing', 0))
    )
    for row in nodes:
        record = NodeRecord(
            path=row.path,
            parent=row.parent_path,
            level=row.level,
            n=row.eligible_count,
            member_count=row.member_count,
            digest=row.membership_digest,
            snapshot_key=row.snapshot_storage_key,
            step=row.step,
            claim=row.admitted_crops,
            propagate_begun=row.step == 'propagating',
            label_op_id=None if row.label_op_id is None else str(row.label_op_id),
        )
        if row.step == 'done':
            kind = str(row.decision)
            record.decision = decision_of(kind, row)
            if kind == 'pass':
                skipped = dict(row.skipped)
                record.propagated = {
                    'written': row.items_propagated,
                    'class_name': row.class_name,
                    'drift': skipped.pop('drift', 0.0),
                    'skipped': skipped,
                }
        elif row.step in ('gated', 'propagating'):
            record.decision = decision_of('pass', row)
        tree.nodes[row.path] = record
    for record in tree.nodes.values():
        parent = tree.nodes.get(record.parent or '')
        if parent is not None:
            parent.children = [*(parent.children or []), record.path]
            parent.child_n[record.path] = record.n
    for record in tree.nodes.values():
        if record.children:
            record.children.sort()
    by_item = {p.item_id: p for p in proposals if p.role == 'primary'}
    for sel in selections:
        node = tree.nodes.get(sel.node_path or '')
        if node is None or sel.purpose not in ASK_PURPOSES:
            continue
        node.crops += 1
        proposal = by_item.get(sel.item_id)
        if proposal is None:
            node.pending.append(sel.item_id)
        elif proposal.outcome == 'answered':
            node.answers[sel.item_id] = answer_of(proposal)
        else:
            node.failed.add(sel.item_id)
    return tree


__all__ = [
    'ASK_PURPOSES',
    'AUDIT',
    'VLM',
    'LoopTree',
    'NodeRecord',
    'answer_of',
    'build_tree',
    'decision_columns',
    'decision_of',
]
