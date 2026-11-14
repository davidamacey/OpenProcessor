"""The nested-document limit of one item document, and the one rule every writer applies.

OpenSearch k-NN issue 3623 (https://github.com/opensearch-project/k-NN/issues/3623; fixed by
PR 3624 for 3.10, which is not released; every 3.x is affected): on an index with derived source
for vectors, a parent document with 150 or more nested Lucene documents, counted across ALL its
nested fields, is read back with ANOTHER document's nested vectors, and any partial update,
scripted update, ``update_by_query`` or reindex then writes those wrong vectors to disk.

Measured on 3.6.0 with the real items index body: at 75 boxes per item (``region_boxes`` plus
``region_box_embeddings`` = 150 nested documents) 96% of the items returned wrong box vectors on
search, GET and mget; at 74 boxes (148) none did. Not measured separately: the nested-in-nested
``assist.boxes[].attributes`` entries. They are counted here because Lucene's block layout gives
every nested object, at any depth, its own document in the parent's block.

Nested documents one item document holds: ``region_boxes`` (one per box), every box-vector
nested list (``region_box_embeddings`` and one ``boxvec_<key>`` per further box space: up to one
entry per box each), ``assist.attributes`` and ``assist.boxes`` with their ``attributes``. The
rule: a write whose result holds :data:`NESTED_DOCS_LIMIT` or more is refused, and so is every
write on an item that already does, except the one repair write (:func:`guard_write`).

Pure: no I/O. The structural fix (box vectors leave the nested field) replaces this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.nested_limit import NESTED_DOCS_LIMIT, max_boxes_per_item
from src.config.region_fields import RegionFields, get_region_fields
from src.domain.embedding_spaces import BOX_SPACE_NESTED_PREFIX


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

ASSIST_FIELD = 'assist'
"""The item field the assist runs write (``src/clients/curation_opensearch/assist_fields.py``)."""

OVER_LIMIT_SCRIPT_REASON = 'op_nested_limit'
"""The message a guarded bulk script throws on an over-limit item (:func:`painless_guard`)."""

ERROR_CODE = 'item_over_nested_limit'


class NestedLimitError(Exception):
    """A write refused because its item holds, or would hold, too many nested documents."""

    def __init__(
        self,
        doc_id: str,
        *,
        nested_docs: int,
        boxes: int,
        reason: str,
    ) -> None:
        self.doc_id = doc_id
        self.nested_docs = nested_docs
        self.boxes = boxes
        self.reason = reason
        super().__init__(
            f'{doc_id}: {reason} ({nested_docs} nested documents, {boxes} boxes; '
            f'the limit is {NESTED_DOCS_LIMIT}, see OpenSearch k-NN issue 3623)'
        )

    def detail(self) -> dict[str, Any]:
        """The 422 body: the limit and the item's numbers."""
        return {
            'error': ERROR_CODE,
            'message': str(self),
            'limit': NESTED_DOCS_LIMIT,
            'nested_docs': self.nested_docs,
            'boxes': self.boxes,
            'crop_id': self.doc_id,
        }


def is_box_vector_path(path: str, F: RegionFields | None = None) -> bool:
    """True for the nested field of a box-vector space: the 0.6.0 one or a ``boxvec_<key>``."""
    F = F or get_region_fields()
    return path == F.box_embeddings or path.startswith(BOX_SPACE_NESTED_PREFIX)


def box_vector_paths(excludes: Collection[str], F: RegionFields | None = None) -> frozenset[str]:
    """Every box-vector nested path of an item, from a ``_source`` exclude list.

    The 0.6.0 path is always one (the default box space keeps it); a further space adds its own.
    """
    F = F or get_region_fields()
    return frozenset({F.box_embeddings, *(p for p in excludes if is_box_vector_path(p, F))})


def _length(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _assist_docs(src: Mapping[str, Any]) -> int:
    assist = src.get(ASSIST_FIELD)
    if not isinstance(assist, dict):
        return 0
    total = _length(assist.get('attributes'))
    boxes = assist.get('boxes')
    if isinstance(boxes, list):
        for box in boxes:
            total += 1 + (_length(box.get('attributes')) if isinstance(box, dict) else 0)
    return total


def nested_doc_count(
    src: Mapping[str, Any],
    *,
    unread_box_vector_paths: Collection[str] = (),
    F: RegionFields | None = None,
) -> int:
    """The nested documents the item document ``src`` holds.

    ``unread_box_vector_paths`` are box-vector lists the read left out of ``src`` (the OCC reads
    exclude every vector): each counts one entry per box, the most it can hold. A list that is
    present counts its real length.
    """
    F = F or get_region_fields()
    boxes = _length(src.get(F.boxes))
    total = boxes
    paths = {k for k in src if is_box_vector_path(k, F)} | set(unread_box_vector_paths)
    for path in paths:
        total += _length(src[path]) if path in src else boxes
    return total + _assist_docs(src)


def _merged(current: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """``current`` after an OpenSearch partial-update ``doc``: objects merge, lists replace."""
    out = dict(current)
    for key, value in update.items():
        old = out.get(key)
        if isinstance(value, dict) and isinstance(old, dict):
            out[key] = _merged(old, value)
        else:
            out[key] = value
    return out


def guard_write(
    doc_id: str,
    current: Mapping[str, Any],
    update: Mapping[str, Any],
    *,
    excludes: Collection[str] = (),
    F: RegionFields | None = None,
) -> dict[str, Any]:
    """The update to write for ``update`` on ``current``, or :class:`NestedLimitError`.

    * A result of :data:`NESTED_DOCS_LIMIT` nested documents or more is refused.
    * On an item that is already at or over the limit every other write is refused too: a
      partial update rewrites the whole document, and the vectors read back from such an item
      are another document's.
    * The one exception, the repair write: a write that sets the box list and brings the item
      under the limit (counting one vector per box, as for any write) also empties every
      box-vector list, so no untrustworthy vector is written again. The box vectors are then
      missing and the box embed stage and backfill (``missing_boxes``) re-create them. The
      returned update carries that clearing.

    ``excludes`` is the ``_source`` exclude list the item was read with.
    """
    if not update:
        return dict(update)
    F = F or get_region_fields()
    paths = box_vector_paths(excludes, F)
    after = _merged(current, update)
    before_docs = nested_doc_count(current, unread_box_vector_paths=paths, F=F)
    after_docs = nested_doc_count(after, unread_box_vector_paths=paths, F=F)
    boxes = _length(after.get(F.boxes))
    if before_docs < NESTED_DOCS_LIMIT:
        if after_docs >= NESTED_DOCS_LIMIT:
            raise NestedLimitError(
                doc_id,
                nested_docs=after_docs,
                boxes=boxes,
                reason='the write would exceed the limit',
            )
        return dict(update)
    if F.boxes in update and after_docs < NESTED_DOCS_LIMIT:
        return {**update, **{p: [] for p in paths}}
    raise NestedLimitError(
        doc_id,
        nested_docs=max(before_docs, after_docs),
        boxes=boxes,
        reason=(
            'the item already holds too many nested documents; only a box-list write that '
            'brings it under the limit is accepted'
        ),
    )


def painless_guard(body: str, *, on_over: str, F: RegionFields | None = None) -> str:
    """``body``, a Painless update script, run only on an item under :data:`NESTED_DOCS_LIMIT`.

    The count reads list sizes from ``ctx._source`` (the vectors in them are not read). ``on_over``
    is ``'noop'`` (``ctx.op = 'noop'``: an ``update_by_query`` counts it in ``noops``) or
    ``'throw'`` (a bulk item fails with :data:`OVER_LIMIT_SCRIPT_REASON` in its cause, which
    :func:`is_over_limit_bulk_error` recognises, so a job can count it apart from real failures).
    """
    F = F or get_region_fields()
    refuse = {
        'noop': "ctx.op = 'noop';",
        'throw': f"throw new IllegalStateException('{OVER_LIMIT_SCRIPT_REASON}');",
    }[on_over]
    count = f"""
int op_nested = 0;
def op_src = ctx._source;
def op_boxes = op_src.get('{F.boxes}');
if (op_boxes != null) {{ op_nested += op_boxes.size(); }}
for (def op_key : op_src.keySet()) {{
  if (op_key == '{F.box_embeddings}' || op_key.startsWith('{BOX_SPACE_NESTED_PREFIX}')) {{
    def op_list = op_src.get(op_key);
    if (op_list != null) {{ op_nested += op_list.size(); }}
  }}
}}
def op_assist = op_src.get('{ASSIST_FIELD}');
if (op_assist != null) {{
  def op_attrs = op_assist.get('attributes');
  if (op_attrs != null) {{ op_nested += op_attrs.size(); }}
  def op_abx = op_assist.get('boxes');
  if (op_abx != null) {{
    for (def op_ab : op_abx) {{
      op_nested += 1;
      def op_abattrs = op_ab.get('attributes');
      if (op_abattrs != null) {{ op_nested += op_abattrs.size(); }}
    }}
  }}
}}
"""
    return f'{count}if (op_nested >= {NESTED_DOCS_LIMIT}) {{ {refuse} }} else {{\n{body}\n}}'


def is_over_limit_bulk_error(item_error: Mapping[str, Any]) -> bool:
    """True for a ``_bulk`` item error a ``'throw'`` :func:`painless_guard` produced."""
    cause = item_error.get('caused_by') or {}
    return OVER_LIMIT_SCRIPT_REASON in str(cause.get('reason', ''))


def over_limit_ids(
    docs: Mapping[str, Mapping[str, Any]],
    *,
    unread_box_vector_paths: Collection[str] = (),
    F: RegionFields | None = None,
) -> dict[str, tuple[int, int]]:
    """``{id: (nested_docs, boxes)}`` for each source in ``docs`` at or over the limit."""
    F = F or get_region_fields()
    out: dict[str, tuple[int, int]] = {}
    for doc_id, src in docs.items():
        n = nested_doc_count(src, unread_box_vector_paths=unread_box_vector_paths, F=F)
        if n >= NESTED_DOCS_LIMIT:
            out[doc_id] = (n, _length(src.get(F.boxes)))
    return out


__all__ = [
    'ASSIST_FIELD',
    'ERROR_CODE',
    'NESTED_DOCS_LIMIT',
    'OVER_LIMIT_SCRIPT_REASON',
    'NestedLimitError',
    'box_vector_paths',
    'guard_write',
    'is_box_vector_path',
    'is_over_limit_bulk_error',
    'max_boxes_per_item',
    'nested_doc_count',
    'over_limit_ids',
    'painless_guard',
]
