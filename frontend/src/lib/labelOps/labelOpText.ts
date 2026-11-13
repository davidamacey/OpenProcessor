/** The words of a recorded label operation in the history list. Every value is the served one. */
import { formatTimestamp } from '$lib/format/date';
import { plural } from '$lib/format/plural';
import { humanizeId } from '$lib/humanizeId';
import type { LabelOp } from '$lib/types_labelOps';

/** "Labelled 24 items as bottle". The count is the served `count`, or `selected` while a write
 *  is still running. */
export function labelOpTitle(op: LabelOp): string {
  const items = plural(op.count ?? op.selected, 'item');
  const cls = op.class_name;
  switch (op.kind) {
    case 'label':
      return cls ? `Labelled ${items} as ${cls}` : `Labelled ${items}`;
    case 'move':
      return cls ? `Moved ${items} to ${cls}` : `Moved ${items}`;
    case 'adopt_detector':
      return `Adopted the detector's classes for ${items}`;
    case 'assist_loop_propagate':
      return cls
        ? `Assist loop labelled ${items} as ${cls}`
        : `Assist loop labelled ${items}`;
    case 'assist_loop_review':
      return `Assist loop review of ${items}`;
  }
}

/** "from cluster cl-3 · by a person · 2026-11-12 23:06:00 UTC"; what was not served is left out. */
export function labelOpSubtitle(op: LabelOp): string {
  return [
    op.target_cluster_uid ? `from cluster ${op.target_cluster_uid}` : null,
    op.label_source
      ? `by ${op.label_source === 'human' ? 'a person' : humanizeId(op.label_source)}`
      : null,
    formatTimestamp(op.at),
  ]
    .filter((p): p is string => p !== null)
    .join(' · ');
}
