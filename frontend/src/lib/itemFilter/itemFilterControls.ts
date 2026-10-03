/**
 * The shared item filter's controls, as specs `ServedFilterField` draws.
 * A route that serves its own `filter_specs` entry for a param (label,
 * options, bounds, help text) overrides the local one; the local specs are
 * the fallback for routes that serve none (`/clusters`, `/export`) and carry
 * only what the contract fixes: the enum value lists (`types_itemFilter`).
 */
import { humanizeId } from '$lib/humanizeId';
import type { ReviewFilterSpec } from '$lib/api';
import { EMBEDDING_STATES, ITEM_ORIGINS, REVIEW_STATUSES } from '$lib/types_itemFilter';

const opts = (values: readonly string[]) =>
  values.map((value) => ({ value, label: humanizeId(value) }));

const base = {
  options: [] as ReviewFilterSpec['options'],
  min: null,
  max: null,
  description: '',
  default: null,
  allows_unset: true,
};

export const ITEM_FILTER_CONTROLS: ReviewFilterSpec[] = [
  { ...base, param: 'class_name', kind: 'class_names', label: 'Class' },
  { ...base, param: 'exclude_class_name', kind: 'class_names', label: 'Not class' },
  {
    ...base,
    param: 'conf_min',
    kind: 'number',
    label: 'Confidence from',
    min: 0,
    max: 1,
  },
  { ...base, param: 'conf_max', kind: 'number', label: 'to', min: 0, max: 1 },
  {
    ...base,
    param: 'min_area',
    kind: 'number',
    label: 'Area from',
    min: 0,
    max: 1,
    description: 'Fraction of the image the item covers.',
  },
  { ...base, param: 'max_area', kind: 'number', label: 'to', min: 0, max: 1 },
  {
    ...base,
    param: 'max_rank',
    kind: 'integer',
    label: 'Largest N per image',
    min: 1,
  },
  {
    ...base,
    param: 'origin',
    kind: 'multi_enum',
    label: 'Origin',
    options: opts(ITEM_ORIGINS),
  },
  {
    ...base,
    param: 'embedding_state',
    kind: 'multi_enum',
    label: 'Embedding',
    options: opts(EMBEDDING_STATES),
  },
  {
    ...base,
    param: 'review_status',
    kind: 'multi_enum',
    label: 'Review status',
    options: opts(REVIEW_STATUSES),
  },
];

export const OPEN_VOCAB_CONTROLS: ReviewFilterSpec[] = [
  { ...base, param: 'open_vocab_set', kind: 'text', label: 'Open-vocabulary set' },
  { ...base, param: 'source_prompt', kind: 'text', label: 'Prompt' },
];

/** The params of the shared item filter (both lists above). */
export const ITEM_FILTER_PARAMS: ReadonlySet<string> = new Set(
  [...ITEM_FILTER_CONTROLS, ...OPEN_VOCAB_CONTROLS].map((c) => c.param),
);

/** The controls to draw: each local spec replaced by the served one when the
 *  route serves an entry for that param. */
export function resolveControls(
  served: readonly ReviewFilterSpec[],
  showOpenVocab: boolean,
): ReviewFilterSpec[] {
  const byParam = new Map(served.map((s) => [s.param, s]));
  const local = showOpenVocab
    ? [...ITEM_FILTER_CONTROLS, ...OPEN_VOCAB_CONTROLS]
    : ITEM_FILTER_CONTROLS;
  return local.map((c) => byParam.get(c.param) ?? c);
}
