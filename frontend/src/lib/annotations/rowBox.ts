/**
 * Which box a card/row is about. A region browse row is the full item plus
 * `region_box_id` (the box that matched the filters); an item-level row
 * has none. Every per-box value a card shows (score, text, detector,
 * thumbnail) comes from this box, never from an item-level scalar — there
 * is none on the wire.
 */
import type { SlotBox, SlotData } from './types';

/** The row's own box, or null when the row is item-level or the id is
 *  not in the served list. */
export function rowBoxOf(
  data: SlotData | null | undefined,
  regionBoxId: string | null | undefined,
): SlotBox | null {
  if (regionBoxId == null) return null;
  return data?.subBoxes?.find((b) => b.boxId === regionBoxId) ?? null;
}

/** The box a card should describe: the row's own box when it has one,
 *  otherwise the item's first box in served display order. */
export function displayBoxOf(
  data: SlotData | null | undefined,
  regionBoxId: string | null | undefined,
): SlotBox | null {
  return rowBoxOf(data, regionBoxId) ?? data?.subBoxes?.[0] ?? null;
}
