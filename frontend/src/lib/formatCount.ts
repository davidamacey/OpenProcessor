/**
 * OpenProcessor d5343cb: several export-status fields (`object_count`,
 * `split_object_counts`, `images_with_unlabeled_items`, …) are `null` on
 * an export written before the backend recorded them — genuinely
 * "unknown", not zero. Rendering `0` there would claim something false
 * (e.g. "0 objects" on an export that has objects, just didn't record
 * the count). Every render site for one of these fields must go through
 * this instead of a bare `.toLocaleString()`.
 */
export function formatCount(n: number | null | undefined): string {
  return n == null ? '—' : n.toLocaleString();
}
