/**
 * DQ-m11 (docs/design/data-quality-pass-2026-09-24.md): `/dashboard`'s
 * class-balance strip is `sort(validated_count desc).slice(0, 30)` — with
 * every class at `validated_count: 0` (the live state throughout the
 * audit: 0 `class_validated` dataset-wide), that's a tie across all 85
 * classes. `Array.prototype.sort` is stable (guaranteed since ES2019), so
 * a tie falls back to the server's own `per_class` order, which is
 * alphabetical — cutting the strip to the first 30 names alphabetically
 * hid `pickup` and `suv` regardless of how many crops they actually have.
 *
 * `sortClassBalance` adds `count` (total crops with that class_id, served
 * alongside `validated_count`) as a tiebreaker before falling back to
 * name — so a validated_count tie surfaces the classes with the most
 * crops (the ones a curator most needs to see) rather than whatever
 * happened to sort first alphabetically.
 */

export interface ClassBalanceRow {
  class_id: number;
  class_name: string;
  count: number;
  validated_count: number;
}

export function sortClassBalance<T extends ClassBalanceRow>(rows: readonly T[]): T[] {
  return [...rows].sort(
    (a, b) =>
      b.validated_count - a.validated_count ||
      b.count - a.count ||
      a.class_name.localeCompare(b.class_name),
  );
}
