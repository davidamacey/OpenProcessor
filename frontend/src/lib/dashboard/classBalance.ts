/**
 * DQ-m11 (docs/design/data-quality-pass-2026-09-24.md): `/dashboard`'s
 * class-balance strip is `sort(validated_count desc).slice(0, 30)` — with
 * every class at `validated_count: 0` (the live state throughout the
 * audit: 0 `class_validated` dataset-wide), that's a tie across all 85
 * classes. `Array.prototype.sort` is stable (guaranteed since ES2019), so
 * a tie falls back to the server's own `per_class` order, which is
 * alphabetical — cutting the strip to the first 30 names alphabetically
 * hid some high-count classes regardless of how many crops they actually have.
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
  /** Served `trainable` (`GET {API_PREFIX}/stats/classes`): validated crops
   *  training can use, test holdout and excluded crops already left out. */
  trainable: number;
}

export interface ClassBalanceBar {
  class_id: number;
  class_name: string;
  validated: number;
  test: number;
  trainable: number;
  /** Bar width, 0..100, of `trainable` against the largest trainable count. */
  pct: number;
  tier: string;
}

export interface ClassBalanceView {
  bars: ClassBalanceBar[];
  /** Classes with 0 validated crops, collapsed to one line. */
  zeroCount: number;
  /** Classes with validated crops beyond `limit`, shown as "+N more". */
  moreCount: number;
}

/**
 * D2 (visual audit 2026-09-24): the balance chart drew a 2% bar for every
 * class at 0 (a wall of identical red "0" bars), silently dropped every
 * class past the first 30, and counted frozen test crops as validated.
 * Bars are now only the classes with validated crops, sized by the served
 * `trainable` with the served test-holdout count shown alongside (display
 * only, never subtracted client-side); zero classes collapse into one
 * count and the overflow into "+N more".
 */
export function buildClassBalance<T extends ClassBalanceRow & { adequacy: string }>(
  rows: readonly T[],
  holdout: Map<number, number>,
  limit = 30,
): ClassBalanceView {
  const sorted = sortClassBalance(rows);
  const nonZero = sorted.filter((r) => r.validated_count > 0);
  const zeroCount = sorted.length - nonZero.length;
  const shown = nonZero.slice(0, limit);
  const withTrainable = shown.map((r) => ({
    r,
    test: holdout.get(r.class_id) ?? 0,
    trainable: r.trainable,
  }));
  const max = Math.max(1, ...withTrainable.map((x) => x.trainable));
  return {
    bars: withTrainable.map(({ r, test, trainable }) => ({
      class_id: r.class_id,
      class_name: r.class_name,
      validated: r.validated_count,
      test,
      trainable,
      pct: Math.round((trainable / max) * 100),
      tier: r.adequacy,
    })),
    zeroCount,
    moreCount: nonZero.length - shown.length,
  };
}

export function sortClassBalance<T extends ClassBalanceRow>(rows: readonly T[]): T[] {
  return [...rows].sort(
    (a, b) =>
      b.validated_count - a.validated_count ||
      b.count - a.count ||
      a.class_name.localeCompare(b.class_name),
  );
}
