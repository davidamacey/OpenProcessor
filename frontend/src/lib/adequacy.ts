/**
 * Single source of truth for class-adequacy chips. Used wherever a
 * class's validated_count is shown — sidebar, /classes, dashboard,
 * cluster cards — so the same count always renders the same color
 * and the same word.
 *
 * Thresholds are intentionally coarse: a class is either "ready"
 * (training-ready, ≥500), "low" (functional but undertrained), or
 * "critical" (almost no labeled data). Tweak in one place if the
 * model cohort grows.
 */

export const ADEQUACY_OK = 500;
export const ADEQUACY_LOW = 100;

export type AdequacyLevel = 'ok' | 'low' | 'critical';

export function adequacyLevel(n: number): AdequacyLevel {
  if (n >= ADEQUACY_OK) return 'ok';
  if (n >= ADEQUACY_LOW) return 'low';
  return 'critical';
}

/** Tailwind classes for an outlined chip showing the count. */
export function adequacyChipClass(n: number): string {
  switch (adequacyLevel(n)) {
    case 'ok':
      return 'bg-green-500/20 text-green-200 border-green-500/40';
    case 'low':
      return 'bg-orange-500/20 text-orange-200 border-orange-500/40';
    case 'critical':
      return 'bg-red-500/20 text-red-200 border-red-500/40';
  }
}

/** Short word for the level — "ok", "low", "critical". */
export function adequacyLabel(n: number): AdequacyLevel {
  return adequacyLevel(n);
}

/** Full sentence for tooltips: "412 validated · low (need ≥500)". */
export function adequacyTooltip(n: number): string {
  const level = adequacyLevel(n);
  if (level === 'ok') return `${n} validated · ok`;
  if (level === 'low') return `${n} validated · low (need ≥${ADEQUACY_OK})`;
  return `${n} validated · critical (need ≥${ADEQUACY_LOW})`;
}
