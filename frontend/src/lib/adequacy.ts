/**
 * Single source of truth for class-adequacy chip styling. Used wherever a
 * class's adequacy is shown — sidebar, /classes, dashboard, cluster cards —
 * so the same server-served tier always renders the same color and word.
 *
 * The tier itself (`block` | `warn` | `ok`) and the thresholds it was
 * computed from (`block_below` / `warn_below`) are served by the backend on
 * `GET {API_PREFIX}/classes` and `GET {API_PREFIX}/stats/classes`
 * (`RegistryClass.adequacy` / `StatsSummary.per_class[].adequacy`,
 * `classesStore.thresholds`) — this module never recomputes the tier from a
 * validated-count number itself.
 */

export type AdequacyLevel = 'block' | 'warn' | 'ok';

/** Tailwind classes for an outlined chip showing the count. Unknown/absent
 *  levels (a stale backend that hasn't shipped `adequacy` yet) render as a
 *  neutral chip rather than guessing a color. */
export function adequacyChipClass(level: string | null | undefined): string {
  switch (level) {
    case 'ok':
      return 'bg-green-500/20 text-green-200 border-green-500/40';
    case 'warn':
      return 'bg-orange-500/20 text-orange-200 border-orange-500/40';
    case 'block':
      return 'bg-red-500/20 text-red-200 border-red-500/40';
    default:
      return 'bg-zinc-800/40 text-zinc-400 border-zinc-700';
  }
}

/** Full sentence for tooltips: "412 validated · block". */
export function adequacyTooltip(level: string | null | undefined, count: number): string {
  return `${count} validated · ${level ?? 'unknown'}`;
}
