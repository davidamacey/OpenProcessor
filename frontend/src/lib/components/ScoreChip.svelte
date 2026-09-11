<script lang="ts">
  /**
   * Small badge for a per-crop curation score (curation-strategy plan
   * §4/§5 — docs/curation-strategy-plan-2026-09.md), e.g. `mistakenness`.
   *
   * Closest existing analog is `DetectorChip.svelte` — same rounded
   * border/bg/text triple, same font-mono + title-tooltip convention —
   * but scores get their own color family (fuchsia) so they read as a
   * distinct kind of provenance from detector chips (which colors by
   * *which model*, not *how confident/anomalous*).
   *
   * The value is rendered as a percentage (every score in the plan is
   * normalized to [0,1] — mistakenness, uniqueness, representativeness).
   * `method`/`version` show on hover via the `title` attribute, matching
   * DetectorChip's `version` convention, so the chip strip stays compact
   * even with several scores side by side.
   */

  interface Props {
    /** Short label, e.g. 'mistakenness'. */
    label: string;
    /** Score value in [0,1]. */
    value: number;
    /** Scorer id/name, e.g. 'mistakenness'. Shown in the tooltip only. */
    method?: string | null;
    /** Scorer version, e.g. 'v1'. Shown as `method@version` on hover. */
    version?: string | null;
    size?: 'sm' | 'md';
  }

  let { label, value, method = null, version = null, size = 'md' }: Props = $props();

  const pct = $derived(`${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`);
  const tooltip = $derived.by(() => {
    if (!method) return `${label}: ${pct}`;
    return `${label}: ${pct} · ${method}${version ? `@${version}` : ''}`;
  });
  const sizeCls = $derived(
    size === 'sm' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2 py-0.5 text-[11px]',
  );
</script>

<span
  class="inline-flex items-center gap-1 rounded border border-fuchsia-500/50 bg-fuchsia-500/10 text-fuchsia-200 {sizeCls} font-mono"
  title={tooltip}
>
  <span class="uppercase tracking-wide opacity-80">{label}</span>
  <span>{pct}</span>
</span>
