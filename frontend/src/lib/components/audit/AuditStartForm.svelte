<!--
  Draw an audit sample. Both inputs are optional: an empty one is not sent
  and the server default applies. A refusal (409 audit_no_candidates, 422)
  is shown as served; the served per-class strata of the last draw follow.
-->
<script lang="ts">
  import type { AuditController } from '$lib/labelConfirmation/auditController.svelte';

  interface Props {
    audit: AuditController;
  }
  let { audit }: Props = $props();

  function num(e: Event): number | null {
    const v = (e.currentTarget as HTMLInputElement).valueAsNumber;
    return Number.isNaN(v) ? null : v;
  }
</script>

<section class="surface flex flex-col gap-3 p-4" data-testid="audit-start">
  <div>
    <h2 class="text-base font-semibold">Draw a sample</h2>
    <p class="text-xs text-zinc-400">
      Picks machine-labelled crops that no human has validated, spread over the detector's
      classes. Labeling them (in Review or from the list below) is what measures the
      detector and the VLM. Drawing a sample never changes a class.
    </p>
  </div>
  <div class="flex flex-wrap items-end gap-3">
    <label class="flex flex-col gap-1 text-xs text-zinc-300">
      Sample size
      <input
        class="input w-32"
        type="number"
        min="1"
        step="50"
        placeholder="server default"
        value={audit.sampleSize ?? ''}
        oninput={(e) => (audit.sampleSize = num(e))}
        data-testid="audit-sample-size"
      />
    </label>
    <label class="flex flex-col gap-1 text-xs text-zinc-300">
      Crops per class
      <input
        class="input w-32"
        type="number"
        min="1"
        step="5"
        placeholder="server default"
        value={audit.minPerClass ?? ''}
        oninput={(e) => (audit.minPerClass = num(e))}
        data-testid="audit-min-per-class"
      />
    </label>
    <button
      type="button"
      class="btn btn-primary"
      disabled={audit.starting}
      onclick={() => void audit.start()}
      data-testid="audit-start-button"
    >
      {audit.starting ? 'Drawing…' : 'Draw sample'}
    </button>
  </div>

  {#if audit.startLines.length > 0}
    <div
      class="rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200"
      data-testid="audit-start-error"
    >
      {#each audit.startLines as line (line)}
        <p>{line}</p>
      {/each}
    </div>
  {/if}

  {#if audit.started}
    {@const s = audit.started}
    <div class="flex flex-col gap-1 text-xs text-zinc-300" data-testid="audit-started">
      <p>Drew {s.sampled} of the {s.requested} requested crops.</p>
      <ul class="flex flex-wrap gap-1.5">
        {#each s.strata as st (st.detector_class)}
          <li
            class="rounded border px-1.5 py-0.5 {st.short_of_floor
              ? 'border-amber-500/40 bg-amber-500/10 text-amber-200'
              : 'border-zinc-700 text-zinc-300'}"
            title={st.short_of_floor
              ? `The budget left ${st.detector_class} fewer than ${s.min_per_class} crops although more exist.`
              : ''}
            data-testid="audit-stratum"
            data-short={st.short_of_floor}
          >
            {st.detector_class}: {st.sampled} of {st.available}
          </li>
        {/each}
      </ul>
    </div>
  {/if}
</section>
